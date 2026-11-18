r"""Moteur de recommandation hybride adaptatif et gestion des cas difficiles (Phases 5 & 6).

Implémente les Sections 8 et 9 :
1. **Pondération adaptative (Section 8.3 & Section 9 - Cold Start / Sparse Data)** :
   Adapte dynamiquement les poids $(\alpha_{\text{als}}, \delta_{\text{session}}, \beta_{\text{content}}, \gamma_{\text{pop}})$
   selon la richesse de l'historique utilisateur :
   - **Cold User sans contexte** ($0$ interaction et $0$ item en session) :
     Repli pur sur Popularité récente diversifiée par catégorie.
   - **Cold User avec session active** (`recent_items` non vide) :
     Priorité à Session-Based ($0.55$) + Content-Based ($0.30$) + Popularity ($0.15$).
   - **Sparse User** ($1 \le \text{interactions} < \text{sparse\_user\_threshold}$) :
     Renforce Session-Based et Content-Based par rapport à ALS.
   - **Warm User** ($\ge \text{sparse\_user\_threshold}$) :
     Utilise les poids nominaux configurés ($\alpha, \delta, \beta, \gamma$).
2. **Learning to Rank (Section 8.4)** :
   Lorsque `ranker` (`LightGBMRankerModel`) et `feature_builder` (`RankingFeatureBuilder`)
   sont attachés et `use_ltr=True`, reclasse les candidats générés via LambdaRank.
3. **Gestion de la Long Tail et Diversité (Section 9)** :
   Applique une pénalité de redondance catégorielle gloutonne lors de la sélection finale
   des top-N produits afin d'éviter que les 10 recommandations appartiennent toutes à la
   même sous-catégorie ultra-populaire.
4. **Politique métier sur les produits déjà achetés (Section 9)** :
   Exclut par défaut les produits déjà achetés (`exclude_already_purchased=True`), mais
   permet de les ré-inclure à la demande (consommables/renouvelables).
"""

from collections import Counter
from typing import Sequence

import numpy as np
import pandas as pd

from recommender.config.model_config import DEFAULT_HYBRID_CONFIG, HybridEngineConfig
from recommender.models.base import BaseRecommender, RecommendationItem
from recommender.models.collaborative import CollaborativeALSRecommender
from recommender.models.content_based import ContentBasedRecommender
from recommender.models.popularity import PopularityRecommender
from recommender.models.session_based import SessionBasedRecommender
from recommender.ranking.candidate_generator import CandidateGenerator
from recommender.ranking.feature_builder import RankingFeatureBuilder
from recommender.ranking.ranker import LightGBMRankerModel
from recommender.ranking.score_normalizer import normalize_scores


class HybridRecommender(BaseRecommender):
    """Moteur hybride combinant Popularity, Content-Based, ALS, Session-Based et LTR."""

    model_name: str = "hybrid_weighted"

    def __init__(
        self,
        popularity_model: PopularityRecommender,
        content_model: ContentBasedRecommender,
        collaborative_model: CollaborativeALSRecommender,
        session_model: SessionBasedRecommender,
        config: HybridEngineConfig = DEFAULT_HYBRID_CONFIG,
        feature_builder: RankingFeatureBuilder | None = None,
        ranker: LightGBMRankerModel | None = None,
        use_ltr: bool = False,
    ) -> None:
        self.popularity_model = popularity_model
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.session_model = session_model
        self.config = config
        self.feature_builder = feature_builder
        self.ranker = ranker
        self.use_ltr = use_ltr
        if use_ltr:
            self.model_name = "hybrid_ltr"

        self.candidate_generator = CandidateGenerator(
            popularity_model=popularity_model,
            content_model=content_model,
            collaborative_model=collaborative_model,
            session_model=session_model,
            config=config,
        )
        self.user_purchased_items_: dict[int, set[int]] = {}
        self.user_viewed_items_: dict[int, set[int]] = {}
        self.user_event_counts_: dict[int, int] = {}

    def fit(self, interactions_df: pd.DataFrame | None = None) -> "HybridRecommender":
        """Indexe de manière vectorisée l'historique d'achat et de consultation."""
        self.user_purchased_items_ = {}
        self.user_viewed_items_ = {}
        self.user_event_counts_ = {}

        if interactions_df is not None and not interactions_df.empty:
            cnt_col = (
                "total_interactions"
                if "total_interactions" in interactions_df.columns
                else None
            )
            if cnt_col:
                u_cnts = interactions_df.groupby("visitorid")[cnt_col].sum()
            else:
                u_cnts = interactions_df.groupby("visitorid").size()
            self.user_event_counts_ = {
                int(uid): int(c) for uid, c in u_cnts.items()
            }

            if "is_confirmed_purchase" in interactions_df.columns:
                tx_mask = interactions_df["is_confirmed_purchase"].astype(bool)
            elif "transaction_count" in interactions_df.columns:
                tx_mask = interactions_df["transaction_count"] > 0
            elif "event" in interactions_df.columns:
                tx_mask = interactions_df["event"] == "transaction"
            else:
                tx_mask = pd.Series(False, index=interactions_df.index)

            tx_sub = interactions_df.loc[tx_mask, ["visitorid", "itemid"]]
            for uid, iid in zip(
                tx_sub["visitorid"].to_numpy(), tx_sub["itemid"].to_numpy()
            ):
                self.user_purchased_items_.setdefault(int(uid), set()).add(int(iid))

            if self.config.exclude_already_viewed:
                for uid, iid in zip(
                    interactions_df["visitorid"].to_numpy(),
                    interactions_df["itemid"].to_numpy(),
                ):
                    self.user_viewed_items_.setdefault(int(uid), set()).add(int(iid))
        return self

    def _resolve_dynamic_weights(
        self,
        user_id: int | None,
        recent_items: Sequence[int] | None,
    ) -> tuple[float, float, float, float]:
        """Retourne les poids adaptatifs `(w_als, w_session, w_content, w_pop)`."""
        known_als = self.collaborative_model.is_known_user(user_id)
        u_events = (
            self.user_event_counts_.get(int(user_id), 0)
            if user_id is not None
            else 0
        )
        has_session = bool(recent_items) or (
            user_id is not None and int(user_id) in self.session_model.user_recent_items_
        )

        # Cas 1 : Nouvel utilisateur sans historique en base (Cold User)
        if not known_als or u_events == 0:
            if has_session:
                return (0.15, 0.50, 0.25, 0.10)
            return (0.0, 0.0, 0.0, 1.0)

        # Cas 2 : Utilisateur à données très éparses (Sparse Data : 1 ou 2 événements)
        if u_events < self.config.sparse_user_threshold:
            return (0.25, 0.38, 0.25, 0.12)

        # Cas 3 : Utilisateur Warm actif
        return (
            self.config.collaborative_weight,
            self.config.session_weight,
            self.config.content_weight,
            self.config.popularity_weight,
        )

    def _resolve_excluded_items(
        self,
        user_id: int | None,
        exclude_items: set[int] | None,
        exclude_purchased_override: bool | None = None,
    ) -> set[int]:
        excluded = set(exclude_items) if exclude_items else set()
        if user_id is None:
            return excluded

        uid = int(user_id)
        excl_purchased = (
            self.config.exclude_already_purchased
            if exclude_purchased_override is None
            else exclude_purchased_override
        )
        if excl_purchased and uid in self.user_purchased_items_:
            excluded.update(self.user_purchased_items_[uid])

        if self.config.exclude_already_viewed and uid in self.user_viewed_items_:
            excluded.update(self.user_viewed_items_[uid])
        return excluded

    def _compute_hybrid_weighted_scores(
        self,
        candidates_df: pd.DataFrame,
        user_id: int | None,
        recent_items: Sequence[int] | None,
    ) -> np.ndarray:
        w_als, w_sess, w_cont, w_pop = self._resolve_dynamic_weights(user_id, recent_items)
        scores = (
            w_als * candidates_df["als_norm"].to_numpy(dtype=np.float64)
            + w_sess * candidates_df["session_norm"].to_numpy(dtype=np.float64)
            + w_cont * candidates_df["content_norm"].to_numpy(dtype=np.float64)
            + w_pop * candidates_df["popularity_norm"].to_numpy(dtype=np.float64)
        )
        # Léger bonus de consensus lorsque plusieurs familles d'algorithmes proposent le même item
        consensus_boost = 0.03 * np.maximum(
            0.0, candidates_df["source_count"].to_numpy(dtype=np.float64) - 1.0
        )
        return scores + consensus_boost

    def _diversify_top_n(
        self,
        item_ids: np.ndarray,
        scores: np.ndarray,
        n: int,
    ) -> list[RecommendationItem]:
        """Sélectionne les top-N avec pénalité de redondance catégorielle (Long Tail / Diversity)."""
        if len(item_ids) == 0 or n <= 0:
            return []

        penalty = float(self.config.diversity_category_penalty)
        cat_map = self.content_model.item_category_map_

        order = np.argsort(-scores, kind="mergesort")
        sorted_items = item_ids[order]
        sorted_scores = scores[order]

        if penalty <= 0.0 or not cat_map:
            return [
                RecommendationItem(
                    itemid=int(iid), score=float(sc), source=self.model_name
                )
                for iid, sc in zip(sorted_items[:n], sorted_scores[:n])
            ]

        # Re-ranking glouton sur le top-3*N pour favoriser la diversité catégorielle
        pool_size = min(len(sorted_items), max(n * 3, 25))
        pool_items = list(sorted_items[:pool_size])
        pool_scores = list(sorted_scores[:pool_size])
        cat_counts: Counter[int] = Counter()
        selected: list[RecommendationItem] = []

        while pool_items and len(selected) < n:
            best_idx = 0
            best_adj_score = -1e18
            for idx, (iid, base_sc) in enumerate(zip(pool_items, pool_scores)):
                cid = cat_map.get(int(iid), -1)
                cat_freq = cat_counts[cid] if cid >= 0 else 0
                adj_sc = float(base_sc) * (1.0 / (1.0 + penalty * cat_freq))
                if adj_sc > best_adj_score:
                    best_adj_score = adj_sc
                    best_idx = idx

            chosen_iid = int(pool_items.pop(best_idx))
            pool_scores.pop(best_idx)
            cid = cat_map.get(chosen_iid, -1)
            if cid >= 0:
                cat_counts[cid] += 1
            selected.append(
                RecommendationItem(
                    itemid=chosen_iid,
                    score=float(best_adj_score),
                    source=self.model_name,
                )
            )
        return selected

    def recommend(
        self,
        user_id: int | None = None,
        n: int = 10,
        exclude_items: set[int] | None = None,
        recent_items: Sequence[int] | None = None,
        exclude_already_purchased: bool | None = None,
    ) -> list[RecommendationItem]:
        """Génère, fusionne, score et diversifie les top-N recommandations hybrides."""
        if n <= 0:
            return []

        excluded = self._resolve_excluded_items(
            user_id=user_id,
            exclude_items=exclude_items,
            exclude_purchased_override=exclude_already_purchased,
        )

        candidates_df = self.candidate_generator.generate_candidates_df(
            user_id=user_id,
            k_per_model=max(self.config.candidates_per_model, n * 2),
            exclude_items=excluded,
            recent_items=recent_items,
        )
        if candidates_df.empty:
            return self.popularity_model.recommend(
                user_id=user_id, n=n, exclude_items=excluded
            )

        weighted_scores = self._compute_hybrid_weighted_scores(
            candidates_df, user_id=user_id, recent_items=recent_items
        )
        candidates_df = candidates_df.assign(
            hybrid_weighted_score=weighted_scores.astype(np.float32)
        )

        if self.use_ltr and self.ranker is not None and self.feature_builder is not None:
            feat_df = self.feature_builder.build_features(candidates_df)
            ltr_raw = self.ranker.predict_scores(feat_df)
            ltr_norm = normalize_scores(ltr_raw, method="min_max")
            # Combinaison convexe entre le score LambdaRank et le score hybride pondéré
            final_scores = 0.65 * ltr_norm + 0.35 * weighted_scores
        else:
            final_scores = weighted_scores

        final_scores = np.where(np.isfinite(final_scores), final_scores, 0.0)
        item_ids = candidates_df["itemid"].to_numpy(dtype=np.int64)
        return self._diversify_top_n(item_ids=item_ids, scores=final_scores, n=n)

    def score_items(
        self,
        user_id: int | None,
        candidate_items: Sequence[int],
        recent_items: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Score un ensemble imposé de `candidate_items` avec le moteur hybride."""
        if not candidate_items:
            return np.zeros(0, dtype=np.float64)

        method = self.config.normalization_method
        s_als = normalize_scores(
            self.collaborative_model.score_items(user_id, candidate_items, recent_items),
            method=method,
        )
        s_sess = normalize_scores(
            self.session_model.score_items(user_id, candidate_items, recent_items),
            method=method,
        )
        s_cont = normalize_scores(
            self.content_model.score_items(user_id, candidate_items, recent_items),
            method=method,
        )
        s_pop = normalize_scores(
            self.popularity_model.score_items(user_id, candidate_items, recent_items),
            method=method,
        )
        w_als, w_sess, w_cont, w_pop = self._resolve_dynamic_weights(user_id, recent_items)
        return w_als * s_als + w_sess * s_sess + w_cont * s_cont + w_pop * s_pop
