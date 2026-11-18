"""Générateur multi-sources de candidats (Candidate Generation - Phase 5, Section 8.1).

Fusionne les listes de candidats issues des 4 composants :
1. Collaborative Filtering (ALS)
2. Session-Based (Transitions séquentielles / PPMI)
3. Content-Based (Similarité cosinus TF-IDF + Taxonomie)
4. Popularity-Based (Décroissance temporelle + Conversion)

Élimine les doublons, évalue tous les candidats retenus sur chaque modèle actif,
et conserve l'origine de génération (`sources_mask`) ainsi que les scores bruts et normalisés.
"""

from typing import Sequence

import numpy as np
import pandas as pd

from recommender.config.model_config import DEFAULT_HYBRID_CONFIG, HybridEngineConfig
from recommender.models.collaborative import CollaborativeALSRecommender
from recommender.models.content_based import ContentBasedRecommender
from recommender.models.popularity import PopularityRecommender
from recommender.models.session_based import SessionBasedRecommender
from recommender.ranking.score_normalizer import normalize_scores


class CandidateGenerator:
    """Orchestre la génération et la fusion sans doublons des produits candidats."""

    def __init__(
        self,
        popularity_model: PopularityRecommender,
        content_model: ContentBasedRecommender,
        collaborative_model: CollaborativeALSRecommender,
        session_model: SessionBasedRecommender,
        config: HybridEngineConfig = DEFAULT_HYBRID_CONFIG,
    ) -> None:
        self.popularity_model = popularity_model
        self.content_model = content_model
        self.collaborative_model = collaborative_model
        self.session_model = session_model
        self.config = config

    def generate_candidates_df(
        self,
        user_id: int | None,
        k_per_model: int | None = None,
        exclude_items: set[int] | None = None,
        recent_items: Sequence[int] | None = None,
    ) -> pd.DataFrame:
        """Génère, déduplique et score un pool unifié de candidats pour un utilisateur."""
        k = k_per_model or self.config.candidates_per_model
        excluded = set(exclude_items) if exclude_items else set()

        als_recs = self.collaborative_model.recommend(
            user_id=user_id, n=k, exclude_items=excluded, recent_items=recent_items
        )
        sess_recs = self.session_model.recommend(
            user_id=user_id, n=k, exclude_items=excluded, recent_items=recent_items
        )
        cont_recs = self.content_model.recommend(
            user_id=user_id, n=k, exclude_items=excluded, recent_items=recent_items
        )
        pop_recs = self.popularity_model.recommend(
            user_id=user_id, n=k, exclude_items=excluded, recent_items=recent_items
        )

        # Union ordonnée sans doublon
        candidate_set: dict[int, set[str]] = {}
        for rec_list in (als_recs, sess_recs, cont_recs, pop_recs):
            for item in rec_list:
                if item.itemid not in excluded:
                    candidate_set.setdefault(int(item.itemid), set()).add(item.source)

        if not candidate_set:
            return pd.DataFrame(
                columns=[
                    "visitorid",
                    "itemid",
                    "als_score",
                    "session_score",
                    "content_score",
                    "popularity_score",
                    "als_norm",
                    "session_norm",
                    "content_norm",
                    "popularity_norm",
                    "source_count",
                    "primary_source",
                ]
            )

        candidate_ids = list(candidate_set.keys())

        raw_als = self.collaborative_model.score_items(
            user_id=user_id, candidate_items=candidate_ids, recent_items=recent_items
        )
        raw_sess = self.session_model.score_items(
            user_id=user_id, candidate_items=candidate_ids, recent_items=recent_items
        )
        raw_cont = self.content_model.score_items(
            user_id=user_id, candidate_items=candidate_ids, recent_items=recent_items
        )
        raw_pop = self.popularity_model.score_items(
            user_id=user_id, candidate_items=candidate_ids, recent_items=recent_items
        )

        method = self.config.normalization_method
        norm_als = normalize_scores(raw_als, method=method)
        norm_sess = normalize_scores(raw_sess, method=method)
        norm_cont = normalize_scores(raw_cont, method=method)
        norm_pop = normalize_scores(raw_pop, method=method)

        source_counts = [len(candidate_set[iid]) for iid in candidate_ids]
        primary_sources = [sorted(candidate_set[iid])[0] for iid in candidate_ids]

        return pd.DataFrame(
            {
                "visitorid": int(user_id) if user_id is not None else -1,
                "itemid": np.array(candidate_ids, dtype=np.int64),
                "als_score": raw_als.astype(np.float32),
                "session_score": raw_sess.astype(np.float32),
                "content_score": raw_cont.astype(np.float32),
                "popularity_score": raw_pop.astype(np.float32),
                "als_norm": norm_als.astype(np.float32),
                "session_norm": norm_sess.astype(np.float32),
                "content_norm": norm_cont.astype(np.float32),
                "popularity_norm": norm_pop.astype(np.float32),
                "source_count": np.array(source_counts, dtype=np.int16),
                "primary_source": primary_sources,
            }
        )
