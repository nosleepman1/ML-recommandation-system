r"""Modèle Content-Based Filtering sur propriétés hachées et taxonomie (Section 7.2).

Principe mathématique :
Comme les propriétés Retailrocket sont anonymisées (`hashed_tokens_signature`) mais que
la hiérarchie des catégories (`categoryid`, `root_categoryid`, `category_path`) est explicite :
1. Chaque produit $i$ est représenté par un document synthétique combinant :
   - Ses jetons taxonomiques explicites (`cat_<id>`, `root_<id>`, noeuds ancêtres du chemin).
   - Ses jetons de propriétés hachées (`p<prop>_<val>`).
2. Un `TfidfVectorizer` (pondéré par IDF sous-linéaire et normalisé $\ell_2$) projette les
   produits dans un espace vectoriel creux $\mathbf{X} \in \mathbb{R}^{|\mathcal{I}| \times D}$.
3. Le profil d'un utilisateur $u$ ayant interagi avec les produits $\{(i_k, w_k)\}$ est le
   barycentre normalisé $\ell_2$ :
   $$\mathbf{p}_u = \frac{\sum_k w_k \mathbf{x}_{i_k}}{\left\|\sum_k w_k \mathbf{x}_{i_k}\right\|_2}$$
4. Le score Content-Based de tout produit candidat $j$ est la similarité cosinus :
   $$\text{sim}(u, j) = \langle \mathbf{p}_u, \mathbf{x}_j \rangle$$
   Ce mécanisme permet également de recommander des produits **Cold-Start** (0 interaction)
   dès lors qu'ils possèdent des métadonnées dans `items_metadata.parquet` !
"""

from typing import Sequence

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from recommender.config.model_config import DEFAULT_CONTENT_CONFIG, ContentBasedConfig
from recommender.models.base import BaseRecommender, RecommendationItem


class ContentBasedRecommender(BaseRecommender):
    """Système de filtrage basé sur le contenu (Taxonomie + Tokens TF-IDF creux)."""

    model_name: str = "content_based"

    def __init__(self, config: ContentBasedConfig = DEFAULT_CONTENT_CONFIG) -> None:
        self.config = config
        self.vectorizer_: TfidfVectorizer | None = None
        self.item_tfidf_matrix_: sparse.csr_matrix | None = None
        self.item_ids_: np.ndarray = np.array([], dtype=np.int64)
        self.item_to_idx_: dict[int, int] = {}
        self.user_history_: dict[int, list[tuple[int, float]]] = {}
        self.category_to_items_: dict[int, list[int]] = {}
        self.item_category_map_: dict[int, int] = {}

    @staticmethod
    def _build_corpus_strings(df: pd.DataFrame) -> list[str]:
        cat_str = np.where(
            df["categoryid"] >= 0,
            "cat_" + df["categoryid"].astype(str) + " cat_" + df["categoryid"].astype(str),
            "cat_unknown",
        )
        root_str = np.where(
            df["root_categoryid"] >= 0,
            "root_" + df["root_categoryid"].astype(str),
            "",
        )
        path_tokens = (
            df["category_path"]
            .fillna("")
            .astype(str)
            .str.replace("/", " node_", regex=False)
        )
        hashed = df.get("hashed_tokens_signature", pd.Series("", index=df.index)).fillna("").astype(str)
        combined = pd.Series(cat_str, index=df.index) + " " + pd.Series(root_str, index=df.index) + " node_" + path_tokens + " " + hashed
        return combined.str.strip().tolist()

    def fit(
        self,
        items_metadata_df: pd.DataFrame,
        interactions_df: pd.DataFrame | None = None,
    ) -> "ContentBasedRecommender":
        """Vectorise le catalogue produits et indexe l'historique pondéré des utilisateurs."""
        if items_metadata_df.empty:
            raise ValueError("items_metadata_df ne peut pas être vide.")

        # On conserve tous les produits ayant des métadonnées ou ayant été interagis
        if "has_metadata" in items_metadata_df.columns and "in_events" in items_metadata_df.columns:
            sub_df = items_metadata_df[
                items_metadata_df["has_metadata"] | items_metadata_df["in_events"]
            ].copy()
        else:
            sub_df = items_metadata_df.copy()

        sub_df = sub_df.sort_values("itemid").reset_index(drop=True)
        self.item_ids_ = sub_df["itemid"].to_numpy(dtype=np.int64)
        self.item_to_idx_ = {int(iid): idx for idx, iid in enumerate(self.item_ids_)}
        self.item_category_map_ = {
            int(iid): int(cid)
            for iid, cid in zip(sub_df["itemid"], sub_df["categoryid"])
        }

        self.category_to_items_ = {}
        for iid, cid in zip(sub_df["itemid"], sub_df["categoryid"]):
            cid_int = int(cid)
            if cid_int >= 0:
                self.category_to_items_.setdefault(cid_int, []).append(int(iid))

        corpus = self._build_corpus_strings(sub_df)
        self.vectorizer_ = TfidfVectorizer(
            max_features=self.config.max_tfidf_features,
            min_df=1 if len(corpus) < 20 else self.config.min_df,
            sublinear_tf=True,
            token_pattern=r"(?u)\b\w+\b",
            dtype=np.float32,
        )
        tfidf_mat = self.vectorizer_.fit_transform(corpus)
        self.item_tfidf_matrix_ = normalize(tfidf_mat, norm="l2", axis=1).tocsr()

        self.user_history_ = {}
        if interactions_df is not None and not interactions_df.empty:
            score_col = (
                "implicit_confidence_score"
                if "implicit_confidence_score" in interactions_df.columns
                else "base_event_weight"
            )
            if score_col not in interactions_df.columns:
                work = interactions_df.assign(_w=1.0)
                score_col = "_w"
            else:
                work = interactions_df

            sorted_inter = work.sort_values(
                by=["visitorid", score_col], ascending=[True, False]
            )
            grouped = sorted_inter.groupby("visitorid").head(
                self.config.max_user_history_items
            )
            for uid, iid, w in zip(
                grouped["visitorid"].to_numpy(),
                grouped["itemid"].to_numpy(),
                grouped[score_col].to_numpy(dtype=np.float64),
            ):
                iid_int = int(iid)
                if iid_int in self.item_to_idx_:
                    self.user_history_.setdefault(int(uid), []).append(
                        (iid_int, float(w))
                    )
        return self

    def _build_user_profile_vector(
        self,
        user_id: int | None,
        recent_items: Sequence[int] | None = None,
    ) -> sparse.csr_matrix | None:
        if self.item_tfidf_matrix_ is None:
            return None

        weighted_items: list[tuple[int, float]] = []
        if recent_items:
            n_rec = len(recent_items)
            for pos, iid in enumerate(recent_items[-self.config.max_user_history_items :]):
                iid_int = int(iid)
                if iid_int in self.item_to_idx_:
                    # Plus l'interaction est récente dans la liste, plus son poids est grand
                    w = float(0.8 ** (n_rec - 1 - pos))
                    weighted_items.append((iid_int, w))

        if user_id is not None and int(user_id) in self.user_history_:
            weighted_items.extend(self.user_history_[int(user_id)])

        if not weighted_items:
            return None

        row_indices = [self.item_to_idx_[iid] for iid, _ in weighted_items]
        weights = np.array([w for _, w in weighted_items], dtype=np.float32)
        sub_mat = self.item_tfidf_matrix_[row_indices]
        profile = weights @ sub_mat
        profile_csr = sparse.csr_matrix(profile, dtype=np.float32)
        return normalize(profile_csr, norm="l2", axis=1).tocsr()

    def recommend(
        self,
        user_id: int | None = None,
        n: int = 10,
        exclude_items: set[int] | None = None,
        recent_items: Sequence[int] | None = None,
    ) -> list[RecommendationItem]:
        """Recommande les top-N produits les plus similaires en contenu au profil utilisateur."""
        if n <= 0 or self.item_tfidf_matrix_ is None:
            return []

        profile_vec = self._build_user_profile_vector(user_id=user_id, recent_items=recent_items)
        if profile_vec is None or profile_vec.nnz == 0:
            return []

        excluded = set(exclude_items) if exclude_items else set()
        if user_id is not None and int(user_id) in self.user_history_ and exclude_items is None:
            excluded.update(iid for iid, _ in self.user_history_[int(user_id)])
        if recent_items and exclude_items is None:
            excluded.update(int(i) for i in recent_items)

        # Produit scalaire creux (Cosine Similarity car les deux vecteurs sont normalisés L2)
        scores_dense = (self.item_tfidf_matrix_ @ profile_vec.T).toarray().ravel()

        top_k = min(len(scores_dense), n + len(excluded) + 15)
        if top_k <= 0:
            return []

        candidate_indices = np.argpartition(-scores_dense, top_k - 1)[:top_k]
        sorted_indices = candidate_indices[np.argsort(-scores_dense[candidate_indices])]

        results: list[RecommendationItem] = []
        for idx in sorted_indices:
            sc = float(scores_dense[idx])
            if sc <= 0.0:
                break
            iid = int(self.item_ids_[idx])
            if iid not in excluded:
                results.append(
                    RecommendationItem(itemid=iid, score=sc, source=self.model_name)
                )
                if len(results) >= n:
                    break
        return results

    def score_items(
        self,
        user_id: int | None,
        candidate_items: Sequence[int],
        recent_items: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Calcule la similarité cosinus entre le profil utilisateur et `candidate_items`."""
        if not candidate_items or self.item_tfidf_matrix_ is None:
            return np.zeros(len(candidate_items), dtype=np.float64)

        profile_vec = self._build_user_profile_vector(user_id=user_id, recent_items=recent_items)
        if profile_vec is None or profile_vec.nnz == 0:
            return np.zeros(len(candidate_items), dtype=np.float64)

        scores = np.zeros(len(candidate_items), dtype=np.float64)
        valid_positions: list[int] = []
        valid_matrix_indices: list[int] = []
        for pos, iid in enumerate(candidate_items):
            idx = self.item_to_idx_.get(int(iid))
            if idx is not None:
                valid_positions.append(pos)
                valid_matrix_indices.append(idx)

        if valid_matrix_indices:
            sub_mat = self.item_tfidf_matrix_[valid_matrix_indices]
            sim = (sub_mat @ profile_vec.T).toarray().ravel()
            scores[valid_positions] = np.clip(sim, 0.0, 1.0)
        return scores
