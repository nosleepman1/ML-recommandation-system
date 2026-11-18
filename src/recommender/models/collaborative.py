r"""Modèle Collaborative Filtering via Implicit ALS sur matrice creuse CSR (Section 7.3).

Principe mathématique (Hu, Koren & Volinsky, ICDM 2008) :
À partir du score d'intérêt implicite $r_{ui} \ge 0$, on définit :
- La préférence binaire : $p_{ui} = \mathbb{I}(r_{ui} > 0)$
- Le niveau de confiance : $c_{ui} = 1 + \alpha \cdot r_{ui}$

L'algorithme Alternating Least Squares (ALS) minimise la fonction objectif régularisée :
$$\mathcal{L}(\mathbf{U}, \mathbf{V}) = \sum_{u, i} c_{ui} \left(p_{ui} - \mathbf{u}_u^\top \mathbf{v}_i\right)^2 + \lambda \left(\sum_u \|\mathbf{u}_u\|_2^2 + \sum_i \|\mathbf{v}_i\|_2^2\right)$$

En fixant tour à tour les facteurs produits $\mathbf{V}$ puis les facteurs utilisateurs $\mathbf{U}$,
chaque sous-problème devient quadratique convexe et admet une solution analytique exploitant
la sparsité de $\mathbf{C}^u - \mathbf{I}$ :
$$\mathbf{u}_u = \left(\mathbf{V}^\top \mathbf{V} + \mathbf{V}^\top (\mathbf{C}^u - \mathbf{I})\mathbf{V} + \lambda \mathbf{I}\right)^{-1} \mathbf{V}^\top \mathbf{C}^u \mathbf{p}_u$$
où $\mathbf{V}^\top \mathbf{V} \in \mathbb{R}^{K \times K}$ est précalculé une seule fois par demi-itération !
"""

import os
from typing import Sequence

import implicit
import numpy as np
import pandas as pd
import threadpoolctl
from scipy import sparse

from recommender.config.model_config import (
    DEFAULT_ALS_CONFIG,
    CollaborativeALSConfig,
)
from recommender.models.base import BaseRecommender, RecommendationItem

# Évite le sur-threading OpenBLAS pendant l'entraînement ALS
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
threadpoolctl.threadpool_limits(1, "blas")


class CollaborativeALSRecommender(BaseRecommender):
    """Filtrage collaboratif implicite par factorisation matricielle ALS."""

    model_name: str = "collaborative_als"

    def __init__(self, config: CollaborativeALSConfig = DEFAULT_ALS_CONFIG) -> None:
        self.config = config
        self.als_model_: implicit.als.AlternatingLeastSquares | None = None
        self.user_item_csr_: sparse.csr_matrix | None = None
        self.user_ids_: np.ndarray = np.array([], dtype=np.int64)
        self.item_ids_: np.ndarray = np.array([], dtype=np.int64)
        self.user_to_idx_: dict[int, int] = {}
        self.item_to_idx_: dict[int, int] = {}

    def fit(self, interactions_df: pd.DataFrame) -> "CollaborativeALSRecommender":
        """Construit la matrice creuse CSR et entraîne le modèle Implicit ALS."""
        if interactions_df.empty:
            raise ValueError("interactions_df ne peut pas être vide pour ALS.")

        df = interactions_df.copy()
        if self.config.exclude_bots and "is_bot_user" in df.columns:
            non_bot = df[~df["is_bot_user"]]
            if not non_bot.empty:
                df = non_bot

        score_col = (
            "implicit_confidence_score"
            if "implicit_confidence_score" in df.columns
            else "base_event_weight"
        )
        if score_col not in df.columns:
            df = df.assign(_score=1.0)
            score_col = "_score"

        # Agrégation au cas où df est un journal d'événements non encore agrégé par (visitorid, itemid)
        pairs_df = df.groupby(["visitorid", "itemid"], as_index=False)[score_col].sum()

        if self.config.min_item_interactions > 1:
            item_counts = pairs_df.groupby("itemid")["visitorid"].transform("count")
            filtered = pairs_df[item_counts >= self.config.min_item_interactions]
            if not filtered.empty:
                pairs_df = filtered

        self.user_ids_ = np.sort(pairs_df["visitorid"].unique().astype(np.int64))
        self.item_ids_ = np.sort(pairs_df["itemid"].unique().astype(np.int64))
        self.user_to_idx_ = {int(u): idx for idx, u in enumerate(self.user_ids_)}
        self.item_to_idx_ = {int(i): idx for idx, i in enumerate(self.item_ids_)}

        row_ind = pairs_df["visitorid"].map(self.user_to_idx_).to_numpy(dtype=np.int32)
        col_ind = pairs_df["itemid"].map(self.item_to_idx_).to_numpy(dtype=np.int32)
        confidence_vals = (
            self.config.alpha * pairs_df[score_col].to_numpy(dtype=np.float32)
        )

        self.user_item_csr_ = sparse.csr_matrix(
            (confidence_vals, (row_ind, col_ind)),
            shape=(len(self.user_ids_), len(self.item_ids_)),
            dtype=np.float32,
        )

        self.als_model_ = implicit.als.AlternatingLeastSquares(
            factors=self.config.factors,
            regularization=self.config.regularization,
            iterations=self.config.iterations,
            random_state=self.config.random_state,
            use_gpu=False,
        )
        self.als_model_.fit(self.user_item_csr_, show_progress=False)
        return self

    def is_known_user(self, user_id: int | None) -> bool:
        return user_id is not None and int(user_id) in self.user_to_idx_

    def _build_temp_user_csr(self, recent_items: Sequence[int]) -> sparse.csr_matrix | None:
        valid_cols = [
            self.item_to_idx_[int(iid)]
            for iid in recent_items
            if int(iid) in self.item_to_idx_
        ]
        if not valid_cols:
            return None
        data = np.full(len(valid_cols), float(self.config.alpha), dtype=np.float32)
        rows = np.zeros(len(valid_cols), dtype=np.int32)
        return sparse.csr_matrix(
            (data, (rows, valid_cols)),
            shape=(1, len(self.item_ids_)),
            dtype=np.float32,
        )

    def recommend(
        self,
        user_id: int | None = None,
        n: int = 10,
        exclude_items: set[int] | None = None,
        recent_items: Sequence[int] | None = None,
    ) -> list[RecommendationItem]:
        """Recommande les top-N produits latents via ALS (avec folding-in pour les sessions)."""
        if n <= 0 or self.als_model_ is None or self.user_item_csr_ is None:
            return []

        excluded = set(exclude_items) if exclude_items else set()
        recalc_user = False

        if user_id is not None and int(user_id) in self.user_to_idx_:
            u_idx = self.user_to_idx_[int(user_id)]
            user_row = self.user_item_csr_[u_idx]
        elif recent_items:
            temp_row = self._build_temp_user_csr(recent_items)
            if temp_row is None:
                return []
            u_idx = 0
            user_row = temp_row
            recalc_user = True
        else:
            return []

        top_k = min(len(self.item_ids_), n + len(excluded) + 10)
        if top_k <= 0:
            return []

        item_indices, scores = self.als_model_.recommend(
            userid=u_idx,
            user_items=user_row,
            N=top_k,
            filter_already_liked_items=exclude_items is None,
            recalculate_user=recalc_user,
        )

        results: list[RecommendationItem] = []
        for idx_val, sc_val in zip(item_indices, scores):
            iid = int(self.item_ids_[int(idx_val)])
            if iid not in excluded and np.isfinite(sc_val):
                results.append(
                    RecommendationItem(
                        itemid=iid,
                        score=float(sc_val),
                        source=self.model_name,
                    )
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
        r"""Calcule le produit scalaire $\langle \mathbf{u}_u, \mathbf{v}_i \rangle$ pour `candidate_items`."""
        scores = np.zeros(len(candidate_items), dtype=np.float64)
        if not candidate_items or self.als_model_ is None or self.user_item_csr_ is None:
            return scores

        if user_id is not None and int(user_id) in self.user_to_idx_:
            u_idx = self.user_to_idx_[int(user_id)]
            user_vec = np.asarray(self.als_model_.user_factors[u_idx]).ravel()
        elif recent_items:
            temp_row = self._build_temp_user_csr(recent_items)
            if temp_row is None:
                return scores
            user_vec = np.asarray(
                self.als_model_.recalculate_user(0, temp_row)
            ).ravel()
        else:
            return scores

        item_factors = self.als_model_.item_factors
        for pos, iid in enumerate(candidate_items):
            i_idx = self.item_to_idx_.get(int(iid))
            if i_idx is not None:
                scores[pos] = float(np.dot(user_vec, item_factors[i_idx]))
        return scores
