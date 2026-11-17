r"""Modèle Popularity-Based avec décroissance temporelle et boost catégoriel (Section 7.1).

Principe mathématique :
Combine trois signaux normalisés sur $[0, 1]$ après transformation $\ln(1 + x)$ :
1. Popularité récente à décroissance exponentielle ($\text{Pop}_{\text{decay}}$).
2. Popularité globale cumulée ($\text{Pop}_{\text{global}}$).
3. Taux de conversion bayésien lissé ($\text{ConvRate}_{\text{EB}}$) multiplié par la vélocité.

Évite de favoriser excessivement les produits anciens qui ne sont plus consultés,
tout en servant de baseline robuste et de solution de repli (fallback) en Cold Start.
"""

from typing import Sequence

import numpy as np
import pandas as pd

from recommender.config.model_config import (
    DEFAULT_POPULARITY_CONFIG,
    PopularityModelConfig,
)
from recommender.models.base import BaseRecommender, RecommendationItem


def _min_max_scale(arr: np.ndarray) -> np.ndarray:
    if arr.size == 0:
        return arr
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax - vmin <= 1e-12:
        return np.zeros_like(arr, dtype=np.float64)
    return (arr - vmin) / (vmax - vmin)


class PopularityRecommender(BaseRecommender):
    """Recommandeur basé sur la popularité pondérée par la récence et la conversion."""

    model_name: str = "popularity"

    def __init__(self, config: PopularityModelConfig = DEFAULT_POPULARITY_CONFIG) -> None:
        self.config = config
        self.ranked_items_: np.ndarray = np.array([], dtype=np.int64)
        self.ranked_scores_: np.ndarray = np.array([], dtype=np.float64)
        self.item_score_map_: dict[int, float] = {}
        self.category_top_items_: dict[int, list[tuple[int, float]]] = {}

    def fit(self, item_features_df: pd.DataFrame) -> "PopularityRecommender":
        """Calcule les scores de popularité composite à partir de `item_features_df`."""
        if item_features_df.empty:
            raise ValueError("item_features_df ne peut pas être vide.")

        df = item_features_df.copy()
        if self.config.only_available_items and "latest_available" in df.columns:
            avail_df = df[df["latest_available"] == 1]
            if not avail_df.empty:
                df = avail_df

        # On ne classe en priorité que les produits ayant au moins une interaction (Warm)
        warm_df = df[df["item_total_events"] > 0].copy()
        if warm_df.empty:
            warm_df = df.copy()

        log_decay = np.log1p(
            np.maximum(0.0, warm_df["item_time_decayed_popularity"].to_numpy(dtype=np.float64))
        )
        log_global = np.log1p(
            np.maximum(0.0, warm_df["item_raw_popularity_score"].to_numpy(dtype=np.float64))
        )
        conv_signal = warm_df["item_smoothed_conversion_rate"].to_numpy(dtype=np.float64) * np.log1p(
            np.maximum(0.0, warm_df["item_transactions_count"].to_numpy(dtype=np.float64))
            + 0.5 * np.maximum(0.0, warm_df["item_carts_count"].to_numpy(dtype=np.float64))
        )

        s_decay = _min_max_scale(log_decay)
        s_global = _min_max_scale(log_global)
        s_conv = _min_max_scale(conv_signal)

        composite = (
            self.config.decay_weight * s_decay
            + self.config.global_weight * s_global
            + self.config.conversion_weight * s_conv
        )

        warm_df = warm_df.assign(pop_score=composite).sort_values(
            by=["pop_score", "item_transactions_count", "item_views_count"],
            ascending=[False, False, False],
        )

        self.ranked_items_ = warm_df["itemid"].to_numpy(dtype=np.int64)
        self.ranked_scores_ = warm_df["pop_score"].to_numpy(dtype=np.float64)
        self.item_score_map_ = {
            int(iid): float(sc)
            for iid, sc in zip(self.ranked_items_, self.ranked_scores_)
        }

        self.category_top_items_ = {}
        if "categoryid" in warm_df.columns:
            for cid, grp in warm_df.groupby("categoryid"):
                cid_int = int(cid)
                if cid_int >= 0:
                    top_grp = grp.head(50)
                    self.category_top_items_[cid_int] = [
                        (int(i), float(s))
                        for i, s in zip(top_grp["itemid"], top_grp["pop_score"])
                    ]
        return self

    def recommend(
        self,
        user_id: int | None = None,
        n: int = 10,
        exclude_items: set[int] | None = None,
        recent_items: Sequence[int] | None = None,
        preferred_category_id: int | None = None,
    ) -> list[RecommendationItem]:
        """Retourne les top-N produits populaires (avec option de filtrage catégoriel)."""
        if n <= 0 or self.ranked_items_.size == 0:
            return []

        excluded = set(exclude_items) if exclude_items else set()
        results: list[RecommendationItem] = []
        seen: set[int] = set()

        if preferred_category_id is not None and preferred_category_id in self.category_top_items_:
            for iid, sc in self.category_top_items_[preferred_category_id]:
                if iid not in excluded and iid not in seen:
                    results.append(
                        RecommendationItem(itemid=iid, score=float(sc), source=self.model_name)
                    )
                    seen.add(iid)
                    if len(results) >= n:
                        return results

        for iid_np, sc_np in zip(self.ranked_items_, self.ranked_scores_):
            iid = int(iid_np)
            if iid not in excluded and iid not in seen:
                results.append(
                    RecommendationItem(itemid=iid, score=float(sc_np), source=self.model_name)
                )
                seen.add(iid)
                if len(results) >= n:
                    break
        return results

    def score_items(
        self,
        user_id: int | None,
        candidate_items: Sequence[int],
        recent_items: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Retourne le score de popularité pour une liste de produits candidats."""
        return np.array(
            [self.item_score_map_.get(int(iid), 0.0) for iid in candidate_items],
            dtype=np.float64,
        )
