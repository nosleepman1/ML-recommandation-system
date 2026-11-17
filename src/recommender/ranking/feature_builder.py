"""Constructeur de matrices de caractéristiques pour le Learning to Rank (Section 8.4).

Combine sans fuite temporelle :
1. Les scores bruts et normalisés des 4 modèles (`als`, `session`, `content`, `popularity`, `source_count`).
2. Les caractéristiques comportementales de l'utilisateur (`user_features`).
3. Les caractéristiques de popularité, vélocité et conversion du produit (`item_features`).
4. Les caractéristiques de compatibilité croisée (`category_match` entre la catégorie
   préférée de l'utilisateur et la catégorie du produit candidat).
"""

import numpy as np
import pandas as pd

RANKING_FEATURE_COLUMNS: tuple[str, ...] = (
    "als_score",
    "session_score",
    "content_score",
    "popularity_score",
    "als_norm",
    "session_norm",
    "content_norm",
    "popularity_norm",
    "source_count",
    "hybrid_weighted_score",
    "user_total_events",
    "user_views_count",
    "user_carts_count",
    "user_transactions_count",
    "user_recency_days",
    "user_category_entropy",
    "user_repeat_interaction_ratio",
    "item_total_events",
    "item_time_decayed_popularity",
    "item_events_last_7d",
    "item_smoothed_conversion_rate",
    "item_trend_velocity_7d_vs_30d",
    "item_recency_days",
    "user_item_category_match",
)


class RankingFeatureBuilder:
    """Assemble les features tabulaires $(u, i)$ pour le modèle LightGBM Ranker."""

    def __init__(
        self,
        user_features_df: pd.DataFrame,
        item_features_df: pd.DataFrame,
    ) -> None:
        u_cols = [
            "visitorid",
            "user_total_events",
            "user_views_count",
            "user_carts_count",
            "user_transactions_count",
            "user_recency_days",
            "user_category_entropy",
            "user_repeat_interaction_ratio",
            "user_top_categoryid",
        ]
        avail_u_cols = [c for c in u_cols if c in user_features_df.columns]
        self.user_df = user_features_df[avail_u_cols].drop_duplicates("visitorid").copy()

        i_cols = [
            "itemid",
            "categoryid",
            "item_total_events",
            "item_time_decayed_popularity",
            "item_events_last_7d",
            "item_smoothed_conversion_rate",
            "item_trend_velocity_7d_vs_30d",
            "item_recency_days",
        ]
        avail_i_cols = [c for c in i_cols if c in item_features_df.columns]
        self.item_df = item_features_df[avail_i_cols].drop_duplicates("itemid").copy()

    def build_features(self, candidates_df: pd.DataFrame) -> pd.DataFrame:
        """Enrichit `candidates_df` avec l'ensemble des `RANKING_FEATURE_COLUMNS`."""
        if candidates_df.empty:
            return candidates_df.copy()

        merged = candidates_df.merge(self.user_df, on="visitorid", how="left")
        merged = merged.merge(self.item_df, on="itemid", how="left")

        if "hybrid_weighted_score" not in merged.columns:
            merged["hybrid_weighted_score"] = (
                0.40 * merged.get("als_norm", 0.0)
                + 0.30 * merged.get("session_norm", 0.0)
                + 0.18 * merged.get("content_norm", 0.0)
                + 0.12 * merged.get("popularity_norm", 0.0)
            ).astype(np.float32)

        u_top_cat = merged.get("user_top_categoryid", pd.Series(-1, index=merged.index)).fillna(-1)
        i_cat = merged.get("categoryid", pd.Series(-2, index=merged.index)).fillna(-2)
        merged["user_item_category_match"] = (
            (u_top_cat >= 0) & (u_top_cat == i_cat)
        ).astype(np.float32)

        defaults = {
            "user_total_events": 0.0,
            "user_views_count": 0.0,
            "user_carts_count": 0.0,
            "user_transactions_count": 0.0,
            "user_recency_days": 999.0,
            "user_category_entropy": 0.0,
            "user_repeat_interaction_ratio": 0.0,
            "item_total_events": 0.0,
            "item_time_decayed_popularity": 0.0,
            "item_events_last_7d": 0.0,
            "item_smoothed_conversion_rate": 0.0,
            "item_trend_velocity_7d_vs_30d": 1.0,
            "item_recency_days": 999.0,
        }
        for col, val in defaults.items():
            if col not in merged.columns:
                merged[col] = np.float32(val)
            else:
                merged[col] = merged[col].fillna(val).astype(np.float32)

        return merged
