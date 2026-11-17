"""Construction des caractéristiques produits (Item Features - Phase 3, Section 6.2).

Toutes les caractéristiques comportementales d'un produit sont calculées uniquement
à partir des interactions `timestamp <= cutoff_timestamp_ms` (zéro fuite temporelle).

Caractéristiques construites :
- Popularité globale : `item_views_count`, `item_carts_count`, `item_transactions_count`,
  `item_total_events`, `item_raw_popularity_score`.
- Popularité récente et dynamique temporelle :
  - `item_events_last_7d`, `item_events_last_14d`, `item_transactions_last_7d`.
  - `item_time_decayed_popularity` : somme pondérée avec décroissance exponentielle
    de demi-vie `item_decay_half_life_days` (par défaut 14 jours).
  - `item_trend_velocity_7d_vs_30d` : rapport entre le rythme quotidien des 7 derniers
    jours et celui des 30 derniers jours (détecte les produits émergents vs obsolètes).
  - `item_recency_days` : jours écoulés depuis la dernière interaction sur le produit.
- Conversion et audience :
  - `item_unique_users_count`, `item_unique_buyers_count`.
  - `item_smoothed_conversion_rate` : taux de conversion lissé par Empirical Bayes
    `(tx + m * global_rate) / (views + m)` pour éviter le bruit sur les produits peu vus.
  - `item_cart_to_tx_rate`.
- Métadonnées catalogue (`items_metadata.parquet`) :
  - `categoryid`, `root_categoryid`, `category_depth`, `has_metadata`,
    `latest_available`, `property_observations_count`, `is_cold_item`.
"""

import numpy as np
import pandas as pd

from recommender.config.model_config import DEFAULT_FEATURE_CONFIG, FeatureConfig
from recommender.data.preprocessing import UNKNOWN_CATEGORY_ID

MS_PER_DAY = 1000.0 * 3600.0 * 24.0


def build_item_features(
    events_df: pd.DataFrame,
    items_metadata_df: pd.DataFrame | None = None,
    cutoff_timestamp_ms: int | None = None,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> pd.DataFrame:
    """Construit la table complète des features produits à l'instant `cutoff_timestamp_ms`."""
    if events_df.empty:
        raise ValueError("events_df ne peut pas être vide pour build_item_features.")

    ref_ts = (
        int(cutoff_timestamp_ms)
        if cutoff_timestamp_ms is not None
        else int(events_df["timestamp"].max())
    )

    observed = events_df[events_df["timestamp"] <= ref_ts].copy()
    if observed.empty:
        raise ValueError(
            f"Aucun événement trouvé avant cutoff_timestamp_ms={ref_ts}."
        )

    if config.exclude_bots_from_item_popularity and "is_bot_user" in observed.columns:
        non_bot = observed[~observed["is_bot_user"]]
        if not non_bot.empty:
            observed = non_bot.copy()

    if "base_event_weight" not in observed.columns:
        weight_map = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}
        observed["base_event_weight"] = (
            observed["event"].map(weight_map).fillna(1.0).astype("float32")
        )

    age_days = np.maximum(
        0.0,
        (float(ref_ts) - observed["timestamp"].to_numpy(dtype=np.float64)) / MS_PER_DAY,
    )
    half_life = max(1e-6, float(config.item_decay_half_life_days))
    decay_factor = np.exp(-np.log(2.0) * age_days / half_life)

    observed["is_view"] = (observed["event"] == "view").astype("int32")
    observed["is_cart"] = (observed["event"] == "addtocart").astype("int32")
    observed["is_tx"] = (observed["event"] == "transaction").astype("int32")
    observed["decayed_weight"] = (
        observed["base_event_weight"].to_numpy(dtype=np.float64) * decay_factor
    ).astype("float32")

    in_7d = age_days <= float(config.short_window_days)
    in_14d = age_days <= float(config.medium_window_days)
    in_30d = age_days <= float(config.long_window_days)

    observed["ev_7d"] = in_7d.astype("int32")
    observed["ev_14d"] = in_14d.astype("int32")
    observed["ev_30d"] = in_30d.astype("int32")
    observed["tx_7d"] = (in_7d & (observed["event"] == "transaction")).astype("int32")

    item_agg = observed.groupby("itemid", as_index=False).agg(
        item_views_count=("is_view", "sum"),
        item_carts_count=("is_cart", "sum"),
        item_transactions_count=("is_tx", "sum"),
        item_total_events=("timestamp", "count"),
        item_raw_popularity_score=("base_event_weight", "sum"),
        item_time_decayed_popularity=("decayed_weight", "sum"),
        item_events_last_7d=("ev_7d", "sum"),
        item_events_last_14d=("ev_14d", "sum"),
        item_events_last_30d=("ev_30d", "sum"),
        item_transactions_last_7d=("tx_7d", "sum"),
        item_unique_users_count=("visitorid", "nunique"),
        item_last_ts_ms=("timestamp", "max"),
    )

    # Acheteurs distincts par produit
    tx_only = observed[observed["is_tx"] == 1]
    if not tx_only.empty:
        buyers_agg = (
            tx_only.groupby("itemid", as_index=False)["visitorid"]
            .nunique()
            .rename(columns={"visitorid": "item_unique_buyers_count"})
        )
        item_agg = item_agg.merge(buyers_agg, on="itemid", how="left")
        item_agg["item_unique_buyers_count"] = (
            item_agg["item_unique_buyers_count"].fillna(0).astype("int32")
        )
    else:
        item_agg["item_unique_buyers_count"] = np.int32(0)

    # Taux de conversion bayésien lissé (Empirical Bayes)
    total_views_global = max(1.0, float(item_agg["item_views_count"].sum()))
    total_tx_global = float(item_agg["item_transactions_count"].sum())
    global_conv_rate = total_tx_global / total_views_global
    prior_m = float(config.bayesian_smoothing_prior_count)

    item_agg["item_smoothed_conversion_rate"] = (
        (
            item_agg["item_transactions_count"].to_numpy(dtype=np.float64)
            + prior_m * global_conv_rate
        )
        / (item_agg["item_views_count"].to_numpy(dtype=np.float64) + prior_m)
    ).astype("float32")

    item_agg["item_cart_to_tx_rate"] = np.where(
        item_agg["item_carts_count"] > 0,
        item_agg["item_transactions_count"].to_numpy(dtype=np.float64)
        / np.maximum(1.0, item_agg["item_carts_count"].to_numpy(dtype=np.float64)),
        0.0,
    ).astype("float32")

    # Récence et vélocité (tendance 7j vs 30j)
    item_agg["item_recency_days"] = np.maximum(
        0.0,
        (float(ref_ts) - item_agg["item_last_ts_ms"].to_numpy(dtype=np.float64))
        / MS_PER_DAY,
    ).astype("float32")

    daily_rate_7d = item_agg["item_events_last_7d"].to_numpy(dtype=np.float64) / max(
        1.0, float(config.short_window_days)
    )
    daily_rate_30d = item_agg["item_events_last_30d"].to_numpy(dtype=np.float64) / max(
        1.0, float(config.long_window_days)
    )
    item_agg["item_trend_velocity_7d_vs_30d"] = (
        (daily_rate_7d + 0.05) / (daily_rate_30d + 0.05)
    ).astype("float32")

    item_agg = item_agg.drop(columns=["item_last_ts_ms", "item_events_last_30d"])

    # Fusion avec le catalogue complet items_metadata_df (inclut aussi les produits Cold-Start)
    if items_metadata_df is not None and not items_metadata_df.empty:
        meta_cols = [
            col
            for col in (
                "itemid",
                "has_metadata",
                "categoryid",
                "root_categoryid",
                "category_depth",
                "latest_available",
                "property_observations_count",
            )
            if col in items_metadata_df.columns
        ]
        merged = items_metadata_df[meta_cols].merge(item_agg, on="itemid", how="outer")
    else:
        merged = item_agg.copy()
        merged["has_metadata"] = False
        merged["categoryid"] = UNKNOWN_CATEGORY_ID
        merged["root_categoryid"] = UNKNOWN_CATEGORY_ID
        merged["category_depth"] = -1
        merged["latest_available"] = -1
        merged["property_observations_count"] = 0

    # Remplissage des valeurs par défaut pour les produits Cold-Start (0 interaction dans le Train)
    zero_int_cols = (
        "item_views_count",
        "item_carts_count",
        "item_transactions_count",
        "item_total_events",
        "item_events_last_7d",
        "item_events_last_14d",
        "item_transactions_last_7d",
        "item_unique_users_count",
        "item_unique_buyers_count",
        "property_observations_count",
    )
    for col in zero_int_cols:
        if col in merged.columns:
            merged[col] = merged[col].fillna(0).astype("int32")

    for col, default_val in (
        ("item_raw_popularity_score", 0.0),
        ("item_time_decayed_popularity", 0.0),
        ("item_smoothed_conversion_rate", float(global_conv_rate)),
        ("item_cart_to_tx_rate", 0.0),
        ("item_recency_days", 999.0),
        ("item_trend_velocity_7d_vs_30d", 1.0),
    ):
        if col in merged.columns:
            merged[col] = merged[col].fillna(default_val).astype("float32")

    merged["has_metadata"] = merged["has_metadata"].fillna(False).astype(bool)
    merged["categoryid"] = (
        merged["categoryid"].fillna(UNKNOWN_CATEGORY_ID).astype("int32")
    )
    merged["root_categoryid"] = (
        merged["root_categoryid"].fillna(UNKNOWN_CATEGORY_ID).astype("int32")
    )
    merged["category_depth"] = merged["category_depth"].fillna(-1).astype("int16")
    merged["latest_available"] = merged["latest_available"].fillna(-1).astype("int8")
    merged["is_cold_item"] = merged["item_total_events"] == 0

    return merged.sort_values("itemid").reset_index(drop=True)
