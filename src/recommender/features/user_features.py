"""Construction des caractéristiques utilisateurs (User Features - Phase 3, Section 6.1).

Toutes les caractéristiques sont strictement bornées par `cutoff_timestamp_ms` :
aucune interaction postérieure à cet instant n'est prise en compte (zéro fuite temporelle).

Caractéristiques construites :
- Volume par type d'interaction : `user_views_count`, `user_carts_count`, `user_transactions_count`,
  `user_total_events`, `user_cumulative_weight`.
- Taux de conversion personnels : `user_view_to_cart_rate`, `user_cart_to_tx_rate`,
  `user_view_to_tx_rate`, `user_is_buyer`, `user_is_cart_abandoner`.
- Préférences catégorielles et produits : `user_top_categoryid` (catégorie connue favorite
  pondérée par l'intention), `user_top_category_share`, `user_top_itemid`.
- Dynamique temporelle : `user_recency_days` (temps depuis la dernière interaction),
  `user_tenure_days` (durée entre la première et la dernière interaction).
- Fréquence d'activité : `user_sessions_count`, `user_active_days_count`,
  `user_events_per_session`, `user_events_per_active_day`.
- Diversité et répétition : `user_unique_items_count`, `user_unique_categories_count`,
  `user_repeat_interaction_ratio`, `user_category_entropy` (Entropie de Shannon).
"""

import numpy as np
import pandas as pd

from recommender.data.preprocessing import UNKNOWN_CATEGORY_ID

MS_PER_DAY = 1000.0 * 3600.0 * 24.0


def build_user_features(
    events_df: pd.DataFrame,
    cutoff_timestamp_ms: int | None = None,
) -> pd.DataFrame:
    """Construit la matrice de features utilisateurs à l'instant `cutoff_timestamp_ms`.

    Args:
        events_df: Journal d'événements nettoyé (contenant `visitorid`, `itemid`,
            `event`, `timestamp`, `base_event_weight`, `global_session_id`, `categoryid`).
        cutoff_timestamp_ms: Timestamp maximal autorisé (en ms). Si None, utilise le
            maximum observé dans `events_df`. Toute ligne `timestamp > cutoff_timestamp_ms`
            est filtrée avant tout calcul.

    Returns:
        DataFrame indexé par colonne `visitorid` contenant les features comportementales.
    """
    if events_df.empty:
        raise ValueError("events_df ne peut pas être vide pour build_user_features.")

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

    if "base_event_weight" not in observed.columns:
        weight_map = {"view": 1.0, "addtocart": 3.0, "transaction": 5.0}
        observed["base_event_weight"] = (
            observed["event"].map(weight_map).fillna(1.0).astype("float32")
        )
    if "global_session_id" not in observed.columns:
        observed["global_session_id"] = 0
    if "categoryid" not in observed.columns:
        observed["categoryid"] = UNKNOWN_CATEGORY_ID

    observed["is_view"] = (observed["event"] == "view").astype("int32")
    observed["is_cart"] = (observed["event"] == "addtocart").astype("int32")
    observed["is_tx"] = (observed["event"] == "transaction").astype("int32")
    observed["calendar_day"] = (observed["timestamp"] // int(MS_PER_DAY)).astype("int32")

    # 1. Agrégations principales par visitorid
    agg_df = observed.groupby("visitorid", as_index=False).agg(
        user_views_count=("is_view", "sum"),
        user_carts_count=("is_cart", "sum"),
        user_transactions_count=("is_tx", "sum"),
        user_total_events=("timestamp", "count"),
        user_cumulative_weight=("base_event_weight", "sum"),
        user_first_ts_ms=("timestamp", "min"),
        user_last_ts_ms=("timestamp", "max"),
        user_sessions_count=("global_session_id", "nunique"),
        user_active_days_count=("calendar_day", "nunique"),
        user_unique_items_count=("itemid", "nunique"),
    )

    # 2. Taux de conversion et indicateurs comportementaux
    views_arr = np.maximum(1.0, agg_df["user_views_count"].to_numpy(dtype=np.float64))
    carts_arr = np.maximum(1.0, agg_df["user_carts_count"].to_numpy(dtype=np.float64))
    total_arr = np.maximum(1.0, agg_df["user_total_events"].to_numpy(dtype=np.float64))

    agg_df["user_view_to_cart_rate"] = (
        agg_df["user_carts_count"].to_numpy(dtype=np.float64) / views_arr
    ).astype("float32")
    agg_df["user_cart_to_tx_rate"] = np.where(
        agg_df["user_carts_count"] > 0,
        agg_df["user_transactions_count"].to_numpy(dtype=np.float64) / carts_arr,
        0.0,
    ).astype("float32")
    agg_df["user_view_to_tx_rate"] = (
        agg_df["user_transactions_count"].to_numpy(dtype=np.float64) / views_arr
    ).astype("float32")

    agg_df["user_is_buyer"] = agg_df["user_transactions_count"] > 0
    agg_df["user_is_cart_abandoner"] = (agg_df["user_carts_count"] > 0) & (
        agg_df["user_transactions_count"] == 0
    )

    # 3. Récence, ancienneté et fréquence d'activité
    agg_df["user_recency_days"] = np.maximum(
        0.0,
        (float(ref_ts) - agg_df["user_last_ts_ms"].to_numpy(dtype=np.float64))
        / MS_PER_DAY,
    ).astype("float32")
    agg_df["user_tenure_days"] = np.maximum(
        0.0,
        (
            agg_df["user_last_ts_ms"].to_numpy(dtype=np.float64)
            - agg_df["user_first_ts_ms"].to_numpy(dtype=np.float64)
        )
        / MS_PER_DAY,
    ).astype("float32")

    agg_df["user_events_per_session"] = (
        total_arr / np.maximum(1.0, agg_df["user_sessions_count"].to_numpy(dtype=np.float64))
    ).astype("float32")
    agg_df["user_events_per_active_day"] = (
        total_arr
        / np.maximum(1.0, agg_df["user_active_days_count"].to_numpy(dtype=np.float64))
    ).astype("float32")

    # 4. Ratio de répétition d'interaction sur les mêmes produits
    agg_df["user_repeat_interaction_ratio"] = np.clip(
        1.0
        - (
            agg_df["user_unique_items_count"].to_numpy(dtype=np.float64) / total_arr
        ),
        0.0,
        1.0,
    ).astype("float32")

    # 5. Produit favori par utilisateur (pondéré par base_event_weight puis récence)
    ui_weights = observed.groupby(["visitorid", "itemid"], as_index=False).agg(
        w=("base_event_weight", "sum"),
        last_ts=("timestamp", "max"),
    )
    top_items = (
        ui_weights.sort_values(
            by=["visitorid", "w", "last_ts"], ascending=[True, False, False]
        )
        .drop_duplicates(subset=["visitorid"], keep="first")
        .rename(columns={"itemid": "user_top_itemid"})[["visitorid", "user_top_itemid"]]
    )
    agg_df = agg_df.merge(top_items, on="visitorid", how="left")
    agg_df["user_top_itemid"] = agg_df["user_top_itemid"].fillna(-1).astype("int64")

    # 6. Catégorie préférée et Entropie de Shannon des intérêts catégoriels
    known_cats = observed[observed["categoryid"] != UNKNOWN_CATEGORY_ID]
    if not known_cats.empty:
        uc_weights = known_cats.groupby(["visitorid", "categoryid"], as_index=False).agg(
            cat_w=("base_event_weight", "sum"),
            last_ts=("timestamp", "max"),
        )
        u_cat_totals = uc_weights.groupby("visitorid")["cat_w"].transform("sum")
        uc_weights["prob"] = uc_weights["cat_w"] / np.maximum(1e-9, u_cat_totals)
        uc_weights["neg_p_log_p"] = -uc_weights["prob"] * np.log(
            np.maximum(1e-12, uc_weights["prob"])
        )

        cat_diversity = uc_weights.groupby("visitorid", as_index=False).agg(
            user_unique_categories_count=("categoryid", "count"),
            user_category_entropy=("neg_p_log_p", "sum"),
        )

        top_cats = (
            uc_weights.sort_values(
                by=["visitorid", "cat_w", "last_ts"], ascending=[True, False, False]
            )
            .drop_duplicates(subset=["visitorid"], keep="first")
            .rename(
                columns={
                    "categoryid": "user_top_categoryid",
                    "prob": "user_top_category_share",
                }
            )[["visitorid", "user_top_categoryid", "user_top_category_share"]]
        )

        agg_df = agg_df.merge(cat_diversity, on="visitorid", how="left")
        agg_df = agg_df.merge(top_cats, on="visitorid", how="left")
    else:
        agg_df["user_unique_categories_count"] = 0
        agg_df["user_category_entropy"] = 0.0
        agg_df["user_top_categoryid"] = UNKNOWN_CATEGORY_ID
        agg_df["user_top_category_share"] = 0.0

    agg_df["user_unique_categories_count"] = (
        agg_df["user_unique_categories_count"].fillna(0).astype("int32")
    )
    agg_df["user_category_entropy"] = (
        agg_df["user_category_entropy"].fillna(0.0).astype("float32")
    )
    agg_df["user_top_categoryid"] = (
        agg_df["user_top_categoryid"].fillna(UNKNOWN_CATEGORY_ID).astype("int32")
    )
    agg_df["user_top_category_share"] = (
        agg_df["user_top_category_share"].fillna(0.0).astype("float32")
    )

    agg_df = agg_df.drop(columns=["user_first_ts_ms", "user_last_ts_ms"])
    return agg_df.sort_values("visitorid").reset_index(drop=True)
