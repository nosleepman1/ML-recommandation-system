r"""Construction des caractéristiques d'interaction et de transitions séquentielles (Section 6.3).

Responsabilités :
1. `build_interaction_features` :
   Pour chaque paire `(visitorid, itemid)` observée avant `cutoff_timestamp_ms`, construit :
   - Le type maximal d'interaction (`ui_max_event_level`), la fréquence par type
     (`ui_views_count`, `ui_carts_count`, `ui_transactions_count`, `ui_total_events`).
   - Le temps écoulé depuis la dernière interaction (`ui_days_since_last`) et depuis
     la première interaction (`ui_days_since_first`).
   - Le score de préférence implicite amorti et pondéré par la récence (`ui_implicit_score`).
   - La part d'attention de l'utilisateur consacrée à ce produit (`ui_user_attention_share`).
   - L'affinité de l'utilisateur pour la catégorie du produit (`ui_user_category_affinity`).
2. `build_sequential_item_transitions` :
   Capture les relations entre produits consultés successivement au sein d'une même session
   (`item_source -> item_target`), avec comptage de co-occurrence, probabilité de transition
   conditionnelle $P(j \mid i)$ et Pointwise Mutual Information positive ($\text{PPMI}$).
"""

import numpy as np
import pandas as pd

from recommender.config.model_config import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_PREPROCESSING_CONFIG,
    FeatureConfig,
    InteractionWeightConfig,
)
from recommender.data.preprocessing import (
    UNKNOWN_CATEGORY_ID,
    build_weighted_user_item_interactions,
)

MS_PER_DAY = 1000.0 * 3600.0 * 24.0


def build_interaction_features(
    events_df: pd.DataFrame,
    cutoff_timestamp_ms: int | None = None,
    weight_config: InteractionWeightConfig = DEFAULT_PREPROCESSING_CONFIG.weights,
) -> pd.DataFrame:
    """Construit les features dyadiques `(visitorid, itemid)` bornées par `cutoff_timestamp_ms`."""
    if events_df.empty:
        raise ValueError(
            "events_df ne peut pas être vide pour build_interaction_features."
        )

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

    if "event_level" not in observed.columns:
        level_map = {"view": 1, "addtocart": 2, "transaction": 3}
        observed["event_level"] = observed["event"].map(level_map).fillna(1).astype("int8")
    if "base_event_weight" not in observed.columns:
        w_map = weight_config.to_weight_map()
        observed["base_event_weight"] = (
            observed["event"].map(w_map).fillna(1.0).astype("float32")
        )
    if "categoryid" not in observed.columns:
        observed["categoryid"] = UNKNOWN_CATEGORY_ID

    weighted_pairs = build_weighted_user_item_interactions(
        observed,
        weight_config=weight_config,
        reference_timestamp_ms=ref_ts,
    )

    # Durée entre la première et la dernière interaction (visitorid, itemid)
    weighted_pairs["ui_days_since_first"] = np.maximum(
        0.0,
        (
            float(ref_ts)
            - weighted_pairs["first_timestamp_ms"].to_numpy(dtype=np.float64)
        )
        / MS_PER_DAY,
    ).astype("float32")
    weighted_pairs["ui_interaction_span_days"] = np.maximum(
        0.0,
        (
            weighted_pairs["last_timestamp_ms"].to_numpy(dtype=np.float64)
            - weighted_pairs["first_timestamp_ms"].to_numpy(dtype=np.float64)
        )
        / MS_PER_DAY,
    ).astype("float32")

    # Part d'attention de l'utilisateur sur ce produit
    user_total_ev = observed.groupby("visitorid")["itemid"].count().rename("u_total")
    weighted_pairs = weighted_pairs.merge(user_total_ev, on="visitorid", how="left")
    weighted_pairs["ui_user_attention_share"] = (
        weighted_pairs["total_interactions"].to_numpy(dtype=np.float64)
        / np.maximum(1.0, weighted_pairs["u_total"].to_numpy(dtype=np.float64))
    ).astype("float32")

    # Catégorie de chaque item et affinité (visitorid, categoryid)
    item_cat = (
        observed[["itemid", "categoryid"]]
        .drop_duplicates(subset=["itemid"], keep="last")
    )
    weighted_pairs = weighted_pairs.merge(item_cat, on="itemid", how="left")
    weighted_pairs["categoryid"] = (
        weighted_pairs["categoryid"].fillna(UNKNOWN_CATEGORY_ID).astype("int32")
    )

    user_cat_w = (
        observed.groupby(["visitorid", "categoryid"], as_index=False)["base_event_weight"]
        .sum()
        .rename(columns={"base_event_weight": "uc_w"})
    )
    user_tot_w = (
        observed.groupby("visitorid", as_index=False)["base_event_weight"]
        .sum()
        .rename(columns={"base_event_weight": "u_tot_w"})
    )
    uc_aff = user_cat_w.merge(user_tot_w, on="visitorid", how="left")
    uc_aff["ui_user_category_affinity"] = np.where(
        uc_aff["categoryid"] != UNKNOWN_CATEGORY_ID,
        uc_aff["uc_w"].to_numpy(dtype=np.float64)
        / np.maximum(1e-6, uc_aff["u_tot_w"].to_numpy(dtype=np.float64)),
        0.0,
    ).astype("float32")

    weighted_pairs = weighted_pairs.merge(
        uc_aff[["visitorid", "categoryid", "ui_user_category_affinity"]],
        on=["visitorid", "categoryid"],
        how="left",
    )
    weighted_pairs["ui_user_category_affinity"] = (
        weighted_pairs["ui_user_category_affinity"].fillna(0.0).astype("float32")
    )

    weighted_pairs = weighted_pairs.drop(
        columns=["u_total", "first_timestamp_ms", "last_timestamp_ms"]
    )
    return weighted_pairs


def build_sequential_item_transitions(
    events_df: pd.DataFrame,
    cutoff_timestamp_ms: int | None = None,
    config: FeatureConfig = DEFAULT_FEATURE_CONFIG,
) -> pd.DataFrame:
    r"""Extrait les transitions séquentielles intra-session `source_itemid -> target_itemid`.

    Pour chaque session (`global_session_id`), considère les paires de produits distincts
    consultés successivement ($i_t \rightarrow i_{t+1}$, avec $i_t \neq i_{t+1}$) avant
    `cutoff_timestamp_ms`, et calcule :
    - `transition_count` : nombre d'occurrences de la transition $i \rightarrow j$.
    - `transition_probability` : $P(j \mid i) = \frac{C(i \rightarrow j)}{\sum_k C(i \rightarrow k)}$.
    - `ppmi_score` : Pointwise Mutual Information positive $\max\left(0, \ln \frac{P(i, j)}{P(i)P(j)}\right)$.
    """
    if events_df.empty:
        raise ValueError("events_df ne peut pas être vide pour build_sequential_item_transitions.")

    ref_ts = (
        int(cutoff_timestamp_ms)
        if cutoff_timestamp_ms is not None
        else int(events_df["timestamp"].max())
    )

    observed = events_df[events_df["timestamp"] <= ref_ts]
    if config.exclude_bots_from_item_popularity and "is_bot_user" in observed.columns:
        non_bot = observed[~observed["is_bot_user"]]
        if not non_bot.empty:
            observed = non_bot

    work = (
        observed[["global_session_id", "timestamp", "itemid"]]
        .sort_values(["global_session_id", "timestamp"], kind="mergesort")
        .copy()
    )

    work["target_itemid"] = work["itemid"].shift(-1)
    work["next_session_id"] = work["global_session_id"].shift(-1)

    valid_transition_mask = (
        (work["global_session_id"] == work["next_session_id"])
        & work["target_itemid"].notna()
        & (work["itemid"] != work["target_itemid"])
    )
    transitions = work.loc[valid_transition_mask, ["itemid", "target_itemid"]].rename(
        columns={"itemid": "source_itemid"}
    )
    if transitions.empty:
        return pd.DataFrame(
            columns=[
                "source_itemid",
                "target_itemid",
                "transition_count",
                "transition_probability",
                "ppmi_score",
            ]
        )

    transitions["target_itemid"] = transitions["target_itemid"].astype("int64")

    pair_counts = transitions.groupby(
        ["source_itemid", "target_itemid"], as_index=False
    ).size().rename(columns={"size": "transition_count"})

    total_transitions = float(pair_counts["transition_count"].sum())
    source_totals = pair_counts.groupby("source_itemid")["transition_count"].transform("sum")
    target_totals = pair_counts.groupby("target_itemid")["transition_count"].transform("sum")

    pair_counts["transition_probability"] = (
        pair_counts["transition_count"].to_numpy(dtype=np.float64)
        / np.maximum(1.0, source_totals.to_numpy(dtype=np.float64))
    ).astype("float32")

    pmi = np.log(
        np.maximum(
            1e-12,
            (pair_counts["transition_count"].to_numpy(dtype=np.float64) * total_transitions)
            / np.maximum(
                1.0,
                source_totals.to_numpy(dtype=np.float64)
                * target_totals.to_numpy(dtype=np.float64),
            ),
        )
    )
    pair_counts["ppmi_score"] = np.maximum(0.0, pmi).astype("float32")

    # Filtrage sur le nombre minimal de co-occurrences et top-K par produit source
    filtered = pair_counts[
        pair_counts["transition_count"] >= config.min_transition_cooccurrence
    ].copy()
    filtered = (
        filtered.sort_values(
            by=["source_itemid", "transition_count", "ppmi_score"],
            ascending=[True, False, False],
        )
        .groupby("source_itemid", as_index=False)
        .head(config.max_transitions_per_item)
        .reset_index(drop=True)
    )
    filtered["transition_count"] = filtered["transition_count"].astype("int32")
    return filtered
