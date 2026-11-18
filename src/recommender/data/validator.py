"""Moteur d'audit complet du dataset Retailrocket (Phase 1 : Sections 4.1 à 4.4).

Ce module réalise l'audit scientifique complet sans modifier les données brutes :
- 4.1 Inspection structurelle et qualité de chaque fichier (lignes, colonnes, types,
  valeurs manquantes, doublons, valeurs aberrantes, mémoire).
- 4.2 Analyse approfondie des interactions (`events.csv`) : entonnoir, utilisateurs,
  produits, distributions temporelles (jour, heure, évolution, récence).
- 4.3 Analyse des produits et taxonomie (`item_properties_part1/2.csv` + `category_tree.csv`) :
  lecture en streaming par chunks, détection des produits sans métadonnées (orphan items),
  propriétés anonymisées vs explicites (`categoryid`, `available`), profondeur de l'arbre.
- 4.4 Analyse comportementale et segmentation utilisateur : occasionnels, actifs,
  acheteurs, abandons de panier, comportements répétitifs, diversité des intérêts.
"""

from collections import Counter
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from recommender.config.settings import AuditSettings, DEFAULT_AUDIT_SETTINGS, ProjectPaths
from recommender.data.loader import (
    iter_item_properties_chunks,
    load_category_tree,
    load_events,
    validate_file_exists,
)
from recommender.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class FileInspectionReport:
    """Rapport d'inspection unitaire d'un fichier CSV (Section 4.1)."""

    file_name: str
    file_size_mb: float
    num_rows: int
    num_columns: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_counts: dict[str, int]
    missing_percentages: dict[str, float]
    exact_duplicate_rows: int
    exact_duplicate_percentage: float
    timestamp_min_ms: int | None
    timestamp_max_ms: int | None
    timestamp_min_utc: str | None
    timestamp_max_utc: str | None
    anomalies: list[str]


def _ms_to_utc_iso(ts_ms: int | float | None) -> str | None:
    """Convertit un timestamp en millisecondes vers une chaîne ISO 8601 UTC."""
    if ts_ms is None or pd.isna(ts_ms):
        return None
    return datetime.fromtimestamp(float(ts_ms) / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def inspect_dataframe_file(
    df: pd.DataFrame,
    file_path: Path,
    timestamp_col: str | None = "timestamp",
) -> FileInspectionReport:
    """Construit le rapport d'inspection (Section 4.1) pour un DataFrame déjà chargé."""
    num_rows = int(len(df))
    num_cols = int(df.shape[1])
    file_size_mb = round(file_path.stat().st_size / (1024**2), 2)

    missing_counts = {col: int(df[col].isna().sum()) for col in df.columns}
    missing_pct = {
        col: round((count / num_rows) * 100.0, 4) if num_rows > 0 else 0.0
        for col, count in missing_counts.items()
    }
    exact_dups = int(df.duplicated().sum())
    dup_pct = round((exact_dups / num_rows) * 100.0, 4) if num_rows > 0 else 0.0

    ts_min: int | None = None
    ts_max: int | None = None
    anomalies: list[str] = []

    if timestamp_col and timestamp_col in df.columns and num_rows > 0:
        ts_series = df[timestamp_col].dropna()
        if len(ts_series) > 0:
            ts_min = int(ts_series.min())
            ts_max = int(ts_series.max())
            if ts_min <= 0:
                anomalies.append(f"Timestamps négatifs ou nuls détectés (min={ts_min}).")

    for col in ("visitorid", "itemid", "categoryid"):
        if col in df.columns and num_rows > 0:
            min_id = df[col].dropna().min()
            if pd.notna(min_id) and int(min_id) < 0:
                anomalies.append(f"Identifiants négatifs détectés dans '{col}' (min={min_id}).")

    return FileInspectionReport(
        file_name=file_path.name,
        file_size_mb=file_size_mb,
        num_rows=num_rows,
        num_columns=num_cols,
        columns=list(df.columns),
        dtypes={col: str(dtype) for col, dtype in df.dtypes.items()},
        missing_counts=missing_counts,
        missing_percentages=missing_pct,
        exact_duplicate_rows=exact_dups,
        exact_duplicate_percentage=dup_pct,
        timestamp_min_ms=ts_min,
        timestamp_max_ms=ts_max,
        timestamp_min_utc=_ms_to_utc_iso(ts_min),
        timestamp_max_utc=_ms_to_utc_iso(ts_max),
        anomalies=anomalies,
    )


def analyze_interactions_and_behavior(
    events_df: pd.DataFrame,
    settings: AuditSettings = DEFAULT_AUDIT_SETTINGS,
) -> dict[str, Any]:
    """Réalise l'audit complet des interactions (4.2) et du comportement utilisateur (4.4)."""
    total_events = int(len(events_df))
    if total_events == 0:
        raise ValueError("Le DataFrame events_df est vide.")

    # 1. Répartition des événements
    event_counts_series = events_df["event"].value_counts(dropna=False)
    event_counts = {str(k): int(v) for k, v in event_counts_series.items()}
    event_percentages = {
        str(k): round((int(v) / total_events) * 100.0, 4)
        for k, v in event_counts_series.items()
    }

    unexpected_events = [
        ev for ev in event_counts if ev not in settings.expected_event_types
    ]

    # Vérification de cohérence métier transactionid <-> event
    tx_events = events_df[events_df["event"] == "transaction"]
    non_tx_events = events_df[events_df["event"] != "transaction"]
    tx_missing_id = int(tx_events["transactionid"].isna().sum())
    non_tx_with_id = int(non_tx_events["transactionid"].notna().sum())

    # Doublons fonctionnels (visitorid, itemid, event, timestamp)
    functional_dups = int(
        events_df.duplicated(subset=["visitorid", "itemid", "event", "timestamp"]).sum()
    )

    # 2. Statistiques globales utilisateurs & produits
    n_unique_users = int(events_df["visitorid"].nunique())
    n_unique_items = int(events_df["itemid"].nunique())
    density_pct = (
        (events_df[["visitorid", "itemid"]].drop_duplicates().shape[0])
        / (n_unique_users * n_unique_items)
        * 100.0
    )

    # 3. Distributions par utilisateur et par produit
    user_event_counts = events_df.groupby("visitorid").size()
    item_event_counts = events_df.groupby("itemid").size()

    def _quantiles_summary(series: pd.Series) -> dict[str, float]:
        q = series.quantile([0.25, 0.50, 0.75, 0.90, 0.95, 0.99, 0.999])
        return {
            "min": float(series.min()),
            "mean": round(float(series.mean()), 4),
            "std": round(float(series.std()), 4),
            "p25": float(q.loc[0.25]),
            "median_p50": float(q.loc[0.50]),
            "p75": float(q.loc[0.75]),
            "p90": float(q.loc[0.90]),
            "p95": float(q.loc[0.95]),
            "p99": float(q.loc[0.99]),
            "p99_9": float(q.loc[0.999]),
            "max": float(series.max()),
        }

    top_users = [
        {"visitorid": int(uid), "interactions": int(cnt)}
        for uid, cnt in user_event_counts.nlargest(settings.top_k_entities).items()
    ]
    top_popular_items = [
        {"itemid": int(iid), "interactions": int(cnt)}
        for iid, cnt in item_event_counts.nlargest(settings.top_k_entities).items()
    ]

    purchased_item_counts = tx_events.groupby("itemid").size()
    top_purchased_items = [
        {"itemid": int(iid), "purchases": int(cnt)}
        for iid, cnt in purchased_item_counts.nlargest(settings.top_k_entities).items()
    ]

    # Concentration Long Tail (Pareto : % d'interactions captées par le top 1%, 5%, 10%, 20% des produits)
    sorted_item_counts = item_event_counts.sort_values(ascending=False)
    cum_share = sorted_item_counts.cumsum() / total_events
    long_tail_concentration = {}
    for pct in (0.01, 0.05, 0.10, 0.20):
        cutoff_idx = max(1, int(np.ceil(n_unique_items * pct)))
        long_tail_concentration[f"top_{int(pct * 100)}pct_items_share_pct"] = round(
            float(cum_share.iloc[cutoff_idx - 1]) * 100.0, 2
        )

    # 4. Segmentation comportementale des utilisateurs (Sections 4.2 & 4.4)
    user_type_Pivot = (
        events_df.groupby(["visitorid", "event"])
        .size()
        .unstack(fill_value=0)
    )
    for ev_col in ("view", "addtocart", "transaction"):
        if ev_col not in user_type_Pivot.columns:
            user_type_Pivot[ev_col] = 0

    user_totals = user_type_Pivot.sum(axis=1)
    users_view_only = int(
        ((user_type_Pivot["view"] > 0)
         & (user_type_Pivot["addtocart"] == 0)
         & (user_type_Pivot["transaction"] == 0)).sum()
    )
    users_cart_abandoners = int(
        ((user_type_Pivot["addtocart"] > 0)
         & (user_type_Pivot["transaction"] == 0)).sum()
    )
    users_buyers = int((user_type_Pivot["transaction"] > 0).sum())
    users_single_interaction = int((user_totals == 1).sum())
    users_casual = int((user_totals <= settings.casual_user_max_events).sum())
    users_active = int((user_totals >= settings.active_user_min_events).sum())

    # Comportements répétitifs vs diversité des intérêts par utilisateur
    user_unique_items = events_df.groupby("visitorid")["itemid"].nunique()
    repeat_ratio_per_user = 1.0 - (user_unique_items / user_totals)
    users_with_repetitive_views = int((user_totals > user_unique_items).sum())
    users_repeat_buyers = int(
        (tx_events.groupby("visitorid").size() > 1).sum()
        if len(tx_events) > 0
        else 0
    )

    # 5. Analyse temporelle (Section 4.2)
    dt_utc = pd.to_datetime(events_df["timestamp"], unit="ms", utc=True)
    dates_series = dt_utc.dt.strftime("%Y-%m-%d")
    hours_series = dt_utc.dt.hour
    dayofweek_series = dt_utc.dt.day_name()

    daily_total = events_df.groupby(dates_series).size()
    daily_by_type = (
        events_df.assign(date=dates_series)
        .groupby(["date", "event"])
        .size()
        .unstack(fill_value=0)
    )
    hourly_distribution = {
        int(h): int(c) for h, c in hours_series.value_counts().sort_index().items()
    }
    dayofweek_distribution = {
        str(d): int(c) for d, c in dayofweek_series.value_counts().items()
    }

    max_ts_ms = int(events_df["timestamp"].max())
    min_ts_ms = int(events_df["timestamp"].min())
    span_days = round((max_ts_ms - min_ts_ms) / (1000.0 * 3600 * 24), 2)

    user_last_ts = events_df.groupby("visitorid")["timestamp"].max()
    user_recency_days = (max_ts_ms - user_last_ts) / (1000.0 * 3600 * 24)

    return {
        "summary": {
            "total_interactions": total_events,
            "unique_visitors": n_unique_users,
            "unique_items_interacted": n_unique_items,
            "unique_user_item_pairs": int(
                events_df[["visitorid", "itemid"]].drop_duplicates().shape[0]
            ),
            "matrix_density_percent": round(density_pct, 6),
            "matrix_sparsity_percent": round(100.0 - density_pct, 6),
            "functional_duplicates_count": functional_dups,
            "unexpected_event_types": unexpected_events,
            "transactions_missing_txid": tx_missing_id,
            "non_transactions_having_txid": non_tx_with_id,
        },
        "event_breakdown": {
            "counts": event_counts,
            "percentages": event_percentages,
            "conversion_rates": {
                "view_to_cart_event_ratio_pct": round(
                    event_counts.get("addtocart", 0)
                    / max(1, event_counts.get("view", 1))
                    * 100.0,
                    3,
                ),
                "cart_to_tx_event_ratio_pct": round(
                    event_counts.get("transaction", 0)
                    / max(1, event_counts.get("addtocart", 1))
                    * 100.0,
                    3,
                ),
                "view_to_tx_event_ratio_pct": round(
                    event_counts.get("transaction", 0)
                    / max(1, event_counts.get("view", 1))
                    * 100.0,
                    3,
                ),
            },
        },
        "distributions": {
            "interactions_per_user": _quantiles_summary(user_event_counts),
            "interactions_per_item": _quantiles_summary(item_event_counts),
            "unique_items_per_user": _quantiles_summary(user_unique_items),
            "long_tail_concentration": long_tail_concentration,
            "top_active_users": top_users,
            "top_popular_items": top_popular_items,
            "top_purchased_items": top_purchased_items,
        },
        "user_behavior_segments": {
            "single_interaction_users_count": users_single_interaction,
            "single_interaction_users_pct": round(
                users_single_interaction / n_unique_users * 100.0, 2
            ),
            "casual_users_le_2_events_count": users_casual,
            "casual_users_le_2_events_pct": round(
                users_casual / n_unique_users * 100.0, 2
            ),
            "active_users_ge_10_events_count": users_active,
            "active_users_ge_10_events_pct": round(
                users_active / n_unique_users * 100.0, 2
            ),
            "view_only_users_count": users_view_only,
            "view_only_users_pct": round(users_view_only / n_unique_users * 100.0, 2),
            "cart_abandoner_users_count": users_cart_abandoners,
            "cart_abandoner_users_pct": round(
                users_cart_abandoners / n_unique_users * 100.0, 2
            ),
            "buyer_users_count": users_buyers,
            "buyer_users_pct": round(users_buyers / n_unique_users * 100.0, 2),
            "users_with_repeat_item_interactions_count": users_with_repetitive_views,
            "users_with_repeat_item_interactions_pct": round(
                users_with_repetitive_views / n_unique_users * 100.0, 2
            ),
            "repeat_buyer_users_count": users_repeat_buyers,
            "mean_repeat_interaction_ratio": round(
                float(repeat_ratio_per_user.mean()), 4
            ),
        },
        "temporal_analysis": {
            "min_timestamp_ms": min_ts_ms,
            "max_timestamp_ms": max_ts_ms,
            "min_datetime_utc": _ms_to_utc_iso(min_ts_ms),
            "max_datetime_utc": _ms_to_utc_iso(max_ts_ms),
            "observation_span_days": span_days,
            "num_active_calendar_days": int(len(daily_total)),
            "daily_interactions_summary": _quantiles_summary(daily_total),
            "hourly_interactions_utc": hourly_distribution,
            "dayofweek_interactions_utc": dayofweek_distribution,
            "user_recency_days_summary": _quantiles_summary(user_recency_days),
            "_daily_by_type_df": daily_by_type,  # utilisé pour générer les graphiques puis retiré du JSON
            "_user_event_counts": user_event_counts,
            "_item_event_counts": item_event_counts,
        },
    }


def analyze_category_tree(category_df: pd.DataFrame) -> dict[str, Any]:
    """Analyse la hiérarchie des catégories (`category_tree.csv`, Section 4.3)."""
    total_rows = int(len(category_df))
    unique_categories = set(category_df["categoryid"].dropna().astype(int).tolist())
    parent_series = category_df["parentid"].dropna().astype(int)
    unique_parents = set(parent_series.tolist())
    all_known_nodes = unique_categories | unique_parents

    root_categories = category_df[category_df["parentid"].isna()]["categoryid"].astype(int).tolist()
    leaf_categories = sorted(unique_categories - unique_parents)
    orphan_parents = sorted(unique_parents - unique_categories)

    # Construction du graphe parent -> calcul de la profondeur et détection de cycles
    parent_map: dict[int, int | None] = {}
    for row in category_df.itertuples(index=False):
        cid = int(row.categoryid)
        pid = int(row.parentid) if pd.notna(row.parentid) else None
        parent_map[cid] = pid

    depths: dict[int, int] = {}
    cycles_detected = 0

    for node in unique_categories:
        visited: set[int] = set()
        curr: int | None = node
        depth = 0
        while curr is not None:
            if curr in visited:
                cycles_detected += 1
                break
            visited.add(curr)
            curr = parent_map.get(curr)
            if curr is not None:
                depth += 1
        depths[node] = depth

    depth_values = list(depths.values()) if depths else [0]
    depth_distribution = dict(sorted(Counter(depth_values).items()))

    return {
        "total_rows": total_rows,
        "unique_category_ids": len(unique_categories),
        "total_distinct_nodes_including_parents": len(all_known_nodes),
        "root_categories_count": len(root_categories),
        "leaf_categories_count": len(leaf_categories),
        "orphan_parent_nodes_count": len(orphan_parents),
        "cycles_detected": cycles_detected,
        "max_tree_depth": int(max(depth_values)),
        "mean_tree_depth": round(float(np.mean(depth_values)), 2),
        "depth_distribution": {str(k): int(v) for k, v in depth_distribution.items()},
    }


def analyze_item_properties_streaming(
    file_paths: Sequence[Path],
    interacted_items: set[int],
    known_category_nodes: set[int],
    settings: AuditSettings = DEFAULT_AUDIT_SETTINGS,
) -> tuple[list[FileInspectionReport], dict[str, Any]]:
    """Audite `item_properties_part1.csv` et `part2.csv` en flux (streaming par chunks).

    Calcule les statistiques par fichier (Section 4.1) et l'analyse globale des
    propriétés produits (Section 4.3) sans jamais saturer la mémoire RAM.
    """
    for p in file_paths:
        validate_file_exists(p)

    file_reports: list[FileInspectionReport] = []
    global_property_counter: Counter[str] = Counter()
    global_items_with_props: set[int] = set()
    items_with_category: set[int] = set()
    items_with_availability: set[int] = set()
    observed_category_values: set[int] = set()
    numeric_prefixed_values_sample = 0
    total_rows_all_parts = 0
    total_numeric_values_count = 0

    for path in file_paths:
        num_rows = 0
        exact_dups = 0
        missing_counts = {"timestamp": 0, "itemid": 0, "property": 0, "value": 0}
        ts_min: int | None = None
        ts_max: int | None = None
        dtypes_map: dict[str, str] = {}
        cols_list: list[str] = []

        for _, chunk in iter_item_properties_chunks([path], chunk_size=settings.chunk_size):
            if not cols_list:
                cols_list = list(chunk.columns)
                dtypes_map = {col: str(dt) for col, dt in chunk.dtypes.items()}

            c_len = int(len(chunk))
            num_rows += c_len
            total_rows_all_parts += c_len

            for col in missing_counts:
                if col in chunk.columns:
                    missing_counts[col] += int(chunk[col].isna().sum())

            exact_dups += int(chunk.duplicated().sum())

            c_ts_min = int(chunk["timestamp"].min())
            c_ts_max = int(chunk["timestamp"].max())
            ts_min = c_ts_min if ts_min is None else min(ts_min, c_ts_min)
            ts_max = c_ts_max if ts_max is None else max(ts_max, c_ts_max)

            # Agrégation des items et des propriétés
            global_items_with_props.update(chunk["itemid"].unique().tolist())
            prop_vc = chunk["property"].value_counts()
            for prop_name, cnt in prop_vc.items():
                global_property_counter[str(prop_name)] += int(cnt)

            # Propriétés explicites : categoryid & available
            cat_mask = chunk["property"] == "categoryid"
            if cat_mask.any():
                cat_sub = chunk.loc[cat_mask, ["itemid", "value"]]
                items_with_category.update(cat_sub["itemid"].unique().tolist())
                valid_cats = pd.to_numeric(cat_sub["value"], errors="coerce").dropna().astype(int)
                observed_category_values.update(valid_cats.tolist())

            avail_mask = chunk["property"] == "available"
            if avail_mask.any():
                items_with_availability.update(
                    chunk.loc[avail_mask, "itemid"].unique().tolist()
                )

            # Détection des valeurs numériques anonymisées Retailrocket (préfixées par 'n')
            val_series = chunk["value"].dropna()
            num_masked = int(val_series.str.startswith("n").sum())
            total_numeric_values_count += num_masked
            numeric_prefixed_values_sample += num_masked

        missing_pct = {
            col: round((cnt / num_rows) * 100.0, 4) if num_rows > 0 else 0.0
            for col, cnt in missing_counts.items()
        }
        file_reports.append(
            FileInspectionReport(
                file_name=path.name,
                file_size_mb=round(path.stat().st_size / (1024**2), 2),
                num_rows=num_rows,
                num_columns=len(cols_list),
                columns=cols_list,
                dtypes=dtypes_map,
                missing_counts=missing_counts,
                missing_percentages=missing_pct,
                exact_duplicate_rows=exact_dups,
                exact_duplicate_percentage=round(
                    (exact_dups / num_rows) * 100.0, 4
                )
                if num_rows > 0
                else 0.0,
                timestamp_min_ms=ts_min,
                timestamp_max_ms=ts_max,
                timestamp_min_utc=_ms_to_utc_iso(ts_min),
                timestamp_max_utc=_ms_to_utc_iso(ts_max),
                anomalies=[],
            )
        )

    # Croisement entre events.csv et item_properties (Section 4.3)
    interacted_with_props = interacted_items & global_items_with_props
    interacted_without_props = interacted_items - global_items_with_props
    interacted_with_cat = interacted_items & items_with_category
    catalog_only_items = global_items_with_props - interacted_items

    total_unique_properties = len(global_property_counter)
    top_properties = [
        {
            "property": prop,
            "occurrences": int(cnt),
            "share_pct": round(cnt / max(1, total_rows_all_parts) * 100.0, 3),
        }
        for prop, cnt in global_property_counter.most_common(settings.top_k_entities)
    ]
    rare_properties_le_5 = sum(1 for _, cnt in global_property_counter.items() if cnt <= 5)
    single_occurrence_properties = sum(
        1 for _, cnt in global_property_counter.items() if cnt == 1
    )

    unmapped_categories = observed_category_values - known_category_nodes

    item_analysis = {
        "total_property_rows_combined": total_rows_all_parts,
        "unique_items_in_properties": len(global_items_with_props),
        "unique_properties_count": total_unique_properties,
        "items_with_categoryid_count": len(items_with_category),
        "items_with_available_flag_count": len(items_with_availability),
        "interacted_items_coverage": {
            "total_interacted_items_in_events": len(interacted_items),
            "interacted_items_with_properties_count": len(interacted_with_props),
            "interacted_items_with_properties_pct": round(
                len(interacted_with_props) / max(1, len(interacted_items)) * 100.0, 2
            ),
            "interacted_items_missing_properties_count": len(interacted_without_props),
            "interacted_items_missing_properties_pct": round(
                len(interacted_without_props) / max(1, len(interacted_items)) * 100.0, 2
            ),
            "interacted_items_with_category_count": len(interacted_with_cat),
            "interacted_items_with_category_pct": round(
                len(interacted_with_cat) / max(1, len(interacted_items)) * 100.0, 2
            ),
            "catalog_items_never_interacted_count": len(catalog_only_items),
        },
        "property_distributions": {
            "top_most_frequent_properties": top_properties,
            "rare_properties_le_5_occurrences": rare_properties_le_5,
            "single_occurrence_properties": single_occurrence_properties,
            "hashed_numeric_values_count": total_numeric_values_count,
            "hashed_numeric_values_pct": round(
                total_numeric_values_count / max(1, total_rows_all_parts) * 100.0, 2
            ),
            "categories_in_props_missing_from_tree_count": len(unmapped_categories),
        },
    }

    return file_reports, item_analysis


def generate_audit_figures(
    interactions_report: dict[str, Any],
    figures_dir: Path,
) -> list[str]:
    """Génère et sauvegarde les visualisations exploratoires de la Phase 1 (Section 4.4)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    figures_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid")
    saved_files: list[str] = []

    # 1. Entonnoir des événements (Event Funnel)
    fig, ax = plt.subplots(figsize=(8, 5))
    ev_counts = interactions_report["event_breakdown"]["counts"]
    labels = list(ev_counts.keys())
    values = list(ev_counts.values())
    bars = ax.bar(labels, values, color=["#2b5c8f", "#d9822b", "#2ca02c"])
    ax.set_title("Répartition des événements e-commerce (Retailrocket)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Nombre d'interactions")
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            f"{int(height):,}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            fontsize=10,
        )
    fig.tight_layout()
    f1 = figures_dir / "01_event_funnel.png"
    fig.savefig(f1, dpi=150)
    plt.close(fig)
    saved_files.append(str(f1))

    # 2. Évolution temporelle quotidienne par type d'événement
    daily_df: pd.DataFrame = interactions_report["temporal_analysis"]["_daily_by_type_df"]
    if not daily_df.empty:
        fig, axes = plt.subplots(2, 1, figsize=(12, 7), sharex=True)
        daily_df.index = pd.to_datetime(daily_df.index)
        if "view" in daily_df.columns:
            axes[0].plot(daily_df.index, daily_df["view"], color="#2b5c8f", label="Consultations (view)")
            axes[0].set_title("Évolution quotidienne des consultations (views)", fontsize=12, fontweight="bold")
            axes[0].legend()
        for col, color in (("addtocart", "#d9822b"), ("transaction", "#2ca02c")):
            if col in daily_df.columns:
                axes[1].plot(daily_df.index, daily_df[col], color=color, label=col)
        axes[1].set_title("Évolution quotidienne des ajouts au panier et achats", fontsize=12, fontweight="bold")
        axes[1].legend()
        fig.tight_layout()
        f2 = figures_dir / "02_daily_temporal_evolution.png"
        fig.savefig(f2, dpi=150)
        plt.close(fig)
        saved_files.append(str(f2))

    # 3. Distributions Log-Log (Loi de puissance / Long Tail) utilisateurs & produits
    user_counts: pd.Series = interactions_report["temporal_analysis"]["_user_event_counts"]
    item_counts: pd.Series = interactions_report["temporal_analysis"]["_item_event_counts"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].hist(np.log10(user_counts.values), bins=40, color="#2b5c8f", edgecolor="white")
    axes[0].set_title("Distribution log10(Interactions par Utilisateur)", fontweight="bold")
    axes[0].set_xlabel("log10(Nombre d'interactions)")
    axes[0].set_ylabel("Nombre d'utilisateurs")

    axes[1].hist(np.log10(item_counts.values), bins=40, color="#d9822b", edgecolor="white")
    axes[1].set_title("Distribution log10(Interactions par Produit)", fontweight="bold")
    axes[1].set_xlabel("log10(Nombre d'interactions)")
    axes[1].set_ylabel("Nombre de produits")
    fig.tight_layout()
    f3 = figures_dir / "03_user_item_long_tail_distributions.png"
    fig.savefig(f3, dpi=150)
    plt.close(fig)
    saved_files.append(str(f3))

    # 4. Distribution horaire (UTC) et segmentation comportementale
    hourly = interactions_report["temporal_analysis"]["hourly_interactions_utc"]
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].plot(list(hourly.keys()), list(hourly.values()), marker="o", color="#2b5c8f")
    axes[0].set_xticks(range(0, 24, 2))
    axes[0].set_title("Activité par heure de la journée (UTC)", fontweight="bold")
    axes[0].set_xlabel("Heure UTC")
    axes[0].set_ylabel("Nombre d'événements")

    seg = interactions_report["user_behavior_segments"]
    seg_labels = [
        "View Only",
        "Abandon Panier",
        "Acheteurs",
        "Occasionnels (<=2)",
        "Actifs (>=10)",
    ]
    seg_pcts = [
        seg["view_only_users_pct"],
        seg["cart_abandoner_users_pct"],
        seg["buyer_users_pct"],
        seg["casual_users_le_2_events_pct"],
        seg["active_users_ge_10_events_pct"],
    ]
    axes[1].barh(seg_labels, seg_pcts, color="#3a7ca5")
    axes[1].set_xlabel("% des utilisateurs uniques")
    axes[1].set_title("Profils comportementaux des utilisateurs (%)", fontweight="bold")
    fig.tight_layout()
    f4 = figures_dir / "04_hourly_and_user_segments.png"
    fig.savefig(f4, dpi=150)
    plt.close(fig)
    saved_files.append(str(f4))

    return saved_files


def run_full_phase1_audit(
    paths: ProjectPaths,
    settings: AuditSettings = DEFAULT_AUDIT_SETTINGS,
    generate_plots: bool = True,
) -> dict[str, Any]:
    """Exécute l'intégralité de l'audit de Phase 1 (4.1, 4.2, 4.3, 4.4) et retourne le rapport."""
    paths.ensure_directories()

    # 1. Chargement et inspection de events.csv
    events_df = load_events(paths.events_path)
    events_file_report = inspect_dataframe_file(
        events_df, paths.events_path, timestamp_col="timestamp"
    )

    # 2. Analyse des interactions (4.2) et du comportement utilisateur (4.4)
    logger.info("Analyse approfondie des interactions et segments comportementaux...")
    interactions_report = analyze_interactions_and_behavior(events_df, settings=settings)
    interacted_items_set = set(events_df["itemid"].unique().tolist())

    # 3. Chargement et analyse de category_tree.csv (4.1 & 4.3)
    category_df = load_category_tree(paths.category_tree_path)
    category_file_report = inspect_dataframe_file(
        category_df, paths.category_tree_path, timestamp_col=None
    )
    category_tree_report = analyze_category_tree(category_df)
    known_category_nodes = set(category_df["categoryid"].dropna().astype(int).tolist()) | set(
        category_df["parentid"].dropna().astype(int).tolist()
    )

    # Libération explicite de events_df après génération éventuelle des figures
    saved_plots: list[str] = []
    if generate_plots:
        logger.info("Génération des visualisations exploratoires dans %s...", paths.figures_dir)
        saved_plots = generate_audit_figures(interactions_report, paths.figures_dir)

    # Nettoyage des objets Pandas temporaires avant sérialisation JSON
    interactions_report["temporal_analysis"].pop("_daily_by_type_df", None)
    interactions_report["temporal_analysis"].pop("_user_event_counts", None)
    interactions_report["temporal_analysis"].pop("_item_event_counts", None)
    del events_df

    # 4. Inspection et analyse par morceaux de item_properties_part1.csv et part2.csv (4.1 & 4.3)
    prop_paths = [
        p for p in paths.item_properties_paths if p.exists()
    ]
    if not prop_paths:
        raise FileNotFoundError(
            "Aucun fichier item_properties_part*.csv trouvé dans data/raw/."
        )

    prop_file_reports, item_properties_report = analyze_item_properties_streaming(
        file_paths=prop_paths,
        interacted_items=interacted_items_set,
        known_category_nodes=known_category_nodes,
        settings=settings,
    )

    all_file_reports = [
        asdict(events_file_report),
        asdict(category_file_report),
        *[asdict(r) for r in prop_file_reports],
    ]

    return {
        "phase": "Phase 1 - Data Comprehension and Audit",
        "files_inspection_4_1": all_file_reports,
        "interactions_and_temporal_4_2": {
            "summary": interactions_report["summary"],
            "event_breakdown": interactions_report["event_breakdown"],
            "distributions": interactions_report["distributions"],
            "temporal_analysis": interactions_report["temporal_analysis"],
        },
        "product_and_taxonomy_4_3": {
            "category_tree": category_tree_report,
            "item_properties": item_properties_report,
        },
        "behavioral_segmentation_4_4": interactions_report["user_behavior_segments"],
        "generated_figures": saved_plots,
    }
