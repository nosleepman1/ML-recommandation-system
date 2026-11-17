"""Pipeline de Data Preprocessing reproductible pour Retailrocket (Phase 2 - Section 5).

Responsabilités :
1. Nettoyage et validation des événements (`events.csv`) :
   - Suppression des doublons exacts `(visitorid, itemid, event, timestamp)`.
   - Filtrage des types d'événements ou identifiants invalides.
   - Tri temporel déterministe et attribution d'identifiants de session (`session_id`).
   - Identification des profils utilisateurs (Cold/Sparse vs Warm vs Bots/Crawlers).
2. Pondération scientifique des interactions `(visitorid, itemid)` :
   - Distinction stricte entre intérêt implicite (`view`, `addtocart`) et préférence
     confirmée (`transaction`).
   - Amortissement logarithmique des consultations répétées pour éviter qu'un simple
     rafraîchissement de page ne surpasse un ajout au panier ou un achat.
   - Pondération par récence à décroissance exponentielle configurable.
3. Consolidation des métadonnées produits (`item_properties_part1/2.csv` & `category_tree.csv`) :
   - Construction de la hiérarchie des catégories (avec prise en charge des 30 catégories
     hors-arbre identifiées lors de l'audit de Phase 1).
   - Extraction en flux (streaming) des propriétés produits tout en conservant le premier
     et le dernier état connu ainsi que leurs timestamps (pour éviter toute fuite temporelle
     lors de l'entraînement).
   - Intégration explicite des 49 815 produits orphelins (`has_metadata = False`) présents
     dans `events.csv` mais absents de `item_properties`.
"""

from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

from recommender.config.model_config import (
    DEFAULT_PREPROCESSING_CONFIG,
    InteractionWeightConfig,
    PreprocessingConfig,
)
from recommender.config.settings import ProjectPaths
from recommender.data.loader import (
    iter_item_properties_chunks,
    load_category_tree,
    load_events,
)
from recommender.utils.logging import get_logger

logger = get_logger(__name__)

EVENT_LEVEL_MAP = {"view": 1, "addtocart": 2, "transaction": 3}
UNKNOWN_CATEGORY_ID = -1


def clean_and_sessionize_events(
    events_df: pd.DataFrame,
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Nettoie, déduplique, trie et segmente en sessions le journal `events.csv`.

    Retourne une nouvelle copie nettoyée sans jamais muter `events_df`.
    """
    raw_count = int(len(events_df))
    if raw_count == 0:
        raise ValueError("Le DataFrame events_df fourni est vide.")

    # 1. Filtrage des valeurs nulles sur les colonnes critiques et identifiants négatifs
    valid_mask = (
        events_df["timestamp"].notna()
        & (events_df["timestamp"] > 0)
        & events_df["visitorid"].notna()
        & (events_df["visitorid"] >= 0)
        & events_df["itemid"].notna()
        & (events_df["itemid"] >= 0)
        & events_df["event"].isin(config.valid_events)
    )
    invalid_rows_dropped = int((~valid_mask).sum())
    cleaned = events_df.loc[valid_mask].copy()

    # 2. Suppression des doublons exacts (visitorid, itemid, event, timestamp)
    before_dedup = int(len(cleaned))
    cleaned = cleaned.drop_duplicates(
        subset=["visitorid", "itemid", "event", "timestamp"], keep="first"
    ).copy()
    duplicates_dropped = before_dedup - int(len(cleaned))

    # 3. Ajout du niveau d'intention numérique et du poids unitaire de base
    weight_map = config.weights.to_weight_map()
    cleaned["event_level"] = cleaned["event"].map(EVENT_LEVEL_MAP).astype("int8")
    cleaned["base_event_weight"] = (
        cleaned["event"].map(weight_map).astype("float32")
    )

    # 4. Tri chronologique déterministe (visitorid, timestamp, event_level, itemid)
    cleaned = cleaned.sort_values(
        by=["visitorid", "timestamp", "event_level", "itemid"],
        ascending=[True, True, True, True],
        kind="mergesort",
    ).reset_index(drop=True)

    # 5. Reconstruction des sessions utilisateur (seuil d'inactivité en millisecondes)
    gap_threshold_ms = int(config.session_inactivity_minutes * 60 * 1000)
    same_user = cleaned["visitorid"] == cleaned["visitorid"].shift(1)
    time_diff_ms = cleaned["timestamp"] - cleaned["timestamp"].shift(1, fill_value=0)
    new_session_flag = (~same_user) | (time_diff_ms > gap_threshold_ms)

    cleaned["user_session_seq"] = (
        new_session_flag.groupby(cleaned["visitorid"]).cumsum().astype("int32")
    )
    cleaned["global_session_id"] = (new_session_flag.cumsum() - 1).astype("int64")

    # 6. Identification de la taille d'historique utilisateur et des bots/crawlers
    user_counts = cleaned.groupby("visitorid")["itemid"].transform("count")
    cleaned["is_warm_user"] = user_counts >= config.min_user_interactions_warm
    cleaned["is_bot_user"] = user_counts > config.bot_max_interactions_threshold

    bot_users_count = int(
        cleaned.loc[cleaned["is_bot_user"], "visitorid"].nunique()
    )
    bot_events_count = int(cleaned["is_bot_user"].sum())

    if config.filter_bot_users_in_clean_events and bot_events_count > 0:
        cleaned = cleaned.loc[~cleaned["is_bot_user"]].reset_index(drop=True)

    stats = {
        "raw_events_count": raw_count,
        "invalid_rows_dropped": invalid_rows_dropped,
        "exact_duplicates_dropped": duplicates_dropped,
        "clean_events_count": int(len(cleaned)),
        "unique_users_clean": int(cleaned["visitorid"].nunique()),
        "unique_items_clean": int(cleaned["itemid"].nunique()),
        "total_sessions_constructed": int(cleaned["global_session_id"].nunique()),
        "warm_users_ge_min_interactions_count": int(
            cleaned.loc[cleaned["is_warm_user"], "visitorid"].nunique()
        ),
        "flagged_bot_users_count": bot_users_count,
        "flagged_bot_events_count": bot_events_count,
        "bots_removed": bool(config.filter_bot_users_in_clean_events),
    }

    return cleaned, stats


def build_weighted_user_item_interactions(
    clean_events_df: pd.DataFrame,
    weight_config: InteractionWeightConfig = DEFAULT_PREPROCESSING_CONFIG.weights,
    reference_timestamp_ms: int | None = None,
) -> pd.DataFrame:
    """Agrège les événements au niveau paire `(visitorid, itemid)` avec pondération métier.

    Implémente les exigences de la Section 5 :
    - Sépare explicitement le comptage des `view`, `addtocart` et `transaction`.
    - Applique un amortissement logarithmique sur les répétitions d'un même type
      d'interaction : `w * (1 + gamma * ln(count))` pour éviter que des consultations
      répétées ne soient confondues avec un achat.
    - Calcule la récence par rapport à `reference_timestamp_ms` (par défaut le max
      du DataFrame fourni, ce qui garantit l'absence de fuite temporelle lorsqu'on
      lui passe uniquement le jeu d'entraînement).
    """
    if clean_events_df.empty:
        raise ValueError("clean_events_df ne peut pas être vide.")

    ref_ts = (
        int(reference_timestamp_ms)
        if reference_timestamp_ms is not None
        else int(clean_events_df["timestamp"].max())
    )

    # Comptage vectorisé par (visitorid, itemid)
    work = clean_events_df[
        ["visitorid", "itemid", "event", "timestamp", "event_level"]
    ].copy()
    work["is_view"] = (work["event"] == "view").astype("int32")
    work["is_cart"] = (work["event"] == "addtocart").astype("int32")
    work["is_tx"] = (work["event"] == "transaction").astype("int32")

    grouped = work.groupby(["visitorid", "itemid"], as_index=False).agg(
        view_count=("is_view", "sum"),
        addtocart_count=("is_cart", "sum"),
        transaction_count=("is_tx", "sum"),
        total_interactions=("timestamp", "count"),
        first_timestamp_ms=("timestamp", "min"),
        last_timestamp_ms=("timestamp", "max"),
        max_event_level=("event_level", "max"),
    )

    gamma = float(weight_config.repeat_damping_factor)

    # Score amorti par type d'interaction : w * (1 + gamma * ln(c)) si c > 0 sinon 0
    view_c = grouped["view_count"].to_numpy(dtype=np.float64)
    cart_c = grouped["addtocart_count"].to_numpy(dtype=np.float64)
    tx_c = grouped["transaction_count"].to_numpy(dtype=np.float64)

    view_score = np.where(
        view_c > 0,
        weight_config.view_weight * (1.0 + gamma * np.log(np.maximum(1.0, view_c))),
        0.0,
    )
    cart_score = np.where(
        cart_c > 0,
        weight_config.addtocart_weight
        * (1.0 + gamma * np.log(np.maximum(1.0, cart_c))),
        0.0,
    )
    tx_score = np.where(
        tx_c > 0,
        weight_config.transaction_weight
        * (1.0 + gamma * np.log(np.maximum(1.0, tx_c))),
        0.0,
    )

    raw_preference_score = view_score + cart_score + tx_score

    # Décroissance temporelle (récence en jours depuis la dernière interaction sur ce produit)
    days_since_last = np.maximum(
        0.0,
        (float(ref_ts) - grouped["last_timestamp_ms"].to_numpy(dtype=np.float64))
        / (1000.0 * 3600.0 * 24.0),
    )
    half_life = max(1e-6, float(weight_config.recency_half_life_days))
    recency_decay = np.exp(-np.log(2.0) * days_since_last / half_life)

    grouped["is_confirmed_purchase"] = grouped["transaction_count"] > 0
    grouped["has_cart_addition"] = grouped["addtocart_count"] > 0
    grouped["raw_preference_score"] = raw_preference_score.astype("float32")
    grouped["days_since_last_interaction"] = days_since_last.astype("float32")
    grouped["recency_weight"] = recency_decay.astype("float32")
    grouped["implicit_confidence_score"] = (
        raw_preference_score * recency_decay
    ).astype("float32")

    return grouped


def build_category_taxonomy_table(
    category_df: pd.DataFrame,
    extra_category_ids: set[int] | None = None,
) -> pd.DataFrame:
    """Construit la table enrichie de la taxonomie des catégories.

    Gère les catégories présentes dans `category_tree.csv` ainsi que les éventuelles
    catégories supplémentaires (`extra_category_ids`, comme les 30 catégories découvertes
    dans `item_properties` lors de l'audit de Phase 1) en les rattachant comme racines.
    """
    parent_map: dict[int, int | None] = {}
    for row in category_df.itertuples(index=False):
        cid = int(row.categoryid)
        pid = int(row.parentid) if pd.notna(row.parentid) else None
        parent_map[cid] = pid

    if extra_category_ids:
        for extra_cid in extra_category_ids:
            if int(extra_cid) not in parent_map and int(extra_cid) >= 0:
                parent_map[int(extra_cid)] = None

    children_counts: Counter[int] = Counter()
    for pid in parent_map.values():
        if pid is not None:
            children_counts[pid] += 1

    records: list[dict[str, Any]] = []
    for cid in sorted(parent_map.keys()):
        lineage: list[int] = []
        visited: set[int] = set()
        curr: int | None = cid
        while curr is not None and curr not in visited:
            visited.add(curr)
            lineage.append(curr)
            curr = parent_map.get(curr)

        lineage_reversed = list(reversed(lineage))
        root_cid = lineage_reversed[0]
        depth = len(lineage_reversed) - 1
        path_str = "/".join(str(node) for node in lineage_reversed)

        records.append(
            {
                "categoryid": int(cid),
                "parentid": parent_map[cid] if parent_map[cid] is not None else pd.NA,
                "root_categoryid": int(root_cid),
                "depth": int(depth),
                "is_leaf": children_counts[cid] == 0,
                "in_original_tree": int(cid) in set(category_df["categoryid"].astype(int)),
                "category_path": path_str,
            }
        )

    out_df = pd.DataFrame(records)
    if not out_df.empty:
        out_df["parentid"] = out_df["parentid"].astype("Int64")
        out_df["categoryid"] = out_df["categoryid"].astype("int64")
        out_df["root_categoryid"] = out_df["root_categoryid"].astype("int64")
        out_df["depth"] = out_df["depth"].astype("int16")
    return out_df


def consolidate_item_metadata_streaming(
    item_properties_paths: Sequence[Path],
    category_df: pd.DataFrame,
    interacted_item_ids: set[int],
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
    cutoff_timestamp_ms: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Consolide en streaming les ~20M de lignes de `item_properties_part1/2.csv`.

    Garantit l'absence de fuite temporelle :
    - Si `cutoff_timestamp_ms` est fourni, seules les observations `timestamp <= cutoff_timestamp_ms`
      (ou le snapshot initial de référence pour les attributs statiques) sont retenues.
    - Produit un catalogue unifié contenant à la fois les produits de `item_properties`
      ET les produits orphelins de `events.csv` (`has_metadata = False`, `categoryid = -1`).
    """
    # Structures mémoire-compactes pour l'agrégation en flux
    # Pour chaque item : on garde le premier et le dernier (timestamp, value) pour categoryid et available
    cat_first_ts: dict[int, int] = {}
    cat_first_val: dict[int, int] = {}
    cat_last_ts: dict[int, int] = {}
    cat_last_val: dict[int, int] = {}

    avail_last_ts: dict[int, int] = {}
    avail_last_val: dict[int, int] = {}
    avail_ever_one: set[int] = set()

    item_prop_count: Counter[int] = Counter()
    property_freq: Counter[str] = Counter()
    # Stocke un ensemble compact de tokens catégoriels/hachés par produit pour le Content-Based
    item_tokens: defaultdict[int, list[str]] = defaultdict(list)
    max_tokens_per_item = 25

    catalog_items_seen: set[int] = set()

    for _, chunk in iter_item_properties_chunks(
        item_properties_paths, chunk_size=config.chunk_size
    ):
        if cutoff_timestamp_ms is not None:
            chunk = chunk[chunk["timestamp"] <= cutoff_timestamp_ms]
            if chunk.empty:
                continue

        catalog_items_seen.update(chunk["itemid"].unique().tolist())

        # Comptage global de propriétés par item
        vc_items = chunk["itemid"].value_counts()
        for iid, cnt in vc_items.items():
            item_prop_count[int(iid)] += int(cnt)

        vc_props = chunk["property"].value_counts()
        for pname, cnt in vc_props.items():
            property_freq[str(pname)] += int(cnt)

        # 1. Traitement vectorisé de categoryid sur le chunk
        cat_rows = chunk[chunk["property"] == "categoryid"]
        if not cat_rows.empty:
            cat_vals = pd.to_numeric(cat_rows["value"], errors="coerce")
            valid_cat = (
                cat_rows.assign(num_val=cat_vals)
                .dropna(subset=["num_val"])
                .sort_values("timestamp")
            )
            if not valid_cat.empty:
                first_cats = valid_cat.drop_duplicates("itemid", keep="first")
                for iid, ts, cval in zip(
                    first_cats["itemid"].to_numpy(),
                    first_cats["timestamp"].to_numpy(),
                    first_cats["num_val"].to_numpy(dtype=np.int64),
                ):
                    iid_int = int(iid)
                    ts_int = int(ts)
                    if iid_int not in cat_first_ts or ts_int < cat_first_ts[iid_int]:
                        cat_first_ts[iid_int] = ts_int
                        cat_first_val[iid_int] = int(cval)

                last_cats = valid_cat.drop_duplicates("itemid", keep="last")
                for iid, ts, cval in zip(
                    last_cats["itemid"].to_numpy(),
                    last_cats["timestamp"].to_numpy(),
                    last_cats["num_val"].to_numpy(dtype=np.int64),
                ):
                    iid_int = int(iid)
                    ts_int = int(ts)
                    if iid_int not in cat_last_ts or ts_int >= cat_last_ts[iid_int]:
                        cat_last_ts[iid_int] = ts_int
                        cat_last_val[iid_int] = int(cval)

        # 2. Traitement vectorisé de available sur le chunk
        avail_rows = chunk[chunk["property"] == "available"]
        if not avail_rows.empty:
            avail_vals = pd.to_numeric(avail_rows["value"], errors="coerce")
            valid_av = (
                avail_rows.assign(num_val=avail_vals)
                .dropna(subset=["num_val"])
                .sort_values("timestamp")
            )
            if not valid_av.empty:
                ones_items = valid_av.loc[valid_av["num_val"] == 1, "itemid"].unique()
                avail_ever_one.update(ones_items.tolist())

                last_av = valid_av.drop_duplicates("itemid", keep="last")
                for iid, ts, aval in zip(
                    last_av["itemid"].to_numpy(),
                    last_av["timestamp"].to_numpy(),
                    last_av["num_val"].to_numpy(dtype=np.int64),
                ):
                    iid_int = int(iid)
                    ts_int = int(ts)
                    if iid_int not in avail_last_ts or ts_int >= avail_last_ts[iid_int]:
                        avail_last_ts[iid_int] = ts_int
                        avail_last_val[iid_int] = int(aval)

        # 3. Extraction vectorisée de tokens représentatifs (hors available et categoryid) pour les items interagis
        other_rows = chunk[
            (~chunk["property"].isin(["categoryid", "available"]))
            & (chunk["itemid"].isin(interacted_item_ids))
        ]
        if not other_rows.empty:
            dedup_other = other_rows.drop_duplicates(subset=["itemid", "property"], keep="last")
            first_val_part = dedup_other["value"].str.strip().str.split(" ", n=1).str[0].fillna("")
            tokens_series = "p" + dedup_other["property"] + "_" + first_val_part
            for iid, tok in zip(
                dedup_other["itemid"].to_numpy(), tokens_series.to_numpy()
            ):
                iid_int = int(iid)
                t_list = item_tokens[iid_int]
                if len(t_list) < max_tokens_per_item and tok not in t_list:
                    t_list.append(str(tok))

    # Construction de la taxonomie enrichie incluant les catégories hors-arbre observées
    observed_cats = set(cat_last_val.values()) | set(cat_first_val.values())
    taxonomy_df = build_category_taxonomy_table(
        category_df, extra_category_ids=observed_cats
    )
    cat_to_root = dict(
        zip(taxonomy_df["categoryid"].tolist(), taxonomy_df["root_categoryid"].tolist())
    )
    cat_to_depth = dict(
        zip(taxonomy_df["categoryid"].tolist(), taxonomy_df["depth"].tolist())
    )
    cat_to_path = dict(
        zip(taxonomy_df["categoryid"].tolist(), taxonomy_df["category_path"].tolist())
    )

    # Union de tous les produits connus (catalogue + événements)
    all_universe_items = sorted(catalog_items_seen | interacted_item_ids)

    item_records: list[dict[str, Any]] = []
    for iid in all_universe_items:
        has_meta = iid in catalog_items_seen
        in_events = iid in interacted_item_ids
        cid = cat_last_val.get(iid, UNKNOWN_CATEGORY_ID)
        root_cid = cat_to_root.get(cid, UNKNOWN_CATEGORY_ID) if cid != UNKNOWN_CATEGORY_ID else UNKNOWN_CATEGORY_ID
        c_depth = cat_to_depth.get(cid, -1) if cid != UNKNOWN_CATEGORY_ID else -1
        c_path = cat_to_path.get(cid, "") if cid != UNKNOWN_CATEGORY_ID else ""

        last_avail = avail_last_val.get(iid, -1)
        ever_avail = 1 if iid in avail_ever_one else (0 if has_meta else -1)
        tokens_str = " ".join(item_tokens[iid]) if iid in item_tokens else ""

        item_records.append(
            {
                "itemid": int(iid),
                "has_metadata": bool(has_meta),
                "in_events": bool(in_events),
                "categoryid": int(cid),
                "root_categoryid": int(root_cid),
                "category_depth": int(c_depth),
                "category_path": c_path,
                "latest_available": int(last_avail),
                "ever_available": int(ever_avail),
                "property_observations_count": int(item_prop_count.get(iid, 0)),
                "hashed_tokens_signature": tokens_str,
            }
        )

    items_metadata_df = pd.DataFrame(item_records)
    items_metadata_df["itemid"] = items_metadata_df["itemid"].astype("int64")
    items_metadata_df["categoryid"] = items_metadata_df["categoryid"].astype("int32")
    items_metadata_df["root_categoryid"] = items_metadata_df["root_categoryid"].astype("int32")
    items_metadata_df["category_depth"] = items_metadata_df["category_depth"].astype("int16")
    items_metadata_df["latest_available"] = items_metadata_df["latest_available"].astype("int8")
    items_metadata_df["ever_available"] = items_metadata_df["ever_available"].astype("int8")
    items_metadata_df["property_observations_count"] = items_metadata_df[
        "property_observations_count"
    ].astype("int32")

    metadata_stats = {
        "total_universe_items": len(all_universe_items),
        "items_in_properties_catalog": len(catalog_items_seen),
        "interacted_items_total": len(interacted_item_ids),
        "interacted_items_with_metadata": int(
            items_metadata_df.loc[
                items_metadata_df["in_events"] & items_metadata_df["has_metadata"]
            ].shape[0]
        ),
        "orphan_interacted_items_without_metadata": int(
            items_metadata_df.loc[
                items_metadata_df["in_events"] & (~items_metadata_df["has_metadata"])
            ].shape[0]
        ),
        "cold_catalog_items_without_events": int(
            items_metadata_df.loc[
                (~items_metadata_df["in_events"]) & items_metadata_df["has_metadata"]
            ].shape[0]
        ),
        "total_taxonomy_categories": int(len(taxonomy_df)),
    }

    return items_metadata_df, taxonomy_df, metadata_stats


def run_phase2_preprocessing_pipeline(
    paths: ProjectPaths,
    config: PreprocessingConfig = DEFAULT_PREPROCESSING_CONFIG,
) -> dict[str, Any]:
    """Exécute le pipeline complet de la Phase 2 et sauvegarde les tables dans `data/processed/`."""
    paths.ensure_directories()

    # 1. Chargement et nettoyage de events.csv
    raw_events = load_events(paths.events_path)
    logger.info("Nettoyage, déduplication et découpage en sessions de events.csv...")
    clean_events, events_stats = clean_and_sessionize_events(raw_events, config=config)
    del raw_events

    # 2. Pondération des interactions (visitorid, itemid)
    logger.info("Calcul de la matrice d'interactions pondérées (User-Item)...")
    weighted_interactions = build_weighted_user_item_interactions(
        clean_events, weight_config=config.weights
    )

    # 3. Consolidation de la taxonomie et des métadonnées produits en streaming
    interacted_items = set(clean_events["itemid"].unique().tolist())
    category_df = load_category_tree(paths.category_tree_path)
    prop_paths = [p for p in paths.item_properties_paths if p.exists()]

    logger.info("Consolidation en streaming des métadonnées produits et catégories...")
    items_metadata_df, taxonomy_df, metadata_stats = consolidate_item_metadata_streaming(
        item_properties_paths=prop_paths,
        category_df=category_df,
        interacted_item_ids=interacted_items,
        config=config,
    )

    # 4. Enrichissement de clean_events avec la catégorie du produit (utile pour User/Item features)
    item_to_cat = dict(
        zip(
            items_metadata_df["itemid"].tolist(),
            items_metadata_df["categoryid"].tolist(),
        )
    )
    clean_events["categoryid"] = (
        clean_events["itemid"].map(item_to_cat).fillna(UNKNOWN_CATEGORY_ID).astype("int32")
    )

    # 5. Sauvegarde au format Parquet dans data/processed/
    events_out = paths.processed_data_dir / "events_clean.parquet"
    interactions_out = paths.processed_data_dir / "user_item_interactions.parquet"
    items_out = paths.processed_data_dir / "items_metadata.parquet"
    taxonomy_out = paths.processed_data_dir / "category_taxonomy.parquet"

    logger.info("Sauvegarde des datasets propres dans %s...", paths.processed_data_dir)
    clean_events.to_parquet(events_out, index=False)
    weighted_interactions.to_parquet(interactions_out, index=False)
    items_metadata_df.to_parquet(items_out, index=False)
    taxonomy_df.to_parquet(taxonomy_out, index=False)

    interaction_summary = {
        "unique_user_item_pairs": int(len(weighted_interactions)),
        "confirmed_purchase_pairs": int(
            weighted_interactions["is_confirmed_purchase"].sum()
        ),
        "cart_addition_pairs": int(weighted_interactions["has_cart_addition"].sum()),
        "raw_preference_score_quantiles": {
            "min": round(float(weighted_interactions["raw_preference_score"].min()), 4),
            "median": round(
                float(weighted_interactions["raw_preference_score"].median()), 4
            ),
            "p90": round(
                float(weighted_interactions["raw_preference_score"].quantile(0.90)), 4
            ),
            "p99": round(
                float(weighted_interactions["raw_preference_score"].quantile(0.99)), 4
            ),
            "max": round(float(weighted_interactions["raw_preference_score"].max()), 4),
        },
        "implicit_confidence_score_quantiles": {
            "min": round(
                float(weighted_interactions["implicit_confidence_score"].min()), 4
            ),
            "median": round(
                float(weighted_interactions["implicit_confidence_score"].median()), 4
            ),
            "p90": round(
                float(
                    weighted_interactions["implicit_confidence_score"].quantile(0.90)
                ),
                4,
            ),
            "p99": round(
                float(
                    weighted_interactions["implicit_confidence_score"].quantile(0.99)
                ),
                4,
            ),
            "max": round(
                float(weighted_interactions["implicit_confidence_score"].max()), 4
            ),
        },
    }

    return {
        "phase": "Phase 2 - Data Preprocessing",
        "config": asdict(config),
        "events_cleaning_and_sessions": events_stats,
        "weighted_interactions_summary": interaction_summary,
        "items_and_taxonomy_summary": metadata_stats,
        "output_files": {
            "events_clean": str(events_out),
            "user_item_interactions": str(interactions_out),
            "items_metadata": str(items_out),
            "category_taxonomy": str(taxonomy_out),
        },
    }
