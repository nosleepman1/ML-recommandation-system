"""Script exécutable de la Phase 3 : Découpage Temporel & Feature Engineering (Sections 6.1, 6.2, 6.3).

Exécute :
1. Le chargement des tables nettoyées de la Phase 2 (`events_clean.parquet`, `items_metadata.parquet`).
2. Le découpage temporel strict Train (80%) / Validation (10%) / Test (10%) sans fuite.
3. La construction des features User, Item, Interaction et Transitions Séquentielles
   en utilisant EXCLUSIVEMENT la fenêtre temporelle d'entraînement (`timestamp <= train_cutoff_ts_ms`).
4. La sauvegarde des tables Parquet et du rapport `reports/phase3_feature_engineering_report.json`.
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommender.config.model_config import (
    DEFAULT_FEATURE_CONFIG,
    DEFAULT_TEMPORAL_SPLIT_CONFIG,
)
from recommender.config.settings import ProjectPaths
from recommender.data.splitting import temporal_train_val_test_split
from recommender.features.interaction_features import (
    build_interaction_features,
    build_sequential_item_transitions,
)
from recommender.features.item_features import build_item_features
from recommender.features.user_features import build_user_features
from recommender.utils.logging import get_logger

logger = get_logger("phase3_features")


def run_phase3_feature_pipeline(paths: ProjectPaths) -> dict[str, Any]:
    """Exécute le découpage temporel et construit toutes les features sur le Train set."""
    events_path = paths.processed_data_dir / "events_clean.parquet"
    items_meta_path = paths.processed_data_dir / "items_metadata.parquet"

    if not events_path.exists() or not items_meta_path.exists():
        raise FileNotFoundError(
            "Fichiers Parquet de Phase 2 introuvables dans data/processed/. "
            "Exécutez d'abord scripts/02_preprocess_data.py."
        )

    logger.info("Chargement de events_clean.parquet et items_metadata.parquet...")
    events_df = pd.read_parquet(events_path)
    items_meta_df = pd.read_parquet(items_meta_path)

    logger.info("Application du découpage temporel strict Train (80%) / Val (10%) / Test (10%)...")
    split = temporal_train_val_test_split(events_df, config=DEFAULT_TEMPORAL_SPLIT_CONFIG)
    del events_df

    # Sauvegarde des 3 splits temporels
    train_out = paths.processed_data_dir / "train_events.parquet"
    val_out = paths.processed_data_dir / "val_events.parquet"
    test_out = paths.processed_data_dir / "test_events.parquet"
    split.train_df.to_parquet(train_out, index=False)
    split.val_df.to_parquet(val_out, index=False)
    split.test_df.to_parquet(test_out, index=False)

    cutoff_ts = split.train_cutoff_ts_ms

    # 6.1 User Features (sur Train uniquement)
    logger.info("Construction des User Features (cutoff=%s)...", split.summary["train_cutoff_utc"])
    user_features_df = build_user_features(split.train_df, cutoff_timestamp_ms=cutoff_ts)

    # 6.2 Item Features (sur Train uniquement + catalogue complet items_metadata)
    logger.info("Construction des Item Features (cutoff=%s)...", split.summary["train_cutoff_utc"])
    item_features_df = build_item_features(
        split.train_df,
        items_metadata_df=items_meta_df,
        cutoff_timestamp_ms=cutoff_ts,
        config=DEFAULT_FEATURE_CONFIG,
    )

    # 6.3 Interaction Features & Sequential Transitions (sur Train uniquement)
    logger.info("Construction des Interaction Features (User x Item)...")
    interaction_features_df = build_interaction_features(
        split.train_df,
        cutoff_timestamp_ms=cutoff_ts,
    )

    logger.info("Construction de la matrice des transitions séquentielles intra-session...")
    transitions_df = build_sequential_item_transitions(
        split.train_df,
        cutoff_timestamp_ms=cutoff_ts,
        config=DEFAULT_FEATURE_CONFIG,
    )

    # Sauvegarde des artefacts de features dans data/processed/
    uf_out = paths.processed_data_dir / "user_features_train.parquet"
    if_out = paths.processed_data_dir / "item_features_train.parquet"
    inter_out = paths.processed_data_dir / "interaction_features_train.parquet"
    seq_out = paths.processed_data_dir / "sequential_transitions_train.parquet"

    user_features_df.to_parquet(uf_out, index=False)
    item_features_df.to_parquet(if_out, index=False)
    interaction_features_df.to_parquet(inter_out, index=False)
    transitions_df.to_parquet(seq_out, index=False)

    report = {
        "phase": "Phase 3 - Temporal Split & Feature Engineering",
        "temporal_split": split.summary,
        "user_features_summary": {
            "num_users": int(len(user_features_df)),
            "num_features": int(user_features_df.shape[1]),
            "columns": list(user_features_df.columns),
            "buyers_in_train_count": int(user_features_df["user_is_buyer"].sum()),
            "cart_abandoners_in_train_count": int(
                user_features_df["user_is_cart_abandoner"].sum()
            ),
            "mean_category_entropy": round(
                float(user_features_df["user_category_entropy"].mean()), 4
            ),
            "p95_category_entropy": round(
                float(user_features_df["user_category_entropy"].quantile(0.95)), 4
            ),
        },
        "item_features_summary": {
            "num_items_total": int(len(item_features_df)),
            "num_features": int(item_features_df.shape[1]),
            "columns": list(item_features_df.columns),
            "warm_items_in_train_count": int((~item_features_df["is_cold_item"]).sum()),
            "cold_items_in_train_count": int(item_features_df["is_cold_item"].sum()),
            "items_active_last_7d_before_cutoff": int(
                (item_features_df["item_events_last_7d"] > 0).sum()
            ),
            "max_time_decayed_popularity": round(
                float(item_features_df["item_time_decayed_popularity"].max()), 4
            ),
        },
        "interaction_and_sequential_summary": {
            "num_user_item_pairs_train": int(len(interaction_features_df)),
            "num_interaction_features": int(interaction_features_df.shape[1]),
            "mean_user_category_affinity": round(
                float(interaction_features_df["ui_user_category_affinity"].mean()), 4
            ),
            "num_sequential_transitions_filtered": int(len(transitions_df)),
            "unique_source_items_with_transitions": int(
                transitions_df["source_itemid"].nunique()
            ),
        },
        "output_files": {
            "train_events": str(train_out),
            "val_events": str(val_out),
            "test_events": str(test_out),
            "user_features_train": str(uf_out),
            "item_features_train": str(if_out),
            "interaction_features_train": str(inter_out),
            "sequential_transitions_train": str(seq_out),
        },
    }
    return report


def print_phase3_summary(report: dict[str, Any]) -> None:
    print("\n" + "=" * 88)
    print(" SMARTRECOMMEND — RAPPORT PHASE 3 : SPLIT TEMPOREL & FEATURE ENGINEERING")
    print("=" * 88)

    sp = report["temporal_split"]
    print("\n[6.0] DÉCOUPAGE TEMPOREL ANTI-FUITE (80% / 10% / 10%)")
    print("-" * 88)
    print(f" • Train Set (<= {sp['train_cutoff_utc']}) : {sp['train_events']:>10,} événements")
    print(f" • Val Set   (<= {sp['val_cutoff_utc']}) : {sp['val_events']:>10,} événements")
    print(f" • Test Set  (>  {sp['val_cutoff_utc']}) : {sp['test_events']:>10,} événements")
    print(
        f" • Utilisateurs Val  (Warm / Cold) : {sp['val_warm_users_in_train_count']:,} Warm | "
        f"{sp['val_cold_users_count']:,} Cold"
    )
    print(
        f" • Utilisateurs Test (Warm / Cold) : {sp['test_warm_users_in_train_count']:,} Warm | "
        f"{sp['test_cold_users_count']:,} Cold"
    )

    uf = report["user_features_summary"]
    print("\n[6.1] USER FEATURES (TRAIN SET UNIQUEMENT)")
    print("-" * 88)
    print(f" • Utilisateurs profilés           : {uf['num_users']:,} ({uf['num_features']} colonnes)")
    print(f" • Acheteurs dans Train            : {uf['buyers_in_train_count']:,}")
    print(f" • Abandonnistes panier dans Train : {uf['cart_abandoners_in_train_count']:,}")
    print(
        f" • Entropie catégorielle (Shannon) : moyenne = {uf['mean_category_entropy']:.4f}, "
        f"P95 = {uf['p95_category_entropy']:.4f}"
    )

    itf = report["item_features_summary"]
    print("\n[6.2] ITEM FEATURES (CATALOGUE UNIFIÉ + DYNAMIQUE TRAIN)")
    print("-" * 88)
    print(f" • Produits profilés (univers)     : {itf['num_items_total']:,} ({itf['num_features']} colonnes)")
    print(f" • Produits Warm (interagis Train) : {itf['warm_items_in_train_count']:,}")
    print(f" • Produits Cold (0 événement Train): {itf['cold_items_in_train_count']:,}")
    print(f" • Produits actifs sur les 7 derniers jours du Train : {itf['items_active_last_7d_before_cutoff']:,}")

    seq = report["interaction_and_sequential_summary"]
    print("\n[6.3] INTERACTION FEATURES & TRANSITIONS SÉQUENTIELLES")
    print("-" * 88)
    print(f" • Paires (User, Item) enrichies   : {seq['num_user_item_pairs_train']:,}")
    print(f" • Affinité catégorielle moyenne   : {seq['mean_user_category_affinity']:.4f}")
    print(
        f" • Transitions séquentielles (i->j): {seq['num_sequential_transitions_filtered']:,} "
        f"paires fortes couvrant {seq['unique_source_items_with_transitions']:,} produits sources"
    )
    print("=" * 88 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(description="Phase 3 : Feature Engineering.")
    parser.add_argument("--data-root", type=Path, default=None)
    args = parser.parse_args()

    paths = ProjectPaths.from_root(args.data_root)
    report = run_phase3_feature_pipeline(paths)

    report_path = paths.reports_dir / "phase3_feature_engineering_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_phase3_summary(report)
    logger.info("Rapport JSON Phase 3 sauvegardé dans : %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
