"""Script exécutable de la Phase 2 : Prétraitement et pondération des données Retailrocket.

Usage :
    python scripts/02_preprocess_data.py
"""

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommender.config.model_config import InteractionWeightConfig, PreprocessingConfig
from recommender.config.settings import ProjectPaths
from recommender.data.preprocessing import run_phase2_preprocessing_pipeline
from recommender.utils.logging import get_logger

logger = get_logger("phase2_preprocessing")


def print_phase2_summary(report: dict) -> None:
    """Affiche une synthèse claire des transformations de la Phase 2."""
    print("\n" + "=" * 88)
    print(" SMARTRECOMMEND — RAPPORT DE PRÉTRAITEMENT DES DONNÉES (PHASE 2)")
    print("=" * 88)

    ev = report["events_cleaning_and_sessions"]
    print("\n[5.1] NETTOYAGE DES ÉVÉNEMENTS & RECONSTRUCTION DES SESSIONS")
    print("-" * 88)
    print(f" • Événements bruts en entrée         : {ev['raw_events_count']:,}")
    print(f" • Lignes invalides écartées          : {ev['invalid_rows_dropped']:,}")
    print(f" • Doublons exacts supprimés          : {ev['exact_duplicates_dropped']:,}")
    print(f" • Événements propres conservés       : {ev['clean_events_count']:,}")
    print(f" • Sessions utilisateurs reconstruites: {ev['total_sessions_constructed']:,}")
    print(f" • Utilisateurs Warm (>=2 événements) : {ev['warm_users_ge_min_interactions_count']:,}")
    print(
        f" • Utilisateurs Bots identifiés (>1000): {ev['flagged_bot_users_count']:,} "
        f"({ev['flagged_bot_events_count']:,} événements)"
    )

    wi = report["weighted_interactions_summary"]
    print("\n[5.2] PONDÉRATION DES INTERACTIONS UTILISATEUR-PRODUIT")
    print("-" * 88)
    print(f" • Paires (visitorid, itemid) uniques : {wi['unique_user_item_pairs']:,}")
    print(f" • Paires avec ajout au panier        : {wi['cart_addition_pairs']:,}")
    print(f" • Paires avec achat confirmé         : {wi['confirmed_purchase_pairs']:,}")
    print(f" • Quantiles Score de Préférence Brut : {wi['raw_preference_score_quantiles']}")
    print(f" • Quantiles Score Confiance + Récence: {wi['implicit_confidence_score_quantiles']}")

    meta = report["items_and_taxonomy_summary"]
    print("\n[5.3] CONSOLIDATION DU CATALOGUE PRODUITS & TAXONOMIE")
    print("-" * 88)
    print(f" • Univers total de produits unifiés  : {meta['total_universe_items']:,}")
    print(f" • Produits interagis avec métadonnées: {meta['interacted_items_with_metadata']:,}")
    print(f" • Produits interagis orphelins       : {meta['orphan_interacted_items_without_metadata']:,}")
    print(f" • Produits catalogue Cold-Start pur  : {meta['cold_catalog_items_without_events']:,}")
    print(f" • Catégories consolidées (avec extra): {meta['total_taxonomy_categories']:,}")

    print("\n[5.4] ARTEFACTS PARQUET GÉNÉRÉS DANS data/processed/")
    print("-" * 88)
    for k, path_str in report["output_files"].items():
        p = Path(path_str)
        size_mb = p.stat().st_size / (1024**2) if p.exists() else 0.0
        print(f" • {k:<24} : {p.name:<32} ({size_mb:>6.2f} Mo)")
    print("=" * 88 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 2 : Prétraitement reproductible et pondération du dataset Retailrocket."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--view-weight", type=float, default=1.0)
    parser.add_argument("--cart-weight", type=float, default=3.0)
    parser.add_argument("--tx-weight", type=float, default=5.0)
    parser.add_argument("--half-life-days", type=float, default=30.0)
    parser.add_argument("--session-minutes", type=int, default=30)
    parser.add_argument("--filter-bots", action="store_true")
    args = parser.parse_args()

    paths = ProjectPaths.from_root(args.data_root)
    weight_cfg = InteractionWeightConfig(
        view_weight=args.view_weight,
        addtocart_weight=args.cart_weight,
        transaction_weight=args.tx_weight,
        recency_half_life_days=args.half_life_days,
    )
    config = PreprocessingConfig(
        weights=weight_cfg,
        session_inactivity_minutes=args.session_minutes,
        filter_bot_users_in_clean_events=args.filter_bots,
    )

    report = run_phase2_preprocessing_pipeline(paths=paths, config=config)
    report_path = paths.reports_dir / "phase2_preprocessing_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_phase2_summary(report)
    logger.info("Rapport JSON Phase 2 sauvegardé dans : %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
