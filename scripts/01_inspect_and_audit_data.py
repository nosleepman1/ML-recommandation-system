"""Script exécutable de la Phase 1 : Inspection et Audit complet du dataset Retailrocket.

Usage :
    python scripts/01_inspect_and_audit_data.py
"""

import argparse
import json
import sys
from pathlib import Path

# Garantit que `src/` est dans le PYTHONPATH même sans `pip install -e .`
PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommender.config.settings import AuditSettings, ProjectPaths
from recommender.data.loader import DataLoadError
from recommender.data.validator import run_full_phase1_audit
from recommender.utils.logging import get_logger

logger = get_logger("phase1_audit")


def print_executive_summary(report: dict) -> None:
    """Affiche dans la console une synthèse structurée et lisible des résultats d'audit."""
    print("\n" + "=" * 88)
    print(" SMARTRECOMMEND — RAPPORT D'AUDIT DES DONNÉES (PHASE 1 : RETAILROCKET)")
    print("=" * 88)

    print("\n[4.1] INSPECTION STRUCTURELLE DES FICHIERS CSV")
    print("-" * 88)
    for f in report["files_inspection_4_1"]:
        print(
            f" • {f['file_name']:<28} | {f['num_rows']:>10,} lignes | "
            f"{f['file_size_mb']:>7.2f} Mo | Doublons exacts: {f['exact_duplicate_rows']:,} "
            f"({f['exact_duplicate_percentage']:.3f}%)"
        )
        print(f"   Colonnes & Types : {f['dtypes']}")
        print(f"   Valeurs manquantes (%) : {f['missing_percentages']}")
        if f["timestamp_min_utc"]:
            print(
                f"   Plage temporelle UTC : {f['timestamp_min_utc']} -> {f['timestamp_max_utc']}"
            )
        if f["anomalies"]:
            print(f"   [!] Anomalies détectées : {f['anomalies']}")

    inter = report["interactions_and_temporal_4_2"]
    s = inter["summary"]
    eb = inter["event_breakdown"]
    dist = inter["distributions"]
    temp = inter["temporal_analysis"]

    print("\n[4.2] ANALYSE DES INTERACTIONS (events.csv)")
    print("-" * 88)
    print(f" • Total interactions          : {s['total_interactions']:,}")
    print(f" • Utilisateurs uniques        : {s['unique_visitors']:,}")
    print(f" • Produits uniques (events)   : {s['unique_items_interacted']:,}")
    print(f" • Paires (User, Item) uniques : {s['unique_user_item_pairs']:,}")
    print(f" • Sparsité de la matrice      : {s['matrix_sparsity_percent']:.6f}% (Densité: {s['matrix_density_percent']:.6f}%)")
    print(f" • Doublons fonctionnels       : {s['functional_duplicates_count']:,}")
    print(f" • Répartition événements      : {eb['counts']} ({eb['percentages']}%)")
    print(f" • Ratios entonnoir            : {eb['conversion_rates']}")
    print(f" • Concentration Long Tail     : {dist['long_tail_concentration']}")
    print(
        f" • Fenêtre temporelle          : {temp['observation_span_days']} jours "
        f"({temp['min_datetime_utc']} -> {temp['max_datetime_utc']})"
    )

    prod = report["product_and_taxonomy_4_3"]
    ct = prod["category_tree"]
    ip = prod["item_properties"]
    cov = ip["interacted_items_coverage"]

    print("\n[4.3] ANALYSE DES PRODUITS & CATÉGORIES")
    print("-" * 88)
    print(
        f" • Arbre des catégories        : {ct['unique_category_ids']:,} catégories, "
        f"{ct['root_categories_count']} racines, profondeur max = {ct['max_tree_depth']}, "
        f"cycles = {ct['cycles_detected']}"
    )
    print(
        f" • Lignes item_properties      : {ip['total_property_rows_combined']:,} lignes cumulées "
        f"pour {ip['unique_items_in_properties']:,} produits distincts et "
        f"{ip['unique_properties_count']:,} propriétés uniques."
    )
    print(
        f" • Couverture métadonnées      : {cov['interacted_items_with_properties_count']:,} / "
        f"{cov['total_interacted_items_in_events']:,} produits interagis ont des propriétés "
        f"({cov['interacted_items_with_properties_pct']:.2f}%). "
        f"Produits orphelins sans propriétés : {cov['interacted_items_missing_properties_count']:,} "
        f"({cov['interacted_items_missing_properties_pct']:.2f}%)."
    )

    beh = report["behavioral_segmentation_4_4"]
    print("\n[4.4] ANALYSE COMPORTEMENTALE & SEGMENTATION UTILISATEURS")
    print("-" * 88)
    print(
        f" • Utilisateurs à 1 seule interaction : {beh['single_interaction_users_count']:,} "
        f"({beh['single_interaction_users_pct']:.2f}%)"
    )
    print(
        f" • Utilisateurs occasionnels (<=2)    : {beh['casual_users_le_2_events_count']:,} "
        f"({beh['casual_users_le_2_events_pct']:.2f}%)"
    )
    print(
        f" • Utilisateurs actifs (>=10)         : {beh['active_users_ge_10_events_count']:,} "
        f"({beh['active_users_ge_10_events_pct']:.2f}%)"
    )
    print(
        f" • Visiteurs 'View Only'              : {beh['view_only_users_count']:,} "
        f"({beh['view_only_users_pct']:.2f}%)"
    )
    print(
        f" • Abandonnistes de panier            : {beh['cart_abandoner_users_count']:,} "
        f"({beh['cart_abandoner_users_pct']:.2f}%)"
    )
    print(
        f" • Acheteurs (au moins 1 transaction) : {beh['buyer_users_count']:,} "
        f"({beh['buyer_users_pct']:.2f}%)"
    )
    print("=" * 88 + "\n")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phase 1 : Inspection et audit scientifique du dataset Retailrocket."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Chemin optionnel vers la racine du projet (par défaut : racine du dépôt).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=500_000,
        help="Taille des chunks pour la lecture de item_properties_part1/2.csv.",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Désactive la génération des graphiques PNG.",
    )
    args = parser.parse_args()

    paths = ProjectPaths.from_root(args.data_root)
    paths.ensure_directories()
    settings = AuditSettings(chunk_size=args.chunk_size)

    try:
        report = run_full_phase1_audit(
            paths=paths,
            settings=settings,
            generate_plots=not args.no_plots,
        )
    except (DataLoadError, FileNotFoundError) as exc:
        logger.error("Audit interrompu — Fichiers bruts manquants : %s", exc)
        logger.info(
            "Fichiers attendus dans '%s' :\n"
            "  1. events.csv\n"
            "  2. item_properties_part1.csv\n"
            "  3. item_properties_part2.csv\n"
            "  4. category_tree.csv",
            paths.raw_data_dir,
        )
        return 1

    output_json = paths.reports_dir / "phase1_data_audit.json"
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print_executive_summary(report)
    logger.info("Rapport JSON complet sauvegardé dans : %s", output_json)
    if report["generated_figures"]:
        logger.info("Graphiques générés dans : %s", paths.figures_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
