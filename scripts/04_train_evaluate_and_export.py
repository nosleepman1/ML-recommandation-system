"""Script exécutable des Phases 4 à 10 : Entraînement, Évaluation Multi-Modèles et Export Production.

Exécute sur les données Retailrocket de `data/processed/` :
1. L'entraînement des 4 modèles de base (`Popularity`, `Content-Based`, `Collaborative ALS`,
   `Session-Based`), du moteur `Hybrid Weighted` et du modèle `Hybrid LightGBM Ranker`.
2. L'évaluation comparative rigoureuse sur le jeu de test temporel indépendant (`test_events.parquet`)
   ainsi que l'analyse par segment (`cold_user`, `casual_user`, `active_user`, `LongTail`).
3. Le suivi des expériences (JSON + MLflow) dans `reports/experiments/`.
4. L'export du moteur de production dans `models/production/` et le test en conditions réelles
   de `RecommendationEngine.load("models/production")` sans réentraînement.
"""

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from recommender.config.settings import ProjectPaths
from recommender.evaluation.evaluator import evaluate_models, prepare_evaluation_cases
from recommender.evaluation.reports import format_evaluation_markdown_table
from recommender.inference.engine import RecommendationEngine
from recommender.training.pipeline import train_all_models
from recommender.training.trainer import log_experiment_run
from recommender.utils.logging import get_logger

logger = get_logger("phase4_10_pipeline")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Phases 4 à 10 : Entraînement, évaluation et export de production."
    )
    parser.add_argument("--data-root", type=Path, default=None)
    parser.add_argument("--eval-users", type=int, default=1200)
    parser.add_argument("--ltr-users", type=int, default=600)
    parser.add_argument("--k", type=int, default=10)
    args = parser.parse_args()

    paths = ProjectPaths.from_root(args.data_root)
    proc = paths.processed_data_dir

    logger.info("Chargement des tables Parquet de la Phase 2 et de la Phase 3...")
    train_events_df = pd.read_parquet(proc / "train_events.parquet")
    val_events_df = pd.read_parquet(proc / "val_events.parquet")
    test_events_df = pd.read_parquet(proc / "test_events.parquet")
    user_features_df = pd.read_parquet(proc / "user_features_train.parquet")
    item_features_df = pd.read_parquet(proc / "item_features_train.parquet")
    items_metadata_df = pd.read_parquet(proc / "items_metadata.parquet")
    interaction_features_df = pd.read_parquet(proc / "interaction_features_train.parquet")
    transitions_df = pd.read_parquet(proc / "sequential_transitions_train.parquet")

    # 1. Entraînement de tous les modèles
    models_bundle, timings = train_all_models(
        train_events_df=train_events_df,
        val_events_df=val_events_df,
        user_features_df=user_features_df,
        item_features_df=item_features_df,
        items_metadata_df=items_metadata_df,
        interaction_features_df=interaction_features_df,
        transitions_df=transitions_df,
        max_ltr_users=args.ltr_users,
    )

    # 2. Préparation des cas d'évaluation sur le Test Set temporel indépendant
    logger.info(
        "Préparation des cas d'évaluation sur test_events.parquet (max_eval_users=%d)...",
        args.eval_users,
    )
    test_cases = prepare_evaluation_cases(
        eval_events_df=test_events_df,
        train_events_df=train_events_df,
        max_eval_users=args.eval_users,
    )

    # Identification des produits Long Tail (hors Top 10% des produits les plus vus dans le Train)
    warm_items = item_features_df[item_features_df["item_total_events"] > 0]
    p90_events = float(warm_items["item_total_events"].quantile(0.90))
    long_tail_items = set(
        item_features_df.loc[
            item_features_df["item_total_events"] <= p90_events, "itemid"
        ].astype(int).tolist()
    )
    item_to_category = dict(
        zip(
            items_metadata_df["itemid"].astype(int).tolist(),
            items_metadata_df["categoryid"].astype(int).tolist(),
        )
    )

    models_for_eval = {
        "1. Popularity Baseline": models_bundle["popularity"],
        "2. Content-Based": models_bundle["content_based"],
        "3. Collaborative ALS": models_bundle["collaborative_als"],
        "4. Session-Based": models_bundle["session_based"],
        "5. Hybrid Weighted": models_bundle["hybrid_weighted"],
        "6. Hybrid LightGBM LTR": models_bundle["hybrid_ltr"],
    }

    logger.info("Évaluation comparative des 6 modèles sur le Test Set temporel...")
    eval_report = evaluate_models(
        models=models_for_eval,
        eval_cases=test_cases,
        item_to_category=item_to_category,
        total_catalog_size=int(len(warm_items)),
        long_tail_items=long_tail_items,
        k=args.k,
    )

    md_table = format_evaluation_markdown_table(eval_report)
    print("\n" + "=" * 96)
    print(" SMARTRECOMMEND — COMPARAISON EXPÉRIMENTALE SUR LE JEU DE TEST TEMPOREL (PHASE 7)")
    print("=" * 96)
    print(md_table)
    print("=" * 96)

    # 3. Sélection du meilleur modèle hybride pour l'export de production
    w_ndcg = eval_report["global_metrics"]["5. Hybrid Weighted"][f"NDCG@{args.k}"]
    ltr_ndcg = eval_report["global_metrics"]["6. Hybrid LightGBM LTR"][f"NDCG@{args.k}"]
    best_hybrid_key = "hybrid_ltr" if ltr_ndcg >= w_ndcg else "hybrid_weighted"
    best_hybrid_model = models_bundle[best_hybrid_key]

    # 4. Sauvegarde et test de rechargement du moteur de production (Phase 10)
    prod_dir = paths.models_dir / "production"
    valid_catalog = set(items_metadata_df["itemid"].astype(int).tolist())
    engine = RecommendationEngine(
        hybrid_model=best_hybrid_model,
        valid_catalog_items=valid_catalog,
    )
    logger.info("Sauvegarde du moteur de production dans %s...", prod_dir)
    prod_metadata = engine.save(prod_dir)

    logger.info("Test de rechargement à froid via RecommendationEngine.load('%s')...", prod_dir)
    reloaded_engine = RecommendationEngine.load(prod_dir)

    sample_warm_user = int(test_cases[0]["visitorid"])
    sample_warm_recs = reloaded_engine.recommend(user_id=sample_warm_user, n=5)
    sample_cold_recs = reloaded_engine.recommend(user_id=999_999_999, n=5)
    sample_session_recs = reloaded_engine.recommend_session(
        session_items=[187946, 461686], n=5
    )

    # 5. Sauvegarde des rapports et du suivi d'expérience
    log_experiment_run(
        experiment_dir=paths.reports_dir / "experiments",
        run_name="phase8_full_benchmark",
        params={"eval_users": len(test_cases), "k": args.k, "selected_production_model": best_hybrid_key},
        metrics=eval_report["global_metrics"]["6. Hybrid LightGBM LTR"],
    )

    md_path = paths.reports_dir / "model_comparison.md"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("# SmartRecommend — Résultats Expérimentaux sur le Test Set Temporel\n\n")
        f.write(md_table + "\n")

    full_output = {
        "phase": "Phases 4 to 10 - Models, Hybrid, Evaluation, Optimization & Production",
        "training_timings_sec": timings,
        "ltr_feature_importances_pct": models_bundle["ranker"].get_feature_importances(),
        "evaluation": eval_report,
        "production_export": {
            "selected_model": best_hybrid_key,
            "metadata": prod_metadata,
            "sample_warm_user_id": sample_warm_user,
            "sample_warm_recommendations": sample_warm_recs,
            "sample_cold_user_recommendations": sample_cold_recs,
            "sample_session_recommendations": sample_session_recs,
        },
    }

    report_path = paths.reports_dir / "phase7_10_evaluation_and_production_report.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(full_output, f, indent=2, ensure_ascii=False)

    logger.info("Rapport complet sauvegardé dans : %s", report_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())
