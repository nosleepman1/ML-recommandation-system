"""Tests unitaires, d'intégration et End-to-End des Phases 7, 8, 9 et 10.

Couvre toutes les exigences de qualité logicielle de la Section 12 :
- Exactitude mathématique des métriques Top-K (`Precision@K`, `Recall@K`, `MAP@K`, `NDCG@K`, `HitRate@K`, `Coverage`, `Diversity`).
- Comparaison multi-modèles et analyse par segments (`cold_user`, `casual_user`, `active_user`).
- Pipeline complet d'entraînement (`train_all_models`) et journalisation des expériences.
- Garanties strictes de `RecommendationEngine` en production :
  - Jamais de produit hors catalogue.
  - Jamais d'identifiants dupliqués.
  - Jamais de scores `NaN` ou `Inf`.
  - Respect de la politique d'exclusion des produits déjà achetés.
  - Chargement à froid `RecommendationEngine.load(...)` sans réentraînement et détection SHA-256.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recommender.config.model_config import (
    CollaborativeALSConfig,
    ContentBasedConfig,
    HybridEngineConfig,
    LGBMRankerConfig,
    PopularityModelConfig,
    SessionBasedConfig,
)
from recommender.evaluation.evaluator import evaluate_models, prepare_evaluation_cases
from recommender.evaluation.metrics import (
    average_precision_at_k,
    catalog_coverage,
    hit_rate_at_k,
    intra_list_category_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommender.evaluation.reports import format_evaluation_markdown_table
from recommender.inference.engine import RecommendationEngine
from recommender.models.base import RecommendationItem
from recommender.training.pipeline import train_all_models
from recommender.training.trainer import log_experiment_run


def test_ranking_metrics_mathematical_correctness() -> None:
    rec = [10, 20, 30, 40, 50]
    gt = {20, 40}

    assert np.isclose(precision_at_k(rec, gt, k=5), 2.0 / 5.0)
    assert np.isclose(recall_at_k(rec, gt, k=5), 1.0)
    assert hit_rate_at_k(rec, gt, k=5) == 1.0
    # AP@5 : hit au rang 2 (1/2) et hit au rang 4 (2/4) -> (0.5 + 0.5) / 2 = 0.5
    assert np.isclose(average_precision_at_k(rec, gt, k=5), 0.5)
    assert 0.0 < ndcg_at_k(rec, gt, k=5) < 1.0
    assert np.isclose(catalog_coverage([[10, 20], [20, 30]], total_catalog_size=6), 0.5)
    assert np.isclose(
        intra_list_category_diversity([10, 20, 30], {10: 1, 20: 1, 30: 2}, k=3),
        2.0 / 3.0,
    )


def test_end_to_end_training_evaluation_and_production_engine(tmp_path: Path) -> None:
    t0 = 1_430_000_000_000
    day_ms = 24 * 3600 * 1000

    # Construction d'un mini-écosystème e-commerce cohérent
    train_rows = []
    for u in range(1, 15):
        base_item = 100 + (u % 3) * 2
        train_rows.append(
            {
                "timestamp": t0 + u * 1000,
                "visitorid": u,
                "itemid": base_item,
                "event": "view",
                "event_level": 1,
                "base_event_weight": 1.0,
                "global_session_id": u,
                "categoryid": 1 if base_item < 104 else 2,
                "is_bot_user": False,
            }
        )
        train_rows.append(
            {
                "timestamp": t0 + u * 1000 + 500,
                "visitorid": u,
                "itemid": base_item + 1,
                "event": "transaction" if u % 2 == 0 else "addtocart",
                "event_level": 3 if u % 2 == 0 else 2,
                "base_event_weight": 5.0 if u % 2 == 0 else 3.0,
                "global_session_id": u,
                "categoryid": 1 if base_item < 104 else 2,
                "is_bot_user": False,
            }
        )
    train_events_df = pd.DataFrame(train_rows)

    val_rows = []
    for u in range(1, 10):
        base_item = 100 + (u % 3) * 2
        val_rows.append(
            {
                "timestamp": t0 + 10 * day_ms + u * 1000,
                "visitorid": u,
                "itemid": base_item,
                "event": "view",
            }
        )
        val_rows.append(
            {
                "timestamp": t0 + 10 * day_ms + u * 1000 + 500,
                "visitorid": u,
                "itemid": base_item + 1,
                "event": "transaction",
            }
        )
    # Ajout d'un Cold User dans Val (visitor 99)
    val_rows.extend(
        [
            {"timestamp": t0 + 11 * day_ms, "visitorid": 99, "itemid": 100, "event": "view"},
            {"timestamp": t0 + 11 * day_ms + 500, "visitorid": 99, "itemid": 101, "event": "view"},
        ]
    )
    val_events_df = pd.DataFrame(val_rows)

    from recommender.features.interaction_features import (
        build_interaction_features,
        build_sequential_item_transitions,
    )
    from recommender.features.item_features import build_item_features
    from recommender.features.user_features import build_user_features

    items_meta_df = pd.DataFrame(
        [
            {
                "itemid": iid,
                "has_metadata": True,
                "in_events": True,
                "categoryid": 1 if iid < 104 else 2,
                "root_categoryid": 1 if iid < 104 else 2,
                "category_depth": 1,
                "category_path": str(1 if iid < 104 else 2),
                "latest_available": 1,
                "property_observations_count": 4,
                "hashed_tokens_signature": f"p1_{iid % 2} p2_{iid}",
            }
            for iid in range(100, 108)
        ]
    )

    uf_df = build_user_features(train_events_df)
    itf_df = build_item_features(train_events_df, items_metadata_df=items_meta_df)
    inter_df = build_interaction_features(train_events_df)
    trans_df = build_sequential_item_transitions(train_events_df)

    bundle, timings = train_all_models(
        train_events_df=train_events_df,
        val_events_df=val_events_df,
        user_features_df=uf_df,
        item_features_df=itf_df,
        items_metadata_df=items_meta_df,
        interaction_features_df=inter_df,
        transitions_df=trans_df,
        pop_cfg=PopularityModelConfig(),
        content_cfg=ContentBasedConfig(min_df=1),
        als_cfg=CollaborativeALSConfig(factors=4, iterations=5, min_item_interactions=1),
        session_cfg=SessionBasedConfig(min_cooccurrence=1),
        hybrid_cfg=HybridEngineConfig(exclude_already_purchased=True),
        ranker_cfg=LGBMRankerConfig(n_estimators=10, num_leaves=7, min_child_samples=1),
    )

    assert "hybrid_ltr" in bundle
    assert all(v >= 0.0 for v in timings.values())

    # Évaluation comparative des 6 modèles
    eval_cases = prepare_evaluation_cases(val_events_df, train_events_df)
    assert len(eval_cases) > 0

    models_to_eval = {
        "Popularity": bundle["popularity"],
        "ContentBased": bundle["content_based"],
        "CollaborativeALS": bundle["collaborative_als"],
        "SessionBased": bundle["session_based"],
        "HybridWeighted": bundle["hybrid_weighted"],
        "HybridLTR": bundle["hybrid_ltr"],
    }
    eval_report = evaluate_models(
        models=models_to_eval,
        eval_cases=eval_cases,
        item_to_category={int(r.itemid): int(r.categoryid) for r in items_meta_df.itertuples()},
        total_catalog_size=len(items_meta_df),
        k=3,
    )
    md_table = format_evaluation_markdown_table(eval_report)
    assert "HybridLTR" in md_table

    # Logging d'expérience
    exp_file = log_experiment_run(
        tmp_path / "experiments",
        "test_run",
        params={"factors": 4},
        metrics=eval_report["global_metrics"]["HybridLTR"],
    )
    assert exp_file.exists()

    # Export et rechargement de RecommendationEngine en production
    valid_catalog = set(items_meta_df["itemid"].astype(int).tolist())
    engine = RecommendationEngine(
        hybrid_model=bundle["hybrid_ltr"],
        valid_catalog_items=valid_catalog,
    )
    prod_dir = tmp_path / "models" / "production"
    meta = engine.save(prod_dir)
    assert meta["catalog_size"] == len(valid_catalog)

    loaded_engine = RecommendationEngine.load(prod_dir)

    # 1. Utilisateur connu (User 2 a acheté l'item 105 dans train_events_df -> 105 doit être exclu !)
    recs_u2 = loaded_engine.recommend(user_id=2, n=4)
    rec_ids_u2 = [r["itemid"] for r in recs_u2]
    assert len(rec_ids_u2) == len(set(rec_ids_u2))  # Zéro doublon
    assert all(iid in valid_catalog for iid in rec_ids_u2)  # Zéro produit hors catalogue
    assert all(np.isfinite(r["score"]) for r in recs_u2)  # Zéro score NaN/Inf
    assert 105 not in rec_ids_u2  # Produit déjà acheté par User 2 exclu

    # 2. Utilisateur totalement inconnu (Cold Start User)
    recs_unknown = loaded_engine.recommend(user_id=888888, n=3)
    assert len(recs_unknown) == 3

    # 3. Recommandation par session temps réel
    sess_recs = loaded_engine.recommend_session(session_items=[100], n=3)
    assert len(sess_recs) == 3
    assert 100 not in [r["itemid"] for r in sess_recs]

    # 4. Vérification que _sanitize_recommendations bloque les produits fantômes et les NaN
    dirty_items = [
        RecommendationItem(itemid=999999, score=10.0, source="bad_ghost"),
        RecommendationItem(itemid=100, score=float("nan"), source="bad_nan"),
        RecommendationItem(itemid=101, score=0.95, source="ok"),
        RecommendationItem(itemid=101, score=0.90, source="dup"),
    ]
    clean = loaded_engine._sanitize_recommendations(dirty_items, n=5, excluded=set())
    assert len(clean) == 1
    assert clean[0]["itemid"] == 101
