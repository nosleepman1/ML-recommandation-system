"""Tests unitaires et d'intégration des Phases 5 et 6 (Hybrid Engine, LTR, Cold Start, Policies)."""

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
from recommender.models.collaborative import CollaborativeALSRecommender
from recommender.models.content_based import ContentBasedRecommender
from recommender.models.hybrid import HybridRecommender
from recommender.models.popularity import PopularityRecommender
from recommender.models.session_based import SessionBasedRecommender
from recommender.ranking.feature_builder import RankingFeatureBuilder
from recommender.ranking.ranker import LightGBMRankerModel
from recommender.ranking.score_normalizer import normalize_scores


def test_score_normalizer_methods() -> None:
    raw = np.array([10.0, 20.0, 5.0, 0.0])
    for method in ("min_max", "z_score", "percentile", "rrf"):
        normed = normalize_scores(raw, method=method)
        assert normed.shape == raw.shape
        assert np.all(np.isfinite(normed))
        assert normed[1] > normed[0] > normed[2] > normed[3]
        assert normed[3] == 0.0


def test_hybrid_engine_cold_start_purchased_policy_and_ltr() -> None:
    item_features = pd.DataFrame(
        [
            {
                "itemid": 10 + i,
                "item_total_events": 50 - i * 5,
                "item_views_count": 40 - i * 4,
                "item_carts_count": 5,
                "item_transactions_count": 3,
                "item_raw_popularity_score": float(60 - i * 5),
                "item_time_decayed_popularity": float(45 - i * 4),
                "item_events_last_7d": 10,
                "item_smoothed_conversion_rate": 0.05,
                "item_trend_velocity_7d_vs_30d": 1.1,
                "item_recency_days": 1.0,
                "categoryid": 1 if i < 3 else 2,
                "latest_available": 1,
            }
            for i in range(6)
        ]
    )
    items_meta = pd.DataFrame(
        [
            {
                "itemid": 10 + i,
                "has_metadata": True,
                "in_events": True,
                "categoryid": 1 if i < 3 else 2,
                "root_categoryid": 1 if i < 3 else 2,
                "category_path": str(1 if i < 3 else 2),
                "hashed_tokens_signature": f"p1_{i % 2} p2_{i}",
            }
            for i in range(6)
        ]
    )
    interactions = pd.DataFrame(
        [
            {"visitorid": 1, "itemid": 10, "implicit_confidence_score": 5.0, "total_interactions": 3, "is_confirmed_purchase": True},
            {"visitorid": 1, "itemid": 11, "implicit_confidence_score": 3.0, "total_interactions": 2, "is_confirmed_purchase": False},
            {"visitorid": 2, "itemid": 11, "implicit_confidence_score": 4.0, "total_interactions": 2, "is_confirmed_purchase": True},
            {"visitorid": 2, "itemid": 12, "implicit_confidence_score": 4.5, "total_interactions": 4, "is_confirmed_purchase": False},
            {"visitorid": 3, "itemid": 13, "implicit_confidence_score": 3.5, "total_interactions": 5, "is_confirmed_purchase": False},
            {"visitorid": 3, "itemid": 14, "implicit_confidence_score": 2.5, "total_interactions": 2, "is_confirmed_purchase": False},
        ]
    )
    transitions = pd.DataFrame(
        [
            {"source_itemid": 10, "target_itemid": 12, "transition_count": 5, "transition_probability": 0.7, "ppmi_score": 1.2},
            {"source_itemid": 11, "target_itemid": 13, "transition_count": 4, "transition_probability": 0.6, "ppmi_score": 1.0},
        ]
    )
    user_features = pd.DataFrame(
        [
            {
                "visitorid": u,
                "user_total_events": 5,
                "user_views_count": 4,
                "user_carts_count": 1,
                "user_transactions_count": 1,
                "user_recency_days": 2.0,
                "user_category_entropy": 0.5,
                "user_repeat_interaction_ratio": 0.2,
                "user_top_categoryid": 1,
            }
            for u in (1, 2, 3)
        ]
    )

    pop = PopularityRecommender(PopularityModelConfig()).fit(item_features)
    cont = ContentBasedRecommender(ContentBasedConfig(min_df=1)).fit(items_meta, interactions)
    als = CollaborativeALSRecommender(
        CollaborativeALSConfig(factors=4, iterations=5, min_item_interactions=1)
    ).fit(interactions)
    sess = SessionBasedRecommender(SessionBasedConfig()).fit(transitions)

    hybrid = HybridRecommender(
        popularity_model=pop,
        content_model=cont,
        collaborative_model=als,
        session_model=sess,
        config=HybridEngineConfig(exclude_already_purchased=True),
    ).fit(interactions)

    # 1. Politique d'exclusion des produits déjà achetés : User 1 a acheté l'item 10 -> 10 ne doit pas sortir
    recs_u1 = hybrid.recommend(user_id=1, n=4)
    rec_ids_u1 = [r.itemid for r in recs_u1]
    assert 10 not in rec_ids_u1
    assert len(rec_ids_u1) == len(set(rec_ids_u1))

    # Si on autorise explicitement le réachat (consommables), 10 peut sortir
    recs_u1_allow = hybrid.recommend(user_id=1, n=6, exclude_already_purchased=False)
    assert 10 in [r.itemid for r in recs_u1_allow]

    # 2. Cold Start User (visitor 99999 inconnu sans session)
    cold_recs = hybrid.recommend(user_id=99999, n=3)
    assert len(cold_recs) == 3
    assert all(np.isfinite(r.score) for r in cold_recs)

    # 3. Entraînement et inférence Learning to Rank (LightGBM Ranker)
    fb = RankingFeatureBuilder(user_features, item_features)
    cand_rows = []
    for uid in (1, 2, 3):
        cdf = hybrid.candidate_generator.generate_candidates_df(user_id=uid, k_per_model=6)
        cdf["relevance_label"] = np.where(cdf["itemid"] % 2 == 0, 2, 0)
        cand_rows.append(cdf)
    train_cand_df = fb.build_features(pd.concat(cand_rows, ignore_index=True))

    ranker = LightGBMRankerModel(
        LGBMRankerConfig(n_estimators=10, num_leaves=7, min_child_samples=1)
    ).fit(train_cand_df)

    hybrid_ltr = HybridRecommender(
        popularity_model=pop,
        content_model=cont,
        collaborative_model=als,
        session_model=sess,
        feature_builder=fb,
        ranker=ranker,
        use_ltr=True,
    ).fit(interactions)

    ltr_recs = hybrid_ltr.recommend(user_id=1, n=3)
    assert len(ltr_recs) == 3
    assert ltr_recs[0].source == "hybrid_ltr"
