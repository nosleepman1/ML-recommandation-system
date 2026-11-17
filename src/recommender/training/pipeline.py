"""Pipeline complet d'entraînement, d'évaluation et d'export de production (Phases 7, 8, 10)."""

import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

import pandas as pd

from recommender.config.model_config import (
    DEFAULT_ALS_CONFIG,
    DEFAULT_CONTENT_CONFIG,
    DEFAULT_HYBRID_CONFIG,
    DEFAULT_POPULARITY_CONFIG,
    DEFAULT_RANKER_CONFIG,
    DEFAULT_SESSION_CONFIG,
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
from recommender.training.trainer import build_ltr_training_dataset
from recommender.utils.logging import get_logger

logger = get_logger(__name__)


def train_all_models(
    train_events_df: pd.DataFrame,
    val_events_df: pd.DataFrame,
    user_features_df: pd.DataFrame,
    item_features_df: pd.DataFrame,
    items_metadata_df: pd.DataFrame,
    interaction_features_df: pd.DataFrame,
    transitions_df: pd.DataFrame,
    pop_cfg: PopularityModelConfig = DEFAULT_POPULARITY_CONFIG,
    content_cfg: ContentBasedConfig = DEFAULT_CONTENT_CONFIG,
    als_cfg: CollaborativeALSConfig = DEFAULT_ALS_CONFIG,
    session_cfg: SessionBasedConfig = DEFAULT_SESSION_CONFIG,
    hybrid_cfg: HybridEngineConfig = DEFAULT_HYBRID_CONFIG,
    ranker_cfg: LGBMRankerConfig = DEFAULT_RANKER_CONFIG,
    max_ltr_users: int = 800,
) -> tuple[dict[str, Any], dict[str, float]]:
    """Entraîne les 4 modèles de base, le moteur Hybrid Weighted et le modèle Hybrid LTR."""
    timings: dict[str, float] = {}

    # 1. Popularity-Based
    t0 = time.perf_counter()
    logger.info("Entraînement de PopularityRecommender...")
    pop_model = PopularityRecommender(pop_cfg).fit(item_features_df)
    timings["popularity_fit_sec"] = round(time.perf_counter() - t0, 3)

    # 2. Content-Based
    t0 = time.perf_counter()
    logger.info("Entraînement de ContentBasedRecommender (TF-IDF + Taxonomie)...")
    content_model = ContentBasedRecommender(content_cfg).fit(
        items_metadata_df=items_metadata_df,
        interactions_df=interaction_features_df,
    )
    timings["content_based_fit_sec"] = round(time.perf_counter() - t0, 3)

    # 3. Collaborative Filtering (Implicit ALS)
    t0 = time.perf_counter()
    logger.info("Entraînement de CollaborativeALSRecommender (Implicit ALS CSR)...")
    als_model = CollaborativeALSRecommender(als_cfg).fit(interaction_features_df)
    timings["collaborative_als_fit_sec"] = round(time.perf_counter() - t0, 3)

    # 4. Session-Based
    t0 = time.perf_counter()
    logger.info("Entraînement de SessionBasedRecommender (Transitions & PPMI)...")
    session_model = SessionBasedRecommender(session_cfg).fit(
        transitions_df=transitions_df,
        train_events_df=train_events_df,
    )
    timings["session_based_fit_sec"] = round(time.perf_counter() - t0, 3)

    # 5. Hybrid Weighted
    t0 = time.perf_counter()
    logger.info("Construction de HybridRecommender (Weighted Adaptive)...")
    hybrid_weighted = HybridRecommender(
        popularity_model=pop_model,
        content_model=content_model,
        collaborative_model=als_model,
        session_model=session_model,
        config=hybrid_cfg,
        use_ltr=False,
    ).fit(interaction_features_df)
    timings["hybrid_weighted_fit_sec"] = round(time.perf_counter() - t0, 3)

    # 6. Hybrid Learning to Rank (LightGBM Ranker)
    t0 = time.perf_counter()
    logger.info("Construction du dataset de classement et entraînement de LightGBMRanker...")
    feature_builder = RankingFeatureBuilder(
        user_features_df=user_features_df,
        item_features_df=item_features_df,
    )
    ltr_train_df = build_ltr_training_dataset(
        hybrid_base=hybrid_weighted,
        feature_builder=feature_builder,
        val_events_df=val_events_df,
        max_users=max_ltr_users,
        candidates_per_model=25,
    )
    ranker_model = LightGBMRankerModel(ranker_cfg)
    if not ltr_train_df.empty:
        ranker_model.fit(ltr_train_df)

    hybrid_ltr = HybridRecommender(
        popularity_model=pop_model,
        content_model=content_model,
        collaborative_model=als_model,
        session_model=session_model,
        config=hybrid_cfg,
        feature_builder=feature_builder,
        ranker=ranker_model,
        use_ltr=not ltr_train_df.empty,
    ).fit(interaction_features_df)
    timings["hybrid_ltr_fit_sec"] = round(time.perf_counter() - t0, 3)

    models_bundle = {
        "popularity": pop_model,
        "content_based": content_model,
        "collaborative_als": als_model,
        "session_based": session_model,
        "hybrid_weighted": hybrid_weighted,
        "hybrid_ltr": hybrid_ltr,
        "ranker": ranker_model,
        "feature_builder": feature_builder,
    }
    return models_bundle, timings
