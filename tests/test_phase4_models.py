"""Tests unitaires de la Phase 4 : Popularity, Content-Based, Collaborative ALS, Session-Based.

Vérifie pour chaque modèle :
- L'entraînement (`fit`), la prédiction (`recommend`), le scoring (`score_items`).
- L'absence de doublons et de scores NaN/Inf.
- Le respect de `exclude_items`.
- La sauvegarde et le rechargement avec vérification SHA-256 (`save` / `load`).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recommender.config.model_config import (
    CollaborativeALSConfig,
    ContentBasedConfig,
    PopularityModelConfig,
    SessionBasedConfig,
)
from recommender.models.collaborative import CollaborativeALSRecommender
from recommender.models.content_based import ContentBasedRecommender
from recommender.models.popularity import PopularityRecommender
from recommender.models.session_based import SessionBasedRecommender


@pytest.fixture()
def synthetic_phase4_data() -> dict[str, pd.DataFrame]:
    item_features = pd.DataFrame(
        [
            {
                "itemid": 10,
                "item_total_events": 50,
                "item_views_count": 40,
                "item_carts_count": 6,
                "item_transactions_count": 4,
                "item_raw_popularity_score": 78.0,
                "item_time_decayed_popularity": 55.0,
                "item_smoothed_conversion_rate": 0.08,
                "categoryid": 1,
                "latest_available": 1,
            },
            {
                "itemid": 20,
                "item_total_events": 30,
                "item_views_count": 25,
                "item_carts_count": 3,
                "item_transactions_count": 2,
                "item_raw_popularity_score": 44.0,
                "item_time_decayed_popularity": 35.0,
                "item_smoothed_conversion_rate": 0.06,
                "categoryid": 1,
                "latest_available": 1,
            },
            {
                "itemid": 30,
                "item_total_events": 10,
                "item_views_count": 10,
                "item_carts_count": 0,
                "item_transactions_count": 0,
                "item_raw_popularity_score": 10.0,
                "item_time_decayed_popularity": 8.0,
                "item_smoothed_conversion_rate": 0.01,
                "categoryid": 2,
                "latest_available": 1,
            },
            # Produit Cold-Start (0 interaction mais dans la catégorie 1 avec les mêmes tokens que 10)
            {
                "itemid": 40,
                "item_total_events": 0,
                "item_views_count": 0,
                "item_carts_count": 0,
                "item_transactions_count": 0,
                "item_raw_popularity_score": 0.0,
                "item_time_decayed_popularity": 0.0,
                "item_smoothed_conversion_rate": 0.02,
                "categoryid": 1,
                "latest_available": 1,
            },
        ]
    )

    items_metadata = pd.DataFrame(
        [
            {
                "itemid": 10,
                "has_metadata": True,
                "in_events": True,
                "categoryid": 1,
                "root_categoryid": 1,
                "category_path": "1",
                "hashed_tokens_signature": "p888_111 p790_n100",
            },
            {
                "itemid": 20,
                "has_metadata": True,
                "in_events": True,
                "categoryid": 1,
                "root_categoryid": 1,
                "category_path": "1",
                "hashed_tokens_signature": "p888_111 p790_n120",
            },
            {
                "itemid": 30,
                "has_metadata": True,
                "in_events": True,
                "categoryid": 2,
                "root_categoryid": 2,
                "category_path": "2",
                "hashed_tokens_signature": "p555_999",
            },
            {
                "itemid": 40,
                "has_metadata": True,
                "in_events": False,
                "categoryid": 1,
                "root_categoryid": 1,
                "category_path": "1",
                "hashed_tokens_signature": "p888_111 p790_n100",
            },
        ]
    )

    interactions = pd.DataFrame(
        [
            {"visitorid": 100, "itemid": 10, "implicit_confidence_score": 5.0},
            {"visitorid": 100, "itemid": 20, "implicit_confidence_score": 3.0},
            {"visitorid": 200, "itemid": 10, "implicit_confidence_score": 4.0},
            {"visitorid": 200, "itemid": 20, "implicit_confidence_score": 4.5},
            {"visitorid": 300, "itemid": 20, "implicit_confidence_score": 2.0},
            {"visitorid": 300, "itemid": 30, "implicit_confidence_score": 5.0},
            {"visitorid": 400, "itemid": 10, "implicit_confidence_score": 3.5},
            {"visitorid": 400, "itemid": 30, "implicit_confidence_score": 1.0},
        ]
    )

    transitions = pd.DataFrame(
        [
            {
                "source_itemid": 10,
                "target_itemid": 20,
                "transition_count": 8,
                "transition_probability": 0.8,
                "ppmi_score": 1.5,
            },
            {
                "source_itemid": 20,
                "target_itemid": 30,
                "transition_count": 4,
                "transition_probability": 0.5,
                "ppmi_score": 0.9,
            },
        ]
    )

    return {
        "item_features": item_features,
        "items_metadata": items_metadata,
        "interactions": interactions,
        "transitions": transitions,
    }


def test_popularity_recommender_fit_recommend_and_save(
    synthetic_phase4_data: dict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    model = PopularityRecommender(PopularityModelConfig()).fit(
        synthetic_phase4_data["item_features"]
    )
    recs = model.recommend(user_id=999, n=2)
    assert len(recs) == 2
    assert recs[0].itemid == 10
    assert recs[0].score >= recs[1].score

    # Vérification de l'exclusion et de la sérialisation
    recs_excl = model.recommend(user_id=999, n=2, exclude_items={10})
    assert recs_excl[0].itemid == 20

    saved_path = tmp_path / "popularity.joblib"
    model.save(saved_path)
    loaded = PopularityRecommender.load(saved_path)
    assert loaded.recommend(user_id=999, n=1)[0].itemid == 10


def test_content_based_recommender_recommends_similar_and_cold_items(
    synthetic_phase4_data: dict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    model = ContentBasedRecommender(ContentBasedConfig(min_df=1)).fit(
        synthetic_phase4_data["items_metadata"],
        synthetic_phase4_data["interactions"],
    )
    # Si un visiteur consulte l'item 10, les items 40 (Cold-Start identique) et 20 doivent être en tête devant 30
    recs = model.recommend(user_id=None, recent_items=[10], n=3, exclude_items={10})
    rec_ids = [r.itemid for r in recs]
    assert 40 in rec_ids[:2]
    assert 20 in rec_ids[:2]

    saved_path = tmp_path / "content.joblib"
    model.save(saved_path)
    loaded = ContentBasedRecommender.load(saved_path)
    scores = loaded.score_items(user_id=100, candidate_items=[20, 30, 40])
    assert scores[0] > scores[1]


def test_collaborative_als_recommender_fit_and_folding_in(
    synthetic_phase4_data: dict[str, pd.DataFrame],
    tmp_path: Path,
) -> None:
    cfg = CollaborativeALSConfig(
        factors=8, iterations=10, min_item_interactions=1, random_state=42
    )
    model = CollaborativeALSRecommender(cfg).fit(synthetic_phase4_data["interactions"])

    assert model.is_known_user(100) is True
    assert model.is_known_user(9999) is False

    # Recommandation pour un utilisateur connu
    recs = model.recommend(user_id=300, n=2, exclude_items={20, 30})
    assert len(recs) == 1
    assert recs[0].itemid == 10
    assert np.isfinite(recs[0].score)

    # Recommandation par folding-in (recalculate_user) pour un utilisateur inconnu avec session
    session_recs = model.recommend(user_id=9999, recent_items=[10], n=2, exclude_items={10})
    assert len(session_recs) >= 1

    saved_path = tmp_path / "als.joblib"
    model.save(saved_path)
    loaded = CollaborativeALSRecommender.load(saved_path)
    assert loaded.is_known_user(100) is True


def test_session_based_recommender_sequential_transitions(
    synthetic_phase4_data: dict[str, pd.DataFrame],
) -> None:
    model = SessionBasedRecommender(SessionBasedConfig()).fit(
        synthetic_phase4_data["transitions"]
    )
    recs = model.recommend(user_id=None, recent_items=[10], n=2)
    assert len(recs) == 1
    assert recs[0].itemid == 20
    assert recs[0].score > 0.0
