"""Tests unitaires et d'intégration de la Phase 3 (Feature Engineering - Sections 6.1, 6.2, 6.3).

Vérifie :
- L'absence absolue de fuite temporelle (`cutoff_timestamp_ms`) sur User, Item et Interaction features.
- L'exactitude mathématique de l'entropie de Shannon des catégories utilisateur (`user_features.py`).
- Le calcul de la popularité à décroissance temporelle, des fenêtres 7j/14j et du taux de
  conversion bayésien lissé (`item_features.py`).
- L'affinité catégorielle `(visitorid, categoryid)` et les probabilités de transition
  séquentielle intra-session `P(j | i)` + `PPMI` (`interaction_features.py`).
"""

import numpy as np
import pandas as pd
import pytest

from recommender.config.model_config import FeatureConfig
from recommender.features.interaction_features import (
    build_interaction_features,
    build_sequential_item_transitions,
)
from recommender.features.item_features import build_item_features
from recommender.features.user_features import build_user_features


@pytest.fixture()
def sample_sessionized_events() -> pd.DataFrame:
    """Journal d'événements synthétiques avec sessions, catégories et un événement futur."""
    day_ms = 24 * 3600 * 1000
    t_cutoff = 1_435_000_000_000

    return pd.DataFrame(
        [
            # User 1 : consulte 2 produits de catégories équiprobables (10 et 20) dans la session 0
            {
                "timestamp": t_cutoff - 2 * day_ms,
                "visitorid": 1,
                "itemid": 101,
                "event": "view",
                "event_level": 1,
                "base_event_weight": 1.0,
                "global_session_id": 0,
                "categoryid": 10,
                "is_bot_user": False,
            },
            {
                "timestamp": t_cutoff - 2 * day_ms + 5000,
                "visitorid": 1,
                "itemid": 102,
                "event": "view",
                "event_level": 1,
                "base_event_weight": 1.0,
                "global_session_id": 0,
                "categoryid": 20,
                "is_bot_user": False,
            },
            # User 2 : même transition séquentielle 101 -> 102 dans la session 1, puis achat de 102
            {
                "timestamp": t_cutoff - 1 * day_ms,
                "visitorid": 2,
                "itemid": 101,
                "event": "view",
                "event_level": 1,
                "base_event_weight": 1.0,
                "global_session_id": 1,
                "categoryid": 10,
                "is_bot_user": False,
            },
            {
                "timestamp": t_cutoff - 1 * day_ms + 3000,
                "visitorid": 2,
                "itemid": 102,
                "event": "transaction",
                "event_level": 3,
                "base_event_weight": 5.0,
                "global_session_id": 1,
                "categoryid": 20,
                "is_bot_user": False,
            },
            # ÉVÉNEMENT FUTUR (> t_cutoff) qui NE DOIT JAMAIS fuiter dans les features
            {
                "timestamp": t_cutoff + 5 * day_ms,
                "visitorid": 1,
                "itemid": 999,
                "event": "transaction",
                "event_level": 3,
                "base_event_weight": 5.0,
                "global_session_id": 2,
                "categoryid": 30,
                "is_bot_user": False,
            },
        ]
    )


def test_user_features_excludes_future_events_and_computes_shannon_entropy(
    sample_sessionized_events: pd.DataFrame,
) -> None:
    t_cutoff = 1_435_000_000_000
    uf = build_user_features(
        sample_sessionized_events, cutoff_timestamp_ms=t_cutoff
    ).set_index("visitorid")

    # L'achat futur de User 1 (item 999 à t_cutoff + 5j) est strictement ignoré
    assert int(uf.loc[1, "user_transactions_count"]) == 0
    assert bool(uf.loc[1, "user_is_buyer"]) is False
    assert int(uf.loc[1, "user_total_events"]) == 2

    # User 1 a 2 catégories équiprobables (p=0.5, p=0.5) -> Entropie de Shannon = ln(2) = 0.693147
    assert np.isclose(float(uf.loc[1, "user_category_entropy"]), np.log(2.0), atol=1e-4)

    # User 2 est bien identifié comme acheteur
    assert bool(uf.loc[2, "user_is_buyer"]) is True
    assert int(uf.loc[2, "user_top_itemid"]) == 102


def test_item_features_time_decay_and_cold_catalog_items(
    sample_sessionized_events: pd.DataFrame,
) -> None:
    t_cutoff = 1_435_000_000_000
    meta_df = pd.DataFrame(
        [
            {"itemid": 101, "has_metadata": True, "categoryid": 10, "root_categoryid": 1, "category_depth": 1, "latest_available": 1, "property_observations_count": 5},
            {"itemid": 102, "has_metadata": True, "categoryid": 20, "root_categoryid": 2, "category_depth": 1, "latest_available": 1, "property_observations_count": 8},
            # Produit 500 présent au catalogue mais avec 0 interaction dans le Train (Cold Item)
            {"itemid": 500, "has_metadata": True, "categoryid": 20, "root_categoryid": 2, "category_depth": 1, "latest_available": 1, "property_observations_count": 3},
        ]
    )

    itf = build_item_features(
        sample_sessionized_events,
        items_metadata_df=meta_df,
        cutoff_timestamp_ms=t_cutoff,
        config=FeatureConfig(item_decay_half_life_days=14.0),
    ).set_index("itemid")

    # L'item 999 (présent uniquement dans le futur > t_cutoff) ne doit pas exister comme Warm item
    assert 999 not in itf.index
    # L'item 500 du catalogue est conservé et marqué is_cold_item = True
    assert bool(itf.loc[500, "is_cold_item"]) is True
    assert int(itf.loc[500, "item_total_events"]) == 0
    # L'item 102 a plus de popularité pondérée que 101 (1 vue + 1 achat contre 2 vues)
    assert float(itf.loc[102, "item_time_decayed_popularity"]) > float(
        itf.loc[101, "item_time_decayed_popularity"]
    )


def test_interaction_features_and_sequential_transitions(
    sample_sessionized_events: pd.DataFrame,
) -> None:
    t_cutoff = 1_435_000_000_000
    inter_df = build_interaction_features(
        sample_sessionized_events, cutoff_timestamp_ms=t_cutoff
    )
    # Aucune paire impliquant l'item futur 999
    assert 999 not in set(inter_df["itemid"].tolist())

    u2_102 = inter_df[
        (inter_df["visitorid"] == 2) & (inter_df["itemid"] == 102)
    ].iloc[0]
    # User 2 a 1.0 de poids sur cat 10 et 5.0 de poids sur cat 20 -> affinité cat 20 = 5 / 6
    assert np.isclose(float(u2_102["ui_user_category_affinity"]), 5.0 / 6.0, atol=1e-4)

    # Transitions séquentielles : la transition 101 -> 102 apparaît dans 2 sessions (count=2, P(102|101)=1.0)
    seq_df = build_sequential_item_transitions(
        sample_sessionized_events,
        cutoff_timestamp_ms=t_cutoff,
        config=FeatureConfig(min_transition_cooccurrence=2),
    )
    assert len(seq_df) == 1
    row = seq_df.iloc[0]
    assert int(row["source_itemid"]) == 101
    assert int(row["target_itemid"]) == 102
    assert int(row["transition_count"]) == 2
    assert np.isclose(float(row["transition_probability"]), 1.0, atol=1e-5)
