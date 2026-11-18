"""Tests unitaires et d'intégration de la Phase 2 (Data Preprocessing - Section 5).

Vérifie :
- La suppression des doublons exacts et des événements invalides.
- La reconstruction des sessions utilisateur selon le seuil d'inactivité (30 min).
- La hiérarchie mathématique des poids d'interaction (amortissement logarithmique des
  consultations répétées vs achat confirmé, décroissance exponentielle de récence).
- La construction de la taxonomie des catégories (y compris les catégories hors-arbre).
- La consolidation des métadonnées produits et le marquage des produits orphelins.
- L'écriture des fichiers Parquet dans `data/processed/` sans altérer `data/raw/`.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from recommender.config.model_config import InteractionWeightConfig, PreprocessingConfig
from recommender.config.settings import ProjectPaths
from recommender.data.preprocessing import (
    UNKNOWN_CATEGORY_ID,
    build_category_taxonomy_table,
    build_weighted_user_item_interactions,
    clean_and_sessionize_events,
    consolidate_item_metadata_streaming,
    run_phase2_preprocessing_pipeline,
)


def test_clean_and_sessionize_events_deduplicates_and_splits_sessions() -> None:
    t0 = 1_430_000_000_000
    min_ms = 60_000
    events_df = pd.DataFrame(
        [
            # Session 1 pour visitor 1 (t0 et t0 + 10 min) + 1 doublon exact
            {"timestamp": t0, "visitorid": 1, "event": "view", "itemid": 10, "transactionid": pd.NA},
            {"timestamp": t0, "visitorid": 1, "event": "view", "itemid": 10, "transactionid": pd.NA},
            {"timestamp": t0 + 10 * min_ms, "visitorid": 1, "event": "addtocart", "itemid": 10, "transactionid": pd.NA},
            # Session 2 pour visitor 1 (t0 + 45 min > seuil de 30 min)
            {"timestamp": t0 + 45 * min_ms, "visitorid": 1, "event": "transaction", "itemid": 10, "transactionid": 99},
            # Ligne invalide (événement inconnu)
            {"timestamp": t0, "visitorid": 2, "event": "invalid_event", "itemid": 20, "transactionid": pd.NA},
        ]
    )

    cleaned, stats = clean_and_sessionize_events(
        events_df, config=PreprocessingConfig(session_inactivity_minutes=30)
    )

    assert stats["raw_events_count"] == 5
    assert stats["invalid_rows_dropped"] == 1
    assert stats["exact_duplicates_dropped"] == 1
    assert stats["clean_events_count"] == 3
    assert stats["total_sessions_constructed"] == 2
    assert cleaned["user_session_seq"].tolist() == [1, 1, 2]


def test_weighted_interactions_respects_intent_hierarchy_and_recency() -> None:
    day_ms = 24 * 3600 * 1000
    ref_ts = 1_440_000_000_000

    # User 1 consulte 10 fois l'item 100 (aucun achat) à t = ref_ts - 30 jours (1 demi-vie)
    views_only = [
        {
            "timestamp": ref_ts - 30 * day_ms,
            "visitorid": 1,
            "event": "view",
            "itemid": 100,
            "transactionid": pd.NA,
            "event_level": 1,
        }
        for _ in range(10)
    ]
    # User 2 achète 1 fois l'item 200 à t = ref_ts (aujourd'hui)
    single_purchase = [
        {
            "timestamp": ref_ts,
            "visitorid": 2,
            "event": "transaction",
            "itemid": 200,
            "transactionid": 777,
            "event_level": 3,
        }
    ]
    df = pd.DataFrame(views_only + single_purchase)

    cfg = InteractionWeightConfig(
        view_weight=1.0,
        addtocart_weight=3.0,
        transaction_weight=5.0,
        repeat_damping_factor=0.5,
        recency_half_life_days=30.0,
    )
    weighted = build_weighted_user_item_interactions(
        df, weight_config=cfg, reference_timestamp_ms=ref_ts
    ).set_index("visitorid")

    # Vérification 1 : 10 vues amorties valent 1 * (1 + 0.5 * ln(10)) = 2.151 < 3.0 (cart) < 5.0 (achat)
    u1_raw = float(weighted.loc[1, "raw_preference_score"])
    u2_raw = float(weighted.loc[2, "raw_preference_score"])
    assert np.isclose(u1_raw, 1.0 * (1.0 + 0.5 * np.log(10.0)), atol=1e-4)
    assert u1_raw < cfg.addtocart_weight < u2_raw
    assert bool(weighted.loc[1, "is_confirmed_purchase"]) is False
    assert bool(weighted.loc[2, "is_confirmed_purchase"]) is True

    # Vérification 2 : Décroissance temporelle après exactement 1 demi-vie (30 jours) = 0.5
    assert np.isclose(float(weighted.loc[1, "recency_weight"]), 0.5, atol=1e-4)
    assert np.isclose(float(weighted.loc[2, "recency_weight"]), 1.0, atol=1e-4)


def test_category_taxonomy_handles_out_of_tree_categories() -> None:
    cat_df = pd.DataFrame(
        [
            {"categoryid": 10, "parentid": pd.NA},
            {"categoryid": 20, "parentid": 10},
        ]
    )
    tax = build_category_taxonomy_table(cat_df, extra_category_ids={999}).set_index("categoryid")
    assert len(tax) == 3
    assert tax.loc[20, "root_categoryid"] == 10
    assert tax.loc[20, "depth"] == 1
    assert tax.loc[20, "category_path"] == "10/20"
    assert tax.loc[999, "root_categoryid"] == 999
    assert bool(tax.loc[999, "in_original_tree"]) is False


def test_full_phase2_pipeline_end_to_end(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure_directories()

    pd.DataFrame(
        [
            {"timestamp": 1433221000000, "visitorid": 1, "event": "view", "itemid": 100, "transactionid": pd.NA},
            {"timestamp": 1433221050000, "visitorid": 1, "event": "transaction", "itemid": 100, "transactionid": 1},
            {"timestamp": 1433222000000, "visitorid": 2, "event": "view", "itemid": 300, "transactionid": pd.NA},
        ]
    ).to_csv(paths.events_path, index=False)

    pd.DataFrame([{"categoryid": 5, "parentid": pd.NA}]).to_csv(
        paths.category_tree_path, index=False
    )

    pd.DataFrame(
        [
            {"timestamp": 1433000000000, "itemid": 100, "property": "categoryid", "value": "5"},
            {"timestamp": 1433000000000, "itemid": 100, "property": "available", "value": "1"},
        ]
    ).to_csv(paths.item_properties_part1_path, index=False)
    pd.DataFrame(
        [
            {"timestamp": 1433000000000, "itemid": 200, "property": "categoryid", "value": "5"},
        ]
    ).to_csv(paths.item_properties_part2_path, index=False)

    report = run_phase2_preprocessing_pipeline(paths=paths)
    assert Path(report["output_files"]["events_clean"]).exists()
    assert Path(report["output_files"]["user_item_interactions"]).exists()
    assert Path(report["output_files"]["items_metadata"]).exists()

    meta_df = pd.read_parquet(report["output_files"]["items_metadata"]).set_index("itemid")
    # Item 300 est dans events.csv mais pas dans item_properties -> orphan item (has_metadata=False, categoryid=-1)
    assert bool(meta_df.loc[300, "has_metadata"]) is False
    assert int(meta_df.loc[300, "categoryid"]) == UNKNOWN_CATEGORY_ID
    # Item 100 a des métadonnées et categoryid=5
    assert bool(meta_df.loc[100, "has_metadata"]) is True
    assert int(meta_df.loc[100, "categoryid"]) == 5
