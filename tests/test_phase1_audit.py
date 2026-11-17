"""Tests unitaires et d'intégration de la Phase 1 (Settings, Loader, Validator).

Vérifie :
- L'immuabilité et la résolution des chemins (`ProjectPaths`).
- La détection des fichiers manquants ou colonnes invalides (`loader.py`).
- L'exactitude mathématique de l'audit des interactions, de la taxonomie des catégories,
  et de la lecture par morceaux (`validator.py`) sur un mini-dataset Retailrocket contrôlé.
"""

from pathlib import Path

import pandas as pd
import pytest

from recommender.config.settings import AuditSettings, ProjectPaths
from recommender.data.loader import (
    DataLoadError,
    inspect_csv_header,
    iter_item_properties_chunks,
    load_category_tree,
    load_events,
    validate_file_exists,
)
from recommender.data.validator import (
    analyze_category_tree,
    analyze_interactions_and_behavior,
    analyze_item_properties_streaming,
    run_full_phase1_audit,
)


@pytest.fixture()
def synthetic_retailrocket_workspace(tmp_path: Path) -> ProjectPaths:
    """Crée un espace de travail temporaire contenant les 4 fichiers Retailrocket."""
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure_directories()

    # 1. events.csv :
    # - visitor 10 : view + addtocart + transaction sur item 100 (acheteur)
    # - visitor 20 : view + addtocart sur item 200 (abandon de panier)
    # - visitor 30 : 1 seul view sur item 300 (view-only, single interaction, orphan item)
    # - 1 doublon exact sur visitor 30
    events_data = pd.DataFrame(
        [
            {"timestamp": 1433221332117, "visitorid": 10, "event": "view", "itemid": 100, "transactionid": pd.NA},
            {"timestamp": 1433221380000, "visitorid": 10, "event": "addtocart", "itemid": 100, "transactionid": pd.NA},
            {"timestamp": 1433221420000, "visitorid": 10, "event": "transaction", "itemid": 100, "transactionid": 5001},
            {"timestamp": 1433300000000, "visitorid": 20, "event": "view", "itemid": 200, "transactionid": pd.NA},
            {"timestamp": 1433300060000, "visitorid": 20, "event": "addtocart", "itemid": 200, "transactionid": pd.NA},
            {"timestamp": 1433400000000, "visitorid": 30, "event": "view", "itemid": 300, "transactionid": pd.NA},
            {"timestamp": 1433400000000, "visitorid": 30, "event": "view", "itemid": 300, "transactionid": pd.NA},
        ]
    )
    events_data.to_csv(paths.events_path, index=False)

    # 2. category_tree.csv : racine 1 -> enfant 2 -> feuille 3
    cat_data = pd.DataFrame(
        [
            {"categoryid": 1, "parentid": pd.NA},
            {"categoryid": 2, "parentid": 1},
            {"categoryid": 3, "parentid": 2},
        ]
    )
    cat_data.to_csv(paths.category_tree_path, index=False)

    # 3. item_properties_part1.csv & part2.csv (item 300 volontairement absent = orphan item)
    prop1 = pd.DataFrame(
        [
            {"timestamp": 1433000000000, "itemid": 100, "property": "categoryid", "value": "3"},
            {"timestamp": 1433000000000, "itemid": 100, "property": "available", "value": "1"},
            {"timestamp": 1433000000000, "itemid": 100, "property": "790", "value": "n150.000"},
        ]
    )
    prop2 = pd.DataFrame(
        [
            {"timestamp": 1433000000000, "itemid": 200, "property": "categoryid", "value": "2"},
            {"timestamp": 1433000000000, "itemid": 200, "property": "available", "value": "0"},
            {"timestamp": 1433000000000, "itemid": 999, "property": "categoryid", "value": "1"},
        ]
    )
    prop1.to_csv(paths.item_properties_part1_path, index=False)
    prop2.to_csv(paths.item_properties_part2_path, index=False)

    return paths


def test_project_paths_immutability(tmp_path: Path) -> None:
    paths = ProjectPaths.from_root(tmp_path)
    paths.ensure_directories()
    assert paths.raw_data_dir.exists()
    assert paths.processed_data_dir.exists()
    with pytest.raises(Exception):
        paths.project_root = tmp_path / "other"  # type: ignore[misc]


def test_validate_file_exists_raises_on_missing(tmp_path: Path) -> None:
    with pytest.raises(DataLoadError):
        validate_file_exists(tmp_path / "non_existent.csv")


def test_inspect_csv_header_detects_missing_columns(tmp_path: Path) -> None:
    bad_csv = tmp_path / "bad_events.csv"
    pd.DataFrame({"visitorid": [1], "itemid": [2]}).to_csv(bad_csv, index=False)
    with pytest.raises(ValueError, match="Colonnes manquantes"):
        inspect_csv_header(bad_csv, expected_columns=("timestamp", "visitorid", "itemid"))


def test_interactions_and_behavioral_segmentation(
    synthetic_retailrocket_workspace: ProjectPaths,
) -> None:
    events_df = load_events(synthetic_retailrocket_workspace.events_path)
    result = analyze_interactions_and_behavior(events_df)

    assert result["summary"]["total_interactions"] == 7
    assert result["summary"]["unique_visitors"] == 3
    assert result["summary"]["unique_items_interacted"] == 3
    assert result["summary"]["functional_duplicates_count"] == 1

    seg = result["user_behavior_segments"]
    assert seg["buyer_users_count"] == 1
    assert seg["cart_abandoner_users_count"] == 1
    assert seg["view_only_users_count"] == 1


def test_category_tree_analysis(
    synthetic_retailrocket_workspace: ProjectPaths,
) -> None:
    cat_df = load_category_tree(synthetic_retailrocket_workspace.category_tree_path)
    report = analyze_category_tree(cat_df)
    assert report["unique_category_ids"] == 3
    assert report["root_categories_count"] == 1
    assert report["leaf_categories_count"] == 1
    assert report["max_tree_depth"] == 2
    assert report["cycles_detected"] == 0


def test_streaming_item_properties_and_full_audit(
    synthetic_retailrocket_workspace: ProjectPaths,
) -> None:
    chunks = list(
        iter_item_properties_chunks(
            synthetic_retailrocket_workspace.item_properties_paths, chunk_size=2
        )
    )
    assert len(chunks) == 4  # 3 lignes par fichier avec chunk_size=2 -> 2 chunks * 2 fichiers

    full_report = run_full_phase1_audit(
        paths=synthetic_retailrocket_workspace,
        settings=AuditSettings(chunk_size=2),
        generate_plots=False,
    )
    cov = full_report["product_and_taxonomy_4_3"]["item_properties"]["interacted_items_coverage"]
    assert cov["total_interacted_items_in_events"] == 3
    assert cov["interacted_items_with_properties_count"] == 2  # items 100 et 200
    assert cov["interacted_items_missing_properties_count"] == 1  # item 300 (orphan)
    assert cov["catalog_items_never_interacted_count"] == 1  # item 999
