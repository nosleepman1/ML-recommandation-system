"""Chargement mémoire-efficace des fichiers bruts du dataset Retailrocket.

Principes d'ingénierie :
- Vérification systématique de l'existence et des colonnes avant chargement complet.
- Lecture par morceaux (chunks) pour `item_properties_part1.csv` et `part2.csv`
  afin d'éviter les pics de consommation RAM (> 20 millions de lignes cumulées).
- Ne modifie jamais les fichiers sources dans `data/raw/`.
"""

from collections.abc import Iterator
from pathlib import Path
from typing import Sequence

import pandas as pd

from recommender.config.settings import DEFAULT_AUDIT_SETTINGS
from recommender.utils.logging import get_logger

logger = get_logger(__name__)

EVENTS_REQUIRED_COLUMNS = ("timestamp", "visitorid", "event", "itemid", "transactionid")
ITEM_PROPERTIES_REQUIRED_COLUMNS = ("timestamp", "itemid", "property", "value")
CATEGORY_TREE_REQUIRED_COLUMNS = ("categoryid", "parentid")


class DataLoadError(FileNotFoundError):
    """Exception levée lorsqu'un fichier brut est introuvable ou invalide."""


def validate_file_exists(file_path: Path) -> Path:
    """Vérifie qu'un fichier existe et n'est pas vide."""
    resolved = Path(file_path)
    if not resolved.exists() or not resolved.is_file():
        raise DataLoadError(
            f"Fichier introuvable : '{resolved}'. "
            "Veuillez déposer les fichiers Retailrocket dans 'data/raw/'."
        )
    if resolved.stat().st_size == 0:
        raise DataLoadError(f"Le fichier '{resolved}' est vide (0 octet).")
    return resolved


def inspect_csv_header(
    file_path: Path,
    expected_columns: Sequence[str] | None = None,
) -> tuple[str, ...]:
    """Lit uniquement l'en-tête d'un CSV et vérifie la présence des colonnes requises."""
    validated_path = validate_file_exists(file_path)
    header_df = pd.read_csv(validated_path, nrows=0)
    actual_columns = tuple(header_df.columns.tolist())

    if expected_columns is not None:
        missing = [col for col in expected_columns if col not in actual_columns]
        if missing:
            raise ValueError(
                f"Colonnes manquantes dans '{validated_path.name}' : {missing}. "
                f"Colonnes trouvées : {list(actual_columns)}"
            )
    return actual_columns


def load_events(events_path: Path) -> pd.DataFrame:
    """Charge `events.csv` avec un typage mémoire-efficace sans altérer les valeurs brutes.

    Note : `transactionid` contient des valeurs manquantes (NaN) pour `view` et `addtocart`,
    d'où l'utilisation du type nullable `Int64` de Pandas.
    """
    inspect_csv_header(events_path, EVENTS_REQUIRED_COLUMNS)
    logger.info("Chargement de %s...", events_path.name)

    df = pd.read_csv(
        events_path,
        dtype={
            "timestamp": "int64",
            "visitorid": "int64",
            "event": "string",
            "itemid": "int64",
            "transactionid": "Int64",
        },
    )
    logger.info(
        "Fichier %s chargé : %d lignes, %.2f Mo en RAM.",
        events_path.name,
        len(df),
        df.memory_usage(deep=True).sum() / (1024**2),
    )
    return df


def load_category_tree(category_tree_path: Path) -> pd.DataFrame:
    """Charge `category_tree.csv` avec support des racines (`parentid` nullable)."""
    inspect_csv_header(category_tree_path, CATEGORY_TREE_REQUIRED_COLUMNS)
    logger.info("Chargement de %s...", category_tree_path.name)

    df = pd.read_csv(
        category_tree_path,
        dtype={
            "categoryid": "int64",
            "parentid": "Int64",
        },
    )
    logger.info("Fichier %s chargé : %d catégories.", category_tree_path.name, len(df))
    return df


def iter_item_properties_chunks(
    file_paths: Sequence[Path],
    chunk_size: int = DEFAULT_AUDIT_SETTINGS.chunk_size,
) -> Iterator[tuple[str, pd.DataFrame]]:
    """Itère par morceaux (chunks) sur un ou plusieurs fichiers `item_properties*.csv`.

    Args:
        file_paths: Séquence des chemins vers `item_properties_part1.csv`, `part2.csv`, etc.
        chunk_size: Nombre de lignes par morceau (par défaut 500 000).

    Yields:
        Un tuple ``(nom_du_fichier, dataframe_chunk)`` permettant un traitement en flux
        à empreinte mémoire constante.
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size doit être strictement positif.")

    for path in file_paths:
        inspect_csv_header(path, ITEM_PROPERTIES_REQUIRED_COLUMNS)
        logger.info(
            "Début de la lecture par morceaux de %s (chunk_size=%d)...",
            path.name,
            chunk_size,
        )
        reader = pd.read_csv(
            path,
            chunksize=chunk_size,
            dtype={
                "timestamp": "int64",
                "itemid": "int64",
                "property": "string",
                "value": "string",
            },
        )
        for chunk in reader:
            yield path.name, chunk
