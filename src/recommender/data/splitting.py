"""Découpage temporel strict (Train / Validation / Test) sans fuite d'information.

Implémente un protocole de séparation chronologique global avec vérification
automatique d'absence de chevauchement temporel (`max(train.ts) < min(val.ts)`
et `max(val.ts) < min(test.ts)`), ainsi que l'analyse du taux d'utilisateurs
et produits Cold-Start dans chaque fenêtre d'évaluation.
"""

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

import pandas as pd

from recommender.config.model_config import (
    DEFAULT_TEMPORAL_SPLIT_CONFIG,
    TemporalSplitConfig,
)


@dataclass(frozen=True)
class TemporalDataSplit:
    """Conteneur immuable des 3 sous-ensembles temporels et de leurs bornes."""

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame
    train_cutoff_ts_ms: int
    val_cutoff_ts_ms: int
    summary: dict[str, Any]


def _ts_to_utc_str(ts_ms: int) -> str:
    return datetime.fromtimestamp(ts_ms / 1000.0, tz=timezone.utc).strftime(
        "%Y-%m-%d %H:%M:%S UTC"
    )


def temporal_train_val_test_split(
    events_df: pd.DataFrame,
    config: TemporalSplitConfig = DEFAULT_TEMPORAL_SPLIT_CONFIG,
) -> TemporalDataSplit:
    """Découpe chronologiquement `events_df` en Train, Validation et Test sans fuite.

    Les seuils de coupure `train_cutoff_ts_ms` et `val_cutoff_ts_ms` sont calculés
    sur les quantiles de `timestamp` de sorte que :
    - Train : `timestamp <= train_cutoff_ts_ms`
    - Validation : `train_cutoff_ts_ms < timestamp <= val_cutoff_ts_ms`
    - Test : `timestamp > val_cutoff_ts_ms`
    """
    if events_df.empty:
        raise ValueError("events_df ne peut pas être vide pour le découpage temporel.")
    if "timestamp" not in events_df.columns:
        raise ValueError("La colonne 'timestamp' est requise.")

    ts_series = events_df["timestamp"]
    train_cutoff_ts = int(ts_series.quantile(config.train_ratio))
    val_cutoff_ts = int(ts_series.quantile(config.train_ratio + config.val_ratio))

    train_mask = ts_series <= train_cutoff_ts
    val_mask = (ts_series > train_cutoff_ts) & (ts_series <= val_cutoff_ts)
    test_mask = ts_series > val_cutoff_ts

    train_df = events_df.loc[train_mask].copy().reset_index(drop=True)
    val_df = events_df.loc[val_mask].copy().reset_index(drop=True)
    test_df = events_df.loc[test_mask].copy().reset_index(drop=True)

    if train_df.empty or val_df.empty or test_df.empty:
        raise ValueError(
            "L'un des sous-ensembles (Train, Val ou Test) est vide après découpage."
        )

    # Vérification stricte d'absence de fuite temporelle
    assert int(train_df["timestamp"].max()) < int(val_df["timestamp"].min()), (
        "Fuite temporelle détectée entre Train et Validation !"
    )
    assert int(val_df["timestamp"].max()) < int(test_df["timestamp"].min()), (
        "Fuite temporelle détectée entre Validation et Test !"
    )

    train_users = set(train_df["visitorid"].unique().tolist())
    train_items = set(train_df["itemid"].unique().tolist())
    val_users = set(val_df["visitorid"].unique().tolist())
    val_items = set(val_df["itemid"].unique().tolist())
    test_users = set(test_df["visitorid"].unique().tolist())
    test_items = set(test_df["itemid"].unique().tolist())

    summary = {
        "train_events": int(len(train_df)),
        "val_events": int(len(val_df)),
        "test_events": int(len(test_df)),
        "train_cutoff_ts_ms": train_cutoff_ts,
        "val_cutoff_ts_ms": val_cutoff_ts,
        "train_cutoff_utc": _ts_to_utc_str(train_cutoff_ts),
        "val_cutoff_utc": _ts_to_utc_str(val_cutoff_ts),
        "val_warm_users_in_train_count": len(val_users & train_users),
        "val_cold_users_count": len(val_users - train_users),
        "val_cold_items_count": len(val_items - train_items),
        "test_warm_users_in_train_count": len(test_users & train_users),
        "test_cold_users_count": len(test_users - train_users),
        "test_cold_items_count": len(test_items - train_items),
    }

    return TemporalDataSplit(
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        train_cutoff_ts_ms=train_cutoff_ts,
        val_cutoff_ts_ms=val_cutoff_ts,
        summary=summary,
    )
