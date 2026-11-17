"""Tests unitaires du découpage temporel anti-fuite (`splitting.py`)."""

import pandas as pd
import pytest

from recommender.config.model_config import TemporalSplitConfig
from recommender.data.splitting import temporal_train_val_test_split


def test_temporal_split_config_validation() -> None:
    with pytest.raises(ValueError, match="somme des ratios"):
        TemporalSplitConfig(train_ratio=0.7, val_ratio=0.1, test_ratio=0.1)


def test_temporal_split_enforces_strict_chronological_boundaries() -> None:
    events = pd.DataFrame(
        [
            {
                "timestamp": 1_430_000_000_000 + i * 1000,
                "visitorid": 1 if i < 90 else 2,
                "itemid": 100 + (i % 5),
                "event": "view",
            }
            for i in range(100)
        ]
    )
    split = temporal_train_val_test_split(
        events, config=TemporalSplitConfig(train_ratio=0.8, val_ratio=0.1, test_ratio=0.1)
    )
    assert len(split.train_df) == 80
    assert len(split.val_df) == 10
    assert len(split.test_df) == 10
    assert split.train_df["timestamp"].max() < split.val_df["timestamp"].min()
    assert split.val_df["timestamp"].max() < split.test_df["timestamp"].min()
    assert split.summary["test_cold_users_count"] == 1  # visitor 2 uniquement dans test
