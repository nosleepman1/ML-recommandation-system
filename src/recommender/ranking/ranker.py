r"""Modèle Learning to Rank basé sur LightGBM (`LGBMRanker` LambdaRank - Section 8.4).

Principe mathématique :
Optimise directement la liste ordonnée des candidats d'un utilisateur via l'objectif
pairwise/listwise `lambdarank` pondéré par le gain $\Delta \text{NDCG}$ lors de
l'inversion de deux candidats de niveaux de pertinence différents :
- Label `3` : Produit acheté (`transaction`) dans la fenêtre de validation.
- Label `2` : Produit ajouté au panier (`addtocart`).
- Label `1` : Produit consulté (`view`).
- Label `0` : Candidat généré mais non interagi (échantillon négatif implicite).
"""

from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd

from recommender.config.model_config import DEFAULT_RANKER_CONFIG, LGBMRankerConfig
from recommender.ranking.feature_builder import RANKING_FEATURE_COLUMNS
from recommender.utils.serialization import load_artifact, save_artifact


class LightGBMRankerModel:
    """Encapsule l'entraînement et l'inférence du modèle `lightgbm.LGBMRanker`."""

    def __init__(self, config: LGBMRankerConfig = DEFAULT_RANKER_CONFIG) -> None:
        self.config = config
        self.model_: lgb.LGBMRanker | None = None
        self.feature_columns_: list[str] = list(RANKING_FEATURE_COLUMNS)

    def fit(self, ranked_candidates_df: pd.DataFrame, label_col: str = "relevance_label") -> "LightGBMRankerModel":
        """Entraîne `LGBMRanker` sur un DataFrame de candidats groupés par `visitorid`."""
        if ranked_candidates_df.empty:
            raise ValueError("ranked_candidates_df ne peut pas être vide pour LGBMRanker.")
        if label_col not in ranked_candidates_df.columns:
            raise ValueError(f"Colonne de label '{label_col}' absente.")

        # Tri obligatoire par groupe (visitorid) pour LightGBM LGBMRanker
        sorted_df = ranked_candidates_df.sort_values("visitorid", kind="mergesort").reset_index(drop=True)
        groups = sorted_df.groupby("visitorid", sort=False).size().to_numpy(dtype=np.int32)

        X = sorted_df[self.feature_columns_].to_numpy(dtype=np.float32)
        y = sorted_df[label_col].to_numpy(dtype=np.int32)

        self.model_ = lgb.LGBMRanker(
            objective="lambdarank",
            metric="ndcg",
            n_estimators=self.config.n_estimators,
            learning_rate=self.config.learning_rate,
            num_leaves=self.config.num_leaves,
            min_child_samples=min(self.config.min_child_samples, max(1, len(sorted_df) // 4)),
            subsample=self.config.subsample,
            colsample_bytree=self.config.colsample_bytree,
            random_state=self.config.random_state,
            n_jobs=-1,
            importance_type="gain",
            verbose=-1,
        )
        self.model_.fit(X, y, group=groups)
        return self

    def predict_scores(self, feature_df: pd.DataFrame) -> np.ndarray:
        """Prédit le score de classement LambdaRank pour chaque candidat de `feature_df`."""
        if feature_df.empty or self.model_ is None:
            return np.zeros(len(feature_df), dtype=np.float64)
        X = feature_df[self.feature_columns_].to_numpy(dtype=np.float32)
        preds = self.model_.predict(X)
        return np.where(np.isfinite(preds), preds, 0.0).astype(np.float64)

    def get_feature_importances(self) -> dict[str, float]:
        """Retourne le gain d'importance de chaque caractéristique."""
        if self.model_ is None:
            return {}
        gains = self.model_.booster_.feature_importance(importance_type="gain")
        total = max(1e-9, float(np.sum(gains)))
        return {
            col: round(float(g) / total * 100.0, 2)
            for col, g in sorted(
                zip(self.feature_columns_, gains), key=lambda kv: kv[1], reverse=True
            )
        }

    def save(self, path: Path) -> dict:
        return save_artifact(self, Path(path), metadata={"model_name": "lightgbm_ranker"})

    @classmethod
    def load(cls, path: Path) -> "LightGBMRankerModel":
        loaded = load_artifact(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(f"Attendu LightGBMRankerModel, reçu {type(loaded).__name__}.")
        return loaded
