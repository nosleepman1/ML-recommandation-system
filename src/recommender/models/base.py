"""Interface abstraite et structures de données communes à tous les modèles (Phase 4)."""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from recommender.utils.serialization import load_artifact, save_artifact


@dataclass(frozen=True)
class RecommendationItem:
    """Représentation immuable d'un produit recommandé avec son score et sa source."""

    itemid: int
    score: float
    source: str


class BaseRecommender(ABC):
    """Contrat d'interface commun à tous les modèles de recommandation SmartRecommend."""

    model_name: str = "base"

    @abstractmethod
    def fit(self, *args, **kwargs) -> "BaseRecommender":
        """Entraîne le modèle sur les données d'entraînement."""

    @abstractmethod
    def recommend(
        self,
        user_id: int | None = None,
        n: int = 10,
        exclude_items: set[int] | None = None,
        recent_items: Sequence[int] | None = None,
    ) -> list[RecommendationItem]:
        """Retourne les top-N produits recommandés (sans doublons ni scores invalides)."""

    @abstractmethod
    def score_items(
        self,
        user_id: int | None,
        candidate_items: Sequence[int],
        recent_items: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Retourne un vecteur numpy 1D des scores attribués aux `candidate_items`."""

    def save(self, path: Path) -> dict:
        """Sauvegarde l'instance du modèle sur disque avec vérification SHA-256."""
        return save_artifact(self, Path(path), metadata={"model_name": self.model_name})

    @classmethod
    def load(cls, path: Path) -> "BaseRecommender":
        """Charge une instance sauvegardée depuis le disque."""
        loaded = load_artifact(Path(path))
        if not isinstance(loaded, cls):
            raise TypeError(
                f"Artefact chargé de type {type(loaded).__name__}, attendu {cls.__name__}."
            )
        return loaded
