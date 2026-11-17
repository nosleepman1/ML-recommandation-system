"""Configurations centralisées et immuables du prétraitement et des modèles ML.

Tous les hyperparamètres et règles métier sont regroupés ici sous forme de
dataclasses gelées (`frozen=True`) pour garantir la reproductibilité et l'immuabilité.
"""

from dataclasses import dataclass
from typing import Mapping, Tuple


@dataclass(frozen=True)
class InteractionWeightConfig:
    """Configuration de la pondération des interactions implicites (Phase 2 - Section 5).

    Attributes:
        view_weight: Poids initial d'une consultation (`view`).
        addtocart_weight: Poids initial d'un ajout au panier (`addtocart`).
        transaction_weight: Poids initial d'un achat confirmé (`transaction`).
        repeat_damping_factor: Facteur d'amortissement logarithmique des répétitions
            d'un même type d'événement sur un même produit par un même visiteur.
            Formule : weight * (1 + repeat_damping_factor * ln(count)).
            Avec 0.5, même 20 vues répétées d'un produit valent 1 * (1 + 0.5*ln(20)) = 2.50,
            ce qui reste inférieur à un ajout au panier (3.0) et à un achat (5.0).
        recency_half_life_days: Demi-vie (en jours) pour la décroissance exponentielle
            de récence : exp(-ln(2) * age_days / recency_half_life_days).
    """

    view_weight: float = 1.0
    addtocart_weight: float = 3.0
    transaction_weight: float = 5.0
    repeat_damping_factor: float = 0.5
    recency_half_life_days: float = 30.0

    def to_weight_map(self) -> Mapping[str, float]:
        """Retourne le dictionnaire associant chaque type d'événement à son poids."""
        return {
            "view": self.view_weight,
            "addtocart": self.addtocart_weight,
            "transaction": self.transaction_weight,
        }


@dataclass(frozen=True)
class PreprocessingConfig:
    """Paramètres du pipeline de préparation des données (Phase 2)."""

    weights: InteractionWeightConfig = InteractionWeightConfig()
    session_inactivity_minutes: int = 30
    min_user_interactions_warm: int = 2
    bot_max_interactions_threshold: int = 1000
    filter_bot_users_in_clean_events: bool = False
    top_k_hashed_properties_to_keep: int = 50
    chunk_size: int = 500_000
    valid_events: Tuple[str, ...] = ("view", "addtocart", "transaction")


@dataclass(frozen=True)
class TemporalSplitConfig:
    """Configuration du découpage temporel strict Train / Validation / Test.

    Garantit qu'aucune interaction future n'est utilisée lors du calcul des
    features ou de l'entraînement des modèles (zéro fuite temporelle).
    """

    train_ratio: float = 0.80
    val_ratio: float = 0.10
    test_ratio: float = 0.10

    def __post_init__(self) -> None:
        total = self.train_ratio + self.val_ratio + self.test_ratio
        if abs(total - 1.0) > 1e-6:
            raise ValueError(
                f"La somme des ratios temporels doit valoir 1.0 (reçu: {total})."
            )
        if min(self.train_ratio, self.val_ratio, self.test_ratio) <= 0.0:
            raise ValueError("Chaque ratio (train, val, test) doit être > 0.")


DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig()
DEFAULT_TEMPORAL_SPLIT_CONFIG = TemporalSplitConfig()

