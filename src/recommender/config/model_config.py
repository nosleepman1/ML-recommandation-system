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


@dataclass(frozen=True)
class FeatureConfig:
    """Configuration du Feature Engineering (Phase 3 - Sections 6.1, 6.2, 6.3).

    Attributes:
        item_decay_half_life_days: Demi-vie en jours pour la popularité produit
            pondérée exponentiellement par le temps.
        short_window_days: Fenêtre récente courte (en jours, par défaut 7j).
        medium_window_days: Fenêtre récente moyenne (en jours, par défaut 14j).
        long_window_days: Fenêtre de référence (en jours, par défaut 30j) pour
            calculer la vélocité/tendance (ratio 7j vs 30j).
        bayesian_smoothing_prior_count: Masse a priori `m` pour le lissage bayésien
            du taux de conversion produit : (tx + m * global_rate) / (views + m).
        min_transition_cooccurrence: Seuil minimal d'observations pour conserver
            une transition séquentielle (item_a -> item_b) intra-session.
        max_transitions_per_item: Nombre maximal de successeurs conservés par produit.
        exclude_bots_from_item_popularity: Si True, ignore les profils identifiés
            comme bots (>1000 interactions) lors du calcul de la popularité produit
            et des transitions séquentielles.
    """

    item_decay_half_life_days: float = 14.0
    short_window_days: int = 7
    medium_window_days: int = 14
    long_window_days: int = 30
    bayesian_smoothing_prior_count: float = 20.0
    min_transition_cooccurrence: int = 2
    max_transitions_per_item: int = 30
    exclude_bots_from_item_popularity: bool = True


@dataclass(frozen=True)
class PopularityModelConfig:
    """Configuration du modèle Popularity-Based avec décroissance temporelle (Section 7.1)."""

    decay_weight: float = 0.60
    global_weight: float = 0.25
    conversion_weight: float = 0.15
    only_available_items: bool = False


@dataclass(frozen=True)
class ContentBasedConfig:
    """Configuration du modèle Content-Based Filtering (Section 7.2)."""

    max_tfidf_features: int = 5_000
    min_df: int = 2
    category_weight: float = 2.0
    root_category_weight: float = 1.0
    tokens_weight: float = 1.5
    max_user_history_items: int = 25


@dataclass(frozen=True)
class CollaborativeALSConfig:
    """Configuration du modèle Implicit ALS (Section 7.3)."""

    factors: int = 64
    regularization: float = 0.05
    alpha: float = 15.0
    iterations: int = 15
    random_state: int = 42
    min_item_interactions: int = 2
    exclude_bots: bool = True


@dataclass(frozen=True)
class SessionBasedConfig:
    """Configuration du modèle Session-Based par transitions et co-occurrences (Section 7.4)."""

    max_neighbors_per_item: int = 40
    min_cooccurrence: int = 2
    recency_position_decay: float = 0.75
    category_fallback_boost: float = 0.15


@dataclass(frozen=True)
class HybridEngineConfig:
    """Configuration du moteur hybride et de la gestion du Cold Start (Sections 8 & 9)."""

    collaborative_weight: float = 0.40
    session_weight: float = 0.30
    content_weight: float = 0.18
    popularity_weight: float = 0.12
    normalization_method: str = "min_max"
    candidates_per_model: int = 50
    exclude_already_purchased: bool = True
    exclude_already_viewed: bool = False
    diversity_category_penalty: float = 0.10
    sparse_user_threshold: int = 3


@dataclass(frozen=True)
class LGBMRankerConfig:
    """Configuration du modèle Learning to Rank LightGBM (Section 8.4)."""

    n_estimators: int = 80
    learning_rate: float = 0.08
    num_leaves: int = 31
    min_child_samples: int = 20
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    random_state: int = 42


DEFAULT_PREPROCESSING_CONFIG = PreprocessingConfig()
DEFAULT_TEMPORAL_SPLIT_CONFIG = TemporalSplitConfig()
DEFAULT_FEATURE_CONFIG = FeatureConfig()
DEFAULT_POPULARITY_CONFIG = PopularityModelConfig()
DEFAULT_CONTENT_CONFIG = ContentBasedConfig()
DEFAULT_ALS_CONFIG = CollaborativeALSConfig()
DEFAULT_SESSION_CONFIG = SessionBasedConfig()
DEFAULT_HYBRID_CONFIG = HybridEngineConfig()
DEFAULT_RANKER_CONFIG = LGBMRankerConfig()



