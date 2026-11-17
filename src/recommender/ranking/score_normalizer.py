"""Stratégies de normalisation de scores hétérogènes (Phase 5 - Section 8.2).

Les scores produits par Implicit ALS (produits scalaires non bornés), Content-Based
(similarités cosinus dans $[0, 1]$), Session-Based (sommes de probabilités/PPMI) et
Popularity ne sont pas sur la même échelle. Ce module fournit 4 stratégies :
- `min_max` : ramène les scores actifs dans $[0, 1]$.
- `z_score` : standardisation centrée-réduite suivie d'une sigmoïde logistique.
- `percentile` : rang empirique cumulé dans $(0, 1]$.
- `rrf` : Reciprocal Rank Fusion $1 / (k_0 + \text{rank})$.
"""

import numpy as np


def normalize_scores(
    scores: np.ndarray,
    method: str = "min_max",
    rrf_k: float = 60.0,
) -> np.ndarray:
    """Normalise un vecteur 1D de scores de candidats sur une échelle comparable $[0, 1]$."""
    arr = np.asarray(scores, dtype=np.float64)
    if arr.size == 0:
        return arr

    # Nettoyage défensif contre NaN / Inf
    clean = np.where(np.isfinite(arr), arr, 0.0)
    positive_mask = clean > 0.0
    if not np.any(positive_mask):
        return np.zeros_like(clean, dtype=np.float64)

    if method == "min_max":
        vmin = float(np.min(clean))
        vmax = float(np.max(clean))
        if vmax - vmin <= 1e-12:
            return np.where(positive_mask, 1.0, 0.0)
        return np.clip((clean - vmin) / (vmax - vmin), 0.0, 1.0)

    if method == "z_score":
        mean = float(np.mean(clean))
        std = float(np.std(clean))
        if std <= 1e-12:
            return np.where(positive_mask, 0.5, 0.0)
        z = np.clip((clean - mean) / std, -8.0, 8.0)
        return np.where(positive_mask, 1.0 / (1.0 + np.exp(-z)), 0.0)

    if method == "percentile":
        order = np.argsort(clean, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(clean) + 1, dtype=np.float64) / float(len(clean))
        return np.where(positive_mask, ranks, 0.0)

    if method == "rrf":
        order = np.argsort(-clean, kind="mergesort")
        ranks = np.empty_like(order, dtype=np.float64)
        ranks[order] = np.arange(1, len(clean) + 1, dtype=np.float64)
        rrf_scores = 1.0 / (rrf_k + ranks)
        return np.where(positive_mask, rrf_scores / (1.0 / (rrf_k + 1.0)), 0.0)

    raise ValueError(
        f"Méthode de normalisation inconnue '{method}'. "
        "Attendu : 'min_max', 'z_score', 'percentile' ou 'rrf'."
    )
