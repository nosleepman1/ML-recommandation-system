r"""Métriques d'évaluation pour systèmes de recommandation Top-K (Phase 7 - Section 10.2).

Implémente :
- $\text{Precision@K} = \frac{|\mathcal{R}_K \cap \mathcal{G}|}{K}$
- $\text{Recall@K} = \frac{|\mathcal{R}_K \cap \mathcal{G}|}{|\mathcal{G}|}$
- $\text{AP@K}$ / $\text{MAP@K} = \frac{1}{\min(K, |\mathcal{G}|)} \sum_{k=1}^{K} \text{Precision@}k \cdot \mathbb{I}(r_k \in \mathcal{G})$
- $\text{NDCG@K} = \frac{\text{DCG@K}}{\text{IDCG@K}}$ avec $\text{DCG@K} = \sum_{k=1}^{K} \frac{\mathbb{I}(r_k \in \mathcal{G})}{\log_2(k + 1)}$
- $\text{HitRate@K} = \mathbb{I}(|\mathcal{R}_K \cap \mathcal{G}| > 0)$
- $\text{CatalogCoverage@K} = \frac{|\bigcup_{u} \mathcal{R}_{u, K}|}{|\mathcal{I}|}$
- $\text{Diversity@K}$ : proportion de catégories distinctes au sein d'une liste Top-K.
"""

from typing import Sequence

import numpy as np


def precision_at_k(recommended: Sequence[int], ground_truth: set[int], k: int = 10) -> float:
    """Calcule la Precision@K."""
    if k <= 0:
        return 0.0
    top_k = list(recommended)[:k]
    if not top_k:
        return 0.0
    hits = sum(1 for item in top_k if item in ground_truth)
    return float(hits) / float(k)


def recall_at_k(recommended: Sequence[int], ground_truth: set[int], k: int = 10) -> float:
    """Calcule le Recall@K."""
    if k <= 0 or not ground_truth:
        return 0.0
    top_k = list(recommended)[:k]
    hits = sum(1 for item in top_k if item in ground_truth)
    return float(hits) / float(len(ground_truth))


def hit_rate_at_k(recommended: Sequence[int], ground_truth: set[int], k: int = 10) -> float:
    """Retourne 1.0 si au moins un élément pertinent est dans le Top-K, sinon 0.0."""
    if k <= 0 or not ground_truth:
        return 0.0
    top_k = list(recommended)[:k]
    return 1.0 if any(item in ground_truth for item in top_k) else 0.0


def average_precision_at_k(
    recommended: Sequence[int],
    ground_truth: set[int],
    k: int = 10,
) -> float:
    """Calcule l'Average Precision at K (AP@K), dont la moyenne sur les utilisateurs donne MAP@K."""
    if k <= 0 or not ground_truth:
        return 0.0
    top_k = list(recommended)[:k]
    hits = 0.0
    sum_precisions = 0.0
    for idx, item in enumerate(top_k, start=1):
        if item in ground_truth:
            hits += 1.0
            sum_precisions += hits / float(idx)
    denom = float(min(k, len(ground_truth)))
    return sum_precisions / denom if denom > 0 else 0.0


def ndcg_at_k(recommended: Sequence[int], ground_truth: set[int], k: int = 10) -> float:
    """Calcule le Normalized Discounted Cumulative Gain (NDCG@K) binaire."""
    if k <= 0 or not ground_truth:
        return 0.0
    top_k = list(recommended)[:k]
    dcg = 0.0
    for idx, item in enumerate(top_k, start=1):
        if item in ground_truth:
            dcg += 1.0 / np.log2(idx + 1.0)

    ideal_hits = min(k, len(ground_truth))
    if ideal_hits == 0:
        return 0.0
    idcg = float(np.sum(1.0 / np.log2(np.arange(1, ideal_hits + 1) + 1.0)))
    return float(dcg / idcg) if idcg > 0 else 0.0


def catalog_coverage(
    all_recommendations: Sequence[Sequence[int]],
    total_catalog_size: int,
) -> float:
    """Calcule la proportion du catalogue couverte par l'ensemble des listes recommandées."""
    if total_catalog_size <= 0 or not all_recommendations:
        return 0.0
    unique_rec = {int(item) for rec_list in all_recommendations for item in rec_list}
    return float(len(unique_rec)) / float(total_catalog_size)


def intra_list_category_diversity(
    recommended: Sequence[int],
    item_to_category: dict[int, int],
    k: int = 10,
) -> float:
    """Calcule la diversité intra-liste (ratio de catégories distinctes dans le Top-K)."""
    top_k = list(recommended)[:k]
    if not top_k:
        return 0.0
    cats = [item_to_category.get(int(iid), -1) for iid in top_k]
    valid_cats = [c for c in cats if c >= 0]
    if not valid_cats:
        return 0.0
    return float(len(set(valid_cats))) / float(len(top_k))
