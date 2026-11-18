"""Évaluateur systématique multi-modèles et analyse par segment (Sections 10.3 & 10.4).

Compare sur un jeu de test temporel indépendant :
1. `PopularityRecommender`
2. `ContentBasedRecommender`
3. `CollaborativeALSRecommender`
4. `SessionBasedRecommender`
5. `HybridRecommender (Weighted)`
6. `HybridRecommender (LightGBM LTR)`

Analyse également les performances par segment utilisateur (Cold, Occasionnel, Actif)
et sur les produits de longue traîne (Long Tail Recall).
"""

import time
from typing import Any, Mapping

import numpy as np
import pandas as pd

from recommender.evaluation.metrics import (
    average_precision_at_k,
    catalog_coverage,
    hit_rate_at_k,
    intra_list_category_diversity,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)
from recommender.models.base import BaseRecommender


def prepare_evaluation_cases(
    eval_events_df: pd.DataFrame,
    train_events_df: pd.DataFrame,
    max_eval_users: int = 1500,
    random_state: int = 42,
) -> list[dict[str, Any]]:
    r"""Construit les cas d'évaluation temporels (contexte de début de session + vérité terrain).

    Pour chaque utilisateur évalué dans `eval_events_df` :
    - Si l'utilisateur a $\ge 2$ produits distincts dans la fenêtre de test, le premier
      produit observé sert d'amorce contextuelle de session (`recent_items`) et les produits
      suivants constituent la vérité terrain future à prédire (`ground_truth`).
    - Si l'utilisateur est Warm (présent dans `train_events_df`), on évalue également sa
      capacité de prédiction à partir de son historique Train.
    """
    sorted_eval = eval_events_df.sort_values(["visitorid", "timestamp"], kind="mergesort")
    train_user_counts = train_events_df.groupby("visitorid").size().to_dict()

    user_items_seq: dict[int, list[int]] = {}
    for uid, iid in zip(
        sorted_eval["visitorid"].to_numpy(), sorted_eval["itemid"].to_numpy()
    ):
        u_list = user_items_seq.setdefault(int(uid), [])
        iid_int = int(iid)
        if not u_list or u_list[-1] != iid_int:
            u_list.append(iid_int)

    eligible_cases: list[dict[str, Any]] = []
    for uid, seq in user_items_seq.items():
        train_cnt = int(train_user_counts.get(uid, 0))
        if len(seq) >= 2:
            prompt_len = max(1, min(3, len(seq) // 2))
            recent = seq[:prompt_len]
            gt = set(seq[prompt_len:]) - set(recent)
        elif train_cnt > 0 and len(seq) == 1:
            recent = []
            gt = {seq[0]}
        else:
            continue

        if not gt:
            continue

        if train_cnt == 0:
            segment = "cold_user"
        elif train_cnt <= 3:
            segment = "casual_user"
        else:
            segment = "active_user"

        eligible_cases.append(
            {
                "visitorid": uid,
                "recent_items": recent,
                "ground_truth": gt,
                "segment": segment,
                "train_events_count": train_cnt,
            }
        )

    # Échantillonnage stratifié reproductible favorisant la présence de Warm, Casual et Cold users
    if len(eligible_cases) > max_eval_users:
        rng = np.random.default_rng(random_state)
        warm_cases = [c for c in eligible_cases if c["segment"] != "cold_user"]
        cold_cases = [c for c in eligible_cases if c["segment"] == "cold_user"]

        n_warm = min(len(warm_cases), int(max_eval_users * 0.65))
        n_cold = min(len(cold_cases), max_eval_users - n_warm)

        chosen_warm_idx = rng.choice(len(warm_cases), size=n_warm, replace=False) if n_warm > 0 else []
        chosen_cold_idx = rng.choice(len(cold_cases), size=n_cold, replace=False) if n_cold > 0 else []

        sampled = [warm_cases[i] for i in chosen_warm_idx] + [
            cold_cases[i] for i in chosen_cold_idx
        ]
        return sampled

    return eligible_cases


def evaluate_models(
    models: Mapping[str, BaseRecommender],
    eval_cases: list[dict[str, Any]],
    item_to_category: dict[int, int],
    total_catalog_size: int,
    long_tail_items: set[int] | None = None,
    k: int = 10,
) -> dict[str, Any]:
    """Évalue chaque modèle sur `eval_cases` (globalement et par segment utilisateur/produit)."""
    global_results: dict[str, dict[str, float]] = {}
    segment_results: dict[str, dict[str, dict[str, float]]] = {}

    for model_label, model in models.items():
        precisions: list[float] = []
        recalls: list[float] = []
        aps: list[float] = []
        ndcgs: list[float] = []
        hits: list[float] = []
        diversities: list[float] = []
        lt_recalls: list[float] = []
        all_rec_lists: list[list[int]] = []

        seg_buckets: dict[str, list[tuple[float, float, float]]] = {
            "cold_user": [],
            "casual_user": [],
            "active_user": [],
        }

        t_start = time.perf_counter()
        for case in eval_cases:
            uid = int(case["visitorid"])
            recent = case["recent_items"]
            gt: set[int] = case["ground_truth"]
            seg = str(case["segment"])

            recs = model.recommend(
                user_id=uid,
                n=k,
                exclude_items=set(recent),
                recent_items=recent,
            )
            rec_ids = [int(r.itemid) for r in recs]
            all_rec_lists.append(rec_ids)

            p_k = precision_at_k(rec_ids, gt, k=k)
            r_k = recall_at_k(rec_ids, gt, k=k)
            ap_k = average_precision_at_k(rec_ids, gt, k=k)
            ndcg_k = ndcg_at_k(rec_ids, gt, k=k)
            hr_k = hit_rate_at_k(rec_ids, gt, k=k)
            div_k = intra_list_category_diversity(rec_ids, item_to_category, k=k)

            precisions.append(p_k)
            recalls.append(r_k)
            aps.append(ap_k)
            ndcgs.append(ndcg_k)
            hits.append(hr_k)
            diversities.append(div_k)

            if long_tail_items:
                gt_lt = gt & long_tail_items
                if gt_lt:
                    lt_recalls.append(recall_at_k(rec_ids, gt_lt, k=k))

            if seg in seg_buckets:
                seg_buckets[seg].append((hr_k, ndcg_k, r_k))

        elapsed_sec = max(1e-6, time.perf_counter() - t_start)
        avg_latency_ms = (elapsed_sec / max(1, len(eval_cases))) * 1000.0
        cov = catalog_coverage(all_rec_lists, total_catalog_size=total_catalog_size)

        global_results[model_label] = {
            f"Precision@{k}": round(float(np.mean(precisions)), 5),
            f"Recall@{k}": round(float(np.mean(recalls)), 5),
            f"MAP@{k}": round(float(np.mean(aps)), 5),
            f"NDCG@{k}": round(float(np.mean(ndcgs)), 5),
            f"HitRate@{k}": round(float(np.mean(hits)), 5),
            "CatalogCoverage": round(float(cov), 5),
            f"Diversity@{k}": round(float(np.mean(diversities)), 5),
            f"LongTailRecall@{k}": round(float(np.mean(lt_recalls)), 5)
            if lt_recalls
            else 0.0,
            "AvgLatencyMs": round(float(avg_latency_ms), 3),
        }

        segment_results[model_label] = {}
        for seg_name, rows in seg_buckets.items():
            if rows:
                arr = np.array(rows, dtype=np.float64)
                segment_results[model_label][seg_name] = {
                    "users_count": len(rows),
                    f"HitRate@{k}": round(float(np.mean(arr[:, 0])), 5),
                    f"NDCG@{k}": round(float(np.mean(arr[:, 1])), 5),
                    f"Recall@{k}": round(float(np.mean(arr[:, 2])), 5),
                }
            else:
                segment_results[model_label][seg_name] = {
                    "users_count": 0,
                    f"HitRate@{k}": 0.0,
                    f"NDCG@{k}": 0.0,
                    f"Recall@{k}": 0.0,
                }

    return {
        "k": k,
        "evaluated_users_count": len(eval_cases),
        "global_metrics": global_results,
        "segment_metrics": segment_results,
    }
