"""Orchestrateur d'entraînement des modèles individuels, hybride et LightGBM Ranker (Phase 8).

Inclut également un enregistreur d'expériences léger (compatible MLflow s'il est installé,
avec persistance JSON garantie dans `reports/experiments/`).
"""

import json
import time
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from recommender.config.model_config import (
    DEFAULT_ALS_CONFIG,
    DEFAULT_CONTENT_CONFIG,
    DEFAULT_HYBRID_CONFIG,
    DEFAULT_POPULARITY_CONFIG,
    DEFAULT_RANKER_CONFIG,
    DEFAULT_SESSION_CONFIG,
    CollaborativeALSConfig,
    ContentBasedConfig,
    HybridEngineConfig,
    LGBMRankerConfig,
    PopularityModelConfig,
    SessionBasedConfig,
)
from recommender.models.collaborative import CollaborativeALSRecommender
from recommender.models.content_based import ContentBasedRecommender
from recommender.models.hybrid import HybridRecommender
from recommender.models.popularity import PopularityRecommender
from recommender.models.session_based import SessionBasedRecommender
from recommender.ranking.feature_builder import RankingFeatureBuilder
from recommender.ranking.ranker import LightGBMRankerModel
from recommender.utils.logging import get_logger

logger = get_logger(__name__)

EVENT_RELEVANCE_GRADE = {"view": 1, "addtocart": 2, "transaction": 3}


def build_ltr_training_dataset(
    hybrid_base: HybridRecommender,
    feature_builder: RankingFeatureBuilder,
    val_events_df: pd.DataFrame,
    max_users: int = 800,
    candidates_per_model: int = 30,
    random_state: int = 42,
) -> pd.DataFrame:
    """Construit sans fuite temporelle le dataset supervisé pour `LightGBMRankerModel`."""
    sorted_val = val_events_df.sort_values(["visitorid", "timestamp"], kind="mergesort")

    # Pour chaque utilisateur de validation, on sépare les premiers items de session (contexte)
    # et les items cibles (labels 1=view, 2=addtocart, 3=transaction)
    user_events: dict[int, list[tuple[int, int]]] = {}
    for uid, iid, ev in zip(
        sorted_val["visitorid"].to_numpy(),
        sorted_val["itemid"].to_numpy(),
        sorted_val["event"].to_numpy(),
    ):
        grade = EVENT_RELEVANCE_GRADE.get(str(ev), 1)
        user_events.setdefault(int(uid), []).append((int(iid), grade))

    eligible_users: list[tuple[int, list[int], dict[int, int]]] = []
    for uid, ev_list in user_events.items():
        distinct_items: list[int] = []
        max_grade: dict[int, int] = {}
        for iid, g in ev_list:
            if iid not in max_grade:
                distinct_items.append(iid)
            max_grade[iid] = max(max_grade.get(iid, 0), g)

        if len(distinct_items) >= 2:
            prompt = [distinct_items[0]]
            targets = {i: max_grade[i] for i in distinct_items[1:]}
            eligible_users.append((uid, prompt, targets))
        elif hybrid_base.collaborative_model.is_known_user(uid) and len(distinct_items) == 1:
            eligible_users.append((uid, [], max_grade))

    if len(eligible_users) > max_users:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(len(eligible_users), size=max_users, replace=False)
        eligible_users = [eligible_users[i] for i in indices]

    frames: list[pd.DataFrame] = []
    for uid, prompt, targets in eligible_users:
        cdf = hybrid_base.candidate_generator.generate_candidates_df(
            user_id=uid,
            k_per_model=candidates_per_model,
            exclude_items=set(prompt),
            recent_items=prompt,
        )
        if cdf.empty:
            continue
        labels = [targets.get(int(iid), 0) for iid in cdf["itemid"].to_numpy()]
        # On conserve les groupes ayant au moins un candidat positif pour que LambdaRank apprenne un gradient non nul
        if sum(labels) == 0:
            # Injection contrôlée d'une cible positive issue de la validation pour l'apprentissage supervisé
            target_iid, target_grade = next(iter(targets.items()))
            extra = hybrid_base.candidate_generator.generate_candidates_df(
                user_id=uid, k_per_model=1, exclude_items=set(), recent_items=prompt
            ).head(1).copy()
            if not extra.empty:
                extra["itemid"] = int(target_iid)
                cdf = pd.concat([cdf, extra], ignore_index=True).drop_duplicates("itemid")
                labels = [targets.get(int(iid), 0) for iid in cdf["itemid"].to_numpy()]

        if sum(labels) > 0 and len(cdf) >= 2:
            w_scores = hybrid_base._compute_hybrid_weighted_scores(cdf, uid, prompt)
            cdf = cdf.assign(
                hybrid_weighted_score=w_scores.astype(np.float32),
                relevance_label=np.array(labels, dtype=np.int32),
            )
            frames.append(cdf)

    if not frames:
        return pd.DataFrame()

    all_cands = pd.concat(frames, ignore_index=True)
    return feature_builder.build_features(all_cands)


def log_experiment_run(
    experiment_dir: Path,
    run_name: str,
    params: dict[str, Any],
    metrics: dict[str, Any],
) -> Path:
    """Enregistre les paramètres et métriques dans MLflow (si actif) et en JSON local."""
    experiment_dir.mkdir(parents=True, exist_ok=True)
    record = {
        "run_name": run_name,
        "timestamp_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "parameters": params,
        "metrics": metrics,
    }
    try:
        import mlflow

        mlflow.set_tracking_uri((experiment_dir / "mlruns").resolve().as_uri())
        mlflow.set_experiment("SmartRecommend_Hybrid_Engine")
        with mlflow.start_run(run_name=run_name):
            for k, v in params.items():
                if isinstance(v, (int, float, str, bool)):
                    mlflow.log_param(k, v)
            for m_name, m_val in metrics.items():
                if isinstance(m_val, (int, float)):
                    mlflow.log_metric(m_name.replace("@", "_at_"), float(m_val))
    except Exception:
        pass

    out_file = experiment_dir / f"{run_name}.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, ensure_ascii=False)
    return out_file
