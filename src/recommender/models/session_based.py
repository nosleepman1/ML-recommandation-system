r"""Modèle Session-Based par transitions séquentielles et co-occurrences (Section 7.4).

Principe mathématique :
Étant donné la séquence d'interactions récentes d'une session $S = (i_1, i_2, \dots, i_L)$,
chaque produit $i_t$ vote pour ses successeurs/co-occurrents $j$ avec un poids décroissant
exponentiellement selon sa distance à la fin de la session :
$$\text{Score}_{\text{session}}(j \mid S) = \sum_{t=1}^{L} \gamma^{L - t} \cdot \left(0.6 \cdot P(j \mid i_t) + 0.4 \cdot \frac{\text{PPMI}(i_t, j)}{1 + \text{PPMI}(i_t, j)}\right)$$
"""

from collections import defaultdict
from typing import Sequence

import numpy as np
import pandas as pd

from recommender.config.model_config import DEFAULT_SESSION_CONFIG, SessionBasedConfig
from recommender.models.base import BaseRecommender, RecommendationItem


class SessionBasedRecommender(BaseRecommender):
    """Recommandeur contextuel de session basé sur les transitions Markoviennes & PPMI."""

    model_name: str = "session_based"

    def __init__(self, config: SessionBasedConfig = DEFAULT_SESSION_CONFIG) -> None:
        self.config = config
        self.transitions_map_: dict[int, list[tuple[int, float]]] = {}
        self.user_recent_items_: dict[int, list[int]] = {}

    def fit(
        self,
        transitions_df: pd.DataFrame,
        train_events_df: pd.DataFrame | None = None,
    ) -> "SessionBasedRecommender":
        """Indexe le graphe de transitions séquentielles et les derniers items par utilisateur."""
        self.transitions_map_ = {}
        if not transitions_df.empty:
            prob = transitions_df["transition_probability"].to_numpy(dtype=np.float64)
            ppmi = transitions_df["ppmi_score"].to_numpy(dtype=np.float64)
            edge_weight = 0.60 * prob + 0.40 * (ppmi / (1.0 + ppmi))

            work = transitions_df.assign(edge_score=edge_weight).sort_values(
                by=["source_itemid", "edge_score"], ascending=[True, False]
            )
            grouped = work.groupby("source_itemid").head(
                self.config.max_neighbors_per_item
            )

            for src, tgt, sc in zip(
                grouped["source_itemid"].to_numpy(),
                grouped["target_itemid"].to_numpy(),
                grouped["edge_score"].to_numpy(dtype=np.float64),
            ):
                self.transitions_map_.setdefault(int(src), []).append(
                    (int(tgt), float(sc))
                )

        self.user_recent_items_ = {}
        if train_events_df is not None and not train_events_df.empty:
            sorted_ev = train_events_df.sort_values(
                ["visitorid", "timestamp"], kind="mergesort"
            )
            tail_ev = sorted_ev.groupby("visitorid").tail(10)
            for uid, iid in zip(
                tail_ev["visitorid"].to_numpy(), tail_ev["itemid"].to_numpy()
            ):
                u_list = self.user_recent_items_.setdefault(int(uid), [])
                iid_int = int(iid)
                if not u_list or u_list[-1] != iid_int:
                    u_list.append(iid_int)
        return self

    def _resolve_session_sequence(
        self,
        user_id: int | None,
        recent_items: Sequence[int] | None,
    ) -> list[int]:
        if recent_items:
            return [int(i) for i in recent_items[-10:]]
        if user_id is not None and int(user_id) in self.user_recent_items_:
            return self.user_recent_items_[int(user_id)]
        return []

    def _compute_session_candidate_scores(
        self, sequence: Sequence[int]
    ) -> dict[int, float]:
        agg_scores: defaultdict[int, float] = defaultdict(float)
        seq_len = len(sequence)
        gamma = float(self.config.recency_position_decay)

        for pos, src_item in enumerate(sequence):
            pos_weight = gamma ** (seq_len - 1 - pos)
            neighbors = self.transitions_map_.get(int(src_item), [])
            for tgt_item, edge_sc in neighbors:
                agg_scores[tgt_item] += pos_weight * edge_sc
        return agg_scores

    def recommend(
        self,
        user_id: int | None = None,
        n: int = 10,
        exclude_items: set[int] | None = None,
        recent_items: Sequence[int] | None = None,
    ) -> list[RecommendationItem]:
        """Recommande les top-N produits successeurs probables de la session en cours."""
        if n <= 0:
            return []

        sequence = self._resolve_session_sequence(user_id, recent_items)
        if not sequence:
            return []

        excluded = set(exclude_items) if exclude_items else set()
        if exclude_items is None:
            excluded.update(sequence)

        agg_scores = self._compute_session_candidate_scores(sequence)
        if not agg_scores:
            return []

        sorted_candidates = sorted(
            agg_scores.items(), key=lambda kv: kv[1], reverse=True
        )
        results: list[RecommendationItem] = []
        for iid, sc in sorted_candidates:
            if iid not in excluded and sc > 0.0:
                results.append(
                    RecommendationItem(
                        itemid=int(iid), score=float(sc), source=self.model_name
                    )
                )
                if len(results) >= n:
                    break
        return results

    def score_items(
        self,
        user_id: int | None,
        candidate_items: Sequence[int],
        recent_items: Sequence[int] | None = None,
    ) -> np.ndarray:
        """Retourne le score de transition de session pour `candidate_items`."""
        if not candidate_items:
            return np.zeros(0, dtype=np.float64)
        sequence = self._resolve_session_sequence(user_id, recent_items)
        if not sequence:
            return np.zeros(len(candidate_items), dtype=np.float64)
        agg_scores = self._compute_session_candidate_scores(sequence)
        return np.array(
            [agg_scores.get(int(iid), 0.0) for iid in candidate_items],
            dtype=np.float64,
        )
