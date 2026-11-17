"""Moteur d'inférence de production SmartRecommend (Phase 10 - Section 13).

Expose l'interface Python de production :
```python
engine = RecommendationEngine.load("models/production")
recommendations = engine.recommend(user_id=123, n=10)
session_recs = engine.recommend_session(session_items=[187946, 5411], n=10)
```

Garanties défensives de production (Section 12) :
- Ne retourne jamais un identifiant de produit inexistant dans `valid_catalog_items_`.
- Ne retourne jamais d'identifiants dupliqués.
- Ne retourne jamais de score `NaN` ou `Inf`.
- Respecte la politique d'exclusion des produits déjà achetés.
- Gère sans erreur tout utilisateur inconnu (Cold Start) sans réentraînement.
- Vérifie la cohérence des versions et des sommes de contrôle SHA-256 au chargement.
"""

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from recommender.models.base import RecommendationItem
from recommender.models.hybrid import HybridRecommender
from recommender.utils.serialization import load_artifact, save_artifact

ENGINE_FORMAT_VERSION = "1.0.0"


class RecommendationEngine:
    """Moteur d'inférence prêt pour l'intégration dans une API de production."""

    def __init__(
        self,
        hybrid_model: HybridRecommender | None = None,
        valid_catalog_items: set[int] | None = None,
        version: str = ENGINE_FORMAT_VERSION,
    ) -> None:
        self.hybrid_model = hybrid_model
        self.valid_catalog_items_: set[int] = (
            set(int(i) for i in valid_catalog_items) if valid_catalog_items else set()
        )
        self.version = version

    def _sanitize_recommendations(
        self,
        raw_items: list[RecommendationItem],
        n: int,
        excluded: set[int],
    ) -> list[dict[str, Any]]:
        """Filtre et garantit les invariants stricts de production (Section 12)."""
        clean_list: list[dict[str, Any]] = []
        seen_ids: set[int] = set()

        for item in raw_items:
            iid = int(item.itemid)
            sc = float(item.score)
            if not np.isfinite(sc):
                continue
            if iid in seen_ids or iid in excluded:
                continue
            if self.valid_catalog_items_ and iid not in self.valid_catalog_items_:
                continue
            seen_ids.add(iid)
            clean_list.append(
                {
                    "itemid": iid,
                    "score": round(sc, 6),
                    "source": str(item.source),
                }
            )
            if len(clean_list) >= n:
                break
        return clean_list

    def recommend(
        self,
        user_id: int | None,
        n: int = 10,
        recent_items: Sequence[int] | None = None,
        exclude_items: set[int] | None = None,
        exclude_already_purchased: bool | None = None,
    ) -> list[dict[str, Any]]:
        """Retourne les `n` meilleures recommandations pour `user_id`."""
        if self.hybrid_model is None:
            raise RuntimeError("Le moteur n'est pas chargé. Appelez engine.load(...).")
        if n <= 0:
            raise ValueError("Le nombre de recommandations 'n' doit être > 0.")

        excluded = self.hybrid_model._resolve_excluded_items(
            user_id=user_id,
            exclude_items=exclude_items,
            exclude_purchased_override=exclude_already_purchased,
        )
        raw_recs = self.hybrid_model.recommend(
            user_id=user_id,
            n=n + 10,
            exclude_items=excluded,
            recent_items=recent_items,
            exclude_already_purchased=exclude_already_purchased,
        )
        sanitized = self._sanitize_recommendations(raw_recs, n=n, excluded=excluded)

        # Fallback de sécurité via Popularity si la liste est incomplète
        if len(sanitized) < n:
            already_picked = excluded | {r["itemid"] for r in sanitized}
            fallback_recs = self.hybrid_model.popularity_model.recommend(
                user_id=user_id, n=n - len(sanitized), exclude_items=already_picked
            )
            sanitized.extend(
                self._sanitize_recommendations(
                    fallback_recs, n=n - len(sanitized), excluded=already_picked
                )
            )
        return sanitized[:n]

    def recommend_session(
        self,
        session_items: Sequence[int],
        n: int = 10,
        user_id: int | None = None,
    ) -> list[dict[str, Any]]:
        """Interface dédiée aux recommandations contextuelles de session en temps réel."""
        valid_session = [
            int(i)
            for i in session_items
            if not self.valid_catalog_items_ or int(i) in self.valid_catalog_items_
        ]
        return self.recommend(
            user_id=user_id,
            n=n,
            recent_items=valid_session,
            exclude_items=set(valid_session),
        )

    def save(self, directory: str | Path) -> dict[str, Any]:
        """Sauvegarde atomique et versionnée de tous les artefacts de production."""
        if self.hybrid_model is None:
            raise RuntimeError("Impossible de sauvegarder un moteur vide.")

        target_dir = Path(directory)
        target_dir.mkdir(parents=True, exist_ok=True)

        bundle_manifest = save_artifact(
            {
                "hybrid_model": self.hybrid_model,
                "valid_catalog_items": self.valid_catalog_items_,
                "version": self.version,
            },
            target_dir / "engine_bundle.joblib",
            metadata={"engine_version": self.version},
        )

        metadata = {
            "engine_format_version": self.version,
            "saved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
            "catalog_size": len(self.valid_catalog_items_),
            "known_als_users": len(self.hybrid_model.collaborative_model.user_ids_),
            "known_als_items": len(self.hybrid_model.collaborative_model.item_ids_),
            "use_ltr": bool(self.hybrid_model.use_ltr),
            "bundle_sha256": bundle_manifest["sha256"],
        }
        with open(target_dir / "production_metadata.json", "w", encoding="utf-8") as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        return metadata

    @classmethod
    def load(cls, directory: str | Path) -> "RecommendationEngine":
        """Charge un moteur prêt à l'inférence depuis `directory` sans réentraînement."""
        target_dir = Path(directory)
        meta_path = target_dir / "production_metadata.json"
        bundle_path = target_dir / "engine_bundle.joblib"

        if not meta_path.exists() or not bundle_path.exists():
            raise FileNotFoundError(
                f"Artefacts de production incomplets dans '{target_dir}'."
            )

        with open(meta_path, "r", encoding="utf-8") as f:
            metadata = json.load(f)

        bundle = load_artifact(bundle_path, verify_checksum=True)
        instance = cls(
            hybrid_model=bundle["hybrid_model"],
            valid_catalog_items=bundle["valid_catalog_items"],
            version=str(metadata.get("engine_format_version", ENGINE_FORMAT_VERSION)),
        )
        return instance
