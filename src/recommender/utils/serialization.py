"""Utilitaires de sérialisation sécurisée et versionnée des artefacts ML (Joblib + JSON)."""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib


def compute_file_sha256(file_path: Path) -> str:
    """Calcule l'empreinte SHA-256 d'un fichier sur disque."""
    hasher = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def save_artifact(
    obj: Any,
    target_path: Path,
    compress: int = 3,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Sauvegarde un composant ML avec Joblib et écrit son manifeste `.meta.json`."""
    resolved = Path(target_path)
    resolved.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(obj, resolved, compress=compress)

    manifest = {
        "artifact_file": resolved.name,
        "saved_at_utc": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC"),
        "sha256": compute_file_sha256(resolved),
        "size_bytes": int(resolved.stat().st_size),
        "metadata": metadata or {},
    }
    meta_path = resolved.with_suffix(resolved.suffix + ".meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, ensure_ascii=False)
    return manifest


def load_artifact(target_path: Path, verify_checksum: bool = True) -> Any:
    """Charge un artefact Joblib et vérifie son intégrité SHA-256 si le manifeste existe."""
    resolved = Path(target_path)
    if not resolved.exists():
        raise FileNotFoundError(f"Artefact introuvable : '{resolved}'.")

    meta_path = resolved.with_suffix(resolved.suffix + ".meta.json")
    if verify_checksum and meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            manifest = json.load(f)
        expected_hash = manifest.get("sha256")
        if expected_hash:
            actual_hash = compute_file_sha256(resolved)
            if actual_hash != expected_hash:
                raise ValueError(
                    f"Corruption détectée pour '{resolved.name}' : "
                    f"SHA-256 attendu={expected_hash}, observé={actual_hash}."
                )
    return joblib.load(resolved)
