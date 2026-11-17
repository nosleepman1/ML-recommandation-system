"""Configuration centralisée et immuable des chemins et constantes du système.

Respecte le principe d'immuabilité (frozen dataclass) afin d'éviter toute
mutation accidentelle de l'état global pendant l'exécution des pipelines.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple


@dataclass(frozen=True)
class ProjectPaths:
    """Chemins canoniques du projet SmartRecommend."""

    project_root: Path
    data_dir: Path
    raw_data_dir: Path
    processed_data_dir: Path
    external_data_dir: Path
    models_dir: Path
    reports_dir: Path
    figures_dir: Path

    @classmethod
    def from_root(cls, root: Path | None = None) -> "ProjectPaths":
        """Construit une instance immuable de ProjectPaths à partir de la racine."""
        resolved_root = (
            Path(root).resolve()
            if root is not None
            else Path(__file__).resolve().parents[3]
        )
        data_dir = resolved_root / "data"
        reports_dir = resolved_root / "reports"
        return cls(
            project_root=resolved_root,
            data_dir=data_dir,
            raw_data_dir=data_dir / "raw",
            processed_data_dir=data_dir / "processed",
            external_data_dir=data_dir / "external",
            models_dir=resolved_root / "models",
            reports_dir=reports_dir,
            figures_dir=reports_dir / "figures",
        )

    @property
    def events_path(self) -> Path:
        return self.raw_data_dir / "events.csv"

    @property
    def item_properties_part1_path(self) -> Path:
        return self.raw_data_dir / "item_properties_part1.csv"

    @property
    def item_properties_part2_path(self) -> Path:
        return self.raw_data_dir / "item_properties_part2.csv"

    @property
    def category_tree_path(self) -> Path:
        return self.raw_data_dir / "category_tree.csv"

    @property
    def item_properties_paths(self) -> Tuple[Path, Path]:
        return (self.item_properties_part1_path, self.item_properties_part2_path)

    def ensure_directories(self) -> None:
        """Crée l'arborescence de dossiers si elle n'existe pas encore."""
        for directory in (
            self.raw_data_dir,
            self.processed_data_dir,
            self.external_data_dir,
            self.models_dir,
            self.reports_dir,
            self.figures_dir,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AuditSettings:
    """Paramètres d'audit et de lecture mémoire-efficace."""

    chunk_size: int = 500_000
    expected_event_types: Tuple[str, ...] = ("view", "addtocart", "transaction")
    top_k_entities: int = 10
    casual_user_max_events: int = 2
    active_user_min_events: int = 10


DEFAULT_PATHS = ProjectPaths.from_root()
DEFAULT_AUDIT_SETTINGS = AuditSettings()
