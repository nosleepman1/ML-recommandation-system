"""Configuration standardisée du logger pour l'ensemble du moteur ML."""

import logging
import sys
from pathlib import Path


def get_logger(
    name: str,
    level: int = logging.INFO,
    log_file: Path | None = None,
) -> logging.Logger:
    """Retourne un logger configuré avec formatage horodaté et sortie console/fichier.

    Args:
        name: Nom du module appelant (ex: ``__name__``).
        level: Niveau de journalisation (par défaut ``logging.INFO``).
        log_file: Chemin optionnel vers un fichier de log.

    Returns:
        Instance de ``logging.Logger`` prête à l'emploi.
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        formatter = logging.Formatter(
            fmt="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        if log_file is not None:
            log_file.parent.mkdir(parents=True, exist_ok=True)
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)

        logger.propagate = False

    return logger
