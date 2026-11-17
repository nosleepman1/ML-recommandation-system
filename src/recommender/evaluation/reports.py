"""Formatage des rapports d'évaluation (Markdown et JSON - Phase 7, Section 10.3)."""

from typing import Any


def format_evaluation_markdown_table(eval_report: dict[str, Any]) -> str:
    """Construit un tableau Markdown comparant tous les modèles évalués."""
    k = eval_report.get("k", 10)
    global_metrics = eval_report.get("global_metrics", {})
    if not global_metrics:
        return "Aucun résultat d'évaluation disponible."

    headers = [
        "Modèle",
        f"HitRate@{k}",
        f"NDCG@{k}",
        f"MAP@{k}",
        f"Recall@{k}",
        f"Precision@{k}",
        f"Diversity@{k}",
        "Coverage",
        "Latence (ms)",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]

    for model_name, m in global_metrics.items():
        lines.append(
            f"| **{model_name}** "
            f"| {m.get(f'HitRate@{k}', 0.0):.4f} "
            f"| {m.get(f'NDCG@{k}', 0.0):.4f} "
            f"| {m.get(f'MAP@{k}', 0.0):.4f} "
            f"| {m.get(f'Recall@{k}', 0.0):.4f} "
            f"| {m.get(f'Precision@{k}', 0.0):.4f} "
            f"| {m.get(f'Diversity@{k}', 0.0):.4f} "
            f"| {m.get('CatalogCoverage', 0.0):.4f} "
            f"| {m.get('AvgLatencyMs', 0.0):.2f} |"
        )
    return "\n".join(lines)
