"""Genera `reports/model_card.md` a partir de un contexto ya calculado (sin templating extra:
el documento no exige Jinja2, f-strings alcanzan)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _dict_to_md_table(rows: list[dict], columns: list[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    sep = "|" + "|".join(["---"] * len(columns)) + "|"
    lines = [header, sep]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    return "\n".join(lines)


def render_model_card(context: dict[str, Any]) -> str:
    parts = []
    parts.append(
        f"# Model Card — {context.get('model', '?')} (run `{context.get('winner_run_id', '?')}`)\n"
    )

    parts.append("## 1. Resumen ejecutivo")
    parts.append(context.get("summary", "") + "\n")

    parts.append("## 2. Dataset y ventana temporal")
    parts.append(
        f"- Filas totales: {context.get('n_rows', '?')}\n"
        f"- Ventana: {context.get('date_start', '?')} a {context.get('date_end', '?')}\n"
        f"- Row counts / prevalencia por split: {context.get('split_row_counts', {})} / "
        f"{context.get('split_prevalence', {})}\n"
    )

    parts.append("## 3. Features usadas y excluidas")
    feature_rows = context.get("feature_decision_table", [])
    if feature_rows:
        parts.append(
            _dict_to_md_table(feature_rows, ["feature", "decision", "riesgo", "evidencia"]) + "\n"
        )
    else:
        parts.append("(tabla incluir/excluir/investigar no provista)\n")

    parts.append("## 4. Modelo ganador")
    parts.append(
        f"- Modelo: `{context.get('model', '?')}` (model_version={context.get('model_version', '?')}, "
        f"hyperparameters_version={context.get('hyperparameters_version', '?')}, "
        f"feature_version={context.get('feature_version', '?')})\n"
        f"- run_id ganador: `{context.get('winner_run_id', '?')}`\n"
        f"- Criterio: mayor PR-AUC de validacion entre los candidatos comparados.\n"
    )
    hyperparameters = context.get("hyperparameters")
    if hyperparameters:
        parts.append(
            "Hiperparametros de la corrida ganadora:\n```json\n"
            + json.dumps(hyperparameters, indent=2, default=str)
            + "\n```\n"
        )
    ensemble_weights = context.get("ensemble_weights")
    if ensemble_weights:
        parts.append(
            "Ponderacion por miembro (aprendida del propio modelo, no configurada a mano: "
            "PR-AUC individual en VALID para `voting_ensemble`, coeficiente del meta-learner "
            "para `stacking_ensemble`):\n"
            + _dict_to_md_table(
                [{"miembro": k, "peso": v} for k, v in ensemble_weights.items()],
                ["miembro", "peso"],
            )
            + "\n"
        )

    parts.append("## 5. Metricas train/valid/test")
    parts.append(
        "TEST se evalua una unica vez, con modelo/preprocesador/calibrador/umbral ya congelados.\n"
    )
    metrics_rows = context.get("metrics_summary_rows", [])
    if metrics_rows:
        parts.append(
            _dict_to_md_table(metrics_rows, ["split", "average_precision", "roc_auc", "brier"])
            + "\n"
        )

    parts.append("## 6. Calibracion (antes / despues)")
    pre = context.get("calibration_pre_metrics", {})
    post = context.get("calibration_post_metrics", {})
    parts.append(
        f"- Metodo: `{context.get('calibration_method', '?')}`\n"
        f"- Brier antes/despues: {pre.get('brier', '?')} / {post.get('brier', '?')}\n"
        f"- ECE antes/despues: {pre.get('ece', '?')} / {post.get('ece', '?')}\n"
        f"- Slope/Intercept (despues): {post.get('slope', '?')} / {post.get('intercept', '?')}\n"
        f"- Reliability diagram: `{context.get('reliability_fig_path', '?')}`\n"
    )

    parts.append("## 7. Umbral y politica de decision")
    parts.append(
        f"- Umbral: {context.get('threshold', '?')}\n"
        f"- Politica: `{context.get('policy_type', '?')}` (defined_by={context.get('policy_defined_by', '?')})\n"
        f"- Metricas en VALID al umbral: {context.get('valid_metrics_at_threshold', {})}\n"
    )

    economic_valid = context.get("economic_metrics_valid", {})
    economic_test = context.get("economic_metrics_test", {})
    if economic_valid or economic_test:
        parts.append("## 8. Ganancia económica")
        parts.append(
            f"- Ganancia VALID al threshold seleccionado: {economic_valid.get('profit_total', '?')}\n"
            f"- Ganancia por transacción en VALID: {economic_valid.get('profit_per_transaction', '?')}\n"
            f"- Monto fraudulento aprobado en VALID: {economic_valid.get('fraud_approved_amount', '?')}\n"
            f"- Ganancia TEST al threshold congelado: {economic_test.get('profit_total', '?')}\n"
            f"- Ganancia por transacción en TEST: {economic_test.get('profit_per_transaction', '?')}\n"
            f"- Monto fraudulento aprobado en TEST: {economic_test.get('fraud_approved_amount', '?')}\n"
        )

    parts.append("## 9. Estabilidad temporal y por segmentos")
    parts.append(
        f"- PR-AUC por ventana temporal: ver `{context.get('temporal_stability_path', '?')}`\n"
        f"- Soporte por segmento: ver `{context.get('segment_support_path', '?')}`\n"
    )

    parts.append("## 10. Latencia")
    latency = context.get("latency", {})
    parts.append(f"- p50: {latency.get('p50_ms', '?')} ms, p95: {latency.get('p95_ms', '?')} ms\n")

    parts.append("## 11. Limitaciones conocidas")
    for limitation in context.get("limitations", []):
        parts.append(f"- {limitation}")
    parts.append("")

    parts.append("## 12. Trazabilidad")
    parts.append(
        f"- git_sha: `{context.get('git_sha', '?')}`\n"
        f"- seed: {context.get('seed', '?')}\n"
        f"- feature_cache_id: `{context.get('feature_cache_id', '?')}`\n"
        f"- parquet/manifest hash: `{context.get('parquet_hash', '?')}`\n"
    )

    parts.append("## 13. Como reproducir")
    parts.append(
        "```bash\n"
        "poetry install\n"
        "poetry run ruff check .\n"
        "poetry run mypy src\n"
        "poetry run pytest\n"
        "poetry run fraud-promote-winner --runs-dir artifacts/runs "
        "--config configs/promotion/v1.yml\n"
        "```\n"
    )

    return "\n".join(parts)


def write_model_card(context: dict[str, Any], output_path: Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(render_model_card(context))
    return output_path
