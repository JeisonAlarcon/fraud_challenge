from __future__ import annotations

from evaluation.model_card import render_model_card, write_model_card


def _minimal_context() -> dict:
    return {
        "model": "logistic_regression",
        "winner_run_id": "run-xyz",
        "summary": "Modelo seleccionado por PR-AUC de validacion.",
        "threshold": 0.42,
        "policy_type": "default_argmax_f1",
        "policy_defined_by": "default_no_business_policy",
        "calibration_method": "platt",
        "calibration_pre_metrics": {"brier": 0.2, "ece": 0.1},
        "calibration_post_metrics": {"brier": 0.05, "ece": 0.02, "slope": 1.0, "intercept": 0.0},
        "limitations": ["'k' excluida por cardinalidad unica."],
        "git_sha": "abc123",
        "seed": 42,
    }


def test_render_model_card_contains_all_sections():
    text = render_model_card(_minimal_context())
    expected_headings = [
        "1. Resumen ejecutivo", "2. Dataset y ventana temporal", "3. Features usadas y excluidas",
        "4. Modelo ganador", "5. Metricas train/valid/test", "6. Calibracion (antes / despues)",
        "7. Umbral y politica de decision", "9. Estabilidad temporal y por segmentos",
        "10. Latencia", "11. Limitaciones conocidas", "12. Trazabilidad", "13. Como reproducir",
    ]
    for heading in expected_headings:
        assert heading in text
    assert "run-xyz" in text
    assert "0.42" in text


def test_write_model_card_creates_file(tmp_path):
    path = write_model_card(_minimal_context(), tmp_path / "reports" / "model_card.md")
    assert path.exists()
    assert "run-xyz" in path.read_text()
