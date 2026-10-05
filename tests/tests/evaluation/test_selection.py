from __future__ import annotations

import json

import pytest

from evaluation.selection import select_winner


def _make_fake_run(runs_dir, run_id: str, valid_pr_auc: float, model: str = "logistic_regression"):
    run_dir = runs_dir / run_id
    (run_dir / "model").mkdir(parents=True)
    (run_dir / "metrics").mkdir(parents=True)
    metadata = {
        "run_id": run_id,
        "model": model,
        "model_version": "v1",
        "hyperparameters_version": "v1",
        "feature_version": "v1",
    }
    (run_dir / "model" / "metadata.json").write_text(json.dumps(metadata))
    epoch_metrics = {
        "final": {"valid": {"average_precision": valid_pr_auc, "roc_auc": 0.8, "brier": 0.1}}
    }
    (run_dir / "metrics" / "epoch_metrics.json").write_text(json.dumps(epoch_metrics))


def test_select_winner_picks_highest_valid_pr_auc(tmp_path):
    runs_dir = tmp_path / "runs"
    _make_fake_run(runs_dir, "run-a", 0.30)
    _make_fake_run(runs_dir, "run-b", 0.55)
    _make_fake_run(runs_dir, "run-c", 0.42)

    result = select_winner(runs_dir)
    assert result.winner_run_id == "run-b"
    assert list(result.comparison_table["run_id"])[0] == "run-b"


def test_select_winner_respects_override(tmp_path):
    runs_dir = tmp_path / "runs"
    _make_fake_run(runs_dir, "run-a", 0.30)
    _make_fake_run(runs_dir, "run-b", 0.55)

    result = select_winner(runs_dir, override_run_id="run-a")
    assert result.winner_run_id == "run-a"
    assert any("no es el de mayor" in w for w in result.warnings)


def test_select_winner_raises_when_no_runs(tmp_path):
    with pytest.raises(ValueError):
        select_winner(tmp_path / "empty_runs")


def test_select_winner_raises_on_unknown_override(tmp_path):
    runs_dir = tmp_path / "runs"
    _make_fake_run(runs_dir, "run-a", 0.30)
    with pytest.raises(ValueError):
        select_winner(runs_dir, override_run_id="does-not-exist")
