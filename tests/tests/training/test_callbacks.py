from __future__ import annotations

from training.callbacks import LiveTrainingMonitor


def _metrics(pr_auc_train: float, pr_auc_valid: float) -> dict:
    return {
        "train": {"average_precision": pr_auc_train},
        "valid": {"average_precision": pr_auc_valid},
    }


def test_monitor_with_total_epochs_advances_progress_bar():
    monitor = LiveTrainingMonitor(total_epochs=3, desc="test")
    assert monitor._pbar is not None
    for epoch in range(1, 4):
        monitor.on_epoch_end(epoch, _metrics(0.1 * epoch, 0.1 * epoch))
    assert monitor._pbar.n == 3
    monitor.close()  # no debe lanzar


def test_monitor_without_total_epochs_falls_back_to_print(capsys):
    monitor = LiveTrainingMonitor(total_epochs=None)
    assert monitor._pbar is None
    monitor.on_epoch_end("final", _metrics(0.5, 0.4))
    captured = capsys.readouterr()
    assert "valid_pr_auc=0.4000" in captured.out
    monitor.close()  # no-op, no debe lanzar
