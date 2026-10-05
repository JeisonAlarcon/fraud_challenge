from __future__ import annotations

from training.history import EpochHistory


def _metrics(pr_auc: float) -> dict:
    return {"valid": {"average_precision": pr_auc}, "train": {"average_precision": pr_auc + 0.1}}


def test_best_epoch_is_first_peak_not_last():
    history = EpochHistory()
    values = [0.5, 0.6, 0.55, 0.5, 0.45]  # pico en epoch 2
    for epoch, value in enumerate(values, start=1):
        history.record(epoch, _metrics(value))
        history.is_best_so_far(epoch)
    assert history.best_epoch == 2


def test_should_early_stop_triggers_at_correct_epoch():
    history = EpochHistory()
    values = [0.5, 0.6, 0.55, 0.5, 0.45]
    stop_epoch = None
    for epoch, value in enumerate(values, start=1):
        history.record(epoch, _metrics(value))
        history.is_best_so_far(epoch)
        if history.should_early_stop(patience=2):
            stop_epoch = epoch
            break
    assert stop_epoch == 4  # epoch 4 - best_epoch(2) == 2 >= patience


def test_no_early_stop_while_still_improving():
    history = EpochHistory()
    for epoch, value in enumerate([0.1, 0.2, 0.3], start=1):
        history.record(epoch, _metrics(value))
        history.is_best_so_far(epoch)
        assert not history.should_early_stop(patience=2)
