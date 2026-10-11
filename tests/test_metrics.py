import numpy as np
import pandas as pd

from saam.metrics import alerts_warned, false_yes_episodes, threshold_for_precision, threshold_for_recall

Y = np.array([0, 0, 1, 0, 1, 1, 0, 1, 1, 1], bool)
S = np.array([0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])


def test_threshold_for_recall_is_the_highest_that_reaches_the_target():
    thr = threshold_for_recall(Y, S, 0.5)  # 3 of 6 positives: scores 0.7, 0.8, 0.9
    assert thr == 0.7
    assert (Y & (S >= thr)).sum() / Y.sum() >= 0.5
    assert threshold_for_recall(Y, S, 1.0) == 0.2  # the lowest positive score


def test_threshold_for_precision_maximises_recall_under_the_constraint():
    thr = threshold_for_precision(Y, S, 0.8)  # >= 0.4 gives 5 of 6 = 83%; >= 0.2 gives 6 of 8 = 75%
    pred = S >= thr
    assert thr == 0.4
    assert (Y & pred).sum() / pred.sum() >= 0.8
    assert threshold_for_precision(np.array([0, 1, 0], bool), np.array([0.9, 0.5, 0.1]), 0.9) is None


def test_alerts_warned_counts_alerts_with_a_yes_in_the_window_before_them():
    times = pd.date_range("2025-03-01 00:00", periods=48, freq="5min", tz="UTC")  # 00:00 .. 03:55
    pred = np.zeros(48, bool)
    pred[times == pd.Timestamp("2025-03-01 01:40", tz="UTC")] = True
    starts = [int(pd.Timestamp(t, tz="UTC").timestamp()) for t in ("2025-03-01 02:00", "2025-03-01 03:00", "2025-03-01 02:10")]
    # 02:00 warned 20 min ahead; 03:00 not warned; 02:10 warned exactly 30 min ahead
    assert alerts_warned(times, pred, starts, 30) == 2 / 3
    assert alerts_warned(times, pred, starts[:1], 15) == 0  # 20 min ahead is outside a 15-min window


def test_false_yes_episodes_counts_runs_without_a_positive():
    times = pd.date_range("2025-03-01", periods=10, freq="5min", tz="UTC")
    pred = np.array([1, 1, 0, 1, 1, 1, 0, 0, 1, 0], bool)
    y = np.array([0, 0, 0, 0, 1, 0, 0, 0, 0, 0], bool)
    assert false_yes_episodes(y, pred, times) == 2  # runs 0-1 and 8; run 3-5 contains a positive
