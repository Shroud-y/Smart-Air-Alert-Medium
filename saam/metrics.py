"""Evaluation metrics shared by baselines and models."""
import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve


def best_f1_threshold(y, score):
    """Threshold with the highest F1; predictions are `score >= threshold`."""
    p, r, t = precision_recall_curve(y, score)
    f1 = 2 * p[:-1] * r[:-1] / np.clip(p[:-1] + r[:-1], 1e-12, None)
    return float(t[int(np.argmax(f1))])


def false_yes_episodes(y, pred, times):
    """Number of false "yes" episodes: maximal runs of consecutive "yes"
    predictions (consecutive grid points) that contain no positive label."""
    y, pred = np.asarray(y, bool), np.asarray(pred, bool)
    if not pred.any():
        return 0
    t = np.asarray(times, "datetime64[s]").astype("int64")
    step = np.min(np.diff(t)) if len(t) > 1 else 0
    idx = np.flatnonzero(pred)
    new_run = np.r_[True, (np.diff(idx) != 1) | (np.diff(t[idx]) != step)]
    run_id = np.cumsum(new_run) - 1
    hit = np.zeros(run_id[-1] + 1, bool)
    np.logical_or.at(hit, run_id, y[idx])
    return int((~hit).sum())


def evaluate(y, score, times, threshold=None, is_probability=True):
    """Metrics for one model on one period. With `threshold=None` the best-F1
    threshold is picked on this same data (optimistic; fine for comparing
    baselines with each other)."""
    y = np.asarray(y, bool)
    score = np.asarray(score, float)
    if threshold is None:
        threshold = best_f1_threshold(y, score)
    pred = score >= threshold
    tp = int((pred & y).sum())
    precision = tp / max(int(pred.sum()), 1)
    recall = tp / max(int(y.sum()), 1)
    t = np.asarray(times, "datetime64[s]").astype("int64")
    days = (t.max() - t.min()) / 86400
    return {
        "pr_auc": average_precision_score(y, score),
        "brier": brier_score_loss(y, score) if is_probability else np.nan,
        "threshold": threshold,
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / max(precision + recall, 1e-12),
        "yes_share": pred.mean(),
        "false_yes_per_day": false_yes_episodes(y, pred, times) / days,
    }
