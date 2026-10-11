"""Assembles features and targets on the project grid and splits them by time."""
import pandas as pd

from . import config as C
from .data import load_alerts
from .features import build_features, build_targets, make_grid


def load_dataset():
    """Features X and targets Y on the 5-minute grid, full history."""
    df = load_alerts()
    grid = make_grid(C.TRAIN_START_FULL, C.TEST_END, C.GRID_STEP_MIN)
    grid = grid[grid >= int(df["started_at"].min().timestamp())]
    return build_features(df, grid), build_targets(df, grid, C.HORIZON_MIN)


def period_mask(index, start, end):
    """Rows in [start, end) whose target window (t, t+H] stays inside the
    period, so that no label of one part looks into the next part."""
    last = pd.Timestamp(end, tz="UTC") - pd.Timedelta(minutes=C.HORIZON_MIN)
    return (index >= pd.Timestamp(start, tz="UTC")) & (index <= last)


def varying_columns(X_train):
    """Columns that take more than one value in the training rows. A feature
    that is constant in training cannot be learned from and distorts linear
    models once it starts to vary later (README, decision 16)."""
    return [c for c in X_train.columns if X_train[c].nunique(dropna=False) > 1]
