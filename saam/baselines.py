"""Baselines every model is compared against (README, "Evaluation")."""
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

from .dataset import varying_columns

# Oblasts that share a border with Kyiv oblast.
ADJACENT = ["chernihiv", "zhytomyr", "cherkasy", "poltava", "vinnytsia"]
NEIGHBOURS = ["kyiv_oblast"] + ADJACENT


class Climatology:
    """Start task: frequency of the target by Kyiv hour x day of week."""

    def fit(self, X, y):
        key = pd.DataFrame({"h": X["hour_kyiv"].to_numpy(), "d": X["dow_kyiv"].to_numpy(), "y": np.asarray(y, float)})
        self.table_ = key.groupby(["h", "d"])["y"].mean()
        self.default_ = float(key["y"].mean())
        return self

    def predict_proba(self, X):
        idx = pd.MultiIndex.from_arrays([X["hour_kyiv"].to_numpy(), X["dow_kyiv"].to_numpy()])
        return self.table_.reindex(idx).fillna(self.default_).to_numpy()


def neighbours_on(X, regions=NEIGHBOURS):
    """Start task, rule: number of the given regions under alert right now.
    `>= 1` reproduces "Kyiv oblast or an adjacent oblast is under alert"."""
    return X[[f"{r}_on" for r in regions]].sum(axis=1).to_numpy()


class StateLogReg:
    """Start task: logistic regression on the yes/no state of every region.
    Flags that are constant in the training rows are left out."""

    def fit(self, X, y):
        flags = [c for c in X.columns if c.endswith("_on") and not c.startswith("ko_") and c != "regions_on"]
        self.cols_ = varying_columns(X[flags])
        self.dropped_ = [c for c in flags if c not in self.cols_]
        self.model_ = LogisticRegression(max_iter=2000).fit(X[self.cols_], y)
        return self

    def predict_proba(self, X):
        return self.model_.predict_proba(X[self.cols_])[:, 1]


class DurationCurve:
    """End task: P(alert ends within H | minutes it has lasted), from the
    empirical frequency in bins of elapsed time."""

    def __init__(self, bin_min=10, max_min=240):
        self.edges = np.r_[np.arange(0, max_min + 1, bin_min), np.inf]

    def _bin(self, X):
        return np.digitize(X["kyiv_on_min"].to_numpy(), self.edges) - 1

    def fit(self, X, y):
        s = pd.Series(np.asarray(y, float)).groupby(self._bin(X)).mean()
        self.default_ = float(np.mean(y))
        self.table_ = s
        return self

    def predict_proba(self, X):
        return self.table_.reindex(self._bin(X)).fillna(self.default_).to_numpy()
