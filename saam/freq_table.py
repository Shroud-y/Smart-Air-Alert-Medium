"""The probabilistic algorithm: a table of conditional frequencies.

Every moment falls into one cell defined by three factors. The forecast is
the share of training moments in that cell that were followed by the event,
pulled towards the overall rate when the cell has little data.
"""
import numpy as np
import pandas as pd

from .baselines import ADJACENT

PRIOR_STRENGTH = 50  # a cell needs about this many points to outweigh the overall rate

TIME_OF_DAY = ["night 22-06", "morning 06-12", "day 12-18", "evening 18-22"]
KYIV_OBLAST = ["quiet", "alert < 15 min", "alert 15-60 min", "alert > 60 min"]
ADJACENT_ON = ["0 adjacent", "1 adjacent", "2 adjacent", "3+ adjacent"]
KYIV_ALERT = ["< 15 min", "15-30 min", "30-60 min", "60-120 min", "> 120 min"]
OBLAST_STATE = ["oblast quiet", "oblast under alert"]


def _time_of_day(X):
    h = X["hour_kyiv"].to_numpy()
    return np.select([(h >= 22) | (h < 6), h < 12, h < 18], [0, 1, 2], default=3)


def _kyiv_oblast(X):
    m = X["kyiv_oblast_oldest_min"].to_numpy()
    return np.where(np.isnan(m), 0, 1 + np.digitize(np.nan_to_num(m), [15, 60]))


def _adjacent(X):
    return np.minimum(X[[f"{r}_on" for r in ADJACENT]].sum(axis=1).to_numpy(), 3).astype(int)


def _kyiv_alert(X):
    return np.digitize(X["kyiv_on_min"].to_numpy(), [15, 30, 60, 120])


def _oblast_state(X):
    return X["kyiv_oblast_on"].to_numpy().astype(int)


FACTORS = {
    "start": [("Kyiv oblast", KYIV_OBLAST, _kyiv_oblast), ("adjacent oblasts", ADJACENT_ON, _adjacent), ("time of day", TIME_OF_DAY, _time_of_day)],
    "end": [("Kyiv alert lasts", KYIV_ALERT, _kyiv_alert), ("Kyiv oblast", OBLAST_STATE, _oblast_state), ("time of day", TIME_OF_DAY, _time_of_day)],
}


class FrequencyTable:
    def __init__(self, task, prior_strength=PRIOR_STRENGTH):
        self.task, self.m = task, prior_strength
        self.factors = FACTORS[task]

    def _cells(self, X):
        return pd.MultiIndex.from_arrays([fn(X) for _, _, fn in self.factors], names=[n for n, _, _ in self.factors])

    def fit(self, X, y):
        y = pd.Series(np.asarray(y, float), index=self._cells(X))
        g = y.groupby(level=list(range(len(self.factors))))
        self.rate_ = float(y.mean())
        t = pd.DataFrame({"n": g.size(), "k": g.sum()})
        t["raw"] = t["k"] / t["n"]
        t["p"] = (t["k"] + self.m * self.rate_) / (t["n"] + self.m)
        self.table_ = t
        return self

    def predict_proba(self, X):
        return self.table_["p"].reindex(self._cells(X)).fillna(self.rate_).to_numpy()

    def readable(self):
        """The table with factor levels spelled out, most likely cells first."""
        t = self.table_.reset_index()
        for name, labels, _ in self.factors:
            t[name] = t[name].map(dict(enumerate(labels)))
        return t.sort_values("p", ascending=False).reset_index(drop=True)
