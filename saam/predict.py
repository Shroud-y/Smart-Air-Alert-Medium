"""Turns a Snapshot into the forecast shown to the user."""
import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np

from . import config as C
from .data import KYIV_CITY
from .features import build_features

MODEL_DIR = Path(__file__).resolve().parent.parent / "models"
MODES = {"recent": "from 2024-07", "full": "full history"}
DEFAULT_MODE = "recent"

# Measured once on the test period, July - 6 September 2026 (README, "Test results").
TEST_NOTE = {
    ("start", "recent"): "on the test period a \"yes\" was right 18% of the time and 64% of alerts were warned of",
    ("start", "full"): "on the test period a \"yes\" was right 27% of the time and 63% of alerts were warned of",
    ("end", "recent"): "on the test period a \"yes\" was right only 45% of the time (target 60%) - a weak hint",
    ("end", "full"): "on the test period a \"yes\" was right only 45% of the time (target 60%) - a weak hint",
}


@dataclass
class Forecast:
    task: str  # "start" or "end"
    kyiv_alert: bool
    kyiv_alert_minutes: float  # how long the Kyiv alert has lasted, NaN if none
    probability: float
    threshold: float
    answer: bool
    mode: str
    table_probability: float
    table_cell: dict  # factor -> level, the readable explanation
    test_note: str
    features: object  # one-row DataFrame, for the debug panel
    top_features: list  # (name, value, share of gain)

    @property
    def question(self):
        if self.task == "start":
            return f"Will an alert start in Kyiv within {C.HORIZON_MIN} min?"
        return f"Will the Kyiv alert end within {C.HORIZON_MIN} min?"


def _file(kind, task, mode):
    return MODEL_DIR / f"{kind}_{task}_{MODES[mode].replace(' ', '_').replace('-', '_')}.joblib"


@lru_cache(maxsize=None)
def _load(task, mode):
    """Model, frequency table and threshold; read once per run."""
    thresholds = json.loads((MODEL_DIR / "thresholds.json").read_text(encoding="utf-8"))
    return joblib.load(_file("lgbm", task, mode)), joblib.load(_file("table", task, mode)), thresholds[f"{task}/{MODES[mode]}"]


def forecast(snapshot, mode=DEFAULT_MODE):
    """Forecast for the moment of the snapshot. Requires the models saved by
    scripts/final.py."""
    grid = np.array([int(snapshot.time.timestamp())], dtype="int64")
    X = build_features(snapshot.alerts, grid)
    row = X.iloc[0]
    kyiv_alert = KYIV_CITY in snapshot.regions_on()
    task = "end" if kyiv_alert else "start"

    model, table, threshold = _load(task, mode)
    p = float(model.predict_proba(X)[0])

    cell = table._cells(X)[0]
    table_cell = {name: labels[int(level)] for (name, labels, _), level in zip(table.factors, cell)}
    imp = model.importance().head(8)
    return Forecast(
        task=task, kyiv_alert=kyiv_alert, kyiv_alert_minutes=float(row["kyiv_on_min"]),
        probability=p, threshold=threshold, answer=p >= threshold, mode=mode,
        table_probability=float(table.predict_proba(X)[0]), table_cell=table_cell,
        test_note=TEST_NOTE[(task, mode)], features=X,
        top_features=[(name, float(row[name]), float(share)) for name, share in imp.items()],
    )
