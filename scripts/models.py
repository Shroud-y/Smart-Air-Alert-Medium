"""Models vs the best baselines on the validation period.

Run from the repository root:  python -X utf8 scripts/models.py
The test period is not touched here.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from saam import config as C
from saam.baselines import DurationCurve, StateLogReg
from saam.dataset import load_dataset, period_mask
from saam.freq_table import FrequencyTable
from saam.data import load_alerts
from saam.features import kyiv_intervals
from saam.metrics import alerts_warned, evaluate, false_yes_episodes, threshold_for_precision, threshold_for_recall
from saam.models import Lgbm

X, Y = load_dataset()
KYIV_STARTS, _ = kyiv_intervals(load_alerts())
val = period_mask(X.index, C.VAL_START, C.VAL_END)
trains = {
    "full history": period_mask(X.index, C.TRAIN_START_FULL, C.TRAIN_END),
    "from 2024-07": period_mask(X.index, C.TRAIN_START_RECENT, C.TRAIN_END),
}
quiet = Y["quiet"].to_numpy()
COLS = ["pr_auc", "brier", "threshold", "precision", "recall", "f1", "yes_share", "false_yes_per_day"]


def operating_point(key, y, p, times):
    """Metrics at the agreed operating point (README, decision 21)."""
    if key == "start":
        thr = threshold_for_recall(y, p, C.START_MIN_RECALL)
    else:
        thr = threshold_for_precision(y, p, C.END_MIN_PRECISION)
    if thr is None:
        return {"threshold": np.nan}
    pred = p >= thr
    t = np.asarray(times, "datetime64[s]").astype("int64")
    out = {"threshold": round(thr, 3), "precision_%": round(100 * (pred & y).sum() / pred.sum(), 1),
           "recall_%": round(100 * (pred & y).sum() / y.sum(), 1), "says_yes_%": round(100 * pred.mean(), 1)}
    if key == "start":
        out["false_yes_per_day"] = round(false_yes_episodes(y, pred, times) / ((t.max() - t.min()) / 86400), 2)
        out["alerts_warned_%"] = round(100 * alerts_warned(times, pred, KYIV_STARTS, C.HORIZON_MIN), 1)
    return out


def show(rows, title):
    t = pd.DataFrame(rows).set_index(["model", "train"])[COLS]
    for c in ("precision", "recall", "f1", "yes_share", "pr_auc"):
        t[c] = (t[c] * 100).round(1)
    t["brier"] = t["brier"].round(4)
    t["threshold"] = t["threshold"].round(3)
    t["false_yes_per_day"] = t["false_yes_per_day"].round(2)
    print(f"\n== {title}")
    print(t.to_string())


for key, task, target, rows_mask, baseline in (
    ("start", "START: will a Kyiv alert start within 30 min", "y_start", quiet, StateLogReg),
    ("end", "END: will the current Kyiv alert end within 30 min", "y_end", ~quiet, DurationCurve),
):
    mv = val & rows_mask
    Xv, yv, tv = X[mv], Y[target][mv].to_numpy(), X.index[mv]
    rows, importances, tables, points = [], {}, {}, []
    for tname, m in trains.items():
        mt = m & rows_mask
        Xt, yt = X[mt], Y[target][mt].to_numpy()
        base = baseline().fit(Xt, yt)
        table = FrequencyTable(key).fit(Xt, yt)
        tables[tname] = table
        model = Lgbm().fit(Xt, yt)
        for mname, fitted in ((f"baseline: {baseline.__name__}", base), ("frequency table", table), ("LightGBM", model)):
            p = fitted.predict_proba(Xv)
            rows.append({"model": mname, "train": tname, **evaluate(yv, p, tv)})
            points.append({"model": mname, "train": tname, **operating_point(key, yv, p, tv)})
        importances[tname] = model
        print(f"  [{target}, {tname}] train rows {mt.sum()}, positives {int(yt.sum())}, features {len(model.cols_)}, constant and left out: {len(model.dropped_)}")
    show(rows, f"{task} (validation, {mv.sum()} rows, base rate {yv.mean():.2%})")
    rule = f"recall >= {C.START_MIN_RECALL:.0%}" if key == "start" else f"precision >= {C.END_MIN_PRECISION:.0%}"
    print(f"\n== {key.upper()}: at the agreed operating point ({rule}), threshold picked on validation")
    print(pd.DataFrame(points).set_index(["model", "train"]).to_string())
    t = tables["from 2024-07"].readable()
    pd.set_option("display.width", 200)
    print(f"\n  frequency table (from 2024-07): {len(t)} cells, overall rate {tables['from 2024-07'].rate_:.2%}; n = training points, k = followed by the event")
    show_cells = pd.concat([t.head(8), t.tail(4)]).assign(raw=lambda d: (d.raw * 100).round(1), p=lambda d: (d.p * 100).round(1))
    print("   " + show_cells.to_string().replace("\n", "\n   "))
    t.to_csv(Path(__file__).resolve().parent.parent / "reports" / f"frequency_table_{key}.csv", index=False)
    for tname, model in importances.items():
        imp = model.importance()
        print(f"\n  feature importance, share of gain ({tname}):")
        print("   " + ", ".join(f"{k} {v:.1%}" for k, v in imp.head(12).items()))
        groups = {"Kyiv oblast and its raions": imp[[c for c in imp.index if c.startswith(("kyiv_oblast_", "ko_"))]].sum(),
                  "Kyiv city itself": imp[[c for c in imp.index if c == "kyiv_on_min"]].sum(),
                  "calendar": imp[[c for c in imp.index if c.startswith(("hour_", "dow_"))]].sum(),
                  "region count": imp[[c for c in imp.index if c == "regions_on"]].sum()}
        groups["other regions"] = 1 - sum(groups.values())
        print("   by group: " + ", ".join(f"{k} {v:.1%}" for k, v in groups.items()))
        kinds = {k: imp[[c for c in imp.index if c.endswith(k) and not c.startswith("ko_")]].sum() for k in ("_on", "_share", "_newest_min", "_oldest_min")}
        print("   by feature kind (regions only): " + ", ".join(f"{k} {v:.1%}" for k, v in kinds.items()))
