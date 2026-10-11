"""How much of the test-period differences could be chance: paired block bootstrap.

Run from the repository root:  python -X utf8 scripts/bootstrap.py

Nothing is tuned or chosen here. LightGBM is loaded from models/. The two
baselines were not saved by scripts/final.py, so they are fitted again with
the same deterministic procedure; the script stops if any reproduced metric
differs from the published one.

Method: the test period is cut into day-long blocks that start at 12:00 Kyiv
time (so a night attack stays in one block). Blocks are drawn with
replacement 2,000 times; both models of a comparison are scored on the same
draw, so their difference is estimated directly. Reported: the 2.5th and
97.5th percentiles and the share of draws in which the difference is above 0.
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from saam import config as C
from saam.baselines import DurationCurve, StateLogReg
from saam.data import KYIV_TZ
from saam.dataset import load_dataset, period_mask
from saam.metrics import threshold_for_precision

N_DRAWS, SEED = 2000, 0
MODES = {"from 2024-07": (C.TRAIN_START_RECENT, "from_2024_07"), "full history": (C.TRAIN_START_FULL, "full_history")}
# Published in README, "Test results" (percent, one decimal).
PUBLISHED = {
    ("start", "from 2024-07"): {"lgbm": 18.9, "base": 17.1},
    ("start", "full history"): {"lgbm": 19.8, "base": 19.4},
    ("end", "from 2024-07"): {"lgbm": 45.2, "base": 52.0},
    ("end", "full history"): {"lgbm": 44.8, "base": 52.0},
}

X, Y = load_dataset()
quiet = Y["quiet"].to_numpy()
val = period_mask(X.index, C.VAL_START, C.VAL_END)
test = period_mask(X.index, C.TEST_START, C.TEST_END)
thresholds = json.loads((ROOT / "models" / "thresholds.json").read_text(encoding="utf-8"))


def blocks_of(index):
    """Block id per point: the Kyiv-time date of (t - 12 h)."""
    local = index.tz_convert(KYIV_TZ) - pd.Timedelta(hours=12)
    return pd.factorize(local.strftime("%Y-%m-%d"))[0]


def precision(y, pred):
    return (pred & y).sum() / pred.sum()


def check(task, mode, which, value):
    want = PUBLISHED[(task, mode)][which]
    if round(100 * value, 1) != want:
        sys.exit(f"STOP: {task} / {mode} / {which} reproduced as {100 * value:.2f}%, published {want}%")


# ---------------------------------------------------------------- predictions
start, end = {}, {}
for mode, (tstart, fname) in MODES.items():
    m_train = period_mask(X.index, tstart, C.TRAIN_END)
    m_final = period_mask(X.index, tstart, C.VAL_END)

    mte = test & quiet
    y = Y["y_start"][mte].to_numpy()
    p_lgbm = joblib.load(ROOT / "models" / f"lgbm_start_{fname}.joblib").predict_proba(X[mte])
    mf = m_final & quiet
    p_base = StateLogReg().fit(X[mf], Y["y_start"][mf].to_numpy()).predict_proba(X[mte])
    check("start", mode, "lgbm", average_precision_score(y, p_lgbm))
    check("start", mode, "base", average_precision_score(y, p_base))
    start[mode] = {"y": y, "lgbm": p_lgbm, "base": p_base, "block": blocks_of(X.index[mte])}

    mte = test & ~quiet
    y = Y["y_end"][mte].to_numpy()
    yes_lgbm = joblib.load(ROOT / "models" / f"lgbm_end_{fname}.joblib").predict_proba(X[mte]) >= thresholds[f"end/{mode}"]
    mt, mv, mf = m_train & ~quiet, val & ~quiet, m_final & ~quiet
    yv = Y["y_end"][mv].to_numpy()
    thr = threshold_for_precision(yv, DurationCurve().fit(X[mt], Y["y_end"][mt].to_numpy()).predict_proba(X[mv]), C.END_MIN_PRECISION)
    yes_base = DurationCurve().fit(X[mf], Y["y_end"][mf].to_numpy()).predict_proba(X[mte]) >= thr
    check("end", mode, "lgbm", precision(y, yes_lgbm))
    check("end", mode, "base", precision(y, yes_base))
    end[mode] = {"y": y, "lgbm": yes_lgbm, "base": yes_base, "block": blocks_of(X.index[mte])}
print("All reproduced metrics equal the published ones; nothing was tuned.")

# ------------------------------------------------------------------ bootstrap
rng = np.random.default_rng(SEED)
rows = []


def summarise(task, metric, what, point, draws):
    lo, hi = np.percentile(draws, [2.5, 97.5])
    row = {"task": task, "metric": metric, "what": what, "point": 100 * point, "ci_low": 100 * lo, "ci_high": 100 * hi}
    if " - " in what:
        row["share_above_0"] = float((draws > 0).mean())
        row["ci_contains_0"] = bool(lo <= 0 <= hi)
    rows.append(row)


# "Start": PR-AUC. Points of both modes are the same test points, so one draw of blocks serves all four score arrays.
ref = start["from 2024-07"]
assert all(np.array_equal(ref["y"], d["y"]) and np.array_equal(ref["block"], d["block"]) for d in start.values())
y, block = ref["y"], ref["block"]
n_blocks = block.max() + 1
members = [np.flatnonzero(block == b) for b in range(n_blocks)]
scores = {(mode, k): start[mode][k] for mode in MODES for k in ("lgbm", "base")}
draws = {k: np.empty(N_DRAWS) for k in scores}
for i in range(N_DRAWS):
    idx = np.concatenate([members[b] for b in rng.integers(0, n_blocks, n_blocks)])
    for k, s in scores.items():
        draws[k][i] = average_precision_score(y[idx], s[idx])
point = {k: average_precision_score(y, s) for k, s in scores.items()}
for mode in MODES:
    summarise("start", "PR-AUC", f"LightGBM, {mode}", point[(mode, "lgbm")], draws[(mode, "lgbm")])
    summarise("start", "PR-AUC", f"logistic regression, {mode}", point[(mode, "base")], draws[(mode, "base")])
    summarise("start", "PR-AUC", f"LightGBM - logistic regression, {mode}", point[(mode, "lgbm")] - point[(mode, "base")], draws[(mode, "lgbm")] - draws[(mode, "base")])
a, b = ("from 2024-07", "lgbm"), ("full history", "lgbm")
summarise("start", "PR-AUC", "LightGBM from 2024-07 - LightGBM full history", point[a] - point[b], draws[a] - draws[b])
a, b = ("from 2024-07", "base"), ("full history", "base")
summarise("start", "PR-AUC", "logistic regression from 2024-07 - logistic regression full history", point[a] - point[b], draws[a] - draws[b])
print(f"start: {len(y)} points, {int(y.sum())} positive, {n_blocks} day blocks, {N_DRAWS} draws")

# "End": precision of "yes" at the frozen thresholds. Per-block counts make the draws cheap.
ref = end["from 2024-07"]
assert all(np.array_equal(ref["y"], d["y"]) and np.array_equal(ref["block"], d["block"]) for d in end.values())
y, block = ref["y"], ref["block"]
n_blocks = block.max() + 1
picks = rng.integers(0, n_blocks, (N_DRAWS, n_blocks))
prec_draws, prec_point = {}, {}
for mode in MODES:
    for k in ("lgbm", "base"):
        yes = end[mode][k]
        tp = np.bincount(block, weights=yes & y, minlength=n_blocks)
        said = np.bincount(block, weights=yes, minlength=n_blocks)
        prec_draws[(mode, k)] = tp[picks].sum(axis=1) / said[picks].sum(axis=1)
        prec_point[(mode, k)] = precision(y, yes)
for mode in MODES:
    summarise("end", "precision of yes", f"LightGBM, {mode}", prec_point[(mode, "lgbm")], prec_draws[(mode, "lgbm")])
    summarise("end", "precision of yes", f"duration curve, {mode}", prec_point[(mode, "base")], prec_draws[(mode, "base")])
    summarise("end", "precision of yes", f"LightGBM - duration curve, {mode}", prec_point[(mode, "lgbm")] - prec_point[(mode, "base")], prec_draws[(mode, "lgbm")] - prec_draws[(mode, "base")])
a, b = ("from 2024-07", "lgbm"), ("full history", "lgbm")
summarise("end", "precision of yes", "LightGBM from 2024-07 - LightGBM full history", prec_point[a] - prec_point[b], prec_draws[a] - prec_draws[b])
base_draws = np.bincount(block, weights=y, minlength=n_blocks)[picks].sum(axis=1) / np.bincount(block, minlength=n_blocks)[picks].sum(axis=1)
summarise("end", "precision of yes", "always yes (base rate)", y.mean(), base_draws)
summarise("end", "precision of yes", "LightGBM, from 2024-07 - always yes", prec_point[("from 2024-07", "lgbm")] - y.mean(), prec_draws[("from 2024-07", "lgbm")] - base_draws)
print(f"end: {len(y)} points, {int(y.sum())} positive, {n_blocks} day blocks with a Kyiv alert, {N_DRAWS} draws")

out = pd.DataFrame(rows)
out.to_csv(ROOT / "reports" / "bootstrap.csv", index=False)
pd.set_option("display.width", 220, "display.max_colwidth", 80)
show = out.copy()
for c in ("point", "ci_low", "ci_high"):
    show[c] = show[c].map(lambda v: f"{v:.1f}")
show["share_above_0"] = out["share_above_0"].map(lambda v: "" if pd.isna(v) else f"{v:.0%}")
show["ci_contains_0"] = out["ci_contains_0"].map(lambda v: "" if pd.isna(v) else ("yes" if v else "no"))
print("\nValues in percentage points; 95% interval = 2.5th..97.5th percentile of the draws.\n")
print(show.to_string(index=False))
