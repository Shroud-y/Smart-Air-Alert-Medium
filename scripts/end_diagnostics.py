"""Why the "end" yes / no answer is worse than the baseline on the test period.

Run from the repository root:  python -X utf8 scripts/end_diagnostics.py

Diagnostics only: nothing is tuned, chosen or changed. LightGBM is loaded from
models/ and is not retrained. The duration curve was not saved, so it is
fitted again by the same deterministic procedure as in scripts/final.py and
checked against the published numbers. Validation figures are quoted from
reports/final_output.txt, because the validation-stage LightGBM was not saved.
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
from saam.baselines import DurationCurve
from saam.data import KYIV_TZ
from saam.dataset import load_dataset, period_mask
from saam.metrics import threshold_for_precision

MODE, FNAME, TSTART = "from 2024-07", "from_2024_07", C.TRAIN_START_RECENT  # the default mode
# reports/final_output.txt, END / VALIDATION, mode "from 2024-07": precision, recall, says yes (percent)
VAL = {"LightGBM": (60.0, 88.2, 68.1), "duration curve": (60.7, 70.7, 54.0)}
VAL_PR_AUC = {"LightGBM": 76.2, "duration curve": 60.8}

X, Y = load_dataset()
quiet = Y["quiet"].to_numpy()
alert = ~quiet
train = period_mask(X.index, TSTART, C.TRAIN_END) & alert
val = period_mask(X.index, C.VAL_START, C.VAL_END) & alert
final = period_mask(X.index, TSTART, C.VAL_END) & alert
test = period_mask(X.index, C.TEST_START, C.TEST_END) & alert
y = Y["y_end"][test].to_numpy()

thr_lgbm = json.loads((ROOT / "models" / "thresholds.json").read_text(encoding="utf-8"))[f"end/{MODE}"]
p_lgbm = joblib.load(ROOT / "models" / f"lgbm_end_{FNAME}.joblib").predict_proba(X[test])
thr_curve = threshold_for_precision(Y["y_end"][val].to_numpy(), DurationCurve().fit(X[train], Y["y_end"][train].to_numpy()).predict_proba(X[val]), C.END_MIN_PRECISION)
p_curve = DurationCurve().fit(X[final], Y["y_end"][final].to_numpy()).predict_proba(X[test])


def stats(pred):
    return 100 * (pred & y).sum() / max(pred.sum(), 1), 100 * (pred & y).sum() / y.sum(), 100 * pred.mean()


got = {"LightGBM": stats(p_lgbm >= thr_lgbm), "duration curve": stats(p_curve >= thr_curve)}
if round(got["LightGBM"][0], 1) != 45.2 or round(got["duration curve"][0], 1) != 52.0:
    sys.exit(f"STOP: reproduced precision {got} differs from the published 45.2% / 52.0%")
print("Reproduced the published test precision (45.2% and 52.0%); nothing was tuned.\n")

print(f"1. Where the threshold rule placed each model (mode {MODE}; test base rate {100 * y.mean():.1f}%)")
print("   rule: among thresholds with validation precision >= 60%, take the one with the highest recall\n")
print("   model            threshold | validation: precision  recall  says yes | test: precision  recall  says yes | PR-AUC validation -> test")
for name, thr, p in (("LightGBM", thr_lgbm, p_lgbm), ("duration curve", thr_curve, p_curve)):
    v, t = VAL[name], got[name]
    print(f"   {name:15s}  {thr:8.3f} | {v[0]:21.1f} {v[1]:7.1f} {v[2]:9.1f} | {t[0]:15.1f} {t[1]:7.1f} {t[2]:9.1f} | {VAL_PR_AUC[name]:.1f} -> {100 * average_precision_score(y, p):.1f}")

print("\n2. The two answers at the same selectiveness on the test")
print("   LightGBM's threshold is moved so that it says yes exactly as often as the duration curve does.")
print("   This uses the forecasts only, not the outcomes, so it is not a tuned result - but it is not the frozen program either.\n")
curve_yes = p_curve >= thr_curve
k = int(curve_yes.sum())
matched = np.zeros(len(y), bool)
matched[np.argsort(-p_lgbm, kind="stable")[:k]] = True
for name, pred in (("duration curve, frozen threshold", curve_yes), ("LightGBM, same share of yes", matched), ("LightGBM, frozen threshold", p_lgbm >= thr_lgbm)):
    a, b, c = stats(pred)
    print(f"   {name:34s} precision {a:5.1f}%   endings caught {b:5.1f}%   says yes {c:5.1f}%")

local = X.index[test].tz_convert(KYIV_TZ) - pd.Timedelta(hours=12)
block = pd.factorize(local.strftime("%Y-%m-%d"))[0]
nb = block.max() + 1
picks = np.random.default_rng(0).integers(0, nb, (2000, nb))


def boot(pred):
    tp = np.bincount(block, weights=pred & y, minlength=nb)[picks].sum(axis=1)
    return tp / np.bincount(block, weights=pred, minlength=nb)[picks].sum(axis=1)


d = 100 * (boot(matched) - boot(curve_yes))
print(f"   LightGBM at the same share of yes minus duration curve: {stats(matched)[0] - stats(curve_yes)[0]:+.1f} points, 95% interval {np.percentile(d, 2.5):+.1f} .. {np.percentile(d, 97.5):+.1f} (day-block bootstrap, 2,000 draws)")

print("\n3. LightGBM's precision along its own curve on the test (the price of being permissive)")
order = np.argsort(-p_lgbm, kind="stable")
hits = np.cumsum(y[order])
for share in (0.10, 0.25, 0.49, 0.68, 0.88, 1.00):
    n = max(int(round(share * len(y))), 1)
    print(f"   says yes for the top {share:4.0%} of moments: precision {100 * hits[n - 1] / n:5.1f}%   endings caught {100 * hits[n - 1] / y.sum():5.1f}%")

print("\n4. Did the probabilities shift? (mean forecast against observed frequency, test)")
for name, p, thr in (("LightGBM", p_lgbm, thr_lgbm), ("duration curve", p_curve, thr_curve)):
    print(f"   {name:15s} mean forecast {100 * p.mean():.1f}%, observed {100 * y.mean():.1f}%; share of moments above its threshold {100 * (p >= thr).mean():.1f}% (validation: {VAL[name][2]:.1f}%)")

print("\n5. What LightGBM leans on, and whether it still means the same")
imp = joblib.load(ROOT / "models" / f"lgbm_end_{FNAME}.joblib").importance().head(6)
print("   top features of the final model: " + ", ".join(f"{k} {v:.1%}" for k, v in imp.items()))
periods = {"train": train, "validation": val, "test": test}
print("\n   during a Kyiv alert, by period:        " + "".join(f"{n:>12s}" for n in periods))
rows = {
    "alert ends within 30 min, %": lambda m: 100 * Y["y_end"][m].mean(),
    "Chernihiv under alert, %": lambda m: 100 * X["chernihiv_on"][m].mean(),
    "Chernihiv: newest alert < 30 min old, %": lambda m: 100 * (X["chernihiv_newest_min"][m] < 30).mean(),
    "Poltava under alert, %": lambda m: 100 * X["poltava_on"][m].mean(),
    "Kyiv alert already lasts > 60 min, %": lambda m: 100 * (X["kyiv_on_min"][m] > 60).mean(),
}
for label, fn in rows.items():
    print(f"   {label:39s}" + "".join(f"{fn(m):12.1f}" for m in periods.values()))

print("\n   chance that the Kyiv alert ends within 30 min, by how recently a new alert started in Chernihiv oblast:")
print("   Chernihiv's newest alert              " + "".join(f"{n:>12s}" for n in periods) + "   (share of test moments)")
bands = [("no alert in Chernihiv", lambda v: v.isna()), ("started < 30 min ago", lambda v: v < 30), ("30-90 min ago", lambda v: (v >= 30) & (v < 90)),
         ("90-180 min ago", lambda v: (v >= 90) & (v < 180)), ("180 min ago or more", lambda v: v >= 180)]
for label, cond in bands:
    cells, share = [], 0.0
    for name, m in periods.items():
        sel = cond(X["chernihiv_newest_min"][m])
        cells.append(f"{100 * Y['y_end'][m][sel].mean():12.1f}" if sel.sum() >= 30 else f"{'-':>12s}")
        if name == "test":
            share = 100 * sel.mean()
    print(f"   {label:39s}" + "".join(cells) + f"   {share:6.1f}%")

print("\n   the same for the duration curve's only input, how long the Kyiv alert has lasted:")
print("   Kyiv alert has lasted                 " + "".join(f"{n:>12s}" for n in periods) + "   (share of test moments)")
for label, lo, hi in (("< 15 min", 0, 15), ("15-30 min", 15, 30), ("30-60 min", 30, 60), ("60-120 min", 60, 120), ("> 120 min", 120, 1e9)):
    cells, share = [], 0.0
    for name, m in periods.items():
        v = X["kyiv_on_min"][m]
        sel = (v >= lo) & (v < hi)
        cells.append(f"{100 * Y['y_end'][m][sel].mean():12.1f}")
        if name == "test":
            share = 100 * sel.mean()
    print(f"   {label:39s}" + "".join(cells) + f"   {share:6.1f}%")
