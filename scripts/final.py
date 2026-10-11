"""Final evaluation: thresholds from validation, retraining on train + validation,
one pass over the test period. Saves the final models to models/.

Run from the repository root:  python -X utf8 scripts/final.py
The protocol is fixed in README, "Frozen before the test".
"""
import json
import sys
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from saam import config as C
from saam.baselines import Climatology, DurationCurve, StateLogReg
from saam.data import load_alerts
from saam.dataset import load_dataset, period_mask
from saam.features import kyiv_intervals
from saam.freq_table import FrequencyTable
from saam.metrics import alerts_warned, false_yes_episodes, threshold_for_precision, threshold_for_recall
from saam.models import LgbmEnsemble

FIG_DIR, MODEL_DIR = ROOT / "reports" / "figures", ROOT / "models"
SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
COLORS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 9,
})

X, Y = load_dataset()
KY_S, KY_E = kyiv_intervals(load_alerts())
quiet = Y["quiet"].to_numpy()
val = period_mask(X.index, C.VAL_START, C.VAL_END)
test = period_mask(X.index, C.TEST_START, C.TEST_END)
MODES = {"from 2024-07": C.TRAIN_START_RECENT, "full history": C.TRAIN_START_FULL}
TASKS = {
    "start": ("y_start", quiet, {"climatology": Climatology, "logistic regression": StateLogReg}),
    "end": ("y_end", ~quiet, {"duration curve": DurationCurve}),
}
pd.set_option("display.width", 220)


def pick_threshold(key, y, p):
    if key == "start":
        return threshold_for_recall(y, p, C.START_MIN_RECALL)
    return threshold_for_precision(y, p, C.END_MIN_PRECISION)


def report(key, y, p, times, thr):
    pred = p >= thr
    t = np.asarray(times, "datetime64[s]").astype("int64")
    out = {"pr_auc_%": 100 * average_precision_score(y, p), "brier": brier_score_loss(y, p), "threshold": thr,
           "precision_%": 100 * (pred & y).sum() / max(pred.sum(), 1), "recall_%": 100 * (pred & y).sum() / y.sum(),
           "says_yes_%": 100 * pred.mean()}
    if key == "start":
        out["false_yes_per_day"] = false_yes_episodes(y, pred, times) / ((t.max() - t.min()) / 86400)
        out["alerts_warned_%"] = 100 * alerts_warned(times, pred, KY_S, C.HORIZON_MIN)
    return out


MODEL_DIR.mkdir(exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)
thresholds, test_scores, final_models = {}, {}, {}

for key, (target, rows_mask, baselines) in TASKS.items():
    mv, mte = val & rows_mask, test & rows_mask
    yv, yte = Y[target][mv].to_numpy(), Y[target][mte].to_numpy()
    n_alerts = int(((KY_S >= X.index[test][0].timestamp()) & (KY_S <= X.index[test][-1].timestamp())).sum())
    print(f"\n{'=' * 100}\n{key.upper()} task | validation: {mv.sum()} points, base rate {yv.mean():.2%} | test: {mte.sum()} points, base rate {yte.mean():.2%}, Kyiv alerts in test: {n_alerts}")
    rows_val, rows_test = [], []
    for mode, tstart in MODES.items():
        m_train = period_mask(X.index, tstart, C.TRAIN_END) & rows_mask
        m_final = period_mask(X.index, tstart, C.VAL_END) & rows_mask
        makers = {**{f"baseline: {n}": cls for n, cls in baselines.items()},
                  "frequency table": lambda: FrequencyTable(key), "LightGBM (5 seeds)": LgbmEnsemble}
        for name, make in makers.items():
            # 1. threshold on validation, model trained on the training period only
            p_val = make().fit(X[m_train], Y[target][m_train].to_numpy()).predict_proba(X[mv])
            thr = pick_threshold(key, yv, p_val)
            rows_val.append({"model": name, "mode": mode, **report(key, yv, p_val, X.index[mv], thr)})
            # 2. retrain on train + validation, apply the frozen threshold to the test
            final = make().fit(X[m_final], Y[target][m_final].to_numpy())
            p_test = final.predict_proba(X[mte])
            rows_test.append({"model": name, "mode": mode, **report(key, yte, p_test, X.index[mte], thr)})
            test_scores[(key, mode, name)] = p_test
            if name.startswith("LightGBM"):
                thresholds[f"{key}/{mode}"] = thr
                final_models[(key, mode)] = final
                joblib.dump(final, MODEL_DIR / f"lgbm_{key}_{mode.replace(' ', '_').replace('-', '_')}.joblib", compress=3)
            if name == "frequency table":
                joblib.dump(final, MODEL_DIR / f"table_{key}_{mode.replace(' ', '_').replace('-', '_')}.joblib", compress=3)
    for title, rows in (("VALIDATION (thresholds picked here, models trained on the training period)", rows_val),
                        ("TEST (models retrained on train + validation, thresholds frozen)", rows_test)):
        print(f"\n-- {key.upper()} / {title}")
        print(pd.DataFrame(rows).set_index(["mode", "model"]).round(3).round({"pr_auc_%": 1, "precision_%": 1, "recall_%": 1, "says_yes_%": 1, "false_yes_per_day": 2, "alerts_warned_%": 1, "brier": 4}).to_string())
    if key == "start":
        rng = np.random.default_rng(0)
        always = np.ones(len(yte))
        print(f"\n   reference on test: always 'yes' -> precision {yte.mean():.1%}, recall 100%; random scores -> PR-AUC {average_precision_score(yte, rng.random(len(yte))):.1%}")
        for mode in MODES:
            imp = final_models[(key, mode)].importance()
            print(f"   final LightGBM ({mode}), top features by gain: " + ", ".join(f"{k} {v:.1%}" for k, v in imp.head(8).items()))

(MODEL_DIR / "thresholds.json").write_text(json.dumps(thresholds, indent=2), encoding="utf-8")
print("\nthresholds saved:", thresholds)
print("model files:", {p.name: f"{p.stat().st_size / 1e6:.1f} MB" for p in sorted(MODEL_DIR.glob("*.joblib"))})

# ------------------------------------------------------------------ figures
mte = test & quiet
yte, tt = Y["y_start"][mte].to_numpy(), X.index[mte]
default = "from 2024-07"
fig, ax = plt.subplots(figsize=(7.5, 4.6))
for (label, k), color in zip((("LightGBM, from 2024-07", ("start", default, "LightGBM (5 seeds)")),
                              ("LightGBM, full history", ("start", "full history", "LightGBM (5 seeds)")),
                              ("logistic regression, from 2024-07", ("start", default, "baseline: logistic regression")),
                              ("frequency table, from 2024-07", ("start", default, "frequency table"))), COLORS):
    p, r, _ = precision_recall_curve(yte, test_scores[k])
    ax.step(r * 100, p * 100, where="post", color=color, lw=2, label=label)
ax.axhline(yte.mean() * 100, color=INK2, lw=1, ls="--")
ax.text(99, yte.mean() * 100 + 0.8, f"base rate {yte.mean():.1%}", color=INK2, ha="right", fontsize=8)
ax.axvline(C.START_MIN_RECALL * 100, color=MUTED, lw=1, ls=":")
ax.set(xlabel="Recall, %", ylabel="Precision, %", xlim=(0, 100), ylim=(0, 60))
ax.set_title("Start task on the test period: precision vs recall")
ax.legend(frameon=False, loc="upper right")
fig.savefig(FIG_DIR / "test_start_pr.png", dpi=140, bbox_inches="tight")
plt.close(fig)

fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
for ax, key, target, mask, lim in ((axes[0], "start", "y_start", test & quiet, 45), (axes[1], "end", "y_end", test & ~quiet, 100)):
    y = Y[target][mask].to_numpy()
    for mode, color in zip(MODES, COLORS):
        frac, mean = calibration_curve(y, test_scores[(key, mode, "LightGBM (5 seeds)")], n_bins=8, strategy="quantile")
        ax.plot(mean * 100, frac * 100, "o-", color=color, lw=2, ms=5, label=f"LightGBM, {mode}")
    ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls="--")
    ax.set(xlim=(0, lim), ylim=(0, lim), xlabel="Predicted probability, %", ylabel="Observed frequency, %")
    ax.set_title(f"Calibration on the test period: {key}")
    ax.legend(frameon=False, loc="upper left", fontsize=8)
fig.savefig(FIG_DIR / "test_calibration.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# Timeline: three busiest consecutive days of the test period, default mode.
p_all = pd.Series(test_scores[("start", default, "LightGBM (5 seeds)")], index=tt)
days = pd.Series(1, index=pd.to_datetime(KY_S, unit="s", utc=True)).loc[C.TEST_START:C.TEST_END].resample("1D").sum()
d0 = days.rolling(3).sum().idxmax() - pd.Timedelta(days=2)
d1 = d0 + pd.Timedelta(days=3)
fig, ax = plt.subplots(figsize=(11, 3.6))
seg = p_all.loc[d0:d1].reindex(pd.date_range(d0, d1, freq="5min", inclusive="left"))
local = seg.index.tz_convert("Europe/Kyiv").tz_localize(None)
ax.plot(local, seg.to_numpy() * 100, color=COLORS[0], lw=1.6, label="forecast: chance of an alert within 30 min")
thr = thresholds[f"start/{default}"] * 100
ax.axhline(thr, color=COLORS[1], lw=1.2, ls="--", label=f"\"yes\" threshold, {thr:.1f}%")
first = True
for s, e in zip(KY_S, KY_E):
    s_ts, e_ts = pd.Timestamp(s, unit="s", tz="UTC"), pd.Timestamp(e, unit="s", tz="UTC")
    if e_ts > d0 and s_ts < d1:
        ax.axvspan(s_ts.tz_convert("Europe/Kyiv").tz_localize(None), e_ts.tz_convert("Europe/Kyiv").tz_localize(None),
                   color="#d03b3b", alpha=0.18, lw=0, label="actual Kyiv alert" if first else None)
        first = False
ax.set(ylabel="Probability, %", xlabel="Kyiv time", ylim=(0, None))
ax.set_title(f"Forecast vs actual alerts, {d0:%d %b} – {d1 - pd.Timedelta(days=1):%d %b %Y} (test period, mode \"{default}\")")
ax.legend(frameon=False, loc="upper left", ncol=3, fontsize=8)
ax.grid(axis="x", visible=False)
fig.savefig(FIG_DIR / "test_timeline.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print("figures: test_start_pr.png, test_calibration.png, test_timeline.png")
