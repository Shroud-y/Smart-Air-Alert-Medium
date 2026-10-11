"""Baselines on the validation period. Prints tables, writes figures.

Run from the repository root:  python -X utf8 scripts/baselines.py
The test period is not touched here.
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.metrics import precision_recall_curve

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from saam import config as C
from saam.baselines import Climatology, DurationCurve, StateLogReg, neighbours_on
from saam.dataset import load_dataset, period_mask
from saam.metrics import evaluate

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
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
val = period_mask(X.index, C.VAL_START, C.VAL_END)
trains = {
    "full history": period_mask(X.index, C.TRAIN_START_FULL, C.TRAIN_END),
    "from 2024-07": period_mask(X.index, C.TRAIN_START_RECENT, C.TRAIN_END),
}
quiet = Y["quiet"].to_numpy()
COLS = ["pr_auc", "brier", "threshold", "precision", "recall", "f1", "yes_share", "false_yes_per_day"]


def show(rows, title):
    t = pd.DataFrame(rows).set_index(["baseline", "train"])[COLS]
    for c in ("precision", "recall", "f1", "yes_share", "pr_auc"):
        t[c] = (t[c] * 100).round(1)
    t["brier"] = t["brier"].round(4)
    t["threshold"] = t["threshold"].round(3)
    t["false_yes_per_day"] = t["false_yes_per_day"].round(2)
    print(f"\n== {title}")
    print(t.to_string())
    return t


# ------------------------------------------------------------ start task
m_val = val & quiet
Xv, yv, tv = X[m_val], Y["y_start"][m_val].to_numpy(), X.index[m_val]
print(f"START task, validation {C.VAL_START} .. {C.VAL_END}: {m_val.sum()} rows, {yv.sum()} positive, base rate {yv.mean():.2%}")
rows, curves = [], {}

rng = np.random.default_rng(0)
rows.append({"baseline": "random", "train": "-", **evaluate(yv, rng.random(len(yv)), tv, is_probability=False)})

for name, regions, thr in (
    ("rule: Kyiv oblast", ["kyiv_oblast"], 1),
    ("rule: Kyiv oblast or adjacent", None, 1),
    ("rule: count of 6 neighbours", None, None),
):
    score = neighbours_on(Xv, regions) if regions else neighbours_on(Xv)
    rows.append({"baseline": name, "train": "-", **evaluate(yv, score, tv, threshold=thr, is_probability=False)})
curves["rule: count of 6 neighbours"] = neighbours_on(Xv)

for tname, m in trains.items():
    mt = m & quiet
    Xt, yt = X[mt], Y["y_start"][mt].to_numpy()
    for name, model in (("climatology", Climatology()), ("logistic regression", StateLogReg())):
        p = model.fit(Xt, yt).predict_proba(Xv)
        rows.append({"baseline": name, "train": tname, **evaluate(yv, p, tv)})
        curves[f"{name} ({tname})"] = p
        if getattr(model, "dropped_", None):
            print(f"  logistic regression ({tname}): constant in training, left out: {model.dropped_}")
    if tname == "from 2024-07":
        coef = pd.Series(model.model_.coef_[0], index=[c[:-3] for c in model.cols_]).sort_values()
start_table = show(rows, "START task: will a Kyiv alert start within 30 min (validation)")
print("\n  logistic regression (from 2024-07), weights of region flags:")
print("  highest:", ", ".join(f"{k} {v:+.2f}" for k, v in coef[::-1][:6].items()))
print("  lowest: ", ", ".join(f"{k} {v:+.2f}" for k, v in coef[:6].items()))

fig, ax = plt.subplots(figsize=(7.5, 4.6))
picks = ["logistic regression (from 2024-07)", "logistic regression (full history)", "rule: count of 6 neighbours", "climatology (from 2024-07)"]
for name, color in zip(picks, COLORS):
    p, r, _ = precision_recall_curve(yv, curves[name])
    ax.step(r * 100, p * 100, where="post", color=color, lw=2, label=name)
ax.axhline(yv.mean() * 100, color=INK2, lw=1, ls="--")
ax.text(99, yv.mean() * 100 + 0.6, f"base rate {yv.mean():.1%}", color=INK2, ha="right", fontsize=8)
ax.set_xlabel("Recall, %")
ax.set_ylabel("Precision, %")
ax.set_xlim(0, 100)
ax.set_ylim(0, 40)
ax.set_title("Start task, baselines: precision vs recall on validation")
ax.legend(frameon=False, loc="upper right")
FIG_DIR.mkdir(parents=True, exist_ok=True)
fig.savefig(FIG_DIR / "baselines_start_pr.png", dpi=140, bbox_inches="tight")
plt.close(fig)

fig, ax = plt.subplots(figsize=(5.2, 4.6))
for name, color in zip(["logistic regression (from 2024-07)", "logistic regression (full history)", "climatology (from 2024-07)"], COLORS):
    frac, mean = calibration_curve(yv, curves[name], n_bins=8, strategy="quantile")
    ax.plot(mean * 100, frac * 100, "o-", color=color, lw=2, ms=5, label=name)
lim = 30
ax.plot([0, lim], [0, lim], color=INK2, lw=1, ls="--")
ax.set_xlim(0, lim)
ax.set_ylim(0, lim)
ax.set_xlabel("Predicted probability, %")
ax.set_ylabel("Observed frequency, %")
ax.set_title("Start task, baselines: calibration on validation")
ax.legend(frameon=False, loc="upper left", fontsize=8)
fig.savefig(FIG_DIR / "baselines_start_calibration.png", dpi=140, bbox_inches="tight")
plt.close(fig)

# -------------------------------------------------------------- end task
m_val = val & ~quiet
Xv, yv, tv = X[m_val], Y["y_end"][m_val].to_numpy(), X.index[m_val]
print(f"\nEND task, validation: {m_val.sum()} rows, {yv.sum()} positive, base rate {yv.mean():.2%}")
rows = [{"baseline": "random", "train": "-", **evaluate(yv, rng.random(len(yv)), tv, is_probability=False)}]
for tname, m in trains.items():
    mt = m & ~quiet
    p = np.full(len(yv), Y["y_end"][mt].mean())
    rows.append({"baseline": "constant (train base rate)", "train": tname, **evaluate(yv, p, tv, threshold=0.5)})
    p = DurationCurve().fit(X[mt], Y["y_end"][mt].to_numpy()).predict_proba(Xv)
    rows.append({"baseline": "duration curve", "train": tname, **evaluate(yv, p, tv)})
end_table = show(rows, "END task: will the current Kyiv alert end within 30 min (validation)")
print("\n  [figures] reports/figures/baselines_start_pr.png, baselines_start_calibration.png")
