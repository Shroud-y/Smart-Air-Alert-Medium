"""Small hyperparameter search for LightGBM on the validation period.

Run from the repository root:  python -X utf8 scripts/tune.py
The test period is not touched here.

20 random combinations from a 3x3x3x3 grid plus the current settings, for each
of the four models (2 tasks x 2 training periods). The current settings are
also refitted with 5 seeds to measure how much the score moves by chance.
"""
import itertools
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from saam import config as C
from saam.dataset import load_dataset, period_mask
from saam.metrics import threshold_for_precision, threshold_for_recall
from saam.models import LGBM_PARAMS, Lgbm

GRID = {
    "n_estimators": [150, 300, 600],
    "learning_rate": [0.02, 0.05, 0.1],
    "num_leaves": [7, 15, 31],
    "min_child_samples": [100, 300, 1000],
}
N_RANDOM, SEEDS = 20, [0, 1, 2, 3, 4]

X, Y = load_dataset()
val = period_mask(X.index, C.VAL_START, C.VAL_END)
quiet = Y["quiet"].to_numpy()
trains = {"full history": C.TRAIN_START_FULL, "from 2024-07": C.TRAIN_START_RECENT}

current = {k: LGBM_PARAMS[k] for k in GRID}
combos = [dict(zip(GRID, v)) for v in itertools.product(*GRID.values())]
rng = np.random.default_rng(0)
picked = [combos[i] for i in rng.choice(len(combos), N_RANDOM, replace=False)]
picked = [current] + [c for c in picked if c != current]


def score(key, y, p):
    out = {"pr_auc": average_precision_score(y, p), "brier": brier_score_loss(y, p)}
    if key == "start":  # precision at the agreed recall
        pred = p >= threshold_for_recall(y, p, C.START_MIN_RECALL)
        out["at_point"] = (pred & y).sum() / pred.sum()
    else:  # recall at the agreed precision
        thr = threshold_for_precision(y, p, C.END_MIN_PRECISION)
        out["at_point"] = np.nan if thr is None else ((p >= thr) & y).sum() / y.sum()
    return out


results = []
t0 = time.time()
for key, target, rows_mask in (("start", "y_start", quiet), ("end", "y_end", ~quiet)):
    mv = val & rows_mask
    Xv, yv = X[mv], Y[target][mv].to_numpy()
    for tname, tstart in trains.items():
        mt = period_mask(X.index, tstart, C.TRAIN_END) & rows_mask
        Xt, yt = X[mt], Y[target][mt].to_numpy()
        for params in picked:
            p = Lgbm(**params).fit(Xt, yt).predict_proba(Xv)
            results.append({"task": key, "train": tname, "kind": "current" if params == current else "candidate", "seed": 0, **params, **score(key, yv, p)})
        for seed in SEEDS[1:]:
            p = Lgbm(random_state=seed).fit(Xt, yt).predict_proba(Xv)
            results.append({"task": key, "train": tname, "kind": "current", "seed": seed, **current, **score(key, yv, p)})
        print(f"  done {key} / {tname} in {time.time() - t0:.0f}s total", flush=True)

res = pd.DataFrame(results)
res.to_csv(Path(__file__).resolve().parent.parent / "reports" / "tuning.csv", index=False)
point_name = {"start": f"precision at recall {C.START_MIN_RECALL:.0%}", "end": f"recall at precision {C.END_MIN_PRECISION:.0%}"}
pd.set_option("display.width", 200)

for (key, tname), g in res.groupby(["task", "train"], sort=False):
    seeds = g[g.kind == "current"]
    cur = seeds[seeds.seed == 0].iloc[0]
    cand = g[g.seed == 0].sort_values("pr_auc", ascending=False)
    best = cand.iloc[0]
    sd = seeds["pr_auc"].std()
    print(f"\n== {key.upper()} / {tname}   (at_point = {point_name[key]})")
    print(f"  current settings over {len(seeds)} seeds: PR-AUC {seeds.pr_auc.mean():.2%} +- {sd:.2%} (min {seeds.pr_auc.min():.2%}, max {seeds.pr_auc.max():.2%}); at_point {seeds.at_point.mean():.1%} +- {seeds.at_point.std():.1%}")
    print(f"  current, seed 0: PR-AUC {cur.pr_auc:.2%}, Brier {cur.brier:.4f}, at_point {cur.at_point:.1%}")
    print(f"  best candidate:  PR-AUC {best.pr_auc:.2%}, Brier {best.brier:.4f}, at_point {best.at_point:.1%}  -> gain {best.pr_auc - cur.pr_auc:+.2%} = {(best.pr_auc - cur.pr_auc) / sd:+.1f} seed-sd")
    show = cand.head(5)[list(GRID) + ["pr_auc", "brier", "at_point"]].copy()
    show["pr_auc"] = (show.pr_auc * 100).round(2)
    show["at_point"] = (show.at_point * 100).round(1)
    show["brier"] = show.brier.round(4)
    print("  top 5 by PR-AUC:")
    print("   " + show.to_string(index=False).replace("\n", "\n   "))
    worst = cand.iloc[-1]
    print(f"  worst candidate: PR-AUC {worst.pr_auc:.2%} ({ {k: worst[k] for k in GRID} })")

print("\n== average PR-AUC rank of each setting across the four models (1 = best)")
r = res[res.seed == 0].copy()
r["rank"] = r.groupby(["task", "train"])["pr_auc"].rank(ascending=False)
avg = r.groupby(list(GRID))["rank"].agg(["mean", "max"]).sort_values("mean").head(6)
print(avg.round(1).to_string())
print("  current settings:", current)
