"""Exploratory analysis. Prints numbers and writes figures to reports/figures/.

Run from the repository root:  python -X utf8 scripts/eda.py
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from saam.data import KYIV_CITY, KYIV_TZ, REGION_EN, load_alerts, region_intervals, state_on_grid

FIG_DIR = Path(__file__).resolve().parent.parent / "reports" / "figures"
H = 30 * 60  # forecast horizon, seconds
STEP = 60  # grid step, seconds
RECENT_FROM = "2025-07-01"

SURFACE, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE = "#2a78d6", "#eb6834"
SEQ = LinearSegmentedColormap.from_list("seq", ["#fcfcfb", "#cde2fb", "#86b6ef", "#2a78d6", "#0d366b"])

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": INK2, "text.color": INK,
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.spines.top": False, "axes.spines.right": False,
    "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6, "axes.axisbelow": True,
    "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlelocation": "left", "font.size": 9,
})


def save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  [figure] reports/figures/{name}")


def epoch(ts):
    return int(pd.Timestamp(ts, tz="UTC").timestamp())


df = load_alerts()
iv = region_intervals(df)
regions = [r for r in REGION_EN if r in iv and r != "Луганська область"]
others = [r for r in regions if r != KYIV_CITY]
en = REGION_EN

t0 = int(df["started_at"].min().timestamp()) // STEP * STEP + STEP
t1 = int(df["finished_at"].max().timestamp()) // STEP * STEP
grid = np.arange(t0, t1 - H, STEP, dtype="int64")
grid_ts = pd.to_datetime(grid, unit="s", utc=True)
state = {r: state_on_grid(iv[r], grid) for r in regions}
ky_s, ky_e = iv[KYIV_CITY]
kyiv = state[KYIV_CITY]

# Target "start": a Kyiv alert starts in (t, t+H]; defined where Kyiv is quiet.
nxt = np.searchsorted(ky_s, grid, "right")
y_start = np.zeros(len(grid), bool)
ok = nxt < len(ky_s)
y_start[ok] = ky_s[nxt[ok]] - grid[ok] <= H
quiet = ~kyiv
# Target "end": the current Kyiv alert ends in (t, t+H]; defined where Kyiv is under alert.
cur = np.searchsorted(ky_s, grid, "right") - 1
y_end = np.zeros(len(grid), bool)
y_end[kyiv] = ky_e[cur[kyiv]] - grid[kyiv] <= H

years = grid_ts.year.to_numpy()
recent = grid >= epoch(RECENT_FROM)

# ---------------------------------------------------------------- 6. quality
print("== 6. Data quality (all regions, after dropping duplicates)")
dur = (df["finished_at"] - df["started_at"]).dt.total_seconds() / 60
print(f"  rows {len(df)}, regions {df['oblast'].nunique()}, Kyiv alerts {len(ky_s)}")
print(f"  duration <= 0: {(dur <= 0).sum()}, < 1 min: {(dur < 1).sum()}, > 24 h: {(dur > 1440).sum()}, max {dur.max() / 60:.1f} h")
long = df.assign(hours=dur / 60).nlargest(3, "hours")
for _, r in long.iterrows():
    print(f"    longest: {en.get(r['oblast'], r['oblast'])} / {r['level']} / {r['started_at']:%Y-%m-%d} / {r['hours']:.0f} h")
key = df.sort_values("started_at").groupby(["oblast", "raion", "hromada"])
overlaps = sum(int((g["started_at"].iloc[1:].to_numpy() < g["finished_at"].iloc[:-1].to_numpy()).sum()) for _, g in key)
print(f"  overlapping intervals within the same oblast/raion/hromada: {overlaps}")
print(f"  Luhansk rows: {(df['oblast'] == 'Луганська область').sum()} (permanent alert is not in the data; region excluded)")

# -------------------------------------------------------------- 5. base rate
print(f"\n== 5. How rare is the event (1-min grid, H = {H // 60} min)")
print(f"  grid {grid_ts[0]:%Y-%m-%d %H:%M} .. {grid_ts[-1]:%Y-%m-%d %H:%M} UTC, {len(grid)} points")
print(f"  Kyiv under alert: {kyiv.mean():.1%} of the time")
print(f"  'start' base rate (quiet minutes followed by a start within H): {y_start[quiet].mean():.2%}")
print(f"  'end' base rate (alert minutes followed by an end within H):    {y_end[kyiv].mean():.2%}")
rows = []
for yr in sorted(set(years)):
    m = years == yr
    rows.append((yr, kyiv[m].mean(), y_start[m & quiet].mean(), y_end[m & kyiv].mean()))
print("  year  under_alert  start_rate  end_rate")
for yr, a, s, e in rows:
    print(f"  {yr}  {a:10.1%}  {s:9.2%}  {e:8.1%}")
for name, a, b in (("train", "2022-03-15", "2026-04-01"), ("validation", "2026-04-01", "2026-07-01"), ("test", "2026-07-01", "2026-09-07")):
    m = (grid >= epoch(a)) & (grid < epoch(b))
    n_starts = int(((ky_s >= epoch(a)) & (ky_s < epoch(b))).sum())
    print(f"  {name:10s} Kyiv starts {n_starts:5d}, start_rate {y_start[m & quiet].mean():.2%}, end_rate {y_end[m & kyiv].mean():.1%}")

# ------------------------------------------------ 1-2. neighbours as a signal
print("\n== 1-2. Other regions as a signal for a Kyiv start")
print("  share_time: share of quiet-Kyiv minutes when the region is under alert")
print("  precision : P(Kyiv start within H | region under alert, Kyiv quiet)")
print("  recall    : share of positive minutes covered by 'region under alert'")
tab = []
for r in others:
    row = {"region": en[r]}
    for tag, m in (("all", quiet), ("recent", quiet & recent)):
        on = m & state[r]
        row[f"share_time_{tag}"] = on.sum() / m.sum()
        row[f"precision_{tag}"] = y_start[on].mean() if on.any() else np.nan
        row[f"recall_{tag}"] = (y_start & on).sum() / (y_start & m).sum()
    tab.append(row)
tab = pd.DataFrame(tab).set_index("region").sort_values("precision_recent", ascending=False)
base_all, base_recent = y_start[quiet].mean(), y_start[quiet & recent].mean()
print(f"  base rate: all {base_all:.2%}, recent (from {RECENT_FROM}) {base_recent:.2%}")
print((tab * 100).round(1).to_string())

print("\n  Lead time: at each Kyiv start, how long the region had already been under alert")
idx_start = np.searchsorted(grid, ky_s, "left")  # first grid point at/after the start
lead_rows, lead_samples = [], {}
for r in others:
    s, e = iv[r]
    k = np.searchsorted(s, ky_s, "right") - 1
    active = (k >= 0) & (e[np.clip(k, 0, None)] > ky_s)
    lead = np.where(active, (ky_s - s[np.clip(k, 0, None)]) / 60, np.nan)
    for tag, m in (("all", np.ones(len(ky_s), bool)), ("recent", ky_s >= epoch(RECENT_FROM))):
        l = lead[m]
        lead_rows.append({"region": en[r], "period": tag, "active_at_start": np.mean(~np.isnan(l)),
                          "lead>=10": np.mean(l >= 10), "lead>=30": np.mean(l >= 30),
                          "median_lead_min": np.nanmedian(l) if np.any(~np.isnan(l)) else np.nan})
    lead_samples[en[r]] = lead[ky_s >= epoch(RECENT_FROM)]
lead_tab = pd.DataFrame(lead_rows)
for tag in ("all", "recent"):
    t = lead_tab[lead_tab.period == tag].set_index("region").drop(columns="period")
    t = t.sort_values("lead>=10", ascending=False).head(10)
    t[["active_at_start", "lead>=10", "lead>=30"]] *= 100
    print(f"  -- {tag}: share of Kyiv starts (%), top 10 by lead>=10")
    print(t.round(1).to_string())

# no warning at all: Kyiv starts with no other region under alert
n_on = np.zeros(len(ky_s), int)
for r in others:
    s, e = iv[r]
    k = np.searchsorted(s, ky_s, "right") - 1
    n_on += (k >= 0) & (e[np.clip(k, 0, None)] > ky_s)
for tag, m in (("all", np.ones(len(ky_s), bool)), ("recent", ky_s >= epoch(RECENT_FROM))):
    print(f"  Kyiv starts with no other region under alert ({tag}): {np.mean(n_on[m] == 0):.1%}; median regions under alert {np.median(n_on[m]):.0f}")

top = tab.index[:12]
fig, ax = plt.subplots(figsize=(8, 4.6))
ypos = np.arange(len(top))[::-1]
ax.barh(ypos + 0.2, tab.loc[top, "precision_all"] * 100, 0.36, color=BLUE, label="2022–2026")
ax.barh(ypos - 0.2, tab.loc[top, "precision_recent"] * 100, 0.36, color=ORANGE, label=f"from {RECENT_FROM}")
ax.axvline(base_recent * 100, color=INK2, lw=1, ls="--")
ax.text(base_recent * 100, len(top) - 0.35, f" base rate {base_recent:.1%}", color=INK2, va="bottom", fontsize=8)
ax.set_yticks(ypos, top)
ax.set_xlabel("P(Kyiv alert starts within 30 min | region under alert, Kyiv quiet), %")
ax.set_title("Which regions signal an upcoming Kyiv alert")
ax.grid(axis="y", visible=False)
ax.legend(frameon=False, loc="lower right")
save(fig, "neighbour_signal.png")

# Regions adjacent to Kyiv oblast plus Sumy; frontline regions are left out
# because they are under alert almost permanently and their lead is meaningless.
box_regions = ["Kyiv oblast", "Zhytomyr", "Cherkasy", "Vinnytsia", "Poltava", "Chernihiv", "Sumy"]
fig, ax = plt.subplots(figsize=(8, 4.2))
data = [np.clip(lead_samples[r][~np.isnan(lead_samples[r])], 0, 240) for r in box_regions]
ax.boxplot(data[::-1], orientation="horizontal", widths=0.55, showfliers=False, patch_artist=True,
           boxprops=dict(facecolor="#cde2fb", edgecolor=BLUE), medianprops=dict(color="#0d366b", lw=2),
           whiskerprops=dict(color=BLUE), capprops=dict(color=BLUE))
ax.set_yticks(range(1, len(box_regions) + 1), box_regions[::-1])
ax.axvline(30, color=INK2, lw=1, ls="--")
ax.text(32, 0.55, "forecast horizon, 30 min", color=INK2, fontsize=8, va="bottom")
ax.set_xlabel("Minutes the region had been under alert when the Kyiv alert started (capped at 240)")
ax.set_title(f"Lead time before a Kyiv alert, from {RECENT_FROM}", pad=12)
ax.grid(axis="y", visible=False)
save(fig, "lead_time.png")

# ------------------------------------------- 3. effect of raion-level alerts
print("\n== 3. Effect of the move to raion-level alerts")
uniq = df.assign(q=df["started_at"].dt.tz_convert(None).dt.to_period("Q"))
mix = uniq.groupby(["q", "level"]).size().unstack(fill_value=0)
print("  rows per quarter by level:")
print(mix.to_string())

month = grid_ts.tz_convert(None).to_period("M")
codes, months = pd.factorize(month)
share = np.array([np.bincount(codes, weights=state[r], minlength=len(months)) / np.bincount(codes, minlength=len(months)) for r in regions])
order = np.argsort(-share.mean(axis=1))
fig, ax = plt.subplots(figsize=(10, 5.6))
im = ax.imshow(share[order] * 100, aspect="auto", cmap=SEQ, vmin=0, vmax=100, interpolation="nearest")
ax.set_yticks(range(len(regions)), [en[regions[i]] for i in order], fontsize=7.5)
ticks = [i for i, m in enumerate(months) if m.month == 1]
ax.set_xticks(ticks, [str(months[i].year) for i in ticks])
ax.grid(False)
ax.set_title("Share of time under alert, by region and month (oblast = any raion or hromada)")
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label("% of the month under alert")
cb.outline.set_visible(False)
save(fig, "alert_share_heatmap.png")

half = pd.Series(grid_ts.year.astype(str) + np.where(grid_ts.month <= 6, "H1", "H2"))
print("  share of time under alert by half-year, % (selected regions):")
sel = ["Київська область", "Чернігівська область", "Житомирська область", "Черкаська область", "Полтавська область", "Сумська область", "Харківська область", KYIV_CITY]
tbl = pd.DataFrame({en[r]: pd.Series(state[r]).groupby(half.values).mean() * 100 for r in sel})
print(tbl.round(1).to_string())
ko = "Київська область"
print("  'Kyiv oblast under alert' as a signal, by half-year:")
print("  half    share_time  precision  recall")
for hname in sorted(set(half)):
    m = (half.values == hname) & quiet
    on = m & state[ko]
    print(f"  {hname}  {on.sum() / m.sum():9.1%}  {y_start[on].mean():9.1%}  {(y_start & on).sum() / max((y_start & m).sum(), 1):6.1%}")

# --------------------------------------------------- 4. drift across years
print("\n== 4. Drift of Kyiv alerts across years")
ky = pd.DataFrame({"start": pd.to_datetime(ky_s, unit="s", utc=True), "dur": (ky_e - ky_s) / 60})
ky["local"] = ky["start"].dt.tz_convert(KYIV_TZ)
ky["year"] = ky["start"].dt.year
print(ky.groupby("year")["dur"].agg(alerts="size", median_min="median", mean_min="mean", p90_min=lambda s: s.quantile(0.9)).round(0).to_string())

pm = ky.set_index(ky["start"].dt.tz_convert(None)).resample("MS")["dur"].agg(["size", "sum"])
fig, axes = plt.subplots(2, 1, figsize=(10, 5), sharex=True)
axes[0].bar(pm.index, pm["size"], width=24, color=BLUE)
axes[0].set_ylabel("alerts per month")
axes[0].set_title("Kyiv city: alerts per month and hours under alert")
axes[1].bar(pm.index, pm["sum"] / 60, width=24, color=BLUE)
axes[1].set_ylabel("hours under alert")
for a in axes:
    a.grid(axis="x", visible=False)
save(fig, "kyiv_alerts_per_month.png")

hh = pd.crosstab(ky["year"], ky["local"].dt.hour).reindex(columns=range(24), fill_value=0)
hh_share = hh.div(hh.sum(axis=1), axis=0) * 100
fig, ax = plt.subplots(figsize=(10, 2.9))
im = ax.imshow(hh_share, aspect="auto", cmap=SEQ, vmin=0, interpolation="nearest")
ax.set_yticks(range(len(hh_share)), hh_share.index)
ax.set_xticks(range(24), range(24))
ax.set_xlabel("Hour of day, Kyiv time")
ax.grid(False)
ax.set_title("Kyiv city: when alerts start, share of the year's alerts by hour")
cb = fig.colorbar(im, ax=ax, pad=0.01)
cb.set_label("% of alerts")
cb.outline.set_visible(False)
save(fig, "kyiv_start_hour_by_year.png")
print("  share of alert starts at night (22:00-05:59 Kyiv time), by year:")
night = ky["local"].dt.hour.isin([22, 23, 0, 1, 2, 3, 4, 5])
print((night.groupby(ky["year"]).mean() * 100).round(1).to_string())

fig, ax = plt.subplots(figsize=(8, 4))
ts = np.arange(0, 241, 5)
for label, m, color in (("2022–2024", ky["year"] <= 2024, BLUE), ("2025–2026", ky["year"] >= 2025, ORANGE)):
    d = ky.loc[m, "dur"].to_numpy()
    p = [np.mean(d[d > t] <= t + 30) * 100 if (d > t).sum() >= 30 else np.nan for t in ts]
    ax.plot(ts, p, color=color, lw=2, label=label)
    print(f"  P(end within 30 min | elapsed) {label}: " + ", ".join(f"{t}m {p[list(ts).index(t)]:.0f}%" for t in (0, 30, 60, 120, 180)))
ax.set_xlabel("Minutes since the alert started")
ax.set_ylabel("P(alert ends within the next 30 min), %")
ax.set_ylim(0, 100)
ax.set_title("Kyiv city: chance that an ongoing alert ends within 30 min")
ax.legend(frameon=False)
save(fig, "kyiv_end_curve.png")
