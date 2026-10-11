"""Features and targets on a time grid.

Every feature at time t is computed only from a snapshot of the state at t:
which alerts are active and when they started. Nothing about alerts that
have already ended is used, because the live API does not report it and the
program is opened for a quick look, not left running. An alert's end time is
used only as "has it ended by t". The same code serves history and live data.

All arithmetic is in UTC epoch seconds; Kyiv-local calendar values are
derived separately (README, "Time", option A).
"""
import numpy as np
import pandas as pd

from .config import MINUTES_CAP
from .data import KYIV_CITY, KYIV_TZ, REGION_EN

EXCLUDED = {"Луганська область"}  # permanent alert, absent from the data
FEATURE_REGIONS = [r for r in REGION_EN if r not in EXCLUDED and r != KYIV_CITY]
KYIV_OBLAST = "Київська область"

# Raions per oblast. Fixed so that history and live data use the same
# denominator; equal to the counts in the dataset and in the ubilling API.
RAION_COUNT = {
    "Вінницька область": 6, "Волинська область": 4, "Дніпропетровська область": 7,
    "Донецька область": 8, "Житомирська область": 4, "Закарпатська область": 6,
    "Запорізька область": 5, "Івано-Франківська область": 6, "Київська область": 7,
    "Кіровоградська область": 4, "Львівська область": 7, "Миколаївська область": 4,
    "Одеська область": 7, "Полтавська область": 4, "Рівненська область": 4,
    "Сумська область": 5, "Тернопільська область": 3, "Харківська область": 7,
    "Херсонська область": 5, "Хмельницька область": 3, "Черкаська область": 4,
    "Чернівецька область": 3, "Чернігівська область": 5,
}
KYIV_OBLAST_RAIONS = {
    "Бориспільський район": "boryspil", "Броварський район": "brovary",
    "Бучанський район": "bucha", "Білоцерківський район": "bila_tserkva",
    "Вишгородський район": "vyshhorod", "Обухівський район": "obukhiv",
    "Фастівський район": "fastiv",
}


def _slug(region):
    return REGION_EN[region].lower().replace(" ", "_").replace("-", "_")


def _epoch(series):
    return series.to_numpy("datetime64[s]").astype("int64")


def make_grid(start, end, step_min):
    """Regular UTC grid of epoch seconds in [start, end)."""
    s = int(pd.Timestamp(start, tz="UTC").timestamp())
    e = int(pd.Timestamp(end, tz="UTC").timestamp())
    return np.arange(s, e, step_min * 60, dtype="int64")


def _elapsed(starts, ends, grid):
    """Minutes since the start of the alert active at each grid instant, NaN
    where none is active. `starts`/`ends` must be sorted and non-overlapping."""
    if len(starts) == 0:
        return np.full(len(grid), np.nan)
    k = np.searchsorted(starts, grid, "right") - 1
    kc = np.clip(k, 0, None)
    active = (k >= 0) & (ends[kc] > grid)
    return np.where(active, (grid - starts[kc]) / 60.0, np.nan)


def _cap(minutes):
    """Cap "minutes since an alert started" at MINUTES_CAP, i.e. "long ago".
    Uncapped, the value for a region with a months-long alert is a day
    counter and lets a model learn the calendar date."""
    return np.minimum(minutes, MINUTES_CAP).astype("float32")


def _region_block(g, grid):
    """Per-raion elapsed minutes for one oblast.

    Returns (by_raion, oblast_level): `by_raion` maps a raion name to the
    elapsed-minutes arrays of every alert series inside it (the raion itself
    and its hromadas); `oblast_level` is the list for oblast-wide alerts.
    """
    by_raion, oblast_level = {}, []
    for (raion, _), u in g.groupby(["raion", "hromada"], sort=False):
        s, e = _epoch(u["started_at"]), _epoch(u["finished_at"])
        order = np.argsort(s, kind="stable")
        el = _elapsed(s[order], e[order], grid)
        (by_raion.setdefault(raion, []) if raion else oblast_level).append(el)
    return by_raion, oblast_level


def _nan_reduce(arrays, fn, n):
    if not arrays:
        return np.full(n, np.nan)
    stacked = np.vstack(arrays)
    out = np.full(n, np.nan)
    has = ~np.isnan(stacked).all(axis=0)
    out[has] = fn(stacked[:, has], axis=0)
    return out


def build_features(df, grid):
    """Feature table for the given grid (epoch seconds, UTC).

    `df` holds alert rows: oblast, raion, hromada, started_at, finished_at.
    """
    n = len(grid)
    cols = {}
    regions_on = np.zeros(n)
    groups = dict(tuple(df.groupby("oblast", sort=False)))

    for region in FEATURE_REGIONS:
        name = _slug(region)
        g = groups.get(region)
        by_raion, oblast_level = _region_block(g, grid) if g is not None else ({}, [])
        whole = _nan_reduce(oblast_level, np.nanmin, n)  # oblast-wide alert active
        raion_el = {r: _nan_reduce(a, np.nanmin, n) for r, a in by_raion.items()}

        everything = oblast_level + [a for arrs in by_raion.values() for a in arrs]
        newest = _nan_reduce(everything, np.nanmin, n)
        oldest = _nan_reduce(everything, np.nanmax, n)
        on = ~np.isnan(newest)

        n_raions_on = sum((~np.isnan(a)).astype(float) for a in raion_el.values()) if raion_el else np.zeros(n)
        share = np.where(~np.isnan(whole), 1.0, n_raions_on / RAION_COUNT[region])

        cols[f"{name}_on"] = on.astype("float32")
        cols[f"{name}_share"] = share.astype("float32")
        cols[f"{name}_newest_min"] = _cap(newest)
        cols[f"{name}_oldest_min"] = _cap(oldest)
        regions_on += on

        if region == KYIV_OBLAST:
            for raion, short in KYIV_OBLAST_RAIONS.items():
                # an oblast-wide alert covers every raion
                parts = [a for a in (raion_el.get(raion), whole) if a is not None]
                el = _nan_reduce(parts, np.nanmin, n)
                cols[f"ko_{short}_on"] = (~np.isnan(el)).astype("float32")
                cols[f"ko_{short}_min"] = _cap(el)

    cols["regions_on"] = regions_on.astype("float32")

    ky_s, ky_e = kyiv_intervals(df)
    cols["kyiv_on_min"] = _elapsed(ky_s, ky_e, grid).astype("float32")

    local = pd.to_datetime(grid, unit="s", utc=True).tz_convert(KYIV_TZ)
    hour = local.hour + local.minute / 60.0
    cols["hour_kyiv"] = local.hour.to_numpy("float32")
    cols["dow_kyiv"] = local.dayofweek.to_numpy("float32")
    cols["hour_sin"] = np.sin(2 * np.pi * hour / 24).astype("float32")
    cols["hour_cos"] = np.cos(2 * np.pi * hour / 24).astype("float32")

    return pd.DataFrame(cols, index=pd.to_datetime(grid, unit="s", utc=True))


def kyiv_intervals(df):
    """Sorted (starts, ends) of Kyiv city alerts, epoch seconds."""
    k = df[df["oblast"] == KYIV_CITY]
    s, e = _epoch(k["started_at"]), _epoch(k["finished_at"])
    order = np.argsort(s, kind="stable")
    return s[order], e[order]


def build_targets(df, grid, horizon_min):
    """Targets for both tasks.

    y_start: a Kyiv alert starts in (t, t+H]; meaningful where `quiet`.
    y_end:   the active Kyiv alert ends in (t, t+H]; meaningful where not `quiet`.
    """
    h = horizon_min * 60
    ky_s, ky_e = kyiv_intervals(df)
    nxt = np.searchsorted(ky_s, grid, "right")
    has_next = nxt < len(ky_s)
    y_start = np.zeros(len(grid), bool)
    y_start[has_next] = ky_s[nxt[has_next]] - grid[has_next] <= h

    cur = nxt - 1
    cc = np.clip(cur, 0, None)
    active = (cur >= 0) & (ky_e[cc] > grid)
    y_end = active & (ky_e[cc] - grid <= h)

    idx = pd.to_datetime(grid, unit="s", utc=True)
    return pd.DataFrame({"quiet": ~active, "y_start": y_start & ~active, "y_end": y_end}, index=idx)
