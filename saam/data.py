"""Loading and cleaning of the historical alerts dataset.

All timestamps stay in UTC. Kyiv-local calendar values are derived separately
(see README, "Time", option A).
"""
from pathlib import Path

import numpy as np
import pandas as pd

DATA_PATH = Path(__file__).resolve().parent.parent / "official_data_uk.csv"

KYIV_TZ = "Europe/Kyiv"
KYIV_CITY = "м. Київ"

# Region names exactly as they appear in the dataset and in the ubilling API.
REGION_EN = {
    "Вінницька область": "Vinnytsia",
    "Волинська область": "Volyn",
    "Дніпропетровська область": "Dnipropetrovsk",
    "Донецька область": "Donetsk",
    "Житомирська область": "Zhytomyr",
    "Закарпатська область": "Zakarpattia",
    "Запорізька область": "Zaporizhzhia",
    "Івано-Франківська область": "Ivano-Frankivsk",
    "Київська область": "Kyiv oblast",
    "Кіровоградська область": "Kirovohrad",
    "Луганська область": "Luhansk",
    "Львівська область": "Lviv",
    "Миколаївська область": "Mykolaiv",
    "Одеська область": "Odesa",
    "Полтавська область": "Poltava",
    "Рівненська область": "Rivne",
    "Сумська область": "Sumy",
    "Тернопільська область": "Ternopil",
    "Харківська область": "Kharkiv",
    "Херсонська область": "Kherson",
    "Хмельницька область": "Khmelnytskyi",
    "Черкаська область": "Cherkasy",
    "Чернівецька область": "Chernivtsi",
    "Чернігівська область": "Chernihiv",
    KYIV_CITY: "Kyiv city",
}


def load_alerts(path=DATA_PATH):
    """Read the raw CSV, drop exact duplicate rows (all 7 columns equal) and
    rows with zero duration."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False)
    df = df.drop_duplicates()
    for col in ("started_at", "finished_at"):
        df[col] = pd.to_datetime(df[col], utc=True)
    df = df[df["finished_at"] > df["started_at"]]
    return df.reset_index(drop=True)


def union_intervals(starts, ends):
    """Union of possibly overlapping [start, end) intervals.

    Intervals are joined only when they overlap or touch; alerts separated by
    any pause stay separate. Returns sorted, disjoint (starts, ends) arrays.
    """
    order = np.argsort(starts, kind="stable")
    starts, ends = starts[order], ends[order]
    out_s, out_e = [], []
    for s, e in zip(starts, ends):
        if out_e and s <= out_e[-1]:
            out_e[-1] = max(out_e[-1], e)
        else:
            out_s.append(s)
            out_e.append(e)
    return np.array(out_s, dtype=starts.dtype), np.array(out_e, dtype=ends.dtype)


def region_intervals(df):
    """Oblast-level alert intervals: an oblast is under alert while at least
    one of its raions or hromadas is. Returns {region: (starts, ends)} as
    int64 epoch seconds (UTC)."""
    out = {}
    for region, g in df.groupby("oblast"):
        s = g["started_at"].to_numpy("datetime64[s]").astype("int64")
        e = g["finished_at"].to_numpy("datetime64[s]").astype("int64")
        out[region] = union_intervals(s, e)
    return out


def state_on_grid(intervals, grid):
    """True where the region is under alert at each grid instant
    (start <= t < end). `grid` is int64 epoch seconds."""
    starts, ends = intervals
    return (np.searchsorted(starts, grid, "right") - np.searchsorted(ends, grid, "right")) > 0
