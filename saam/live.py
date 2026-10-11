"""Where the current state comes from: the ubilling API or the historical dataset.

Both sources produce the same thing, a Snapshot: the alerts active at one
moment with their start times. Features are computed from it by the same
code as in training (README, decision 18).
"""
from dataclasses import dataclass, field

import pandas as pd
import requests

from .config import MINUTES_CAP
from .data import REGION_EN

UBILLING_URL = "https://ubilling.net.ua/aerialalerts/?raw"
FAR_FUTURE = pd.Timestamp("2100-01-01", tz="UTC")
COLUMNS = ["oblast", "raion", "hromada", "started_at", "finished_at"]


@dataclass
class Snapshot:
    time: pd.Timestamp  # UTC moment the state refers to
    alerts: pd.DataFrame  # active alerts, columns as in the dataset
    source: str
    levels: dict = field(default_factory=dict)  # region -> "yellow" / "red", live only
    notes: list = field(default_factory=list)

    def regions_on(self):
        return set(self.alerts["oblast"])


def _frame(rows):
    df = pd.DataFrame(rows, columns=COLUMNS)
    for c in ("started_at", "finished_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def parse_ubilling(payload, now):
    """Turn the `?raw` answer of ubilling into a Snapshot. An oblast is under
    alert if the oblast itself or any of its districts is enabled."""
    rows, levels, notes = [], {}, []

    def start_of(item, where):
        if item.get("enabled_at"):
            return pd.Timestamp(item["enabled_at"]).tz_convert("UTC")
        notes.append(f"{where}: start time unknown, treated as long ago")
        return now - pd.Timedelta(minutes=MINUTES_CAP)

    for region, state in payload["raw"].items():
        if region not in REGION_EN:
            continue
        found = []
        if state.get("enabled"):
            rows.append((region, "", "", start_of(state, region), FAR_FUTURE))
            found.append(state.get("alert_level"))
        for raion, d in (state.get("districts") or {}).items():
            if d.get("enabled"):
                rows.append((region, raion, "", start_of(d, f"{region} / {raion}"), FAR_FUTURE))
                found.append(d.get("alert_level"))
        if found:
            levels[region] = "red" if "red" in found else ("yellow" if "yellow" in found else None)
    return Snapshot(now, _frame(rows), "ubilling.net.ua (live)", levels, notes)


def fetch_live(timeout=15):
    r = requests.get(UBILLING_URL, timeout=timeout, headers={"User-Agent": "saam-mvp"})
    r.raise_for_status()
    return parse_ubilling(r.json(), pd.Timestamp.now(tz="UTC").floor("s"))


def from_history(df, when):
    """Snapshot of the dataset at `when` (UTC): only the alerts active at that
    moment, with their end hidden - exactly what a live source would show."""
    when = pd.Timestamp(when)
    when = when.tz_localize("UTC") if when.tzinfo is None else when.tz_convert("UTC")
    active = df[(df["started_at"] <= when) & (df["finished_at"] > when)][COLUMNS].copy()
    active["finished_at"] = FAR_FUTURE
    return Snapshot(when, active.reset_index(drop=True), "dataset (replay)")
