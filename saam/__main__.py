"""Console entry point:  python -m saam [--replay "2026-08-27 21:00"] [--mode recent|full] [--debug]

Without --replay the current state is taken from the ubilling API.
--replay takes a moment in Kyiv time and plays it back from the dataset.
"""
import argparse
import sys

import pandas as pd

from .data import KYIV_CITY, KYIV_TZ, REGION_EN, load_alerts
from .live import fetch_live, from_history
from .predict import DEFAULT_MODE, MODES, forecast


def get_snapshot(args):
    if args.replay:
        when = pd.Timestamp(args.replay).tz_localize(KYIV_TZ).tz_convert("UTC")
        return from_history(load_alerts(), when)
    return fetch_live()


def print_plain(snap, fc, debug):
    local = snap.time.tz_convert(KYIV_TZ)
    print(f"State at {local:%Y-%m-%d %H:%M} Kyiv time | source: {snap.source} | mode: {MODES[fc.mode]}")
    on = sorted(REGION_EN[r] + (f" ({snap.levels[r]})" if snap.levels.get(r) else "") for r in snap.regions_on() if r != KYIV_CITY)
    print(f"Regions under alert ({len(on)}): " + (", ".join(on) if on else "none"))
    if fc.kyiv_alert:
        print(f"Kyiv city: ALERT for {fc.kyiv_alert_minutes:.0f} min" + (f" ({snap.levels[KYIV_CITY]})" if snap.levels.get(KYIV_CITY) else ""))
    else:
        print("Kyiv city: no alert")
    print(f"\n{fc.question}  ->  {'YES' if fc.answer else 'NO'}")
    if debug:
        print(f"\n[debug] probability {fc.probability:.1%}, threshold {fc.threshold:.1%}")
        print(f"[debug] {fc.test_note}")
        print(f"[debug] frequency table: {fc.table_probability:.1%} for " + ", ".join(f"{k}: {v}" for k, v in fc.table_cell.items()))
        print("[debug] main features of the model (value now, share of the model's gain):")
        for name, value, share in fc.top_features:
            shown = "-" if pd.isna(value) else f"{value:g}"
            print(f"          {name:28s} {shown:>8s}   {share:.1%}")
        for note in snap.notes:
            print(f"[debug] note: {note}")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="saam", description="Air raid alert forecast for Kyiv.")
    ap.add_argument("--replay", metavar="DATETIME", help='moment in Kyiv time to play back from the dataset, e.g. "2026-08-27 21:00"')
    ap.add_argument("--mode", choices=list(MODES), default=DEFAULT_MODE, help="training period of the model: recent = from 2024-07 (default), full = full history")
    ap.add_argument("--debug", action="store_true", help="show the probability, the threshold and what the model looked at")
    args = ap.parse_args(argv)
    try:
        snap = get_snapshot(args)
    except Exception as e:  # network or API trouble: say so and point to the fallback
        print(f"Could not get the current state: {e}\nTry the replay mode, e.g.:  python -m saam --replay \"2026-08-27 21:00\"", file=sys.stderr)
        return 1
    print_plain(snap, forecast(snap, args.mode), args.debug)
    return 0


if __name__ == "__main__":
    sys.exit(main())
