"""Console entry point:  python -m saam [options]

By default the program keeps running: it takes the current state from the
ubilling API and refreshes the screen about once a minute until Ctrl+C.
With --replay it plays the dataset back from the given moment (Kyiv time),
moving forward in steps. --once prints a single screen and exits.
"""
import argparse
import sys
import time

import pandas as pd

from .data import KYIV_CITY, KYIV_TZ, REGION_EN, load_alerts
from .live import fetch_live, from_history
from .predict import DEFAULT_MODE, MODES, forecast

LIVE_INTERVAL_S = 60  # ubilling allows 2 requests per second; once a minute is plenty
REPLAY_INTERVAL_S = 2
REPLAY_STEP_MIN = 5


def snapshot_source(args):
    """Returns a function that gives the next Snapshot on every call."""
    if not args.replay:
        return fetch_live
    df = load_alerts()
    start = pd.Timestamp(args.replay).tz_localize(KYIV_TZ).tz_convert("UTC")
    end = df["finished_at"].max()
    tick = iter(range(10**9))

    def next_snapshot():
        when = start + pd.Timedelta(minutes=args.step * next(tick))
        if when > end:
            raise EOFError(f"the dataset ends on {end.tz_convert(KYIV_TZ):%Y-%m-%d %H:%M} Kyiv time")
        return from_history(df, when)

    return next_snapshot


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
    if fc.caveat:
        print(fc.caveat)
    if debug:
        print(f"\n[debug] probability {fc.probability:.1%}, threshold {fc.threshold:.1%}")
        if not fc.caveat:  # for "end" the same line is already under the answer
            print(f"[debug] {fc.test_note}")
        print(f"[debug] frequency table: {fc.table_probability:.1%} for " + ", ".join(f"{k}: {v}" for k, v in fc.table_cell.items()))
        print("[debug] global feature importance - the same list for every forecast, not an explanation of this one")
        print("[debug] (feature, value now, share of the model's total gain over all training data):")
        for name, value, share in fc.top_features:
            shown = "-" if pd.isna(value) else f"{value:g}"
            print(f"          {name:28s} {shown:>8s}   {share:.1%}")
        for note in snap.notes:
            print(f"[debug] note: {note}")


def watch(args, next_snapshot, console, sleep=time.sleep, max_ticks=None):
    """Refresh the screen until Ctrl+C. If an update fails, the last good
    screen stays and the failure is shown under it. Returns the number of
    successful updates."""
    from rich.live import Live

    from .ui import render

    interval = args.interval or (REPLAY_INTERVAL_S if args.replay else LIVE_INTERVAL_S)
    how = f"replay, +{args.step} min every {interval:g} s" if args.replay else f"refreshes every {interval:g} s"
    last, done, ticks = None, 0, 0
    with Live(console=console, auto_refresh=False, screen=False) as live:
        try:
            while max_ticks is None or ticks < max_ticks:
                ticks += 1
                try:
                    snap = next_snapshot()
                    last = (snap, forecast(snap, args.mode))
                    done += 1
                    status = f"{how} - Ctrl+C to exit"
                except EOFError as e:  # replay reached the end of the dataset
                    if last is None:
                        raise
                    live.update(render(*last, debug=args.debug, as_list=args.list, status=f"replay finished: {e}"), refresh=True)
                    break
                except Exception as e:  # network or API trouble: keep the last screen
                    if last is None:
                        raise
                    status = f"update failed ({type(e).__name__}), showing the state above; retrying in {interval:g} s - Ctrl+C to exit"
                live.update(render(*last, debug=args.debug, as_list=args.list, status=status), refresh=True)
                sleep(interval)
        except KeyboardInterrupt:
            pass
    return done


def main(argv=None):
    ap = argparse.ArgumentParser(prog="saam", description="Air raid alert forecast for Kyiv.")
    ap.add_argument("--replay", metavar="DATETIME", help='play the dataset back from this moment in Kyiv time, e.g. "2026-08-27 21:00"')
    ap.add_argument("--mode", choices=list(MODES), default=DEFAULT_MODE, help="training period of the model: recent = from 2024-07 (default), full = full history")
    ap.add_argument("--debug", action="store_true", help="show the probability, the threshold and what the model looked at")
    ap.add_argument("--once", action="store_true", help="print one screen and exit instead of refreshing")
    ap.add_argument("--interval", type=float, metavar="SECONDS", help=f"seconds between refreshes (default {LIVE_INTERVAL_S} live, {REPLAY_INTERVAL_S} in replay)")
    ap.add_argument("--step", type=int, default=REPLAY_STEP_MIN, metavar="MINUTES", help=f"replay only: minutes of history per refresh (default {REPLAY_STEP_MIN})")
    ap.add_argument("--list", action="store_true", help="show regions under alert as one line instead of the wall")
    ap.add_argument("--plain", action="store_true", help="plain text without colours; implies --once")
    args = ap.parse_args(argv)

    fallback = 'Try the replay mode, e.g.:  python -m saam --replay "2026-08-27 21:00"'
    try:
        next_snapshot = snapshot_source(args)
        if args.plain or args.once:
            snap = next_snapshot()
            fc = forecast(snap, args.mode)
            if args.plain:
                print_plain(snap, fc, args.debug)
            else:
                from rich.console import Console

                from .ui import render

                Console().print(render(snap, fc, debug=args.debug, as_list=args.list))
        else:
            from rich.console import Console

            watch(args, next_snapshot, Console())
    except Exception as e:  # could not get even the first state: say so and point to the fallback
        print(f"Could not get the current state: {e}\n{fallback}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
