"""Console rendering with rich: wall of regions and the forecast panel."""
import pandas as pd
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .data import KYIV_CITY, KYIV_TZ, REGION_EN

CRIMEA = "АР Крим"
NAMES = {**REGION_EN, CRIMEA: "Crimea"}
# Kyiv city first, then the rest by English name.
WALL_ORDER = [KYIV_CITY] + sorted((r for r in NAMES if r != KYIV_CITY), key=NAMES.get)
NO_DATA = {CRIMEA}  # never reported by the sources used here
NO_DATA_IN_REPLAY = {"Луганська область"}  # permanent alert, absent from the dataset
WALL_COLS, BRICK_W = 4, 19

TRAINED_ON = {"recent": "data from 2024-07", "full": "the full history"}
STYLE = {"red": "bold white on red3", "yellow": "bold black on yellow3", "alert": "bold white on red3",
         "quiet": "grey70 on grey23", "nodata": "grey42 on grey11"}


def region_state(region, snap):
    if region in NO_DATA or (region in NO_DATA_IN_REPLAY and "replay" in snap.source):
        return "nodata"
    if region not in snap.regions_on():
        return "quiet"
    return snap.levels.get(region) or "alert"


def render_wall(snap):
    """All regions as a wall of equal bricks, coloured by state."""
    out = Text()
    for i, region in enumerate(WALL_ORDER):
        out.append(f" {NAMES[region]}".ljust(BRICK_W), style=STYLE[region_state(region, snap)])
        last_in_row = (i + 1) % WALL_COLS == 0
        out.append("\n" if last_in_row or i == len(WALL_ORDER) - 1 else " ")
    legend = Text("\n")
    live = bool(snap.levels) or "live" in snap.source
    items = [(" red ", "red", " missile threat  "), (" yellow ", "yellow", " drone threat  ")] if live else [(" alert ", "alert", "  ")]
    for label, key, after in items + [(" quiet ", "quiet", "  "), (" n/a ", "nodata", " no data")]:
        legend.append(label, style=STYLE[key])
        legend.append(after)
    return Group(out, legend)


def render_list(snap):
    """Regions under alert as one line of text."""
    on = sorted(REGION_EN[r] + (f" ({snap.levels[r]})" if snap.levels.get(r) else "") for r in snap.regions_on() if r in REGION_EN)
    return Text(f"Regions under alert ({len(on)}): " + (", ".join(on) if on else "none"))


def render_forecast(snap, fc):
    if fc.kyiv_alert:
        level = f", {snap.levels[KYIV_CITY]}" if snap.levels.get(KYIV_CITY) else ""
        state = Text(f"Kyiv city: ALERT for {fc.kyiv_alert_minutes:.0f} min{level}", style="bold red3")
    else:
        state = Text("Kyiv city: no alert", style="bold green3")
    # "yes" is the bad news for the start task and the good news for the end task
    good = fc.answer == (fc.task == "end")
    answer = Text.assemble((fc.question + "   "), (" YES " if fc.answer else " NO ", "bold white on green4" if good else "bold white on red3"))
    lines = [state, Text(), answer]
    if fc.caveat:
        lines += [Text(), Text(fc.caveat, style="yellow3")]
    return Panel(Group(*lines), title="Forecast", title_align="left", padding=(1, 2), expand=False)


def render_debug(snap, fc):
    t = Table(show_header=False, box=None, padding=(0, 2, 0, 0))
    t.add_row("probability", f"{fc.probability:.1%}")
    t.add_row("threshold for \"yes\"", f"{fc.threshold:.1%}")
    if not fc.caveat:  # for "end" the same line is already under the answer
        t.add_row("measured quality", fc.test_note)
    t.add_row("frequency table", f"{fc.table_probability:.1%}  ({', '.join(f'{k}: {v}' for k, v in fc.table_cell.items())})")
    f = Table(title="Global feature importance", title_justify="left", title_style="bold", box=None, padding=(0, 2, 0, 0))
    f.add_column("feature")
    f.add_column("value now", justify="right")
    f.add_column("global importance", justify="right")
    for name, value, share in fc.top_features:
        f.add_row(name, "-" if pd.isna(value) else f"{value:.3g}", f"{share:.1%}")
    what = Text("Share of the model's total gain over all training data. "
                "The same list for every forecast, not an explanation of this one.", style="grey62")
    notes = [Text(f"note: {n}", style="grey62") for n in snap.notes]
    return Panel(Group(t, Text(), f, what, *notes), title="Debug", title_align="left", padding=(1, 2), expand=False)


def render(snap, fc, debug=False, as_list=False, status=None):
    local = snap.time.tz_convert(KYIV_TZ)
    header = Text.assemble(("Air raid alerts in Ukraine", "bold"), f"   {local:%Y-%m-%d %H:%M} Kyiv time\n",
                           (f"source: {snap.source}   model: trained on {TRAINED_ON[fc.mode]}\n", "grey62"))
    parts = [header, render_list(snap) if as_list else render_wall(snap), Text(), render_forecast(snap, fc)]
    if debug:
        parts.append(render_debug(snap, fc))
    if status:
        parts.append(Text(status, style="grey62"))
    return Group(*parts)
