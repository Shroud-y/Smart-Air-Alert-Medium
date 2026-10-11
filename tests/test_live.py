import joblib
import numpy as np
import pandas as pd
import pytest

from saam.data import KYIV_CITY, load_alerts
from saam.features import build_features
from saam.live import from_history, parse_ubilling
from saam.predict import MODEL_DIR, _file, forecast

NOW = pd.Timestamp("2026-10-11 09:00", tz="UTC")


def district(on, at=None, level=None):
    return {"enabled": on, "type": "district", "enabled_at": at, "disabled_at": None, "alert_level": level}


PAYLOAD = {"source": "x", "cachedat": "x", "raw": {
    "АР Крим": {"enabled": True, "districts": [], "enabled_at": "2022-12-10T22:22:00+00:00", "alert_level": None},
    "Київська область": {"enabled": False, "enabled_at": None, "alert_level": None, "districts": {
        "Броварський район": district(True, "2026-10-11T08:40:00.000Z", "yellow"),
        "Вишгородський район": district(True, "2026-10-11T08:55:00.000Z", "red"),
        "Бучанський район": district(False)}},
    "Чернігівська область": {"enabled": True, "enabled_at": None, "alert_level": "yellow", "districts": {}},
    "м. Київ": {"enabled": True, "districts": [], "enabled_at": "2026-10-11T08:50:00.000Z", "alert_level": "yellow"},
    "Львівська область": {"enabled": False, "enabled_at": None, "alert_level": None, "districts": {"Львівський район": district(False)}},
}}


def test_parse_ubilling_gives_the_same_features_as_the_dataset_would():
    snap = parse_ubilling(PAYLOAD, NOW)
    assert snap.regions_on() == {"Київська область", "Чернігівська область", KYIV_CITY}  # Crimea is not a feature region
    assert snap.levels == {"Київська область": "red", "Чернігівська область": "yellow", KYIV_CITY: "yellow"}

    f = build_features(snap.alerts, np.array([int(NOW.timestamp())])).iloc[0]
    assert f.kyiv_oblast_on == 1 and f.kyiv_oblast_share == pytest.approx(2 / 7)
    assert f.kyiv_oblast_newest_min == 5 and f.kyiv_oblast_oldest_min == 20
    assert f.ko_brovary_on == 1 and f.ko_brovary_min == 20 and f.ko_bucha_on == 0
    assert f.kyiv_on_min == 10
    assert f.lviv_on == 0 and f.regions_on == 2


def test_alert_without_a_start_time_is_treated_as_long_ago_and_noted():
    snap = parse_ubilling(PAYLOAD, NOW)
    f = build_features(snap.alerts, np.array([int(NOW.timestamp())])).iloc[0]
    assert f.chernihiv_on == 1 and f.chernihiv_share == 1 and f.chernihiv_newest_min == 180
    assert any("Чернігівська область" in n for n in snap.notes)


@pytest.mark.skipif(not (MODEL_DIR / "thresholds.json").exists(), reason="models are not built; run scripts/final.py")
@pytest.mark.parametrize("when", ["2026-08-27 18:00", "2026-08-27 22:30", "2026-07-15 09:05"])
def test_replay_forecast_equals_the_model_on_the_training_grid(when):
    """The program must compute, from a snapshot, exactly the probability the
    model was evaluated with on the dataset grid."""
    df = load_alerts()
    t = pd.Timestamp(when, tz="UTC")
    fc = forecast(from_history(df, t), "recent")

    X = build_features(df, np.array([int(t.timestamp())]))
    expected = joblib.load(_file("lgbm", fc.task, "recent")).predict_proba(X)[0]
    assert fc.probability == pytest.approx(expected, abs=1e-12)
    assert fc.answer == (fc.probability >= fc.threshold)


def test_wall_shows_every_region_once_and_colours_it_by_state():
    from saam.ui import NAMES, WALL_ORDER, region_state, render_list, render_wall

    snap = parse_ubilling(PAYLOAD, NOW)
    assert len(WALL_ORDER) == len(set(WALL_ORDER)) == 26 and WALL_ORDER[0] == KYIV_CITY
    text = render_wall(snap).renderables[0].plain
    for region in WALL_ORDER:
        assert text.count(f" {NAMES[region]} ") == 1

    assert region_state("Київська область", snap) == "red"  # the most severe level among its districts
    assert region_state("Чернігівська область", snap) == "yellow"
    assert region_state("Львівська область", snap) == "quiet"
    assert region_state("АР Крим", snap) == "nodata"
    replay = from_history(load_alerts(), pd.Timestamp("2026-08-27 20:40", tz="UTC"))
    assert region_state("Луганська область", replay) == "nodata"  # permanent alert is absent from the dataset
    assert region_state("Харківська область", replay) == "alert"  # no levels in the dataset
    assert "Kyiv oblast (red)" in render_list(snap).plain


def _watch(argv, next_snapshot=None, max_ticks=3):
    import argparse
    import io

    from rich.console import Console

    import saam.__main__ as M

    args = argparse.Namespace(replay=None, mode="recent", debug=False, list=False, interval=None, step=5)
    for k, v in argv.items():
        setattr(args, k, v)
    console = Console(file=io.StringIO(), force_terminal=True, width=100, record=True)
    slept = []
    done = M.watch(args, next_snapshot or M.snapshot_source(args), console, sleep=slept.append, max_ticks=max_ticks)
    return done, slept, console.export_text()


@pytest.mark.skipif(not (MODEL_DIR / "thresholds.json").exists(), reason="models are not built; run scripts/final.py")
def test_watch_in_replay_moves_forward_in_steps():
    done, slept, text = _watch({"replay": "2026-08-27 23:40", "step": 10})
    assert done == 3 and slept == [2, 2, 2]  # default replay interval
    assert "2026-08-28 00:00 Kyiv time" in text  # 23:40 + 2 steps of 10 min


@pytest.mark.skipif(not (MODEL_DIR / "thresholds.json").exists(), reason="models are not built; run scripts/final.py")
def test_watch_keeps_the_last_screen_when_an_update_fails():
    good = from_history(load_alerts(), pd.Timestamp("2026-08-27 20:40", tz="UTC"))
    calls = iter([good, ConnectionError("down"), good])

    def source():
        item = next(calls)
        if isinstance(item, Exception):
            raise item
        return item

    done, slept, text = _watch({}, source)
    assert done == 2 and slept == [60, 60, 60]  # default live interval
    assert "Forecast" in text


@pytest.mark.skipif(not (MODEL_DIR / "thresholds.json").exists(), reason="models are not built; run scripts/final.py")
def test_replay_stops_at_the_end_of_the_dataset():
    done, _, text = _watch({"replay": "2026-09-07 08:20", "step": 5}, max_ticks=20)
    assert done < 20 and "replay finished" in text


def _screen(when, debug):
    import io

    from rich.console import Console

    from saam.ui import render

    snap = from_history(load_alerts(), pd.Timestamp(when, tz="UTC"))
    fc = forecast(snap, "recent")
    console = Console(file=io.StringIO(), force_terminal=True, width=120, record=True)
    console.print(render(snap, fc, debug=debug))
    return snap, fc, " ".join(console.export_text().replace("│", " ").split())  # undo wrapping inside panels


END_MOMENT, START_MOMENT = "2026-08-27 17:30", "2026-08-27 20:40"  # UTC: during a Kyiv alert / between alerts
CAVEAT = 'Weak hint: on the test period a "yes" was right only 45% of the time (target 60%).'


@pytest.mark.skipif(not (MODEL_DIR / "thresholds.json").exists(), reason="models are not built; run scripts/final.py")
@pytest.mark.parametrize("debug", [False, True])
def test_end_forecast_always_shows_its_test_precision_under_the_answer(debug, capsys):
    import saam.__main__ as M

    snap, fc, text = _screen(END_MOMENT, debug)
    assert fc.task == "end" and fc.caveat == CAVEAT
    assert text.count("right only 45% of the time") == 1  # shown once, with or without --debug

    M.print_plain(snap, fc, debug)
    assert capsys.readouterr().out.count("right only 45% of the time") == 1


@pytest.mark.skipif(not (MODEL_DIR / "thresholds.json").exists(), reason="models are not built; run scripts/final.py")
def test_start_forecast_has_no_caveat_and_reports_quality_only_in_debug(capsys):
    import saam.__main__ as M

    snap, fc, text = _screen(START_MOMENT, False)
    assert fc.task == "start" and fc.caveat is None
    assert "on the test period" not in text
    assert "64% of alerts were warned of" in _screen(START_MOMENT, True)[2]

    M.print_plain(snap, fc, False)
    assert "on the test period" not in capsys.readouterr().out


@pytest.mark.skipif(not (MODEL_DIR / "thresholds.json").exists(), reason="models are not built; run scripts/final.py")
def test_debug_names_the_feature_list_as_global_importance(capsys):
    import saam.__main__ as M

    snap, fc, text = _screen(START_MOMENT, True)
    assert "Global feature importance" in text and "not an explanation of this one" in text
    assert "Main features" not in text

    M.print_plain(snap, fc, True)
    out = capsys.readouterr().out
    assert "global feature importance" in out and "main features" not in out
