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
