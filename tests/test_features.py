import numpy as np
import pandas as pd
import pytest

from saam.data import KYIV_CITY, load_alerts
from saam.features import build_features, build_targets, make_grid

KO = "Київська область"
CN = "Чернігівська область"


def rows(*items):
    """items: (oblast, raion, hromada, start, end) with 'YYYY-MM-DD HH:MM' in UTC."""
    df = pd.DataFrame(items, columns=["oblast", "raion", "hromada", "started_at", "finished_at"])
    for c in ("started_at", "finished_at"):
        df[c] = pd.to_datetime(df[c], utc=True)
    return df


def at(frame, ts):
    return frame.loc[pd.Timestamp(ts, tz="UTC")]


def test_targets_match_hand_computed_example():
    df = rows((KYIV_CITY, "", "", "2025-03-01 10:00", "2025-03-01 10:40"))
    t = build_targets(df, make_grid("2025-03-01 09:00", "2025-03-01 12:00", 5), horizon_min=30)

    assert not at(t, "2025-03-01 09:25").y_start  # start is 35 min away
    assert at(t, "2025-03-01 09:30").y_start  # exactly 30 min away
    assert at(t, "2025-03-01 09:55").y_start
    assert at(t, "2025-03-01 09:55").quiet
    assert not at(t, "2025-03-01 10:00").quiet  # alert is active from its start instant
    assert not at(t, "2025-03-01 10:00").y_start  # no "start" target during an alert

    assert not at(t, "2025-03-01 10:05").y_end  # ends in 35 min
    assert at(t, "2025-03-01 10:10").y_end  # ends in exactly 30 min
    assert at(t, "2025-03-01 10:40").quiet  # end instant is already quiet
    assert not at(t, "2025-03-01 10:40").y_end
    assert not at(t, "2025-03-01 11:00").y_start  # nothing follows


def test_oblast_is_on_when_any_raion_or_hromada_is():
    df = rows(
        (KO, "Броварський район", "", "2025-03-01 10:00", "2025-03-01 11:00"),
        (KO, "Вишгородський район", "Some hromada", "2025-03-01 10:30", "2025-03-01 10:50"),
        (CN, "", "", "2025-03-01 10:20", "2025-03-01 10:25"),
        (KYIV_CITY, "", "", "2025-03-01 08:00", "2025-03-01 08:30"),
    )
    f = build_features(df, make_grid("2025-03-01 09:00", "2025-03-01 12:00", 5))

    r = at(f, "2025-03-01 09:55")
    assert r.kyiv_oblast_on == 0 and r.kyiv_oblast_share == 0 and np.isnan(r.kyiv_oblast_newest_min)

    r = at(f, "2025-03-01 10:10")  # one raion of seven
    assert r.kyiv_oblast_on == 1
    assert r.kyiv_oblast_share == pytest.approx(1 / 7)
    assert r.ko_brovary_on == 1 and r.ko_vyshhorod_on == 0
    assert r.ko_brovary_min == 10

    r = at(f, "2025-03-01 10:40")  # raion + a hromada of another raion
    assert r.kyiv_oblast_share == pytest.approx(2 / 7)
    assert r.kyiv_oblast_newest_min == 10 and r.kyiv_oblast_oldest_min == 40
    assert r.ko_vyshhorod_on == 1

    r = at(f, "2025-03-01 10:20")  # oblast-wide alert covers every raion
    assert r.chernihiv_on == 1 and r.chernihiv_share == 1
    assert at(f, "2025-03-01 10:25").chernihiv_on == 0  # end instant is off

    assert at(f, "2025-03-01 10:10").regions_on == 1
    assert at(f, "2025-03-01 10:20").regions_on == 2


def test_oblast_wide_alert_switches_on_every_kyiv_oblast_raion():
    df = rows((KO, "", "", "2024-03-01 10:00", "2024-03-01 11:00"), (KYIV_CITY, "", "", "2024-03-01 08:00", "2024-03-01 08:30"))
    r = at(build_features(df, make_grid("2024-03-01 10:00", "2024-03-01 10:30", 5)), "2024-03-01 10:15")
    assert r.ko_bucha_on == 1 and r.ko_bucha_min == 15 and r.kyiv_oblast_share == 1


def test_kyiv_city_own_features():
    df = rows((KYIV_CITY, "", "", "2025-03-01 10:00", "2025-03-01 10:40"))
    f = build_features(df, make_grid("2025-03-01 09:00", "2025-03-01 12:00", 5))
    assert np.isnan(at(f, "2025-03-01 09:55").kyiv_on_min)
    assert at(f, "2025-03-01 10:15").kyiv_on_min == 15
    assert np.isnan(at(f, "2025-03-01 11:00").kyiv_on_min)


def test_calendar_features_use_kyiv_time():
    df = rows((KYIV_CITY, "", "", "2025-01-01 00:00", "2025-01-01 00:10"))
    summer = build_features(df, make_grid("2025-07-15 21:30", "2025-07-15 21:35", 5)).iloc[0]
    winter = build_features(df, make_grid("2025-12-15 21:30", "2025-12-15 21:35", 5)).iloc[0]
    assert (summer.hour_kyiv, summer.dow_kyiv) == (0, 2)  # 00:30 Wednesday, UTC+3
    assert (winter.hour_kyiv, winter.dow_kyiv) == (23, 0)  # 23:30 Monday, UTC+2


@pytest.mark.parametrize("day, hours", [("2025-10-26", 25), ("2025-03-30", 23), ("2025-06-10", 24)])
def test_clock_change_days_have_25_and_23_hours(day, hours):
    df = rows((KYIV_CITY, "", "", "2025-01-01 00:00", "2025-01-01 00:10"))
    grid = make_grid(pd.Timestamp(day) - pd.Timedelta(days=1), pd.Timestamp(day) + pd.Timedelta(days=2), 60)
    f = build_features(df, grid)
    assert (f.index.tz_convert("Europe/Kyiv").strftime("%Y-%m-%d") == day).sum() == hours
    # the grid itself stays regular in UTC
    assert set(np.diff(grid)) == {3600}


def _random_alerts(seed=0):
    rng = np.random.default_rng(seed)
    base = pd.Timestamp("2025-05-01", tz="UTC")
    units = [(KYIV_CITY, "", ""), (KO, "", ""), (KO, "Броварський район", ""), (KO, "Бучанський район", "H1"),
             (CN, "Ніжинський район", ""), (CN, "", ""), ("Сумська область", "Сумський район", "")]
    out = []
    for unit in units:
        t = base
        for _ in range(60):
            t += pd.Timedelta(minutes=int(rng.integers(5, 400)))
            end = t + pd.Timedelta(minutes=int(rng.integers(1, 300)))
            out.append((*unit, t, end))
            t = end
    return pd.DataFrame(out, columns=["oblast", "raion", "hromada", "started_at", "finished_at"])


def test_features_do_not_depend_on_the_future():
    """Features up to `cut` must be identical when everything after `cut` is
    unknown: later alerts removed, and alerts still active at `cut` given an
    arbitrary far-away end."""
    df = _random_alerts()
    grid = make_grid("2025-05-01", "2025-05-12", 5)
    full = build_features(df, grid)

    for cut in ("2025-05-03 07:15", "2025-05-06 12:00", "2025-05-09 23:40"):
        cut_ts = pd.Timestamp(cut, tz="UTC")
        past = df[df["started_at"] <= cut_ts].copy()
        still_on = past["finished_at"] > cut_ts
        past.loc[still_on, "finished_at"] = pd.Timestamp("2030-01-01", tz="UTC")
        known = build_features(past, grid)
        pd.testing.assert_frame_equal(full.loc[:cut_ts], known.loc[:cut_ts])
        assert still_on.any()  # the case is actually exercised


def test_features_ignore_alerts_that_already_ended():
    """A snapshot at t shows only active alerts, so removing every alert that
    ended before t must not change the features at t."""
    df = _random_alerts()
    full = build_features(df, make_grid("2025-05-01", "2025-05-12", 5))
    for cut in ("2025-05-03 07:15", "2025-05-06 12:00", "2025-05-09 23:40"):
        cut_ts = pd.Timestamp(cut, tz="UTC")
        active = df[(df["started_at"] <= cut_ts) & (df["finished_at"] > cut_ts)]
        snap = build_features(active, np.array([int(cut_ts.timestamp())]))
        pd.testing.assert_frame_equal(full.loc[[cut_ts]], snap, check_freq=False)
        assert len(active) > 0


def test_targets_do_depend_on_the_future():
    """Sanity check of the test above: the same truncation must change targets."""
    df = _random_alerts()
    grid = make_grid("2025-05-01", "2025-05-12", 5)
    cut_ts = pd.Timestamp("2025-05-06 12:00", tz="UTC")
    past = df[df["started_at"] <= cut_ts]
    full, known = build_targets(df, grid, 30), build_targets(past, grid, 30)
    assert not full.loc[cut_ts - pd.Timedelta(minutes=30):].equals(known.loc[cut_ts - pd.Timedelta(minutes=30):])


def test_loader_drops_exact_duplicates_and_zero_duration(tmp_path):
    p = tmp_path / "a.csv"
    p.write_text(
        "oblast,raion,hromada,level,started_at,finished_at,source\n"
        "A,,,oblast,2025-01-01 10:00:00+00:00,2025-01-01 11:00:00+00:00,official\n"
        "A,,,oblast,2025-01-01 10:00:00+00:00,2025-01-01 11:00:00+00:00,official\n"
        "A,,,oblast,2025-01-01 10:00:00+00:00,2025-01-01 11:30:00+00:00,official\n"
        "A,,,oblast,2025-01-02 10:00:00+00:00,2025-01-02 10:00:00+00:00,official\n",
        encoding="utf-8",
    )
    df = load_alerts(p)
    assert len(df) == 2  # the copy and the zero-length row are gone, the differing row stays
