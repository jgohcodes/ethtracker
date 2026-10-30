import pandas as pd

from ethtracker import levelstudy


def frame(rows):
    idx = pd.date_range("2026-01-01", periods=len(rows), freq="D", tz="UTC")
    return pd.DataFrame(rows, columns=["open", "high", "low", "close"], index=idx).assign(volume=1.0)


def test_a_wick_into_support_that_closes_back_above_and_rallies_counts_as_rejected_held_and_bounced():
    rows = [(110, 112, 108, 110), (110, 111, 99.8, 103)] + [(103, 106 + i, 102, 104 + i) for i in range(6)]
    r = levelstudy._judge(frame(rows), 1, 100.0, "support")
    assert r == (True, True, True)


def test_a_close_through_support_breaks_it():
    rows = [(110, 112, 108, 110), (110, 111, 99, 99.5), (99, 99, 96, 97)] + [(97, 98, 95, 96)] * 5
    rejected, held, bounced = levelstudy._judge(frame(rows), 1, 100.0, "support")
    assert not rejected and not held and not bounced


def test_no_touch_without_approaching_from_the_near_side():
    rows = [(100.5, 101, 100, 100.5)] * 8  # already sitting on the level
    assert levelstudy._judge(frame(rows), 1, 100.0, "support") is None


def test_swings_are_only_known_once_confirmed():
    path = [100, 120, 140, 130, 115, 110, 125, 135, 129, 128]
    df = frame([(p, p, p, p) for p in path])
    (hi_day, hi_conf, hi), (lo_day, lo_conf, lo) = levelstudy.confirmed_swings(df)
    assert (hi, lo) == (140.0, 110.0) and hi_conf > hi_day and lo_conf > lo_day
    assert hi_conf == df.index[4]  # 115 is the first low 15% under 140
