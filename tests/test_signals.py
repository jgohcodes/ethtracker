import numpy as np
import pandas as pd
import pytest

from ethtracker import signals
from ethtracker.config import Config
from tests.conftest import synthetic_candles

CFG = Config()


def bars(closes, highs=None, lows=None, volume=100.0) -> pd.DataFrame:
    idx = pd.date_range("2026-01-01", periods=len(closes), freq="D", tz="UTC")
    close = pd.Series(closes, dtype=float, index=idx)
    pick = lambda v, default: default if v is None else pd.Series(v, dtype=float, index=idx)
    return pd.DataFrame({"open": close, "high": pick(highs, close + 1), "low": pick(lows, close - 1),
                         "close": close, "volume": volume})


def test_anchored_vwap_is_the_volume_weighted_typical_price_from_the_anchor():
    df = bars([10, 20, 30], highs=[12, 22, 33], lows=[8, 18, 27])
    df["volume"] = [1.0, 2.0, 3.0]
    typical = [(12 + 8 + 10) / 3, (22 + 18 + 20) / 3, (33 + 27 + 30) / 3]
    assert signals.anchored_vwap(df, df.index[0]) == pytest.approx((typical[0] + 2 * typical[1] + 3 * typical[2]) / 6)
    assert signals.anchored_vwap(df, df.index[1]) == pytest.approx((2 * typical[1] + 3 * typical[2]) / 5)


def test_anchored_vwap_is_none_without_volume_or_after_the_last_candle():
    df = bars([10, 20, 30], volume=0.0)
    assert signals.anchored_vwap(df, df.index[0]) is None
    assert signals.anchored_vwap(bars([10, 20]), pd.Timestamp("2030-01-01", tz="UTC")) is None


def test_volume_profile_finds_the_crowded_price_and_a_value_area_around_it():
    closes = [100.0] * 24 + [150.0]
    df = bars(closes, highs=[101] * 24 + [151], lows=[99] * 24 + [149])
    df["volume"] = [100.0] * 24 + [10.0]
    prof = signals.volume_profile(df)
    assert 99 <= prof.val <= prof.poc <= prof.vah <= 101.1
    assert prof.poc == pytest.approx(100, abs=1.0)


def test_volume_profile_needs_enough_candles_and_volume():
    assert signals.volume_profile(bars([100.0] * 10)) is None
    assert signals.volume_profile(bars([100.0] * 30, volume=0.0)) is None


def test_volume_profile_handles_zero_range_candles():
    df = bars([100.0] * 30, highs=[100.0] * 30, lows=[100.0] * 29 + [90.0])
    prof = signals.volume_profile(df)
    assert prof is not None and 90 <= prof.poc <= 100


def test_stretched_flag_fires_after_a_jump_above_the_20d_mean():
    ts = signals.technical_series(CFG, bars([100.0] * 40 + [125.0]))
    assert bool(ts["ext_hot"].iloc[-1]) and not bool(ts["ext_hot"].iloc[-2])


def test_volatility_spike_fires_on_an_outsized_range_but_not_on_calm_days():
    closes = [100.0] * 60 + [103.0]
    highs = [101.0] * 60 + [120.0]  # a day ranging far beyond the ~2-point ATR
    lows = [99.0] * 60 + [98.0]
    ts = signals.technical_series(CFG, bars(closes, highs=highs, lows=lows))
    assert bool(ts["vol_spike"].iloc[-1]) and ts["range_ratio"].iloc[-1] >= CFG.range_spike_atr
    assert not ts["vol_spike"].iloc[-2]


def test_rsi_flag_covers_overbought_and_a_rollover_from_overbought():
    rising = list(np.linspace(100, 160, 40))
    ts = signals.technical_series(CFG, bars(rising))
    assert bool(ts["rsi_hot"].iloc[-1])  # RSI pinned near 100
    rolled = rising + [rising[-1] * f for f in (0.97, 0.94, 0.91, 0.88, 0.85, 0.82)]
    ts = signals.technical_series(CFG, bars(rolled))
    last = ts.iloc[-1]
    assert last["rsi"] < CFG.rsi_overbought and last["rsi_peak"] >= CFG.rsi_overbought
    assert bool(last["rsi_hot"])  # momentum rolled over within 10 days of being overbought
    calm = signals.technical_series(CFG, bars(list(100 + np.sin(np.arange(60)))))
    assert not calm["rsi_hot"].iloc[-1]


def test_risk_count_is_the_sum_of_the_three_flags():
    ts = signals.technical_series(CFG, synthetic_candles(300))
    expected = ts[["ext_hot", "rsi_hot", "vol_spike"]].sum(axis=1)
    assert (ts["risk_count"] == expected).all()


def test_live_signals_use_the_same_risk_flags_as_the_series():
    df = bars([100.0] * 60 + [103.0], highs=[101.0] * 60 + [120.0], lows=[99.0] * 60 + [98.0])
    ts = signals.technical_series(CFG, df).iloc[-1]
    sigs = signals.compute(CFG, df)
    flagged = {s.name for s in sigs.signals if s.risk}
    assert ("volatility" in flagged) == bool(ts["vol_spike"])
    assert ("extension" in flagged) == bool(ts["ext_hot"])
    assert ("rsi" in flagged) == bool(ts["rsi_hot"])


def test_compute_reports_levels_and_notes_what_it_cannot_do():
    df = synthetic_candles(500)
    sigs = signals.compute(CFG, df)
    for level in ("sma_20", "sma_50", "sma_100", "sma_200", "avwap_low", "avwap_year", "poc", "vah", "val"):
        assert level in sigs.levels
    names = {s.name for s in sigs.signals}
    assert {"trend", "long_trend", "extension", "rsi", "volatility", "avwap_low", "poc"} <= names
    assert any("no derivatives data" in n for n in sigs.notes)
    short = signals.compute(CFG, synthetic_candles(30))
    assert any("trend signals need 50" in n for n in short.notes)


def test_without_volume_the_volume_based_signals_are_dropped_with_a_note():
    sigs = signals.compute(CFG, synthetic_candles(500).assign(volume=0.0))
    assert not {"avwap_low", "poc", "volume_pressure"} & {s.name for s in sigs.signals}
    assert any("no volume" in n for n in sigs.notes)


def test_correlated_signals_share_a_family_and_vote_once():
    sigs = signals.compute(CFG, synthetic_candles(500))  # a tape that is below every reference level
    below_levels = [s for s in sigs.signals if s.family == "levels"]
    assert len(below_levels) >= 5 and all(s.bias == -1 for s in below_levels)
    votes = sigs.votes()
    assert votes["levels"] == -1  # six bearish signals, one vote
    assert sum(votes.values()) > -len(below_levels)


def test_votes_take_the_sign_of_each_familys_summed_bias_and_unfamilied_signals_stand_alone():
    s = lambda name, bias, family="": signals.Signal(name, bias, False, name, family=family)
    sigs = signals.SignalSet(100.0, (s("a", -1, "x"), s("b", -1, "x"), s("c", 1, "x"), s("d", 1, "y"), s("e", 0), s("f", 1)), {}, ())
    assert sigs.votes() == {"x": -1, "y": 1, "e": 0, "f": 1}


def test_every_named_signal_that_can_vote_belongs_to_a_family():
    sigs = signals.compute(CFG, synthetic_candles(500), derivs_frame(hours=24 * 40, oi=(100, 130), funding=0.0006))
    for s in sigs.signals:
        if s.bias != 0:
            assert s.family == signals.FAMILIES[s.name]


def test_a_chosen_anchor_replaces_the_365d_low():
    df = synthetic_candles(500)
    anchor = df.index[100]
    sigs = signals.compute(CFG, df, anchor=anchor)
    assert sigs.levels["avwap_low"] == pytest.approx(signals.anchored_vwap(df, anchor))
    assert "chosen anchor" in sigs.get("avwap_low").headline


def derivs_frame(hours=48, oi=(100.0, 100.0), funding=0.0001) -> pd.DataFrame:
    idx = pd.date_range("2026-10-01", periods=hours, freq="h", tz="UTC")
    return pd.DataFrame({"open_interest": np.linspace(oi[0], oi[1], hours), "funding_8h": funding}, index=idx)


def run_derivs(frame, price_change=0.03):
    out, notes = [], []
    signals._derivatives(CFG, frame, price_change, out, notes)
    return {s.name: s for s in out}, notes


@pytest.mark.parametrize("funding,bias,risk", [(0.0006, -1, True), (-0.0006, 1, True), (0.0001, 0, False), (0.0005, -1, True)])
def test_funding_thresholds(funding, bias, risk):
    found, _ = run_derivs(derivs_frame(funding=funding))
    assert (found["funding"].bias, found["funding"].risk) == (bias, risk)
    assert found["funding"].unvalidated


def test_open_interest_spike_with_rising_price_is_new_longs():
    found, _ = run_derivs(derivs_frame(oi=(100, 125)), price_change=0.04)
    assert found["oi_change"].bias == 1 and "new longs" in found["oi_change"].headline


def test_open_interest_spike_with_falling_price_is_bearish():
    found, _ = run_derivs(derivs_frame(oi=(100, 125)), price_change=-0.04)
    assert found["oi_change"].bias == -1


def test_falling_open_interest_is_a_flush_or_short_covering_not_a_direction():
    flush, _ = run_derivs(derivs_frame(oi=(125, 100)), price_change=-0.04)
    cover, _ = run_derivs(derivs_frame(oi=(125, 100)), price_change=0.04)
    assert flush["oi_change"].bias == 0 and "flushed" in flush["oi_change"].headline
    assert cover["oi_change"].bias == 0 and "short covering" in cover["oi_change"].headline


def test_open_interest_needs_24_hours_before_it_reports_a_change():
    found, notes = run_derivs(derivs_frame(hours=10))
    assert "oi_change" not in found and any("needs 24h" in n for n in notes)


def test_open_interest_percentile_needs_30_days_and_flags_crowding():
    short, notes = run_derivs(derivs_frame(hours=24 * 10))
    assert "oi_level" not in short and any("percentile needs 30 days" in n for n in notes)
    crowded, _ = run_derivs(derivs_frame(hours=24 * 40, oi=(100, 200)))  # latest OI is the highest ever recorded
    assert crowded["oi_level"].risk and "100th percentile" in crowded["oi_level"].headline
    ordinary, _ = run_derivs(derivs_frame(hours=24 * 40, oi=(200, 100)))
    assert not ordinary["oi_level"].risk


def liquidation_frame(spike: bool) -> pd.DataFrame:
    idx = pd.date_range("2026-09-01", periods=40, freq="D", tz="UTC")
    rng = np.random.default_rng(3)
    long_liq = 1_000_000 + rng.normal(0, 50_000, 40)
    if spike:
        long_liq[-1] = 5_000_000
    return pd.DataFrame({"long_liq_usd": long_liq, "short_liq_usd": 1_000_000 + rng.normal(0, 50_000, 40)}, index=idx)


def test_a_long_liquidation_spike_is_forced_selling_flagged_unvalidated():
    found, _ = run_derivs(liquidation_frame(spike=True))
    assert found["long_liquidations"].bias == 1 and found["long_liquidations"].unvalidated
    assert "forced selling" in found["long_liquidations"].headline
    quiet, _ = run_derivs(liquidation_frame(spike=False))
    assert quiet["long_liquidations"].bias == 0


def test_a_short_liquidation_spike_is_a_risk_flag_not_a_direction():
    frame = liquidation_frame(spike=False)
    frame.iloc[-1, frame.columns.get_loc("short_liq_usd")] = 6_000_000
    found, _ = run_derivs(frame)
    assert found["short_liquidations"].risk and found["short_liquidations"].bias == 0


def test_too_little_liquidation_history_is_noted_not_guessed():
    found, notes = run_derivs(liquidation_frame(spike=True).tail(5))
    assert "long_liquidations" not in found and any("need 16 days" in n for n in notes)


def test_absent_derivatives_data_is_a_note_not_an_error():
    found, notes = run_derivs(None)
    assert not found and any("no derivatives data" in n for n in notes)


def test_forward_outcomes_measure_the_return_and_the_worst_drawdown():
    df = bars([100] * 5, lows=[100, 90, 100, 100, 100])
    out = signals.forward_outcomes(df, 2)
    assert list(out["fwd_return"]) == [0.0, 0.0, 0.0]
    assert list(out["drawdown"]) == pytest.approx([0.10, 0.0, 0.0])


def test_warning_study_buckets_partition_the_warmed_up_days():
    df = synthetic_candles(500)
    buckets = {b.label: b for b in signals.warning_study(CFG, df, 14)}
    per_count = sum(b.n for label, b in buckets.items() if label.endswith(("flag", "flags")) and label != "2+ flags")
    assert per_count == buckets["all days"].n
    assert buckets["all days"].n == 500 - 14 - signals.STUDY_WARMUP
    assert all(0 <= b.drawdown_20 <= b.drawdown_15 <= 1 for b in buckets.values())


def test_count_episodes_counts_runs_not_days():
    flags = pd.Series([False, True, True, True, False, True, False, False, True, True])
    assert signals.count_episodes(flags) == 3
    assert signals.count_episodes(pd.Series([True, True, True])) == 1  # a run that starts on the first day
    assert signals.count_episodes(pd.Series([False, False])) == 0


def test_study_buckets_report_fewer_episodes_than_days_for_clustered_flags():
    buckets = {b.label: b for b in signals.warning_study(CFG, synthetic_candles(500), 14)}
    assert all(1 <= b.episodes <= b.n for b in buckets.values())
    assert any(b.episodes < b.n for label, b in buckets.items() if label != "all days")  # flagged days cluster


def test_flagged_days_are_a_small_share_of_the_sample():
    """Why n is printed beside every figure: the flagged buckets are small, so their rates are noisy."""
    df = synthetic_candles(500)
    buckets = {b.label: b for b in signals.warning_study(CFG, df, 14)}
    assert buckets["2+ flags"].n < buckets["all days"].n / 10
