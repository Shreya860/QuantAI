import numpy as np
import pandas as pd

from src.features.technical_indicators import (build_features, make_dataset, make_target, rsi, drawdown)


def test_no_lookahead_features_unchanged_when_future_removed(synth):
    d = next(iter(synth.values()))
    full = build_features(d)
    cut = d.iloc[:600]
    part = build_features(cut)
    pd.testing.assert_frame_equal(full.iloc[:600], part, check_exact=False, atol=1e-12)


def test_future_shock_does_not_change_past_features(synth):
    d = next(iter(synth.values())).copy()
    base = build_features(d)
    d.iloc[-50:, d.columns.get_loc("Close")] *= 3
    shocked = build_features(d)
    pd.testing.assert_frame_equal(base.iloc[:-50], shocked.iloc[:-50], check_exact=False, atol=1e-12)


def test_target_is_next_day_return(synth):
    d = next(iter(synth.values()))
    y = make_target(d["Close"])
    t = d.index[100]
    assert np.isclose(y[t], d["Close"].iloc[101] / d["Close"].iloc[100] - 1)
    assert np.isnan(y.iloc[-1])


def test_dataset_aligned_no_nans(synth):
    X, y = make_dataset(next(iter(synth.values())))
    assert X.index.equals(y.index) and not X.isna().any().any() and not y.isna().any()


def test_expected_feature_columns(synth):
    cols = set(build_features(next(iter(synth.values()))).columns)
    for c in ["ret_1d", "logret_1d", "sma20_ratio", "sma50_ratio", "ema20_ratio", "rsi14", "macd",
              "vol_20", "mom_20", "bb_pctb", "volume_ratio", "drawdown"]:
        assert c in cols


def test_rsi_bounds_and_extremes():
    up = pd.Series(np.arange(1, 100, dtype=float))
    assert rsi(up).dropna().iloc[-1] == 100
    assert 0 <= rsi(pd.Series(np.random.default_rng(0).normal(100, 1, 200))).dropna().min()


def test_drawdown_nonpositive_and_zero_at_peak():
    dd = drawdown(pd.Series([1, 2, 1, 3, 1.5]))
    assert (dd <= 0).all() and dd.iloc[1] == 0 and np.isclose(dd.iloc[-1], -0.5)
