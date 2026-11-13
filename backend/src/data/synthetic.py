"""Synthetic OHLCV generator for offline testing (A1/A8).

Everything produced here carries the ticker prefix ``SYNTH_`` and
``df.attrs['synthetic'] = True`` so it can never be mistaken for real data.
"""
import numpy as np
import pandas as pd

SYNTH_PREFIX = "SYNTH_"


def generate_synthetic_ohlcv(tickers, dates=None, n_days=1500, seed=42,
                             annual_drift=0.10, annual_vol=0.25, market_corr=0.5):
    """One-factor correlated GBM. Returns {SYNTH_<ticker>: OHLCV DataFrame}."""
    rng = np.random.default_rng(seed)
    if dates is None:
        dates = pd.bdate_range("2015-01-01", periods=n_days)
    dates = pd.DatetimeIndex(dates)
    n = len(dates)
    dt = 1 / 252
    market = rng.standard_normal(n)
    out = {}
    for i, t in enumerate(tickers):
        idio = rng.standard_normal(n)
        z = np.sqrt(market_corr) * market + np.sqrt(1 - market_corr) * idio
        vol = annual_vol * (0.8 + 0.1 * (i % 5))
        r = (annual_drift - 0.5 * vol ** 2) * dt + vol * np.sqrt(dt) * z
        close = 100 * (1 + i * 0.2) * np.exp(np.cumsum(r))
        open_ = np.r_[close[0], close[:-1]] * (1 + 0.002 * rng.standard_normal(n))
        spread = np.abs(0.004 * rng.standard_normal(n)) + 0.001
        high = np.maximum(open_, close) * (1 + spread)
        low = np.minimum(open_, close) * (1 - spread)
        vol_ = rng.integers(5e5, 5e6, n).astype(float)
        df = pd.DataFrame({"Open": open_, "High": high, "Low": low, "Close": close,
                           "Volume": vol_}, index=dates)
        df.index.name = "Date"
        df.attrs.update(synthetic=True, source="synthetic")
        out[SYNTH_PREFIX + t.replace(".NS", "")] = df
    return out
