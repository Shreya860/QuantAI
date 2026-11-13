"""Technical features (A3). Every feature at row t uses ONLY data up to and including day t
(rolling/ewm windows look backwards). The target is next-day return, built separately and
aligned so that row t's label is the return realised on day t+1.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    """Wilder's RSI."""
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.where(dn != 0, 100.0).where(up.notna() & dn.notna())


def macd(close: pd.Series, fast=12, slow=26, signal=9):
    line = close.ewm(span=fast, adjust=False).mean() - close.ewm(span=slow, adjust=False).mean()
    sig = line.ewm(span=signal, adjust=False).mean()
    return line, sig, line - sig


def bollinger(close: pd.Series, n=20, k=2.0):
    mid = close.rolling(n).mean()
    sd = close.rolling(n).std()
    upper, lower = mid + k * sd, mid - k * sd
    pct_b = (close - lower) / (upper - lower)
    width = (upper - lower) / mid
    return pct_b, width


def drawdown(close: pd.Series) -> pd.Series:
    """Current drawdown from the running peak (<= 0)."""
    return close / close.cummax() - 1


def build_features(ohlcv: pd.DataFrame) -> pd.DataFrame:
    """Feature matrix for one asset (index = date). Scale-free features only, so one model can
    be compared across assets of very different price levels."""
    c, v = ohlcv["Close"], ohlcv["Volume"]
    f = pd.DataFrame(index=ohlcv.index)
    f["ret_1d"] = c.pct_change()
    f["logret_1d"] = np.log(c).diff()
    f["ret_lag2"] = f["ret_1d"].shift(1)
    f["ret_lag3"] = f["ret_1d"].shift(2)
    f["mom_5"] = c.pct_change(5)
    f["mom_10"] = c.pct_change(10)
    f["mom_20"] = c.pct_change(20)
    sma20, sma50 = c.rolling(20).mean(), c.rolling(50).mean()
    ema20 = c.ewm(span=20, adjust=False, min_periods=20).mean()
    f["sma20_ratio"] = c / sma20 - 1
    f["sma50_ratio"] = c / sma50 - 1
    f["sma20_50"] = sma20 / sma50 - 1
    f["ema20_ratio"] = c / ema20 - 1
    f["rsi14"] = rsi(c, 14)
    m, s, h = macd(c)
    f["macd"], f["macd_signal"], f["macd_hist"] = m / c, s / c, h / c   # price-normalised
    f["vol_10"] = f["ret_1d"].rolling(10).std()
    f["vol_20"] = f["ret_1d"].rolling(20).std()
    f["bb_pctb"], f["bb_width"] = bollinger(c)
    f["volume_ratio"] = v / v.rolling(20).mean().replace(0, np.nan)
    f["drawdown"] = drawdown(c)
    return f.replace([np.inf, -np.inf], np.nan)


def make_target(close: pd.Series) -> pd.Series:
    """y_t = return realised on day t+1 (known only after t+1 closes). Last row is NaN."""
    return close.pct_change().shift(-1).rename("target")


def make_dataset(ohlcv: pd.DataFrame, dropna_target: bool = True):
    """(X, y) aligned on date. Warm-up rows (NaN features) are dropped; the final row is kept in
    X only when dropna_target=False (that's the row you predict *tomorrow* from)."""
    X = build_features(ohlcv).dropna()
    y = make_target(ohlcv["Close"]).reindex(X.index)
    if dropna_target:
        keep = y.notna()
        return X[keep], y[keep]
    return X, y
