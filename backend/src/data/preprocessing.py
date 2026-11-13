"""Cleaning + calendar alignment (A2)."""
from __future__ import annotations

import numpy as np
import pandas as pd

COMMON_SPLITS = (2, 3, 4, 5, 10)


def detect_splits(close: pd.Series, min_ratio=1.8, tol=0.12, max_residual=0.25) -> pd.Series:
    """Find unadjusted stock splits / bonuses: day-over-day close ratio near an integer k>=2
    whose residual move (after dividing by k) is a plausible daily move. NSE circuit limits are
    <=20%, so real moves never give a ratio above ~1.25; ratios >= 1.8 are corporate actions.
    Returns Series of factors (k) indexed by the split date (reverse splits give 1/k)."""
    ratio = close.shift(1) / close
    found = {}
    for dt, r in ratio.dropna().items():
        for inv, rr in ((False, r), (True, 1 / r)):
            if rr >= min_ratio:
                k = min(COMMON_SPLITS, key=lambda s: abs(s - rr))
                if abs(rr - k) / k < tol and abs(rr / k - 1) < max_residual:
                    found[dt] = (1 / k) if inv else k
    return pd.Series(found, dtype=float)


def adjust_for_splits(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Back-adjust OHLC (divide) and Volume (multiply) before each detected split."""
    splits = detect_splits(df["Close"])
    out = df.copy()
    for dt, k in splits.items():
        before = out.index < dt
        out.loc[before, ["Open", "High", "Low", "Close"]] /= k
        out.loc[before, "Volume"] *= k
    return out, splits


def clean_ohlcv(df: pd.DataFrame, max_ffill: int = 3) -> tuple[pd.DataFrame, dict]:
    """Sort, de-duplicate dates, drop impossible rows, fill short gaps, adjust splits."""
    rep = {"rows_in": len(df)}
    out = df[["Open", "High", "Low", "Close", "Volume"]].apply(pd.to_numeric, errors="coerce")
    out.index = pd.to_datetime(out.index)
    out = out[~out.index.isna()].sort_index()
    rep["duplicate_dates"] = int(out.index.duplicated().sum())
    out = out[~out.index.duplicated(keep="last")]
    bad = (out[["Open", "High", "Low", "Close"]] <= 0).any(axis=1)
    rep["nonpositive_price_rows"] = int(bad.sum())
    out.loc[bad, ["Open", "High", "Low", "Close"]] = np.nan
    rep["missing_before"] = int(out.isna().sum().sum())
    out[["Open", "High", "Low", "Close"]] = out[["Open", "High", "Low", "Close"]].ffill(limit=max_ffill)
    out["Volume"] = out["Volume"].fillna(0)
    out = out.dropna(subset=["Close"])
    out["High"] = out[["High", "Open", "Close"]].max(axis=1)   # repair inconsistent bars
    out["Low"] = out[["Low", "Open", "Close"]].min(axis=1)
    out, splits = adjust_for_splits(out)
    rep["splits_adjusted"] = {str(k.date()): float(v) for k, v in splits.items()}
    rep["rows_out"] = len(out)
    return out, rep


def build_price_panel(ohlcv: dict, max_ffill: int = 3) -> pd.DataFrame:
    """Close prices, dates as rows / tickers as columns, on the common calendar
    (intersection of each asset's date range, union of trading days, short gaps filled)."""
    close = pd.DataFrame({t: d["Close"] for t, d in ohlcv.items()})
    start = max(d.index[0] for d in ohlcv.values())
    end = min(d.index[-1] for d in ohlcv.values())
    close = close.loc[start:end].sort_index().ffill(limit=max_ffill)
    return close.dropna(how="any")


def compute_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns (first row dropped)."""
    return prices.pct_change().dropna(how="all")


def align_ohlcv(ohlcv: dict, index: pd.DatetimeIndex, max_ffill: int = 3) -> dict:
    """Reindex every asset's OHLCV onto the shared calendar (short gaps forward-filled,
    volume 0 on filled days) so features and returns are computed on identical dates."""
    out = {}
    for t, d in ohlcv.items():
        a = d.reindex(index)
        a[["Open", "High", "Low", "Close"]] = a[["Open", "High", "Low", "Close"]].ffill(limit=max_ffill)
        a["Volume"] = a["Volume"].fillna(0)
        a.attrs.update(d.attrs)
        out[t] = a.dropna(subset=["Close"])
    return out
