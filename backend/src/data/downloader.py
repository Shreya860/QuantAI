"""Data acquisition (A1).

Priority (config data.source_priority): Kaggle CSV folder -> yfinance (cached, with retries).
Synthetic data is only used as a *labeled* top-up/fallback, and a loud warning is always issued.
Never silent: every asset gets an entry in ``MarketData.sources`` and fake series are prefixed ``SYNTH_``.
"""
from __future__ import annotations

import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from .preprocessing import clean_ohlcv
from .synthetic import generate_synthetic_ohlcv

log = logging.getLogger(__name__)
OHLCV = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class MarketData:
    ohlcv: dict                                   # ticker -> cleaned OHLCV DataFrame
    sources: dict                                 # ticker -> 'kaggle' | 'yfinance' | 'synthetic'
    reports: dict = field(default_factory=dict)   # ticker -> cleaning report
    @property
    def is_synthetic(self):
        return {t: s == "synthetic" for t, s in self.sources.items()}
    @property
    def any_synthetic(self):
        return any(self.is_synthetic.values())
    def summary(self) -> pd.DataFrame:
        rows = [{"ticker": t, "source": self.sources[t], "start": d.index[0].date(),
                 "end": d.index[-1].date(), "rows": len(d)} for t, d in self.ohlcv.items()]
        return pd.DataFrame(rows).set_index("ticker")


# ---------------------------------------------------------------- Kaggle
def read_kaggle_csv(path) -> pd.DataFrame:
    """Read one Kaggle NIFTY-50 file (columns: Date, Symbol, Open, High, Low, Close, Volume, ...)."""
    df = pd.read_csv(path, parse_dates=["Date"]).set_index("Date")
    missing = [c for c in OHLCV if c not in df.columns]
    if missing:
        raise ValueError(f"{path}: missing columns {missing}")
    df = df[OHLCV].copy()
    df.attrs.update(synthetic=False, source="kaggle")
    return df


def load_kaggle_dir(kaggle_dir, symbols=None) -> dict:
    kaggle_dir = Path(kaggle_dir)
    out = {}
    for p in sorted(kaggle_dir.glob("*.csv")):
        sym = p.stem.upper()
        if symbols and sym not in {s.upper() for s in symbols}:
            continue
        try:
            out[sym] = read_kaggle_csv(p)
        except Exception as e:  # a bad file must not hide the others, but must be reported
            warnings.warn(f"Skipping {p.name}: {e}")
    return out


# --------------------------------------------------------------- yfinance
def download_yfinance(tickers, years=5, cache_dir="data/raw", retries=3, backoff=2.0,
                      start=None, end=None) -> dict:
    """Download daily OHLCV, caching one CSV per ticker. Returns only tickers that succeeded."""
    cache_dir = Path(cache_dir); cache_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for t in tickers:
        cache = cache_dir / f"{t}.csv"
        if cache.exists():
            df = pd.read_csv(cache, index_col=0, parse_dates=True)
            df.attrs.update(synthetic=False, source="yfinance(cache)")
            out[t] = df
            continue
        for attempt in range(1, retries + 1):
            try:
                import yfinance as yf  # lazy: offline/test runs don't need it
                end_ = pd.Timestamp(end) if end else pd.Timestamp.today().normalize()
                start_ = pd.Timestamp(start) if start else end_ - pd.DateOffset(years=years)
                df = yf.download(t, start=start_, end=end_, auto_adjust=True,
                                 progress=False, multi_level_index=False)
                if df is None or df.empty:
                    raise RuntimeError("empty response")
                df = df[OHLCV]; df.index.name = "Date"
                df.to_csv(cache)
                df.attrs.update(synthetic=False, source="yfinance")
                out[t] = df
                break
            except ImportError:
                log.warning("yfinance not installed; skipping download (pip install yfinance)")
                return out
            except Exception as e:
                log.warning("yfinance %s attempt %d/%d failed: %s", t, attempt, retries, e)
                time.sleep(backoff ** attempt if attempt < retries else 0)
    return out


# ------------------------------------------------------------- orchestrator
def get_market_data(cfg: dict) -> MarketData:
    d = cfg["data"]
    raw, sources = {}, {}
    for src in d["source_priority"]:
        if src == "kaggle" and Path(d["kaggle_dir"]).exists():
            for k, v in load_kaggle_dir(d["kaggle_dir"], d.get("kaggle_symbols")).items():
                raw.setdefault(k, v); sources.setdefault(k, "kaggle")
        elif src == "yfinance" and len(raw) < d["min_assets"]:
            for k, v in download_yfinance(d["tickers"], d["years"], d["cache_dir"],
                                          start=d.get("start"), end=d.get("end")).items():
                raw.setdefault(k, v); sources.setdefault(k, "yfinance")

    # Labeled synthetic top-up so portfolio strategies have enough assets.
    n_real = len(raw)
    if n_real < d["min_assets"]:
        if not d["allow_synthetic"]:
            raise RuntimeError(f"Only {n_real} real assets found, need {d['min_assets']}; "
                               "add more CSVs to data/kaggle or enable data.allow_synthetic.")
        need = d["min_assets"] - n_real
        ref = next(iter(raw.values())).index.unique().sort_values() if raw else None
        fake = generate_synthetic_ohlcv([f"PEER{i + 1}" for i in range(need)], dates=ref,
                                        seed=d["seed"])
        for k, v in fake.items():
            raw[k] = v; sources[k] = "synthetic"
        warnings.warn(
            f"\n{'!' * 70}\nSYNTHETIC DATA IN USE: only {n_real} real asset(s) found; added "
            f"{need} FAKE series ({', '.join(fake)}).\nPortfolio results are NOT real market "
            f"evidence. Add more Kaggle CSVs to {d['kaggle_dir']}.\n{'!' * 70}", stacklevel=2)

    clean, reports = {}, {}
    for k, v in raw.items():
        c, rep = clean_ohlcv(v, max_ffill=d["max_ffill_days"])
        c.attrs.update(v.attrs)
        clean[k], reports[k] = c, rep
    md = MarketData(clean, sources, reports)
    for t, r in reports.items():
        log.info("cleaned %s: %s", t, r)
    return md
