import numpy as np
import pandas as pd
import pytest

from src.data.preprocessing import clean_ohlcv, detect_splits, build_price_panel
from src.data.synthetic import generate_synthetic_ohlcv, SYNTH_PREFIX
from src.data.downloader import get_market_data, read_kaggle_csv
from config.settings import load_config


def _df(n=60):
    idx = pd.bdate_range("2020-01-01", periods=n)
    c = np.linspace(100, 120, n)
    return pd.DataFrame({"Open": c, "High": c + 1, "Low": c - 1, "Close": c, "Volume": 1000.0}, index=idx)


def test_duplicates_and_order_fixed():
    d = _df(); d = pd.concat([d, d.iloc[[5]]]).sample(frac=1, random_state=0)
    out, rep = clean_ohlcv(d)
    assert out.index.is_monotonic_increasing and not out.index.has_duplicates
    assert rep["duplicate_dates"] == 1


def test_missing_filled_only_up_to_limit():
    d = _df(); d.iloc[10:12, d.columns.get_loc("Close")] = np.nan
    d.iloc[30:38, d.columns.get_loc("Close")] = np.nan
    out, _ = clean_ohlcv(d, max_ffill=3)
    assert out["Close"].notna().all()           # unfillable rows dropped, none left NaN
    assert len(out) < len(d)                    # the 8-day gap beyond the limit was dropped


def test_split_detected_and_adjusted():
    d = _df(); d.loc[d.index[30]:, ["Open", "High", "Low", "Close"]] /= 5
    assert list(detect_splits(d["Close"]).values) == [5.0]
    out, rep = clean_ohlcv(d)
    assert out["Close"].pct_change().abs().max() < 0.05


def test_big_but_real_moves_not_flagged_as_split():
    d = _df(); d.loc[d.index[30]:, "Close"] *= 0.8      # -20% circuit move
    assert len(detect_splits(d["Close"])) == 0


def test_synthetic_is_labeled():
    s = generate_synthetic_ohlcv(["X.NS"], n_days=50)
    (k, v), = s.items()
    assert k.startswith(SYNTH_PREFIX) and v.attrs["synthetic"] is True


def test_fallback_never_silent(tmp_path):
    cfg = load_config()
    cfg["data"].update(kaggle_dir=str(tmp_path), source_priority=["kaggle"], min_assets=3)
    with pytest.warns(UserWarning, match="SYNTHETIC DATA IN USE"):
        md = get_market_data(cfg)
    assert md.any_synthetic and all(t.startswith(SYNTH_PREFIX) for t in md.ohlcv)


def test_synthetic_disallowed_raises(tmp_path):
    cfg = load_config()
    cfg["data"].update(kaggle_dir=str(tmp_path), source_priority=["kaggle"], allow_synthetic=False)
    with pytest.raises(RuntimeError):
        get_market_data(cfg)


def test_kaggle_csv_roundtrip(tmp_path):
    d = _df(); d.index.name = "Date"; d["Symbol"] = "X"
    p = tmp_path / "X.csv"; d.reset_index().to_csv(p, index=False)
    out = read_kaggle_csv(p)
    assert list(out.columns) == ["Open", "High", "Low", "Close", "Volume"] and out.attrs["synthetic"] is False


def test_panel_common_calendar(synth):
    panel = build_price_panel(synth)
    assert not panel.isna().any().any() and panel.shape[1] == len(synth)
