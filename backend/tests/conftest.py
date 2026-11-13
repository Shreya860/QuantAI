import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data.synthetic import generate_synthetic_ohlcv


@pytest.fixture(scope="session")
def synth():
    return generate_synthetic_ohlcv(["A", "B", "C", "D"], n_days=900, seed=7)


@pytest.fixture(scope="session")
def returns(synth):
    import pandas as pd
    return pd.DataFrame({k: v["Close"] for k, v in synth.items()}).pct_change().dropna()
