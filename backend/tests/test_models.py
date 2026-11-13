import numpy as np
import pandas as pd

from src.features.technical_indicators import make_dataset
from src.models.forecasting import build_models, time_split, fit_models, feature_importance
from src.models.evaluation import (forecast_metrics, walk_forward_predict, evaluate_split)


def _xy(synth):
    return make_dataset(next(iter(synth.values())))


def test_split_chronological_sizes_no_overlap(synth):
    X, y = _xy(synth)
    (a, _), (b, _), (c, _) = time_split(X, y)
    assert a.index.max() < b.index.min() and b.index.max() < c.index.min()
    assert len(a) + len(b) + len(c) == len(X)
    assert abs(len(a) / len(X) - 0.6) < 0.01


def test_all_models_output_shape_and_finite(synth):
    X, y = _xy(synth)
    models = fit_models(X.iloc[:500], y.iloc[:500])
    assert set(models) == {"mean_baseline", "zero_baseline", "random_forest", "xgboost"}
    for m in models.values():
        p = m.predict(X.iloc[500:600])
        assert p.shape == (100,) and np.isfinite(p).all()


def test_baselines_values(synth):
    X, y = _xy(synth)
    m = fit_models(X.iloc[:300], y.iloc[:300])
    assert np.allclose(m["zero_baseline"].predict(X.iloc[:5]), 0)
    assert np.allclose(m["mean_baseline"].predict(X.iloc[:5]), y.iloc[:300].mean())


def test_feature_importance(synth):
    X, y = _xy(synth)
    m = fit_models(X.iloc[:500], y.iloc[:500])
    fi = feature_importance(m["random_forest"], X.columns)
    assert np.isclose(fi.sum(), 1) and len(fi) == X.shape[1]
    assert feature_importance(m["zero_baseline"], X.columns) is None


def test_metrics_known_values():
    m = forecast_metrics([1, -1, 1, -1], [1, 1, 1, -1])
    assert np.isclose(m["MAE"], 0.5) and np.isclose(m["DirAcc"], 0.75)
    assert np.isclose(m["RMSE"], np.sqrt(1))


def test_walk_forward_is_strictly_out_of_sample(synth):
    X, y = _xy(synth)
    P, imp = walk_forward_predict(X, y, min_train=300, step=100)
    assert P.index.min() == X.index[300] and P.index.is_monotonic_increasing
    assert not P.index.has_duplicates and np.isfinite(P.values).all()


def test_walk_forward_ignores_future_labels(synth):
    """Corrupting labels after a block must not change that block's predictions."""
    X, y = _xy(synth)
    P1, _ = walk_forward_predict(X.iloc[:500], y.iloc[:500], min_train=300, step=100)
    y2 = y.iloc[:500].copy(); y2.iloc[400:] = 99.0
    P2, _ = walk_forward_predict(X.iloc[:500], y2, min_train=300, step=100)
    first_blocks = P1.index[:100]
    pd.testing.assert_frame_equal(P1.loc[first_blocks], P2.loc[first_blocks])


def test_shuffle_free_split_eval_runs(synth):
    X, y = _xy(synth)
    out = evaluate_split(X, y)
    assert {"validation", "test", "importance"} <= set(out)
