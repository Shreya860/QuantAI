"""Walk-forward evaluation (A5): MAE, RMSE, R2, directional accuracy, feature importance."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .forecasting import build_models, feature_importance, time_split


def forecast_metrics(y_true, y_pred, y_naive=None) -> dict:
    """y_naive: expanding-mean forecasts, used for an out-of-sample R2 (Campbell-Thompson style):
    R2_oos = 1 - SSE(model)/SSE(hist-mean). >0 means the model beats just predicting the mean."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    out = {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "R2": r2_score(y_true, y_pred),
        # sign agreement; exact-zero predictions count as wrong (the zero baseline gets ~0 by design)
        "DirAcc": float(np.mean(np.sign(y_pred) == np.sign(y_true))),
    }
    if y_naive is not None:
        sse_m = np.sum((y_true - y_pred) ** 2); sse_n = np.sum((y_true - np.asarray(y_naive)) ** 2)
        out["R2_oos_vs_mean"] = 1 - sse_m / sse_n
    return out


def walk_forward_predict(X: pd.DataFrame, y: pd.Series, cfg: dict | None = None,
                         min_train: int = 504, step: int = 63, expanding: bool = True,
                         seed: int = 42):
    """Expanding-window walk-forward. At each block start the models are refit on rows strictly
    before the block, then predict the next `step` rows. Returns (preds DataFrame, importance dict).
    Row s of y is the return on s+1, so training on rows < block start only uses labels
    realised no later than the block's first feature date."""
    preds = {k: [] for k in build_models(cfg)}
    idx, imps = [], {}
    for start in range(min_train, len(X), step):
        end = min(start + step, len(X))
        lo = 0 if expanding else start - min_train
        models = build_models(cfg, seed)
        for name, m in models.items():
            m.fit(X.iloc[lo:start], y.iloc[lo:start])
            preds[name].append(pd.Series(m.predict(X.iloc[start:end]), index=X.index[start:end]))
        idx.append(X.index[start:end])
        for name in ("random_forest", "xgboost"):
            imps.setdefault(name, []).append(feature_importance(models[name], X.columns))
    P = pd.DataFrame({k: pd.concat(v) for k, v in preds.items()})
    imp = {k: pd.concat(v, axis=1).mean(axis=1).sort_values(ascending=False) for k, v in imps.items()}
    return P, imp


def evaluate_walk_forward(X, y, cfg=None, min_train=504, step=63, seed=42):
    """Metrics table (rows = models) + importances + OOS predictions."""
    P, imp = walk_forward_predict(X, y, cfg, min_train, step, seed=seed)
    yt = y.loc[P.index]
    naive = P["mean_baseline"]
    table = pd.DataFrame({k: forecast_metrics(yt, P[k], naive) for k in P}).T
    return table, imp, P


def evaluate_split(X, y, cfg=None, ratios=(0.6, 0.2, 0.2), seed=42) -> dict:
    """Static 60/20/20 chronological split: fit on train, report validation and test metrics."""
    from .forecasting import fit_models
    (Xtr, ytr), (Xv, yv), (Xte, yte) = time_split(X, y, ratios)
    models = fit_models(Xtr, ytr, cfg, seed)
    out = {}
    for nm, (Xs, ys) in {"validation": (Xv, yv), "test": (Xte, yte)}.items():
        base = models["mean_baseline"].predict(Xs)
        out[nm] = pd.DataFrame({k: forecast_metrics(ys, m.predict(Xs), base)
                                for k, m in models.items()}).T
    out["importance"] = {k: feature_importance(models[k], X.columns)
                         for k in ("random_forest", "xgboost")}
    return out
