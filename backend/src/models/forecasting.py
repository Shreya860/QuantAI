"""Forecasting models (A4): 2 baselines + Random Forest + XGBoost, all sklearn-compatible."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from xgboost import XGBRegressor


class MeanBaseline(BaseEstimator, RegressorMixin):
    """Predict the historical average return of the training window."""
    def fit(self, X, y):
        self.mean_ = float(np.mean(y)); return self
    def predict(self, X):
        return np.full(len(X), self.mean_)


class ZeroBaseline(BaseEstimator, RegressorMixin):
    """Predict 0 (random-walk, no-drift)."""
    def fit(self, X, y):
        return self
    def predict(self, X):
        return np.zeros(len(X))


def build_models(cfg: dict | None = None, seed: int = 42) -> dict:
    """Fresh, unfitted estimators keyed by name."""
    m = (cfg or {}).get("model", {})
    rf = {"n_estimators": 200, "max_depth": 4, "min_samples_leaf": 20, "n_jobs": -1, **m.get("rf", {})}
    xgb = {"n_estimators": 200, "max_depth": 3, "learning_rate": 0.03, "subsample": 0.8,
           "colsample_bytree": 0.8, "min_child_weight": 20, **m.get("xgb", {})}
    return {
        "mean_baseline": MeanBaseline(),
        "zero_baseline": ZeroBaseline(),
        "random_forest": Pipeline([("imp", SimpleImputer(strategy="median")),
                                   ("model", RandomForestRegressor(random_state=seed, **rf))]),
        "xgboost": Pipeline([("imp", SimpleImputer(strategy="median")),
                             ("model", XGBRegressor(random_state=seed, n_jobs=1, verbosity=0, **xgb))]),
    }


def time_split(X: pd.DataFrame, y: pd.Series, ratios=(0.6, 0.2, 0.2)):
    """Chronological train/val/test split. Never shuffles."""
    assert abs(sum(ratios) - 1) < 1e-9 and len(X) == len(y)
    n = len(X)
    i1, i2 = int(n * ratios[0]), int(n * (ratios[0] + ratios[1]))
    return (X.iloc[:i1], y.iloc[:i1]), (X.iloc[i1:i2], y.iloc[i1:i2]), (X.iloc[i2:], y.iloc[i2:])


def fit_models(X_train, y_train, cfg=None, seed=42) -> dict:
    models = build_models(cfg, seed)
    for mdl in models.values():
        mdl.fit(X_train, y_train)
    return models


def feature_importance(model, columns) -> pd.Series | None:
    est = model.named_steps["model"] if isinstance(model, Pipeline) else model
    if hasattr(est, "feature_importances_"):
        return pd.Series(est.feature_importances_, index=columns).sort_values(ascending=False)
    return None
