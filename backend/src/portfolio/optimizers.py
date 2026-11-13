"""Portfolio construction strategies used by the backtester.
(Person C owns the deeper theory/extensions; these are the minimal, tested versions the
backtester needs.) Weights are always a pandas Series summing to 1, long-only.

Strategy interface:   strategy.target_weights(date, ret_hist) -> pd.Series
where ret_hist = daily simple returns with index <= date (never anything later).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf

from src.features.technical_indicators import build_features
from src.models.forecasting import build_models


def shrunk_covariance(returns: pd.DataFrame, trading_days=252) -> pd.DataFrame:
    """Ledoit-Wolf shrinkage covariance (annualised): far more stable than the raw sample matrix."""
    lw = LedoitWolf().fit(returns.dropna().values)
    return pd.DataFrame(lw.covariance_ * trading_days, index=returns.columns, columns=returns.columns)


def _solve(obj, n, max_weight):
    cap = max(max_weight, 1.0 / n)
    x0 = np.full(n, 1.0 / n)
    res = minimize(obj, x0, method="SLSQP", bounds=[(0, cap)] * n,
                   constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1}],
                   options={"maxiter": 500, "ftol": 1e-10})
    w = np.clip(res.x if res.success else x0, 0, None)
    return w / w.sum()


def min_variance_weights(cov: pd.DataFrame, max_weight=1.0) -> pd.Series:
    S = cov.values
    return pd.Series(_solve(lambda w: w @ S @ w, len(cov), max_weight), index=cov.columns)


def max_sharpe_weights(mu: pd.Series, cov: pd.DataFrame, rf=0.0, max_weight=1.0) -> pd.Series:
    """Long-only max-Sharpe. If no asset beats rf the Sharpe objective is meaningless, so fall
    back to minimum variance."""
    mu = mu.reindex(cov.columns)
    if (mu - rf).max() <= 0:
        return min_variance_weights(cov, max_weight)
    S, m = cov.values, mu.values
    w = _solve(lambda w: -(w @ m - rf) / np.sqrt(w @ S @ w + 1e-12), len(mu), max_weight)
    return pd.Series(w, index=mu.index)


class EqualWeight:
    name = "Equal-Weight"
    def target_weights(self, date, ret_hist):
        n = ret_hist.shape[1]
        return pd.Series(1.0 / n, index=ret_hist.columns)


class MeanVariance:
    """Classical Markowitz max-Sharpe with trailing-window historical mean + Ledoit-Wolf covariance."""
    name = "Mean-Variance"
    def __init__(self, rf=0.07, lookback=504, max_weight=0.5, trading_days=252):
        self.rf, self.lookback, self.max_weight, self.td = rf, lookback, max_weight, trading_days
    def _inputs(self, ret_hist):
        window = ret_hist.iloc[-self.lookback:]
        return window.mean() * self.td, shrunk_covariance(window, self.td)
    def target_weights(self, date, ret_hist):
        mu, cov = self._inputs(ret_hist)
        return max_sharpe_weights(mu, cov, self.rf, self.max_weight)


class AIAssisted(MeanVariance):
    """Same optimiser and covariance as Mean-Variance; ONLY the expected-return input changes:
        mu = blend * mu_ML + (1 - blend) * mu_hist
    mu_ML = annualised next-day forecast of a per-asset RF/XGBoost model trained on rows whose
    labels were fully realised by `date`. Isolating the mu input makes the comparison a clean test
    of 'does ML-estimated return help?'."""
    name = "AI-Assisted"
    def __init__(self, ohlcv: dict, cfg: dict | None = None, model="random_forest", blend=0.5,
                 retrain_every=3, min_train=504, clip=0.5, **kw):
        super().__init__(**kw)
        self.feats = {t: build_features(d).dropna() for t, d in ohlcv.items()}
        self.cfg, self.model_name, self.blend = cfg, model, blend
        self.retrain_every, self.min_train, self.clip = retrain_every, min_train, clip
        self._models, self._calls = {}, 0
        self.signals = {}          # date -> Series of annualised ML mu (for audit / dashboard)

    def ml_mu(self, date, ret_hist) -> pd.Series:
        refit = (self._calls % self.retrain_every == 0) or not self._models
        mu = {}
        for t, X in self.feats.items():
            Xh = X.loc[:date]
            if date not in Xh.index or len(Xh) < self.min_train + 1:
                mu[t] = np.nan; continue
            Xtr = Xh.iloc[:-1]                                   # rows strictly before `date`
            ytr = ret_hist[t].shift(-1).reindex(Xtr.index)       # label_s = return on s+1 <= date
            ok = ytr.notna()
            if refit or t not in self._models:
                m = build_models(self.cfg)[self.model_name]
                m.fit(Xtr[ok], ytr[ok]); self._models[t] = m
            mu[t] = float(self._models[t].predict(Xh.iloc[[-1]])[0]) * self.td
        self._calls += 1
        return pd.Series(mu).clip(-self.clip, self.clip)

    def target_weights(self, date, ret_hist):
        mu_h, cov = self._inputs(ret_hist)
        mu_ml = self.ml_mu(date, ret_hist).reindex(mu_h.index)
        mu = (self.blend * mu_ml + (1 - self.blend) * mu_h).fillna(mu_h)
        self.signals[date] = mu_ml
        return max_sharpe_weights(mu, cov, self.rf, self.max_weight)
