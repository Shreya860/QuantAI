import numpy as np
import pandas as pd
import pytest

from src.backtesting.engine import run_backtest, monthly_rebalance_dates
from src.portfolio.optimizers import (EqualWeight, MeanVariance, AIAssisted, max_sharpe_weights,
                                      shrunk_covariance)

MH = 300


def _strats(synth):
    return [EqualWeight(), MeanVariance(max_weight=0.5, lookback=250),
            AIAssisted(synth, None, "random_forest", 0.5, 6, MH, lookback=250, max_weight=0.5)]


@pytest.fixture(scope="module")
def result(synth, returns):
    return run_backtest(returns, _strats(synth), cost_rate=0.001, min_history=MH)


def test_dates_ordered_and_identical_for_all_strategies(result):
    idx = result.net_returns.index
    assert idx.is_monotonic_increasing and not idx.has_duplicates
    assert result.equity.index.equals(idx) and result.equity.notna().all().all()


def test_oos_starts_after_warmup_and_rebalances_month_end(returns, result):
    assert returns.index.get_loc(result.rebalance_dates[0]) >= MH - 1
    assert result.net_returns.index[0] > result.rebalance_dates[0]
    for d in result.rebalance_dates:
        nxt = returns.index[returns.index.get_loc(d) + 1]
        assert nxt.month != d.month               # last trading day of its month


def test_weights_valid(result):
    for w in result.weights.values():
        assert np.allclose(w.sum(axis=1), 1) and (w >= -1e-9).all().all()
    assert (result.weights["Mean-Variance"] <= 0.5 + 1e-6).all().all()


def test_zero_cost_means_zero_deduction(synth, returns):
    r = run_backtest(returns, _strats(synth), cost_rate=0.0, min_history=MH)
    assert (r.costs.values == 0).all()
    pd.testing.assert_frame_equal(r.net_returns, r.gross_returns)


def test_positive_cost_lowers_returns_and_scales_with_turnover(synth, returns, result):
    free = run_backtest(returns, _strats(synth), cost_rate=0.0, min_history=MH)
    assert (result.metrics["Final Equity"] < free.metrics["Final Equity"]).all()
    # total cost ~ rate * total turnover
    tot_turn = result.turnover.sum()
    assert np.allclose(result.costs.sum(), 0.001 * tot_turn)


def test_first_rebalance_costs_full_investment(result):
    assert np.allclose(result.turnover.iloc[0], 1.0)     # from cash to fully invested


def test_equal_weight_turnover_small_after_start(result):
    # equal-weight only trades to undo drift; optimised portfolios churn more
    assert result.turnover["Equal-Weight"].iloc[1:].mean() < result.turnover["Mean-Variance"].iloc[1:].mean()


def test_no_leakage_future_changes_dont_alter_past(synth, returns):
    """Perturb the last 120 days: everything up to the perturbation point must be identical."""
    cut = len(returns) - 120
    r2 = returns.copy(); r2.iloc[cut:] = r2.iloc[cut:] * -3 + 0.05
    synth2 = {k: v.copy() for k, v in synth.items()}
    # rebuild perturbed closes consistently from perturbed returns
    for k in synth2:
        c = synth[k]["Close"].copy()
        pert = (1 + r2[k]).cumprod() * synth[k]["Close"].iloc[0]
        c.iloc[1:] = pert.values
        synth2[k]["Close"] = c
        synth2[k]["High"] = np.maximum(synth2[k]["High"], c); synth2[k]["Low"] = np.minimum(synth2[k]["Low"], c)
    ra = run_backtest(returns, _strats(synth), 0.001, MH)
    rb = run_backtest(r2, _strats(synth2), 0.001, MH)
    cutoff = returns.index[cut - 1]
    # weights chosen on/before the cutoff must match exactly
    for n in ra.weights:
        wa, wb = ra.weights[n].loc[:cutoff], rb.weights[n].loc[:cutoff]
        pd.testing.assert_frame_equal(wa, wb, check_exact=False, atol=1e-8)
    pd.testing.assert_frame_equal(ra.net_returns.loc[:cutoff], rb.net_returns.loc[:cutoff],
                                  check_exact=False, atol=1e-10)


def test_ai_signal_uses_only_past(synth, returns):
    ai = AIAssisted(synth, None, "random_forest", 1.0, 1, MH, lookback=250)
    d = returns.index[MH + 50]
    ai.target_weights(d, returns.loc[:d])
    assert max(ai.signals) == d


def test_optimizer_properties():
    rng = np.random.default_rng(0)
    R = pd.DataFrame(rng.normal(0.0005, 0.01, (500, 4)), columns=list("ABCD"))
    cov = shrunk_covariance(R)
    mu = pd.Series([0.2, 0.1, 0.05, -0.1], index=cov.columns)
    w = max_sharpe_weights(mu, cov, 0.02, 0.4)
    assert np.isclose(w.sum(), 1) and (w >= -1e-9).all() and (w <= 0.4 + 1e-6).all()
    assert w["A"] >= w["D"]
    w_neg = max_sharpe_weights(mu - 1, cov, 0.02, 1.0)     # nothing beats rf -> min variance, still valid
    assert np.isclose(w_neg.sum(), 1)


def test_monthly_rebalance_helper():
    idx = pd.bdate_range("2020-01-01", "2021-12-31")
    d = monthly_rebalance_dates(idx, 100)
    assert d.is_monotonic_increasing and (d.isin(idx)).all()
