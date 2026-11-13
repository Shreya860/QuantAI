"""Walk-forward backtest engine (A6).

Timeline convention (no look-ahead):
  * Rebalance dates = last trading day of each month, once `min_history` days exist.
  * Weights for a rebalance at close of day t are computed from returns up to and including t.
  * Trades happen at close of t; the new weights earn the returns from t+1 onward.
  * Transaction cost = cost_rate * turnover, turnover = sum|w_target - w_drifted| (traded notional
    as a fraction of portfolio value), charged on the first holding day t+1. Zero cost => net == gross.
  * All strategies share the same dates, the same warm-up and the same first rebalance.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .metrics import performance_metrics


@dataclass
class BacktestResult:
    net_returns: pd.DataFrame
    gross_returns: pd.DataFrame
    equity: pd.DataFrame            # growth of 1 (net of costs)
    weights: dict                   # strategy -> DataFrame (rebalance date x asset), target weights
    turnover: pd.DataFrame          # rebalance date x strategy
    costs: pd.DataFrame             # daily cost deducted, per strategy
    metrics: pd.DataFrame
    rebalance_dates: pd.DatetimeIndex


def monthly_rebalance_dates(index: pd.DatetimeIndex, min_history: int) -> pd.DatetimeIndex:
    s = pd.Series(index, index=index)
    last = s.groupby([index.year, index.month]).max()
    dates = pd.DatetimeIndex(last.values)
    pos = index.get_indexer(dates)
    keep = (pos >= min_history - 1) & (pos < len(index) - 1)    # need >= 1 day left to hold
    return dates[keep]


def run_backtest(returns: pd.DataFrame, strategies: list, cost_rate=0.001, min_history=504,
                 rf=0.07, td=252) -> BacktestResult:
    returns = returns.sort_index()
    assert returns.index.is_monotonic_increasing and not returns.index.has_duplicates
    reb = monthly_rebalance_dates(returns.index, min_history)
    if len(reb) == 0:
        raise ValueError("Not enough history for even one rebalance.")
    start_pos = returns.index.get_loc(reb[0]) + 1
    oos = returns.index[start_pos:]
    reb_set = set(reb)
    names = [s.name for s in strategies]
    net = pd.DataFrame(0.0, index=oos, columns=names)
    gross, costs = net.copy(), net.copy()
    turn = pd.DataFrame(np.nan, index=reb, columns=names)
    wlog = {n: [] for n in names}

    for strat in strategies:
        n_assets = returns.shape[1]
        w_hold = pd.Series(0.0, index=returns.columns)       # start in cash
        pending_cost = 0.0
        # decide at t for every date in [reb[0], end); earn from t+1
        for t in returns.index[start_pos - 1: -1]:
            if t in reb_set:
                w_new = strat.target_weights(t, returns.loc[:t]).reindex(returns.columns).fillna(0.0)
                assert abs(w_new.sum() - 1) < 1e-6 and (w_new >= -1e-9).all(), "invalid weights"
                tv = float((w_new - w_hold).abs().sum())
                turn.loc[t, strat.name] = tv
                pending_cost = cost_rate * tv
                w_hold = w_new
                wlog[strat.name].append(w_new.rename(t))
            nxt = returns.index[returns.index.get_loc(t) + 1]
            r = returns.loc[nxt]
            g = float(w_hold @ r)
            gross.loc[nxt, strat.name] = g
            costs.loc[nxt, strat.name] = pending_cost
            net.loc[nxt, strat.name] = (1 + g) * (1 - pending_cost) - 1 if pending_cost else g
            pending_cost = 0.0
            w_hold = w_hold * (1 + r) / (1 + g)              # drift

    m = pd.DataFrame({n: performance_metrics(net[n], rf, td) for n in names}).T
    m["Avg Turnover/Rebalance"] = turn.mean()
    m["Total Cost Paid"] = costs.sum()
    m["Final Equity"] = (1 + net).prod()
    return BacktestResult(net, gross, (1 + net).cumprod(), {n: pd.DataFrame(v) for n, v in wlog.items()},
                          turn, costs, m, reb)
