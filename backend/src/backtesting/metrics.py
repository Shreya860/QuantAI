"""Performance metrics for a daily net-return series."""
import numpy as np
import pandas as pd


def performance_metrics(r: pd.Series, rf=0.07, td=252) -> dict:
    r = r.dropna()
    eq = (1 + r).cumprod()
    years = len(r) / td
    cagr = eq.iloc[-1] ** (1 / years) - 1
    vol = r.std() * np.sqrt(td)
    ex = r - rf / td
    downside = np.sqrt(np.mean(np.minimum(ex, 0) ** 2)) * np.sqrt(td)
    mdd = (eq / eq.cummax() - 1).min()
    return {
        "Total Return": eq.iloc[-1] - 1,
        "CAGR": cagr,
        "Ann. Volatility": vol,
        "Sharpe": ex.mean() * td / vol if vol > 0 else np.nan,
        "Sortino": ex.mean() * td / downside if downside > 0 else np.nan,
        "Max Drawdown": mdd,
        "Calmar": cagr / abs(mdd) if mdd < 0 else np.nan,
    }
