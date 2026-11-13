"""End-to-end backend run: data -> features -> ML evaluation -> backtest -> outputs/.
Usage:  python run_pipeline.py [--model random_forest|xgboost] [--cost 0.001] [--no-ml-eval]
"""
import argparse
import logging
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from config.settings import ROOT, load_config
from src.backtesting.engine import run_backtest
from src.data.downloader import get_market_data
from src.data.preprocessing import align_ohlcv, build_price_panel, compute_returns
from src.features.technical_indicators import make_dataset
from src.models.evaluation import evaluate_split, evaluate_walk_forward
from src.portfolio.optimizers import AIAssisted, EqualWeight, MeanVariance

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--cost", type=float, default=None)
    ap.add_argument("--no-ml-eval", action="store_true")
    a = ap.parse_args()
    cfg = load_config()
    out = ROOT / "outputs"; out.mkdir(exist_ok=True)
    fin, mod, pf, bt = cfg["finance"], cfg["model"], cfg["portfolio"], cfg["backtest"]
    cost = fin["transaction_cost"] if a.cost is None else a.cost

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        md = get_market_data(cfg)
    for x in w:
        print(x.message)
    print("\n=== DATA ===\n", md.summary())
    pd.DataFrame(md.reports).T.to_csv(out / "cleaning_report.csv")

    prices = build_price_panel(md.ohlcv, cfg["data"]["max_ffill_days"])
    rets = compute_returns(prices)
    ohlcv = align_ohlcv(md.ohlcv, prices.index)
    prices.to_csv(ROOT / "data/processed/prices.csv"); rets.to_csv(ROOT / "data/processed/returns.csv")
    print(f"\nCommon calendar: {prices.index[0].date()} -> {prices.index[-1].date()} ({len(prices)} days)")

    # ---- ML evaluation (real assets only: scoring forecasts on fake data would be meaningless)
    if not a.no_ml_eval:
        for t in [t for t in ohlcv if md.sources[t] != "synthetic"]:
            X, y = make_dataset(ohlcv[t])
            tbl, imp, _ = evaluate_walk_forward(X, y, cfg, mod["min_train_days"], mod["wf_step_days"])
            sp = evaluate_split(X, y, cfg, tuple(mod["split"]))
            print(f"\n=== {t}: walk-forward OOS ({len(X)} rows) ===\n", tbl.round(5))
            print(f"--- {t}: static 60/20/20 split, TEST ---\n", sp["test"].round(5))
            tbl.to_csv(out / f"{t}_walkforward_metrics.csv")
            sp["test"].to_csv(out / f"{t}_split_test_metrics.csv")
            pd.DataFrame(imp).to_csv(out / f"{t}_feature_importance.csv")
            fig, ax = plt.subplots(figsize=(7, 5))
            imp["random_forest"].head(10)[::-1].plot.barh(ax=ax); ax.set_title(f"{t} RF importance (WF mean)")
            fig.tight_layout(); fig.savefig(out / f"{t}_importance.png", dpi=120); plt.close(fig)

    # ---- backtest
    strategies = [EqualWeight(),
                  MeanVariance(fin["risk_free_rate"], pf["cov_lookback_days"], pf["max_weight"]),
                  AIAssisted(ohlcv, cfg, a.model or pf["ai_model"], pf["ai_blend"],
                             pf["retrain_every_rebalances"], bt["min_history_days"],
                             rf=fin["risk_free_rate"], lookback=pf["cov_lookback_days"],
                             max_weight=pf["max_weight"])]
    res = run_backtest(rets, strategies, cost, bt["min_history_days"], fin["risk_free_rate"])
    print(f"\n=== BACKTEST (cost={cost:.4f}, OOS {res.net_returns.index[0].date()} -> "
          f"{res.net_returns.index[-1].date()}, {len(res.rebalance_dates)} rebalances) ===")
    pd.set_option("display.width", 200)
    print(res.metrics.round(4))
    if md.any_synthetic:
        print("\n*** WARNING: portfolio included SYNTHETIC assets -> results are a pipeline demo, NOT evidence. ***")
    res.metrics.to_csv(out / "backtest_metrics.csv"); res.equity.to_csv(out / "equity_curves.csv")
    res.turnover.to_csv(out / "turnover.csv")
    for n, wdf in res.weights.items():
        wdf.to_csv(out / f"weights_{n}.csv")
    fig, ax = plt.subplots(figsize=(9, 5)); res.equity.plot(ax=ax)
    ax.set_title("Out-of-sample equity (net of costs)" + (" [SYNTHETIC ASSETS INCLUDED]" if md.any_synthetic else ""))
    fig.tight_layout(); fig.savefig(out / "equity_curves.png", dpi=120)


if __name__ == "__main__":
    main()
