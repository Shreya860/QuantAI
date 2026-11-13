# QuantAI (backend: Person A)

Data -> features -> ML -> optimisation -> backtest. Base repo attribution: DakshSave/Quantitative-Finance-Projects (MIT) — keep LICENSE.

## Run
```bash
pip install -r requirements.txt
# put Kaggle NIFTY-50 CSVs (one per stock) in data/kaggle/  (ADANIPORTS.csv is included)
python run_pipeline.py            # results in outputs/
python -m pytest -q               # 35 tests
```
**Need >= 3 real assets for portfolio results.** With fewer, the loader adds SYNTH_PEERn series, prints a loud
warning, and every output is flagged. Drop more Kaggle files in data/kaggle/ (or use yfinance) and the
synthetic top-up disappears automatically.

## Module map (PRD task -> file)
A1 data/downloader.py (+ synthetic.py) · A2 data/preprocessing.py · A3 features/technical_indicators.py ·
A4 models/forecasting.py · A5 models/evaluation.py · A6 backtesting/engine.py, metrics.py ·
A7 tests/ · A8 config/, requirements.txt, .gitignore ·
portfolio/optimizers.py = minimal EW / Mean-Variance / AI-Assisted strategies for Person C to extend.

## Key design decisions (viva material)
* **No look-ahead:** features at day t use data <= t; label y_t = return on t+1; a model trained for a
  decision at t only sees rows s < t (labels realised by t). Tested by perturbing the future.
* **Backtest timing:** decide at close of month-end t, trade at close t, earn from t+1. Cost = rate x
  traded notional, charged on day t+1. Same warm-up and dates for all strategies.
* **AI-assisted = Mean-Variance with only the expected-return input replaced:**
  mu = blend*mu_ML + (1-blend)*mu_hist (same covariance, same constraints) -> clean attribution.
* **Split adjustment:** Kaggle files are not split-adjusted (ADANIPORTS 5:1 on 2010-09-23); detected and fixed.
* **Metrics:** also reports R2 vs expanding-mean baseline (R2_oos_vs_mean); plain R2 is ~0 for daily returns.
