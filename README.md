# NIFTY-50 Investment Intelligence Platform

This repository implements the second Cult Open Projects 2026 problem statement: **Data-Driven Investment Intelligence Using NIFTY-50 Market Data**.

The goal is not just stock-price prediction. The project turns historical NIFTY-50 data into a practical decision-support prototype with forecasting, portfolio construction, risk assessment, explainability, anomaly detection, and a dashboard.

## Features

- Historical NIFTY-50 data ingestion from the provided Kaggle CSV files.
- Technical indicators: moving averages, EMA, RSI, MACD, Bollinger position, momentum, volume signals, and volatility.
- Stock predictor engine using a time-based train/test split and a ridge regression return forecaster implemented with `numpy`.
- Evaluation with MAE, RMSE, R2, and directional accuracy.
- Portfolio construction for conservative, balanced, and aggressive investor profiles.
- Risk analytics: volatility, Sharpe ratio, Sortino ratio, maximum drawdown, and risk-adjusted return.
- Explainable allocation notes for every recommended portfolio weight.
- Market anomaly detection for volatility spikes and unusual volume behavior.
- Optional Streamlit dashboard prototype.

## Repository Structure

```text
.
├── app.py
├── requirements.txt
├── scripts/
│   ├── generate_pdfs.py
│   └── run_pipeline.py
├── src/
│   └── investment_intelligence/
│       ├── anomaly.py
│       ├── data_loader.py
│       ├── features.py
│       ├── forecasting.py
│       ├── pipeline.py
│       ├── portfolio.py
│       └── risk.py
├── tests/
│   └── test_core.py
└── docs/
    ├── REPORT.md
    ├── REPORT.pdf
    ├── PRESENTATION.md
    └── PRESENTATION.pdf
```

## Dataset Setup

Download the organizer-provided dataset:

- NIFTY-50 Stock Market Data: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data/data
- Additional allowed dataset: https://www.kaggle.com/datasets/stoicstatic/india-stock-data-nse-1990-2020

Place the CSV files in:

```text
data/raw/
```

The code supports common Kaggle columns such as `Date`, `Symbol`, `Open`, `High`, `Low`, `Close`, `Volume`, and `Turnover`. If no real dataset is present, the pipeline uses deterministic sample data so reviewers can still run the project immediately.

## Environment Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with:

```bash
source .venv/bin/activate
```

## Run the Full Pipeline

```bash
python scripts/run_pipeline.py --data-dir data/raw --profile balanced
```

Investor profiles:

- `conservative`: favors lower volatility and smaller concentration.
- `balanced`: balances signal strength, risk, and Sharpe ratio.
- `aggressive`: gives more weight to expected return and permits higher concentration.

Generated outputs are saved to:

```text
reports/generated/
models/ridge_return_model.json
```

## Run the Dashboard

```bash
streamlit run app.py
```

The dashboard includes:

- Historical stock charts.
- Forecasted stock opportunities.
- Portfolio allocation view.
- Risk metrics.
- Explainable recommendations.
- Market anomaly table.

## Methodology

1. Load and normalize NIFTY-50 historical OHLCV data.
2. Engineer technical indicators from historical prices and volumes.
3. Split each stock chronologically into train and test sets to avoid look-ahead leakage.
4. Train a ridge regression model to forecast next-day returns.
5. Evaluate prediction quality using MAE, RMSE, R2, and directional accuracy.
6. Estimate stock-level risk and risk-adjusted performance.
7. Build profile-aware portfolios using predicted return, volatility, and Sharpe ratio.
8. Generate explanations and anomaly flags for transparent decision support.

## Reproducing Results

```bash
python scripts/run_pipeline.py --profile conservative
python scripts/run_pipeline.py --profile balanced
python scripts/run_pipeline.py --profile aggressive
```

Use the generated CSV files in `reports/generated/` for the technical report and presentation.

To regenerate the PDF deliverables:

```bash
python scripts/generate_pdfs.py
```

## Competition Alignment

This project covers the mandatory tasks:

- Stock Predictor Engine.
- Portfolio Construction Module.
- Risk Assessment Module.

It also includes optional elements:

- Explainable AI framework.
- Personalized investment profiles.
- Market anomaly detection.
- Interactive dashboard prototype.

## Disclaimer

This project is for educational and competition purposes only. It is not financial advice and should not be used for real investment decisions without professional review and live risk controls.
