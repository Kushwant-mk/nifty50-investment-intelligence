# Presentation Draft

## Slide 1: Problem Overview

Build an investment intelligence platform for NIFTY-50 market data that supports forecasting, portfolio construction, risk assessment, explainability, and decision-making.

## Slide 2: Dataset

- Daily OHLCV data for NIFTY-50 companies.
- Covers long-term historical stock behavior.
- Organizer-provided datasets only.
- Optional company metadata can support sector-level insights.

## Slide 3: Feature Engineering

- Returns and momentum.
- Moving averages and EMA.
- RSI and MACD.
- Bollinger band position.
- Volatility and volume signals.

## Slide 4: Stock Predictor Engine

- Chronological train/test split.
- Ridge regression next-day return model.
- Metrics: MAE, RMSE, R2, directional accuracy.
- Designed as a transparent baseline.

## Slide 5: Portfolio Construction

- Conservative, balanced, and aggressive profiles.
- Combines expected return, Sharpe ratio, and volatility penalty.
- Applies concentration caps.
- Produces allocation weights and explanations.

## Slide 6: Risk Analytics

- Annualized return.
- Volatility.
- Sharpe ratio.
- Sortino ratio.
- Maximum drawdown.
- Portfolio-level risk metrics.

## Slide 7: Explainability and Anomalies

- Human-readable allocation explanations.
- Return and volume z-score anomaly detection.
- Helps users understand risk events and recommendation drivers.

## Slide 8: Results and Next Steps

- Prototype runs end to end with sample data and Kaggle CSV files.
- Dashboard enables interactive exploration.
- Next improvements: walk-forward validation, sector constraints, richer optimization, and feature attribution.
