# Technical Report: Data-Driven Investment Intelligence Using NIFTY-50 Market Data

## 1. Problem Understanding

The objective is to build an AI-powered investment intelligence platform using only the organizer-provided NIFTY-50 datasets. The platform should help investors analyze historical performance, forecast stock behavior, assess risk, construct portfolios for different risk profiles, and understand why recommendations are produced.

The solution focuses on decision support rather than isolated price prediction. Forecasts are combined with risk-adjusted analytics and explainability so that users can compare opportunities and trade-offs.

## 2. Data and Exploratory Data Analysis

The expected input consists of daily stock records containing date, open, high, low, close, volume, and turnover. The project normalizes multiple Kaggle CSV formats into a common schema.

Recommended EDA checks:

- Coverage by stock and date range.
- Missing values and duplicate records.
- Price and volume distributions.
- Stock-level return distribution.
- Sector-level behavior where metadata is available.
- Volatility clusters and drawdown periods.
- Correlation between NIFTY-50 constituents.

Initial insights typically expected from NIFTY-50 data:

- Banking and financial stocks often show higher systematic market sensitivity.
- IT stocks may show strong long-term momentum but can experience sharp corrections.
- Volume spikes frequently coincide with high absolute returns.
- Drawdowns and volatility clustering are more informative for decision support than raw price level alone.

## 3. Feature Engineering

The feature pipeline creates:

- Daily, weekly, and monthly momentum.
- Moving average ratios.
- Exponential moving average ratios.
- 21-day annualized volatility.
- Volume z-score.
- RSI.
- MACD.
- Bollinger band position.

The target variable is next-day return. A direction label is also produced for directional accuracy.

## 4. Model Architecture

The baseline predictor is ridge regression implemented with `numpy`. Ridge regression was selected because it is fast, transparent, reproducible, and robust for a first investment intelligence prototype.

The split is chronological within each stock to avoid look-ahead leakage. The model trains on the earlier 80 percent of observations and tests on the latest 20 percent.

## 5. Evaluation

The model reports:

- MAE.
- RMSE.
- R2 score.
- Directional accuracy.

For investment intelligence, directional accuracy and downstream portfolio behavior are as important as raw RMSE. A model with modest point forecast accuracy can still be useful if it ranks opportunities and risk trade-offs sensibly.

## 6. Portfolio Construction

The system supports three investor profiles:

- Conservative: prioritizes low volatility and caps concentration.
- Balanced: balances expected return, volatility, and Sharpe ratio.
- Aggressive: gives more weight to expected return and allows higher concentration.

Portfolio scoring combines:

- Predicted next-period return.
- Historical Sharpe ratio.
- Historical volatility penalty.

Weights are normalized and profile-specific concentration caps are applied.

## 7. Risk Assessment

The platform computes:

- Annualized return.
- Annualized volatility.
- Sharpe ratio.
- Sortino ratio.
- Maximum drawdown.
- Portfolio-level risk metrics.

These metrics help users understand the risk behind each recommendation instead of relying only on forecasted upside.

## 8. Explainability

Each portfolio allocation includes a plain-language explanation based on:

- Forecasted return signal.
- Volatility.
- Sharpe ratio.
- Maximum drawdown.
- Investor profile.

Example:

`TCS receives 18.2% in the balanced profile because it has a positive next-period signal, volatility of 21.4%, Sharpe ratio of 1.10, and max drawdown of -18.7%.`

## 9. Anomaly Detection

The anomaly module identifies unusual market behavior using rolling z-scores for returns and volume changes. This flags sudden volatility spikes, extreme drawdowns, and unusual trading activity.

## 10. Limitations and Future Work

Limitations:

- The baseline model is intentionally simple.
- It does not use live market data because the problem statement prohibits it.
- Forecasts are short-horizon and should not be treated as financial advice.
- Transaction costs, liquidity constraints, and tax effects are not modeled.

Future improvements:

- Add walk-forward validation.
- Compare tree-based models and sequence models.
- Add sector-aware constraints.
- Add covariance-based mean-variance optimization.
- Add richer explainability with feature attribution.
- Export final report and dashboard screenshots for submission.
