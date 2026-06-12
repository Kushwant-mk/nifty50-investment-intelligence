from __future__ import annotations

import numpy as np
import pandas as pd


def maximum_drawdown(close: pd.Series) -> float:
    wealth = close / close.iloc[0]
    peak = wealth.cummax()
    drawdown = wealth / peak - 1
    return float(drawdown.min())


def stock_risk_table(data: pd.DataFrame, risk_free_rate: float = 0.04) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    daily_rf = risk_free_rate / 252
    for symbol, group in data.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        returns = group["close"].pct_change().dropna()
        downside = returns[returns < daily_rf]
        annual_return = float((1 + returns.mean()) ** 252 - 1)
        annual_volatility = float(returns.std() * np.sqrt(252))
        downside_volatility = float(downside.std() * np.sqrt(252)) if len(downside) > 1 else np.nan
        sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility else np.nan
        sortino = (annual_return - risk_free_rate) / downside_volatility if downside_volatility else np.nan
        rows.append(
            {
                "symbol": symbol,
                "annual_return": annual_return,
                "volatility": annual_volatility,
                "sharpe_ratio": float(sharpe),
                "sortino_ratio": float(sortino),
                "max_drawdown": maximum_drawdown(group["close"]),
            }
        )
    return pd.DataFrame(rows).sort_values("sharpe_ratio", ascending=False).reset_index(drop=True)


def portfolio_metrics(returns: pd.DataFrame, weights: pd.Series, risk_free_rate: float = 0.04) -> dict[str, float]:
    aligned = returns[weights.index].dropna()
    portfolio_returns = aligned @ weights
    annual_return = float((1 + portfolio_returns.mean()) ** 252 - 1)
    annual_volatility = float(portfolio_returns.std() * np.sqrt(252))
    sharpe = (annual_return - risk_free_rate) / annual_volatility if annual_volatility else np.nan
    wealth = (1 + portfolio_returns).cumprod()
    drawdown = wealth / wealth.cummax() - 1
    return {
        "annual_return": annual_return,
        "volatility": annual_volatility,
        "sharpe_ratio": float(sharpe),
        "max_drawdown": float(drawdown.min()),
    }
