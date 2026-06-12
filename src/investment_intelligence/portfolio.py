from __future__ import annotations

import numpy as np
import pandas as pd

from .risk import portfolio_metrics, stock_risk_table


PROFILE_CONFIG = {
    "conservative": {"return_weight": 0.20, "risk_weight": 0.55, "sharpe_weight": 0.25, "max_weight": 0.25},
    "balanced": {"return_weight": 0.35, "risk_weight": 0.30, "sharpe_weight": 0.35, "max_weight": 0.35},
    "aggressive": {"return_weight": 0.55, "risk_weight": 0.15, "sharpe_weight": 0.30, "max_weight": 0.50},
}


def returns_matrix(data: pd.DataFrame) -> pd.DataFrame:
    prices = data.pivot(index="date", columns="symbol", values="close").sort_index()
    return prices.pct_change(fill_method=None).dropna(how="all")


def _minmax(series: pd.Series) -> pd.Series:
    if series.max() == series.min():
        return pd.Series(0.5, index=series.index)
    return (series - series.min()) / (series.max() - series.min())


def construct_portfolio(data: pd.DataFrame, forecasts: pd.DataFrame, profile: str = "balanced") -> tuple[pd.DataFrame, dict[str, float]]:
    if profile not in PROFILE_CONFIG:
        raise ValueError(f"Unknown profile '{profile}'. Choose from {list(PROFILE_CONFIG)}")

    config = PROFILE_CONFIG[profile]
    risk = stock_risk_table(data).merge(forecasts[["symbol", "predicted_next_return"]], on="symbol", how="left")
    risk = risk.fillna({"predicted_next_return": 0.0})
    score = (
        config["return_weight"] * _minmax(risk["predicted_next_return"])
        + config["sharpe_weight"] * _minmax(risk["sharpe_ratio"].fillna(0))
        + config["risk_weight"] * (1 - _minmax(risk["volatility"]))
    )
    risk["score"] = score.clip(lower=0)

    if risk["score"].sum() == 0:
        risk["weight"] = 1 / len(risk)
    else:
        raw = risk["score"] / risk["score"].sum()
        capped = raw.clip(upper=config["max_weight"])
        risk["weight"] = capped / capped.sum()

    allocation = risk.sort_values("weight", ascending=False)[
        ["symbol", "weight", "predicted_next_return", "annual_return", "volatility", "sharpe_ratio", "max_drawdown", "score"]
    ].reset_index(drop=True)
    metrics = portfolio_metrics(returns_matrix(data), allocation.set_index("symbol")["weight"])
    return allocation, metrics


def explain_allocation(row: pd.Series, profile: str) -> str:
    direction = "positive" if row["predicted_next_return"] >= 0 else "negative"
    return (
        f"{row['symbol']} receives {row['weight']:.1%} in the {profile} profile because it has a "
        f"{direction} next-period signal, volatility of {row['volatility']:.1%}, "
        f"Sharpe ratio of {row['sharpe_ratio']:.2f}, and max drawdown of {row['max_drawdown']:.1%}."
    )
