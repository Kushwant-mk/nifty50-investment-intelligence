from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from .anomaly import detect_market_anomalies
from .data_loader import load_market_data
from .features import add_technical_indicators
from .forecasting import evaluate_predictions, fit_ridge_return_model, forecast_latest, time_based_split
from .portfolio import construct_portfolio, explain_allocation
from .risk import stock_risk_table


def run_pipeline(data_dir: str | Path = "data/raw", profile: str = "balanced") -> dict[str, object]:
    raw = load_market_data(data_dir)
    featured = add_technical_indicators(raw)
    train, test = time_based_split(featured)
    model = fit_ridge_return_model(train)
    predictions = model.predict(test)
    metrics = evaluate_predictions(test["target_next_return"].to_numpy(), predictions)
    forecasts = forecast_latest(featured, model)
    risk = stock_risk_table(raw)
    allocation, portfolio = construct_portfolio(raw, forecasts, profile=profile)
    allocation["explanation"] = allocation.apply(lambda row: explain_allocation(row, profile), axis=1)
    anomalies = detect_market_anomalies(raw)

    return {
        "raw": raw,
        "featured": featured,
        "train": train,
        "test": test,
        "model": model,
        "model_metrics": metrics,
        "forecasts": forecasts,
        "risk": risk,
        "allocation": allocation,
        "portfolio_metrics": portfolio,
        "anomalies": anomalies,
    }


def save_outputs(
    results: dict[str, object],
    output_dir: str | Path = "reports/generated",
    model_dir: str | Path = "models",
) -> None:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    for name in ["forecasts", "risk", "allocation", "anomalies"]:
        value = results[name]
        if isinstance(value, pd.DataFrame):
            value.to_csv(path / f"{name}.csv", index=False)
    pd.DataFrame([results["model_metrics"]]).to_csv(path / "model_metrics.csv", index=False)
    pd.DataFrame([results["portfolio_metrics"]]).to_csv(path / "portfolio_metrics.csv", index=False)

    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    model = results["model"]
    artifact = {
        "model_type": "ridge_regression_next_day_return",
        "feature_columns": model.feature_columns,
        "mean": model.mean_.tolist(),
        "scale": model.scale_.tolist(),
        "coef": model.coef_.tolist(),
        "metrics": results["model_metrics"],
        "note": "Generated from the current data source. If data/raw is empty, this artifact is based on deterministic sample data.",
    }
    (model_path / "ridge_return_model.json").write_text(json.dumps(artifact, indent=2), encoding="utf-8")
