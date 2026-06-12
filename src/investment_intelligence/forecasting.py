from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from .features import FEATURE_COLUMNS


@dataclass
class RidgeModel:
    feature_columns: list[str]
    mean_: np.ndarray
    scale_: np.ndarray
    coef_: np.ndarray

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        x = frame[self.feature_columns].to_numpy(dtype=float)
        x_scaled = (x - self.mean_) / self.scale_
        design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
        return design @ self.coef_


def time_based_split(featured: pd.DataFrame, test_fraction: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    for _, group in featured.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        cutoff = max(1, int(len(group) * (1 - test_fraction)))
        train_parts.append(group.iloc[:cutoff])
        test_parts.append(group.iloc[cutoff:])
    return pd.concat(train_parts).reset_index(drop=True), pd.concat(test_parts).reset_index(drop=True)


def fit_ridge_return_model(train: pd.DataFrame, alpha: float = 2.0) -> RidgeModel:
    x = train[FEATURE_COLUMNS].to_numpy(dtype=float)
    y = train["target_next_return"].to_numpy(dtype=float)
    mean = x.mean(axis=0)
    scale = x.std(axis=0)
    scale[scale == 0] = 1
    x_scaled = (x - mean) / scale
    design = np.column_stack([np.ones(len(x_scaled)), x_scaled])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0
    coef = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    return RidgeModel(FEATURE_COLUMNS.copy(), mean, scale, coef)


def evaluate_predictions(actual: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    error = predicted - actual
    rmse = float(np.sqrt(np.mean(error**2)))
    mae = float(np.mean(np.abs(error)))
    ss_res = float(np.sum(error**2))
    ss_tot = float(np.sum((actual - actual.mean()) ** 2))
    r2 = 1 - ss_res / ss_tot if ss_tot else 0.0
    directional_accuracy = float(np.mean((predicted > 0) == (actual > 0)))
    return {"mae": mae, "rmse": rmse, "r2": r2, "directional_accuracy": directional_accuracy}


def forecast_latest(featured: pd.DataFrame, model: RidgeModel) -> pd.DataFrame:
    latest = featured.loc[featured.groupby("symbol")["date"].idxmax()].copy()
    latest["predicted_next_return"] = model.predict(latest)
    latest["predicted_direction"] = np.where(latest["predicted_next_return"] >= 0, "Up", "Down")
    latest["confidence"] = latest["predicted_next_return"].abs().rank(pct=True)
    return latest.sort_values("predicted_next_return", ascending=False).reset_index(drop=True)
