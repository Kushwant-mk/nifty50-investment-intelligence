from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS = [
    "return_1d",
    "return_5d",
    "return_21d",
    "ma_10_ratio",
    "ma_30_ratio",
    "ema_12_ratio",
    "ema_26_ratio",
    "volatility_21d",
    "volume_zscore_21d",
    "rsi_14",
    "macd",
    "bollinger_position",
]


def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = -delta.clip(upper=0).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def add_technical_indicators(data: pd.DataFrame) -> pd.DataFrame:
    """Add technical indicators and next-day prediction targets."""
    frames: list[pd.DataFrame] = []
    for symbol, group in data.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        g = group.copy()
        close = g["close"]
        volume = g["volume"]

        g["return_1d"] = close.pct_change()
        g["return_5d"] = close.pct_change(5)
        g["return_21d"] = close.pct_change(21)
        g["ma_10_ratio"] = close / close.rolling(10).mean() - 1
        g["ma_30_ratio"] = close / close.rolling(30).mean() - 1
        g["ema_12_ratio"] = close / close.ewm(span=12, adjust=False).mean() - 1
        g["ema_26_ratio"] = close / close.ewm(span=26, adjust=False).mean() - 1
        g["volatility_21d"] = g["return_1d"].rolling(21).std() * np.sqrt(252)
        g["volume_zscore_21d"] = (volume - volume.rolling(21).mean()) / volume.rolling(21).std()
        g["rsi_14"] = _rsi(close)

        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        g["macd"] = (ema_12 - ema_26) / close

        middle = close.rolling(20).mean()
        width = 2 * close.rolling(20).std()
        g["bollinger_position"] = (close - (middle - width)) / (2 * width)

        g["target_next_return"] = close.pct_change().shift(-1)
        g["target_direction"] = (g["target_next_return"] > 0).astype(int)
        g["symbol"] = symbol
        frames.append(g)

    featured = pd.concat(frames, ignore_index=True)
    featured[FEATURE_COLUMNS] = featured[FEATURE_COLUMNS].replace([np.inf, -np.inf], np.nan)
    return featured.dropna(subset=FEATURE_COLUMNS + ["target_next_return"]).reset_index(drop=True)


def latest_feature_rows(featured: pd.DataFrame) -> pd.DataFrame:
    idx = featured.groupby("symbol")["date"].idxmax()
    return featured.loc[idx].sort_values("symbol").reset_index(drop=True)
