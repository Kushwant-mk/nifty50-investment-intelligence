from __future__ import annotations

import pandas as pd


def detect_market_anomalies(data: pd.DataFrame, z_threshold: float = 3.0) -> pd.DataFrame:
    rows = []
    for symbol, group in data.sort_values(["symbol", "date"]).groupby("symbol", sort=False):
        g = group.copy()
        returns = g["close"].pct_change()
        volume_change = g["volume"].pct_change()
        g["return_zscore"] = (returns - returns.rolling(60).mean()) / returns.rolling(60).std()
        g["volume_zscore"] = (volume_change - volume_change.rolling(60).mean()) / volume_change.rolling(60).std()
        flagged = g[(g["return_zscore"].abs() >= z_threshold) | (g["volume_zscore"].abs() >= z_threshold)].copy()
        flagged["symbol"] = symbol
        rows.append(flagged[["date", "symbol", "close", "volume", "return_zscore", "volume_zscore"]])
    if not rows:
        return pd.DataFrame(columns=["date", "symbol", "close", "volume", "return_zscore", "volume_zscore"])
    return pd.concat(rows, ignore_index=True).sort_values(["date", "symbol"], ascending=[False, True])
