from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


CANONICAL_COLUMNS = ["date", "symbol", "open", "high", "low", "close", "volume", "turnover"]


def _normalize_columns(frame: pd.DataFrame, fallback_symbol: str | None = None) -> pd.DataFrame:
    """Normalize Kaggle-style NIFTY-50 files into one canonical schema."""
    lookup = {column.lower().strip(): column for column in frame.columns}

    def pick(*names: str) -> str | None:
        for name in names:
            if name.lower() in lookup:
                return lookup[name.lower()]
        return None

    date_col = pick("Date")
    symbol_col = pick("Symbol", "Name")
    open_col = pick("Open")
    high_col = pick("High")
    low_col = pick("Low")
    close_col = pick("Close", "Last")
    volume_col = pick("Volume", "Traded Volume")
    turnover_col = pick("Turnover")

    required = [date_col, open_col, high_col, low_col, close_col, volume_col]
    if any(column is None for column in required):
        raise ValueError(f"Missing required market columns in file with columns: {list(frame.columns)}")

    normalized = pd.DataFrame(
        {
            "date": pd.to_datetime(frame[date_col], errors="coerce"),
            "symbol": frame[symbol_col].astype(str) if symbol_col else fallback_symbol,
            "open": pd.to_numeric(frame[open_col], errors="coerce"),
            "high": pd.to_numeric(frame[high_col], errors="coerce"),
            "low": pd.to_numeric(frame[low_col], errors="coerce"),
            "close": pd.to_numeric(frame[close_col], errors="coerce"),
            "volume": pd.to_numeric(frame[volume_col], errors="coerce"),
            "turnover": pd.to_numeric(frame[turnover_col], errors="coerce") if turnover_col else np.nan,
        }
    )
    normalized = normalized.dropna(subset=["date", "symbol", "open", "high", "low", "close", "volume"])
    normalized["symbol"] = normalized["symbol"].str.upper().str.replace(".CSV", "", regex=False)
    return normalized.sort_values(["symbol", "date"]).reset_index(drop=True)


def _is_price_file(frame: pd.DataFrame) -> bool:
    columns = {column.lower().strip() for column in frame.columns}
    required = {"date", "open", "high", "low", "volume"}
    has_close = "close" in columns or "last" in columns
    return required.issubset(columns) and has_close


def _load_metadata(files: list[Path]) -> pd.DataFrame | None:
    for file in files:
        frame = pd.read_csv(file, nrows=5)
        columns = {column.lower().strip(): column for column in frame.columns}
        if {"symbol", "company name"}.issubset(columns):
            metadata = pd.read_csv(file)
            metadata = metadata.rename(
                columns={
                    columns["symbol"]: "symbol",
                    columns["company name"]: "company_name",
                    columns.get("industry", "Industry"): "industry",
                }
            )
            keep = [column for column in ["symbol", "company_name", "industry"] if column in metadata.columns]
            metadata = metadata[keep].drop_duplicates("symbol")
            metadata["symbol"] = metadata["symbol"].astype(str).str.upper()
            return metadata
    return None


def load_market_data(data_dir: str | Path = "data/raw") -> pd.DataFrame:
    """Load all CSV files from data/raw. Falls back to deterministic sample data."""
    path = Path(data_dir)
    files = sorted(path.glob("*.csv")) if path.exists() else []
    if not files:
        return make_sample_market_data()

    frames: list[pd.DataFrame] = []
    for file in files:
        raw = pd.read_csv(file)
        if not _is_price_file(raw):
            continue
        frames.append(_normalize_columns(raw, fallback_symbol=file.stem))

    if not frames:
        raise ValueError(f"No OHLCV price CSV files were found in {path}.")

    data = pd.concat(frames, ignore_index=True)
    metadata = _load_metadata(files)
    if metadata is not None:
        data = data.merge(metadata, on="symbol", how="left")
    return data.drop_duplicates(["symbol", "date"]).sort_values(["symbol", "date"]).reset_index(drop=True)


def make_sample_market_data(seed: int = 7, days: int = 360) -> pd.DataFrame:
    """Create realistic enough sample data so the project runs without the Kaggle download."""
    rng = np.random.default_rng(seed)
    symbols = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK", "HINDUNILVR"]
    sectors = {
        "RELIANCE": "Energy",
        "TCS": "Information Technology",
        "HDFCBANK": "Banking",
        "INFY": "Information Technology",
        "ICICIBANK": "Banking",
        "HINDUNILVR": "Consumer Goods",
    }
    dates = pd.bdate_range("2020-01-01", periods=days)
    rows: list[dict[str, object]] = []

    for index, symbol in enumerate(symbols):
        price = 900 + index * 180
        drift = 0.00025 + index * 0.00003
        volatility = 0.012 + index * 0.0015
        shocks = rng.normal(drift, volatility, size=len(dates))
        if symbol in {"RELIANCE", "ICICIBANK"}:
            shocks[125:135] -= 0.035
        close = price * np.exp(np.cumsum(shocks))
        open_price = close * (1 + rng.normal(0, 0.004, size=len(dates)))
        high = np.maximum(open_price, close) * (1 + rng.uniform(0.001, 0.02, size=len(dates)))
        low = np.minimum(open_price, close) * (1 - rng.uniform(0.001, 0.02, size=len(dates)))
        volume = rng.integers(450_000, 4_500_000, size=len(dates)) * (1 + index * 0.08)

        for date, opn, hi, lo, cls, vol in zip(dates, open_price, high, low, close, volume):
            rows.append(
                {
                    "date": date,
                    "symbol": symbol,
                    "open": round(float(opn), 2),
                    "high": round(float(hi), 2),
                    "low": round(float(lo), 2),
                    "close": round(float(cls), 2),
                    "volume": int(vol),
                    "turnover": round(float(cls * vol), 2),
                    "sector": sectors[symbol],
                }
            )

    return pd.DataFrame(rows).sort_values(["symbol", "date"]).reset_index(drop=True)
