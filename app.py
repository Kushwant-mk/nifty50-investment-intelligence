from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from investment_intelligence.pipeline import run_pipeline


def main() -> None:
    try:
        import plotly.express as px
        import streamlit as st
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "Dashboard dependencies are missing. Install them with: pip install -r requirements.txt"
        ) from exc

    st.set_page_config(page_title="NIFTY-50 Investment Intelligence", layout="wide")
    st.title("NIFTY-50 Investment Intelligence")
    st.caption("Forecasting, risk analytics, portfolio construction, explainability, and anomaly detection.")

    with st.sidebar:
        profile = st.selectbox("Investor profile", ["conservative", "balanced", "aggressive"], index=1)
        data_dir = st.text_input("Dataset folder", "data/raw")
        st.info("Place the Kaggle NIFTY-50 CSV files in data/raw. If empty, the app uses sample data.")

    results = run_pipeline(data_dir=data_dir, profile=profile)
    data = results["raw"]
    forecasts = results["forecasts"]
    allocation = results["allocation"]
    risk = results["risk"].copy()
    risk["risk_chart_size"] = risk["sharpe_ratio"].fillna(0).clip(lower=0) + 0.1

    metric_cols = st.columns(4)
    for col, (label, value) in zip(metric_cols, results["model_metrics"].items()):
        col.metric(label.replace("_", " ").title(), f"{value:.3f}")

    st.subheader("Historical Performance")
    selected = st.multiselect("Stocks", sorted(data["symbol"].unique()), default=sorted(data["symbol"].unique())[:4])
    chart_data = data[data["symbol"].isin(selected)]
    st.plotly_chart(px.line(chart_data, x="date", y="close", color="symbol"), width="stretch")

    st.subheader("Forecasted Opportunities")
    st.dataframe(
        forecasts[["symbol", "date", "close", "predicted_next_return", "predicted_direction", "confidence"]],
        width="stretch",
    )

    st.subheader("Portfolio Construction")
    c1, c2 = st.columns([1.2, 1])
    with c1:
        st.plotly_chart(px.pie(allocation, values="weight", names="symbol", hole=0.45), width="stretch")
    with c2:
        st.dataframe(allocation[["symbol", "weight", "volatility", "sharpe_ratio", "max_drawdown"]], width="stretch")

    st.subheader("Why These Allocations?")
    for explanation in allocation["explanation"]:
        st.write(f"- {explanation}")

    st.subheader("Risk Dashboard")
    st.dataframe(risk.drop(columns=["risk_chart_size"]), width="stretch")
    st.plotly_chart(
        px.scatter(
            risk,
            x="volatility",
            y="annual_return",
            size="risk_chart_size",
            hover_name="symbol",
            color="sharpe_ratio",
        ),
        width="stretch",
    )

    st.subheader("Market Anomalies")
    st.dataframe(results["anomalies"].head(50), width="stretch")


if __name__ == "__main__":
    main()
