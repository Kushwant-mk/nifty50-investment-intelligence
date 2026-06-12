from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from investment_intelligence.data_loader import make_sample_market_data
from investment_intelligence.features import add_technical_indicators
from investment_intelligence.pipeline import run_pipeline


def test_feature_engineering_creates_targets():
    data = make_sample_market_data(days=120)
    featured = add_technical_indicators(data)
    assert not featured.empty
    assert {"target_next_return", "target_direction"}.issubset(featured.columns)


def test_pipeline_outputs_required_modules():
    results = run_pipeline(profile="balanced")
    assert results["model_metrics"]["rmse"] >= 0
    assert abs(results["allocation"]["weight"].sum() - 1) < 1e-9
    assert {"sharpe_ratio", "max_drawdown"}.issubset(results["portfolio_metrics"])
