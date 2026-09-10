import importlib.util
from pathlib import Path

import pytest
import torch


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/weather/flood_modeling/hydrographnet/formal_si5m_forecast_metrics.py"
)
SPEC = importlib.util.spec_from_file_location(
    "formal_si5m_forecast_metrics", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_formal_forecast_metrics_use_si_units_and_high_zone_scope():
    target_depth = torch.tensor([[0.0, 0.0], [0.1, 0.2], [0.2, 0.3]])
    predicted_depth = target_depth + torch.tensor(
        [[0.0, 0.1], [0.0, 0.1], [0.0, 0.1]]
    )
    target_volume = torch.zeros_like(target_depth)
    predicted_volume = torch.ones_like(target_depth)
    metrics = MODULE.compute_formal_forecast_metrics(
        predicted_depth,
        target_depth,
        predicted_volume,
        target_volume,
        torch.tensor([0, 3]),
        water_depth_mean=0.0,
        water_depth_std=2.0,
        volume_mean=10.0,
        volume_std=3.0,
        delta_t_seconds=300.0,
        wet_depth_threshold_m=0.01,
    )

    assert metrics["forecast_horizon_minutes"] == 15.0
    assert metrics["all_depth_rmse_m"] == pytest.approx(0.2 / 2**0.5)
    assert metrics["high_depth_rmse_m"] == pytest.approx(0.2)
    assert metrics["all_volume_rmse_m3"] == pytest.approx(3.0)
    assert metrics["high_node_count"] == 1
    assert metrics["high_wet_recall"] == 1.0


def test_formal_forecast_metrics_penalize_missed_arrival_at_horizon():
    target_depth = torch.tensor([[0.0], [0.1], [0.2]])
    predicted_depth = torch.zeros_like(target_depth)
    metrics = MODULE.compute_formal_forecast_metrics(
        predicted_depth,
        target_depth,
        torch.zeros_like(target_depth),
        torch.zeros_like(target_depth),
        torch.tensor([3]),
        water_depth_mean=0.0,
        water_depth_std=1.0,
        volume_mean=0.0,
        volume_std=1.0,
        delta_t_seconds=300.0,
    )

    assert metrics["all_arrival_time_mae_minutes"] == 10.0
    assert metrics["high_wet_recall"] == 0.0
