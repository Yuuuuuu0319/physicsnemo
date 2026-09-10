#!/usr/bin/env python3
"""Common physical-unit forecast metrics for the formal SI five-minute study."""

from __future__ import annotations

import torch


def _safe_ratio(numerator: torch.Tensor, denominator: torch.Tensor) -> float:
    return float((numerator / torch.clamp(denominator, min=1.0)).item())


def _scope_metrics(
    predicted_depth_m: torch.Tensor,
    target_depth_m: torch.Tensor,
    predicted_volume_m3: torch.Tensor,
    target_volume_m3: torch.Tensor,
    node_mask: torch.Tensor,
    *,
    wet_depth_threshold_m: float,
    delta_t_seconds: float,
) -> dict[str, float | int]:
    predicted_depth = predicted_depth_m[:, node_mask]
    target_depth = target_depth_m[:, node_mask]
    predicted_volume = predicted_volume_m3[:, node_mask]
    target_volume = target_volume_m3[:, node_mask]
    depth_error = predicted_depth - target_depth
    volume_error = predicted_volume - target_volume

    predicted_wet = predicted_depth >= wet_depth_threshold_m
    target_wet = target_depth >= wet_depth_threshold_m
    true_positive = torch.sum(predicted_wet & target_wet).to(torch.float64)
    false_positive = torch.sum(predicted_wet & ~target_wet).to(torch.float64)
    false_negative = torch.sum(~predicted_wet & target_wet).to(torch.float64)
    precision = _safe_ratio(true_positive, true_positive + false_positive)
    recall = _safe_ratio(true_positive, true_positive + false_negative)
    f1 = 2.0 * precision * recall / max(precision + recall, 1.0e-12)

    target_wet_any = torch.any(target_wet, dim=0)
    arrival_mae_minutes = 0.0
    if torch.any(target_wet_any):
        steps = predicted_depth.shape[0]
        predicted_arrival = torch.argmax(predicted_wet.to(torch.int64), dim=0)
        target_arrival = torch.argmax(target_wet.to(torch.int64), dim=0)
        predicted_arrival = torch.where(
            torch.any(predicted_wet, dim=0),
            predicted_arrival,
            torch.full_like(predicted_arrival, steps),
        )
        arrival_mae_minutes = float(
            torch.mean(
                torch.abs(
                    predicted_arrival[target_wet_any]
                    - target_arrival[target_wet_any]
                ).to(torch.float64)
            ).item()
            * delta_t_seconds
            / 60.0
        )

    peak_depth_error = torch.max(predicted_depth, dim=0).values - torch.max(
        target_depth, dim=0
    ).values
    total_volume_error = torch.sum(volume_error, dim=1)
    return {
        "node_count": int(torch.sum(node_mask).item()),
        "depth_rmse_m": float(torch.sqrt(torch.mean(depth_error.square())).item()),
        "depth_mae_m": float(torch.mean(torch.abs(depth_error)).item()),
        "depth_bias_m": float(torch.mean(depth_error).item()),
        "volume_rmse_m3": float(
            torch.sqrt(torch.mean(volume_error.square())).item()
        ),
        "volume_mae_m3": float(torch.mean(torch.abs(volume_error)).item()),
        "peak_depth_rmse_m": float(
            torch.sqrt(torch.mean(peak_depth_error.square())).item()
        ),
        "peak_depth_bias_m": float(torch.mean(peak_depth_error).item()),
        "wet_precision": precision,
        "wet_recall": recall,
        "wet_f1": f1,
        "gt_wet_point_count": int(torch.sum(target_wet).item()),
        "gt_wet_node_count": int(torch.sum(target_wet_any).item()),
        "arrival_time_mae_minutes": arrival_mae_minutes,
        "total_volume_rmse_m3": float(
            torch.sqrt(torch.mean(total_volume_error.square())).item()
        ),
        "total_volume_bias_m3": float(torch.mean(total_volume_error).item()),
    }


def compute_formal_forecast_metrics(
    predicted_depth_normalized: torch.Tensor,
    target_depth_normalized: torch.Tensor,
    predicted_volume_normalized: torch.Tensor,
    target_volume_normalized: torch.Tensor,
    zone_label: torch.Tensor,
    *,
    water_depth_mean: float,
    water_depth_std: float,
    volume_mean: float,
    volume_std: float,
    delta_t_seconds: float,
    wet_depth_threshold_m: float = 0.01,
    high_zone_code: int = 3,
) -> dict[str, float | int]:
    """Return event-level metrics in SI units for all and high-zone nodes."""

    shapes = {
        tuple(predicted_depth_normalized.shape),
        tuple(target_depth_normalized.shape),
        tuple(predicted_volume_normalized.shape),
        tuple(target_volume_normalized.shape),
    }
    if len(shapes) != 1 or predicted_depth_normalized.ndim != 2:
        raise ValueError("Forecast arrays must share shape [time, node].")
    if zone_label.reshape(-1).numel() != predicted_depth_normalized.shape[1]:
        raise ValueError("zone_label must align with the forecast node dimension.")
    if delta_t_seconds <= 0.0 or wet_depth_threshold_m < 0.0:
        raise ValueError("Time step must be positive and wet threshold nonnegative.")

    predicted_depth_m = (
        predicted_depth_normalized * water_depth_std + water_depth_mean
    )
    target_depth_m = target_depth_normalized * water_depth_std + water_depth_mean
    predicted_volume_m3 = predicted_volume_normalized * volume_std + volume_mean
    target_volume_m3 = target_volume_normalized * volume_std + volume_mean
    zone_label = zone_label.reshape(-1).to(predicted_depth_normalized.device)
    masks = {
        "all": torch.ones_like(zone_label, dtype=torch.bool),
        "high": zone_label == int(high_zone_code),
    }
    if not torch.any(masks["high"]):
        raise ValueError(f"No nodes use high-zone code {high_zone_code}.")

    metrics: dict[str, float | int] = {
        "forecast_delta_t_seconds": float(delta_t_seconds),
        "forecast_horizon_minutes": float(
            predicted_depth_normalized.shape[0] * delta_t_seconds / 60.0
        ),
        "wet_depth_threshold_m": float(wet_depth_threshold_m),
        "high_zone_code": int(high_zone_code),
    }
    for scope, mask in masks.items():
        scope_metrics = _scope_metrics(
            predicted_depth_m,
            target_depth_m,
            predicted_volume_m3,
            target_volume_m3,
            mask,
            wet_depth_threshold_m=wet_depth_threshold_m,
            delta_t_seconds=delta_t_seconds,
        )
        metrics.update(
            {f"{scope}_{name}": value for name, value in scope_metrics.items()}
        )
    return metrics
