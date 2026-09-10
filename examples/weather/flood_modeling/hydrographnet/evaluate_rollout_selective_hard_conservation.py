# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Rollout diagnostic for exact selective hard-conservation edge fluxes.

The edge head is corrected with the differentiable componentwise PyTorch layer
using train-split-only per-face RMS-squared weights. Future HEC-RAS targets are
used only for evaluation metrics, never to construct the inference correction.
"""

import argparse
import csv
import hashlib
from pathlib import Path

import torch

from evaluate_pretrained_edge_flux_head import (
    add_metric,
    add_residual_metrics,
    compute_divergence,
    finalize_metrics,
    rmse,
)
from formal_si5m_forecast_metrics import compute_formal_forecast_metrics
from evaluate_formal_si5m_node_rollout import verify_split_contract
from hecras_projection import (
    selected_node_mask,
)
from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.utils import load_checkpoint
from selective_hard_conservation import project_connected_control_volume_flux
from train import HecRasEdgeFluxHead
from utils import compute_area_extrapolated_storage_delta


def update_rollout_state(
    x_iter: torch.Tensor,
    pred: torch.Tensor,
    inflow_value: torch.Tensor,
    precip_value: torch.Tensor,
    n_time_steps: int,
    volume_delta_override: torch.Tensor | None = None,
) -> torch.Tensor:
    static_part = x_iter[:, :12]
    water_depth_window = x_iter[:, 12 : 12 + n_time_steps]
    volume_window = x_iter[:, 12 + n_time_steps : 12 + 2 * n_time_steps]
    new_wd = water_depth_window[:, -1] + pred[:, 0]
    volume_delta = pred[:, 1] if volume_delta_override is None else volume_delta_override
    new_volume = volume_window[:, -1] + volume_delta
    water_depth_updated = torch.cat(
        [water_depth_window[:, 1:], new_wd[:, None]], dim=1
    )
    volume_updated = torch.cat([volume_window[:, 1:], new_volume[:, None]], dim=1)
    static_updated = static_part.clone()
    static_updated[:, 10:11] = inflow_value.expand(x_iter.shape[0], 1)
    static_updated[:, 11:12] = precip_value.expand(x_iter.shape[0], 1)
    return torch.cat([static_updated, water_depth_updated, volume_updated], dim=1)


def rollout_local_source_delta(
    dataset,
    x_iter: torch.Tensor,
    target_precipitation: torch.Tensor,
    delta_t: float,
    device: torch.device,
    target_node_precipitation: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return inference-available precipitation/IP source volume for one step."""

    if target_node_precipitation is not None:
        precipitation_rate = target_node_precipitation.to(device).reshape(-1)
    else:
        previous_precip_norm = x_iter[0, 11]
        target_precip_norm = target_precipitation.reshape(-1)[0].to(device)
        average_precip_norm = 0.5 * (previous_precip_norm + target_precip_norm)
        precip_mean = float(dataset.dynamic_stats["precipitation"]["mean"])
        precip_std = float(dataset.dynamic_stats["precipitation"]["std"])
        precipitation_rate = average_precip_norm * precip_std + precip_mean
    area = graph_area = dataset.hecras_face_graph.get("node_surface_area")
    if graph_area is None:
        area = dataset.static_data["area_denorm"].reshape(-1)
    area = torch.as_tensor(area, dtype=x_iter.dtype, device=device)
    if dataset.local_source_runoff_mode == "full_area":
        runoff_fraction = torch.ones_like(area)
    else:
        runoff_percentage = dataset.denormalize(
            dataset.static_data["infiltration"],
            dataset.static_stats["infiltration"]["mean"],
            dataset.static_stats["infiltration"]["std"],
        ).reshape(-1)
        runoff_fraction = torch.as_tensor(
            runoff_percentage / 100.0,
            dtype=x_iter.dtype,
            device=device,
        )
    return precipitation_rate * area * runoff_fraction * float(delta_t)


def attach_rollout_state_attrs(
    graph,
    x_iter: torch.Tensor,
    local_source_delta: torch.Tensor,
    n_time_steps: int,
    delta_t: float,
) -> None:
    """Refresh edge-head attributes that change during autoregressive rollout."""

    graph.x = x_iter
    if hasattr(graph, "current_surface_elevation") and hasattr(
        graph, "current_water_depth_denorm"
    ):
        terrain_elevation = (
            graph.current_surface_elevation - graph.current_water_depth_denorm
        )
        latest_wd_norm = x_iter[:, 12 + n_time_steps - 1]
        wd_mean = graph.water_depth_mean.reshape(-1)[0].to(x_iter.device)
        wd_std = graph.water_depth_std.reshape(-1)[0].to(x_iter.device)
        current_wd = latest_wd_norm * wd_std + wd_mean
        graph.current_water_depth_denorm = current_wd
        graph.current_surface_elevation = terrain_elevation + current_wd
    graph.local_source_rate = local_source_delta / float(delta_t)


def manning_face_volume_heuristic(graph, delta_t: float) -> torch.Tensor:
    """Return an untuned Manning-style face-volume control in HDF orientation."""

    required = (
        "hecras_face_index",
        "hecras_face_length",
        "hecras_edge_physical_features",
        "current_water_depth_denorm",
        "current_surface_elevation",
    )
    missing = [name for name in required if not hasattr(graph, name)]
    if missing:
        raise AttributeError("Manning face heuristic requires: " + ", ".join(missing))
    src, dst = graph.hecras_face_index
    depth = graph.current_water_depth_denorm
    surface = graph.current_surface_elevation
    features = graph.hecras_edge_physical_features
    distance = torch.clamp(features[:, 0], min=1.0e-6)
    manning = torch.clamp(features[:, 3], min=1.0e-4)
    hydraulic_depth = torch.clamp(0.5 * (depth[src] + depth[dst]), min=0.0)
    surface_slope = (surface[dst] - surface[src]) / distance
    discharge = (
        -torch.sign(surface_slope)
        * graph.hecras_face_length
        * hydraulic_depth.pow(5.0 / 3.0)
        * torch.sqrt(torch.abs(surface_slope))
        / manning
    )
    return discharge * float(delta_t)


def projection_target(
    graph,
    pred_delta: torch.Tensor,
    mode: str,
    device: torch.device,
    boundary_source_delta: torch.Tensor | None = None,
    local_source_delta: torch.Tensor | None = None,
) -> torch.Tensor:
    if mode == "node_delta":
        return pred_delta.reshape(-1)
    if mode == "hecras_boundary_source":
        if boundary_source_delta is None:
            raise ValueError(
                "projection-target=hecras_boundary_source requires a per-step "
                "boundary/source delta."
            )
        return pred_delta.reshape(-1) - boundary_source_delta.to(device).reshape(-1)
    if mode == "initial_boundary_source":
        if not hasattr(graph, "hecras_edge_boundary_source_delta"):
            raise AttributeError(
                "projection-target=initial_boundary_source requires "
                "graph.hecras_edge_boundary_source_delta."
            )
        boundary_source = graph.hecras_edge_boundary_source_delta.to(device).reshape(-1)
        return pred_delta.reshape(-1) - boundary_source
    if mode == "known_local_source":
        if local_source_delta is None:
            raise ValueError(
                "projection-target=known_local_source requires the current "
                "precipitation/IP source delta."
            )
        return pred_delta.reshape(-1) - local_source_delta.to(device).reshape(-1)
    raise ValueError(f"Unknown projection target: {mode!r}")


def rollout_boundary_source_delta(dataset, hydrograph_id: str, transition_index: int):
    edge_flow = getattr(dataset, "hecras_edge_flow_delta_by_hydrograph", {}).get(
        hydrograph_id
    )
    if not isinstance(edge_flow, dict):
        raise AttributeError(
            "projection-target=hecras_boundary_source requires split edge targets."
        )
    key = "boundary" if "boundary" in edge_flow else "boundary_selected"
    if key not in edge_flow:
        raise AttributeError("No boundary source exists in the edge targets.")
    if transition_index < 0 or transition_index >= edge_flow[key].shape[0]:
        raise IndexError(
            f"Boundary/source transition {transition_index} is outside "
            f"0..{edge_flow[key].shape[0] - 1} for {hydrograph_id}."
        )
    if key == "boundary":
        values = edge_flow[key][transition_index]
    else:
        values = torch.zeros(
            dataset.static_data["xy_coords"].shape[0], dtype=torch.float
        )
        values[torch.as_tensor(edge_flow["selected_nodes"], dtype=torch.long)] = (
            torch.as_tensor(edge_flow[key][transition_index], dtype=torch.float)
        )
    return torch.as_tensor(values, dtype=torch.float)


def expanded_node_target(
    edge_targets: dict,
    target_name: str,
    transition_index: int,
    num_nodes: int,
    *,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """Expand a sharded selected-node target or read a legacy full target."""
    if target_name in edge_targets:
        return torch.as_tensor(
            edge_targets[target_name][transition_index],
            dtype=dtype,
            device=device,
        )
    selected_name = f"{target_name}_selected"
    if selected_name not in edge_targets or "selected_nodes" not in edge_targets:
        raise KeyError(f"No {target_name!r} target is available.")
    values = torch.zeros(num_nodes, dtype=dtype, device=device)
    selected_nodes = torch.as_tensor(
        edge_targets["selected_nodes"], dtype=torch.long, device=device
    )
    values[selected_nodes] = torch.as_tensor(
        edge_targets[selected_name][transition_index],
        dtype=dtype,
        device=device,
    )
    return values


def feedback_node_mask(graph, args) -> torch.Tensor:
    """Return nodes where projected volume is fed back into rollout state."""

    mode = args.feedback_mode
    device = args.device_obj
    if mode == "selected":
        return selected_node_mask(graph, args.projection_mode, device)
    if mode == "all":
        return torch.ones(graph.x.shape[0], dtype=torch.bool, device=device)

    if not hasattr(graph, "zone_label"):
        raise AttributeError(f"feedback_mode={mode!r} requires graph.zone_label")
    zone_label = graph.zone_label.to(device)
    if mode == "high":
        return zone_label == 3
    if mode in ("high_interior", "high_interior_control_volume"):
        mask = zone_label == 3
        if hasattr(graph, "hecras_boundary_node_mask"):
            mask = mask & (~graph.hecras_boundary_node_mask.to(device))
        return mask
    raise ValueError(f"Unknown feedback mode: {mode!r}")


def aggregate_high_interior_control_volumes(
    graph, values: torch.Tensor
) -> torch.Tensor:
    """Sum node values within each fixed high-interior connected component."""

    if not hasattr(graph, "hecras_high_interior_control_volume_label"):
        raise AttributeError(
            "Control-volume metrics require "
            "graph.hecras_high_interior_control_volume_label."
        )
    labels = graph.hecras_high_interior_control_volume_label.to(values.device)
    selected = labels >= 0
    aggregates = []
    for label in torch.unique(labels[selected]):
        aggregates.append(torch.sum(values[labels == label]))
    return torch.stack(aggregates)


def aggregate_weighted_high_interior_control_volumes(
    graph, values: torch.Tensor
) -> torch.Tensor:
    """Match the connected-control-volume training loss node-count weighting."""
    if not hasattr(graph, "hecras_high_interior_control_volume_label"):
        raise AttributeError(
            "Weighted control-volume metrics require "
            "graph.hecras_high_interior_control_volume_label."
        )
    labels = graph.hecras_high_interior_control_volume_label.to(values.device)
    selected = labels >= 0
    selected_labels = labels[selected].long()
    if selected_labels.numel() == 0:
        raise ValueError("No high-interior control-volume nodes are available.")
    num_components = int(torch.max(selected_labels).item()) + 1
    counts = torch.zeros(num_components, dtype=values.dtype, device=values.device)
    aggregates = torch.zeros_like(counts)
    counts.index_add_(0, selected_labels, torch.ones_like(values[selected]))
    aggregates.index_add_(0, selected_labels, values[selected])
    nonempty = counts > 0
    return aggregates[nonempty] / torch.sqrt(counts[nonempty])


def evaluate_rollout(model, edge_head, dataset, args) -> list[dict[str, float | str | int]]:
    rows = []
    model.eval()
    edge_head.eval()
    with torch.no_grad():
        for idx in range(len(dataset)):
            graph, rollout_data = dataset[idx]
            graph = graph.to(args.device_obj)
            x_iter = graph.x.to(args.device_obj)
            edge_features = graph.edge_attr.to(args.device_obj)
            zone_label = graph.zone_label.to(args.device_obj)
            volume_std = graph.volume_std.reshape(-1)[0].to(args.device_obj)

            inflow_seq = rollout_data["inflow"].to(args.device_obj)
            precip_seq = rollout_data["precipitation"].to(args.device_obj)
            local_precip_seq = rollout_data.get("local_precipitation")
            if local_precip_seq is not None:
                local_precip_seq = local_precip_seq.to(args.device_obj)
            wd_gt_seq = rollout_data["water_depth_gt"].to(args.device_obj)
            volume_gt_seq = rollout_data["volume_gt"].to(args.device_obj)
            sums: dict[str, float] = {}
            cg_fallbacks = 0
            hydrograph_id = dataset.hydrograph_ids[idx]
            ground_truth_previous_volume = x_iter[
                :, 12 + args.n_time_steps : 12 + 2 * args.n_time_steps
            ][:, -1]
            ground_truth_previous_water_depth = x_iter[
                :, 12 : 12 + args.n_time_steps
            ][:, -1]
            terrain_elevation = (
                graph.current_surface_elevation - graph.current_water_depth_denorm
            )
            water_depth_mean = graph.water_depth_mean.reshape(-1)[0].to(
                args.device_obj
            )
            water_depth_std = graph.water_depth_std.reshape(-1)[0].to(
                args.device_obj
            )
            ground_truth_previous_surface = terrain_elevation + (
                ground_truth_previous_water_depth * water_depth_std
                + water_depth_mean
            )
            edge_targets = dataset.hecras_edge_flow_delta_by_hydrograph.get(
                hydrograph_id
            )
            if not isinstance(edge_targets, dict):
                raise AttributeError(
                    "Formal edge evaluation requires paired internal, boundary, "
                    "and face targets."
                )
            face_src, face_dst = graph.hecras_face_index
            high_face_mask = (zone_label[face_src] == 3) | (
                zone_label[face_dst] == 3
            )
            high_high_face_mask = (zone_label[face_src] == 3) & (
                zone_label[face_dst] == 3
            )
            low_low_face_mask = (zone_label[face_src] != 3) & (
                zone_label[face_dst] != 3
            )
            control_volume_label = (
                graph.hecras_high_interior_control_volume_label.to(args.device_obj)
            )
            high_interior_touch_face_mask = (
                (control_volume_label[face_src] >= 0)
                | (control_volume_label[face_dst] >= 0)
            )
            high_interior_mask = (zone_label == 3) & (
                ~graph.hecras_boundary_node_mask.to(args.device_obj)
            )
            active_face_mask = edge_head.active_face_mask(graph)
            if not torch.any(active_face_mask):
                raise ValueError(
                    "The configured edge-head active face scope selected no faces."
                )
            predicted_depth_sequence = []
            predicted_volume_sequence = []
            for step in range(args.rollout_length):
                water_depth_window = x_iter[:, 12 : 12 + args.n_time_steps]
                volume_window = x_iter[
                    :, 12 + args.n_time_steps : 12 + 2 * args.n_time_steps
                ]
                local_source_delta = rollout_local_source_delta(
                    dataset,
                    x_iter,
                    precip_seq[step],
                    args.delta_t,
                    args.device_obj,
                    (
                        local_precip_seq[step]
                        if local_precip_seq is not None
                        else None
                    ),
                )
                attach_rollout_state_attrs(
                    graph,
                    x_iter,
                    local_source_delta,
                    args.n_time_steps,
                    args.delta_t,
                )
                pred = model(x_iter, edge_features, graph)
                pred_dataset_delta = pred[:, 1] * volume_std
                gt_dataset_delta_norm = (
                    volume_gt_seq[step] - ground_truth_previous_volume
                )
                gt_dataset_delta = gt_dataset_delta_norm * volume_std
                if args.storage_delta_mode == "area_extrapolated_volume":
                    pred_delta = compute_area_extrapolated_storage_delta(
                        pred,
                        graph,
                    )
                    gt_prediction = torch.stack(
                        [
                            wd_gt_seq[step] - ground_truth_previous_water_depth,
                            gt_dataset_delta_norm,
                        ],
                        dim=1,
                    )
                    gt_delta = compute_area_extrapolated_storage_delta(
                        gt_prediction,
                        graph,
                        current_surface_elevation=ground_truth_previous_surface,
                    )
                elif args.storage_delta_mode in ("dataset_volume", "capped_volume"):
                    pred_delta = pred_dataset_delta
                    gt_delta = gt_dataset_delta
                else:
                    raise ValueError(
                        f"Unknown storage_delta_mode: {args.storage_delta_mode!r}"
                    )
                ground_truth_previous_volume = volume_gt_seq[step]
                ground_truth_previous_water_depth = wd_gt_seq[step]
                ground_truth_previous_surface = terrain_elevation + (
                    ground_truth_previous_water_depth * water_depth_std
                    + water_depth_mean
                )
                new_wd = water_depth_window[:, -1] + pred[:, 0]
                original_new_volume = volume_window[:, -1] + pred[:, 1]

                edge_flux = edge_head(graph, pred).reshape(-1)
                boundary_source_delta = None
                if args.projection_target == "hecras_boundary_source":
                    boundary_source_delta = rollout_boundary_source_delta(
                        dataset,
                        hydrograph_id,
                        dataset.hecras_edge_flow_time_offset
                        + args.n_time_steps
                        - 1
                        + step,
                    ).to(args.device_obj)
                target = projection_target(
                    graph,
                    pred_delta,
                    args.projection_target,
                    args.device_obj,
                    boundary_source_delta,
                    local_source_delta,
                )
                if not hasattr(graph, "hecras_internal_face_flow_rms"):
                    raise AttributeError(
                        "Selective hard conservation requires train-split-only "
                        "per-face RMS statistics."
                    )
                hard_result = project_connected_control_volume_flux(
                    edge_flux,
                    target,
                    graph.hecras_face_index,
                    graph.hecras_high_interior_control_volume_label,
                    node_batch=getattr(graph, "batch", None),
                    active_face_mask=active_face_mask,
                    face_correction_weight=(
                        graph.hecras_internal_face_flow_rms.to(
                            edge_flux.device
                        ).reshape(-1).square()
                    ),
                    ridge=0.0,
                )
                projected_flux = hard_result.edge_flux.reshape(-1)
                correction_rms = float(
                    hard_result.face_correction.float().square().mean().sqrt().item()
                )

                original_divergence = compute_divergence(graph, edge_flux)
                projected_divergence = compute_divergence(graph, projected_flux)
                target_transition_index = (
                    dataset.hecras_edge_flow_time_offset
                    + args.n_time_steps
                    - 1
                    + step
                )
                true_internal_delta = expanded_node_target(
                    edge_targets,
                    "internal",
                    target_transition_index,
                    pred_delta.numel(),
                    dtype=pred_delta.dtype,
                    device=args.device_obj,
                )
                true_boundary_delta = expanded_node_target(
                    edge_targets,
                    "boundary",
                    target_transition_index,
                    pred_delta.numel(),
                    dtype=pred_delta.dtype,
                    device=args.device_obj,
                )
                true_face_delta = torch.as_tensor(
                    edge_targets["face"][target_transition_index],
                    dtype=edge_flux.dtype,
                    device=args.device_obj,
                )
                raw_hecras_face_delta = torch.as_tensor(
                    edge_targets.get("raw_face", edge_targets["face"])[
                        target_transition_index
                    ],
                    dtype=edge_flux.dtype,
                    device=args.device_obj,
                )
                manning_face_delta = manning_face_volume_heuristic(
                    graph, args.delta_t
                )
                additive_source_delta = torch.zeros_like(projected_divergence)
                if args.projection_target == "hecras_boundary_source":
                    additive_source_delta = boundary_source_delta
                elif args.projection_target == "initial_boundary_source":
                    additive_source_delta = graph.hecras_edge_boundary_source_delta.to(
                        args.device_obj
                    ).reshape(-1)
                elif args.projection_target == "known_local_source":
                    additive_source_delta = local_source_delta
                projected_total_delta = projected_divergence + additive_source_delta
                if args.storage_delta_mode == "area_extrapolated_volume":
                    extrapolated_storage_correction = pred_delta - pred_dataset_delta
                    projected_dataset_delta = (
                        projected_total_delta - extrapolated_storage_correction
                    )
                else:
                    projected_dataset_delta = projected_total_delta
                projected_volume_delta_norm = projected_dataset_delta / volume_std
                projected_new_volume = volume_window[:, -1] + projected_volume_delta_norm
                blended_projected_new_volume = (
                    original_new_volume
                    + args.projected_volume_alpha
                    * (projected_new_volume - original_new_volume)
                )
                feedback_mask = feedback_node_mask(graph, args)
                blended_projected_new_volume = torch.where(
                    feedback_mask,
                    blended_projected_new_volume,
                    original_new_volume,
                )
                blended_projected_volume_delta_norm = (
                    blended_projected_new_volume - volume_window[:, -1]
                )

                selected_new_volume = (
                    blended_projected_new_volume
                    if args.state_update == "projected_volume"
                    else original_new_volume
                )
                predicted_depth_sequence.append(new_wd)
                predicted_volume_sequence.append(selected_new_volume)
                wd_error = new_wd - wd_gt_seq[step]
                original_volume_error = original_new_volume - volume_gt_seq[step]
                projected_volume_error = projected_new_volume - volume_gt_seq[step]
                selected_volume_error = selected_new_volume - volume_gt_seq[step]
                original_combined_error = torch.stack(
                    [wd_error, original_volume_error], dim=1
                )
                projected_combined_error = torch.stack(
                    [wd_error, projected_volume_error], dim=1
                )
                selected_combined_error = torch.stack(
                    [wd_error, selected_volume_error], dim=1
                )
                add_metric(sums, "rollout_rmse", rmse(selected_combined_error))
                add_metric(sums, "rollout_wd_rmse", rmse(wd_error))
                add_metric(sums, "rollout_volume_rmse", rmse(selected_volume_error))
                add_metric(sums, "original_state_rollout_rmse", rmse(original_combined_error))
                add_metric(sums, "original_state_rollout_volume_rmse", rmse(original_volume_error))
                add_metric(sums, "projected_state_rollout_rmse", rmse(projected_combined_error))
                add_metric(sums, "projected_state_rollout_volume_rmse", rmse(projected_volume_error))
                add_residual_metrics(
                    sums,
                    "original_projection_closure",
                    original_divergence - target,
                    target,
                )
                add_residual_metrics(
                    sums,
                    "projected_projection_closure",
                    projected_divergence - target,
                    target,
                )
                control_volume_target = aggregate_high_interior_control_volumes(
                    graph, target
                )
                add_residual_metrics(
                    sums,
                    "control_volume_original_projection_closure",
                    aggregate_high_interior_control_volumes(
                        graph, original_divergence - target
                    ),
                    control_volume_target,
                )
                weighted_control_volume_target = (
                    aggregate_weighted_high_interior_control_volumes(
                        graph, target
                    )
                )
                add_residual_metrics(
                    sums,
                    "control_volume_weighted_original_projection_closure",
                    aggregate_weighted_high_interior_control_volumes(
                        graph, original_divergence - target
                    ),
                    weighted_control_volume_target,
                )
                add_residual_metrics(
                    sums,
                    "control_volume_projected_projection_closure",
                    aggregate_high_interior_control_volumes(
                        graph, projected_divergence - target
                    ),
                    control_volume_target,
                )
                add_residual_metrics(
                    sums,
                    "control_volume_weighted_projected_projection_closure",
                    aggregate_weighted_high_interior_control_volumes(
                        graph, projected_divergence - target
                    ),
                    weighted_control_volume_target,
                )
                add_residual_metrics(
                    sums,
                    "projection_delta",
                    projected_flux - edge_flux,
                    edge_flux,
                )
                add_metric(sums, "edge_flux_rms_m3", rmse(edge_flux))
                add_metric(sums, "projected_edge_flux_rms_m3", rmse(projected_flux))
                add_residual_metrics(
                    sums,
                    "edge_face_error",
                    edge_flux - true_face_delta,
                    true_face_delta,
                )
                add_residual_metrics(
                    sums,
                    "active_edge_face_error",
                    (edge_flux - true_face_delta)[active_face_mask],
                    true_face_delta[active_face_mask],
                )
                add_residual_metrics(
                    sums,
                    "high_interior_touch_edge_face_error",
                    (edge_flux - true_face_delta)[high_interior_touch_face_mask],
                    true_face_delta[high_interior_touch_face_mask],
                )
                add_residual_metrics(
                    sums,
                    "high_touch_edge_face_error",
                    (edge_flux - true_face_delta)[high_face_mask],
                    true_face_delta[high_face_mask],
                )
                add_residual_metrics(
                    sums,
                    "high_high_edge_face_error",
                    (edge_flux - true_face_delta)[high_high_face_mask],
                    true_face_delta[high_high_face_mask],
                )
                add_residual_metrics(
                    sums,
                    "hard_edge_face_error",
                    projected_flux - true_face_delta,
                    true_face_delta,
                )
                add_residual_metrics(
                    sums,
                    "hard_active_edge_face_error",
                    (projected_flux - true_face_delta)[active_face_mask],
                    true_face_delta[active_face_mask],
                )
                add_residual_metrics(
                    sums,
                    "hard_high_interior_touch_edge_face_error",
                    (projected_flux - true_face_delta)[
                        high_interior_touch_face_mask
                    ],
                    true_face_delta[high_interior_touch_face_mask],
                )
                add_residual_metrics(
                    sums,
                    "raw_hecras_high_interior_touch_edge_face_error",
                    (edge_flux - raw_hecras_face_delta)[
                        high_interior_touch_face_mask
                    ],
                    raw_hecras_face_delta[high_interior_touch_face_mask],
                )
                add_residual_metrics(
                    sums,
                    "raw_hecras_zero_flow_control_error",
                    -raw_hecras_face_delta[high_interior_touch_face_mask],
                    raw_hecras_face_delta[high_interior_touch_face_mask],
                )
                add_residual_metrics(
                    sums,
                    "raw_hecras_manning_control_error",
                    (manning_face_delta - raw_hecras_face_delta)[
                        high_interior_touch_face_mask
                    ],
                    raw_hecras_face_delta[high_interior_touch_face_mask],
                )
                add_residual_metrics(
                    sums,
                    "raw_hecras_hard_high_interior_touch_edge_face_error",
                    (projected_flux - raw_hecras_face_delta)[
                        high_interior_touch_face_mask
                    ],
                    raw_hecras_face_delta[high_interior_touch_face_mask],
                )
                add_residual_metrics(
                    sums,
                    "target_projection_high_interior_touch_face_correction",
                    (true_face_delta - raw_hecras_face_delta)[
                        high_interior_touch_face_mask
                    ],
                    raw_hecras_face_delta[high_interior_touch_face_mask],
                )
                directional_mask = high_interior_touch_face_mask & (
                    torch.abs(raw_hecras_face_delta) > 1.0e-6
                )
                if torch.any(directional_mask):
                    add_metric(
                        sums,
                        "raw_hecras_active_direction_accuracy",
                        torch.mean(
                            (
                                torch.sign(edge_flux[directional_mask])
                                == torch.sign(raw_hecras_face_delta[directional_mask])
                            ).to(torch.float32)
                        ),
                    )
                add_residual_metrics(
                    sums,
                    "internal_divergence_error",
                    original_divergence - true_internal_delta,
                    true_internal_delta,
                )
                add_residual_metrics(
                    sums,
                    "high_interior_internal_divergence_error",
                    (original_divergence - true_internal_delta)[high_interior_mask],
                    true_internal_delta[high_interior_mask],
                )
                true_storage_budget = (
                    local_source_delta + true_boundary_delta + true_internal_delta
                )
                add_residual_metrics(
                    sums,
                    "predicted_storage_budget",
                    pred_delta - true_storage_budget,
                    true_storage_budget,
                )
                add_residual_metrics(
                    sums,
                    "high_interior_predicted_storage_budget",
                    (pred_delta - true_storage_budget)[high_interior_mask],
                    true_storage_budget[high_interior_mask],
                )
                add_residual_metrics(
                    sums,
                    "ground_truth_storage_budget",
                    gt_delta - true_storage_budget,
                    true_storage_budget,
                )
                add_residual_metrics(
                    sums,
                    "high_interior_ground_truth_storage_budget",
                    (gt_delta - true_storage_budget)[high_interior_mask],
                    true_storage_budget[high_interior_mask],
                )
                add_residual_metrics(
                    sums,
                    "control_volume_predicted_storage_budget",
                    aggregate_high_interior_control_volumes(
                        graph, pred_delta - true_storage_budget
                    ),
                    aggregate_high_interior_control_volumes(
                        graph, true_storage_budget
                    ),
                )
                add_residual_metrics(
                    sums,
                    "control_volume_ground_truth_storage_budget",
                    aggregate_high_interior_control_volumes(
                        graph, gt_delta - true_storage_budget
                    ),
                    aggregate_high_interior_control_volumes(
                        graph, true_storage_budget
                    ),
                )
                if torch.any(low_low_face_mask):
                    low_low_correction = (
                        projected_flux - edge_flux
                    )[low_low_face_mask]
                    add_metric(
                        sums,
                        "low_low_projection_correction_rms_m3",
                        rmse(low_low_correction),
                    )
                    sums["low_low_projection_correction_max_abs_m3"] = sums.get(
                        "low_low_projection_correction_max_abs_m3", 0.0
                    ) + torch.max(torch.abs(low_low_correction)).detach().item()
                sums["projection_correction_rms_m3"] = (
                    sums.get("projection_correction_rms_m3", 0.0) + correction_rms
                )

                for zone in range(4):
                    node_mask = zone_label == zone
                    if torch.any(node_mask):
                        zone_state_error = selected_combined_error[node_mask]
                        add_metric(
                            sums,
                            f"zone_{zone}_rollout_rmse",
                            rmse(zone_state_error),
                        )
                        add_metric(
                            sums,
                            f"zone_{zone}_rollout_wd_rmse",
                            rmse(wd_error[node_mask]),
                        )
                        add_metric(
                            sums,
                            f"zone_{zone}_rollout_volume_rmse",
                            rmse(selected_volume_error[node_mask]),
                        )
                        add_metric(
                            sums,
                            f"zone_{zone}_original_state_rollout_rmse",
                            rmse(original_combined_error[node_mask]),
                        )
                        add_metric(
                            sums,
                            f"zone_{zone}_original_state_rollout_volume_rmse",
                            rmse(original_volume_error[node_mask]),
                        )
                        add_metric(
                            sums,
                            f"zone_{zone}_projected_state_rollout_rmse",
                            rmse(projected_combined_error[node_mask]),
                        )
                        add_metric(
                            sums,
                            f"zone_{zone}_projected_state_rollout_volume_rmse",
                            rmse(projected_volume_error[node_mask]),
                        )
                        add_residual_metrics(
                            sums,
                            f"zone_{zone}_original_projection_closure",
                            (original_divergence - target)[node_mask],
                            target[node_mask],
                        )
                        add_residual_metrics(
                            sums,
                            f"zone_{zone}_projected_projection_closure",
                            (projected_divergence - target)[node_mask],
                            target[node_mask],
                        )

                x_iter = update_rollout_state(
                    x_iter,
                    pred,
                    inflow_seq[step],
                    precip_seq[step],
                    args.n_time_steps,
                    blended_projected_volume_delta_norm
                    if args.state_update == "projected_volume"
                    else None,
                )

            row = finalize_metrics(sums, volume_unit_suffix="m3")
            for key in list(row):
                if not key.endswith(
                    (
                        "_relative_rmse",
                        "_rmse_m3",
                        "_mae_m3",
                        "_bias_m3",
                        "_target_rms_m3",
                    )
                ):
                    row[key] = row[key] / max(args.rollout_length, 1)
            row.update(
                compute_formal_forecast_metrics(
                    torch.stack(predicted_depth_sequence),
                    wd_gt_seq[: args.rollout_length],
                    torch.stack(predicted_volume_sequence),
                    volume_gt_seq[: args.rollout_length],
                    zone_label,
                    water_depth_mean=dataset.dynamic_stats["water_depth"]["mean"],
                    water_depth_std=dataset.dynamic_stats["water_depth"]["std"],
                    volume_mean=dataset.dynamic_stats["volume"]["mean"],
                    volume_std=dataset.dynamic_stats["volume"]["std"],
                    delta_t_seconds=args.delta_t,
                    wet_depth_threshold_m=args.wet_depth_threshold_m,
                )
            )
            row["hydrograph_id"] = hydrograph_id
            row["rollout_length"] = args.rollout_length
            row["projection_mode"] = args.projection_mode
            row["projection_target"] = args.projection_target
            row["projection_algorithm"] = (
                "differentiable_componentwise_training_face_rms_squared_exact"
            )
            row["storage_delta_mode"] = args.storage_delta_mode
            row["projection_ridge"] = 0.0
            row["state_update"] = args.state_update
            row["projected_volume_alpha"] = args.projected_volume_alpha
            row["feedback_mode"] = args.feedback_mode
            row["projection_high_weight"] = args.projection_high_weight
            row["projection_low_weight"] = args.projection_low_weight
            row["projection_solver"] = "closed_form_componentwise_torch"
            row["future_hecras_face_flow_used_for_inference"] = (
                args.projection_target
                in {"initial_boundary_source", "hecras_boundary_source"}
                or args.edge_head_use_previous_face_flow
            )
            row["edge_head_active_face_mode"] = edge_head.active_face_mode
            row["num_total_faces"] = int(active_face_mask.numel())
            row["num_active_faces"] = int(torch.sum(active_face_mask).item())
            row["active_face_fraction"] = float(
                torch.mean(active_face_mask.to(torch.float32)).item()
            )
            row["num_high_interior_touch_faces"] = int(
                torch.sum(high_interior_touch_face_mask).item()
            )
            row["zone_mask_sha256"] = args.zone_mask_sha256
            row["high_zone_code_label"] = 3
            row["num_high_zone_nodes"] = args.num_high_zone_nodes
            row["num_high_interior_nodes"] = args.num_high_interior_nodes
            row["num_high_interior_control_volumes"] = int(
                torch.unique(
                    graph.hecras_high_interior_control_volume_label[
                        graph.hecras_high_interior_control_volume_label >= 0
                    ]
                ).numel()
            )
            row["num_high_touch_faces"] = int(torch.sum(high_face_mask).item())
            row["num_high_high_faces"] = int(torch.sum(high_high_face_mask).item())
            row["num_low_low_faces"] = int(torch.sum(low_low_face_mask).item())
            row["cg_fallbacks"] = cg_fallbacks
            rows.append(row)
            print(
                f"{row['hydrograph_id']}: rollout_rmse={row['rollout_rmse']:.6f} "
                "control_volume_closure_rel="
                f"{row['control_volume_projected_projection_closure_relative_rmse']:.6f}"
            )
    return rows


def average_rows(rows: list[dict[str, float | str | int]]) -> dict[str, float | str | int]:
    summary: dict[str, float | str | int] = {
        "hydrograph_id": "MEAN",
        "rollout_length": rows[0]["rollout_length"],
        "projection_mode": rows[0]["projection_mode"],
        "projection_target": rows[0]["projection_target"],
        "projection_algorithm": rows[0]["projection_algorithm"],
        "storage_delta_mode": rows[0]["storage_delta_mode"],
        "projection_ridge": rows[0]["projection_ridge"],
        "state_update": rows[0]["state_update"],
        "projected_volume_alpha": rows[0]["projected_volume_alpha"],
        "feedback_mode": rows[0]["feedback_mode"],
        "projection_high_weight": rows[0]["projection_high_weight"],
        "projection_low_weight": rows[0]["projection_low_weight"],
        "projection_solver": rows[0]["projection_solver"],
        "edge_head_active_face_mode": rows[0]["edge_head_active_face_mode"],
        "num_total_faces": rows[0]["num_total_faces"],
        "num_active_faces": rows[0]["num_active_faces"],
        "active_face_fraction": rows[0]["active_face_fraction"],
        "num_high_interior_touch_faces": rows[0][
            "num_high_interior_touch_faces"
        ],
        "future_hecras_face_flow_used_for_inference": rows[0][
            "future_hecras_face_flow_used_for_inference"
        ],
        "zone_mask_sha256": rows[0]["zone_mask_sha256"],
        "high_zone_code_label": rows[0]["high_zone_code_label"],
        "num_high_zone_nodes": rows[0]["num_high_zone_nodes"],
        "num_high_interior_nodes": rows[0]["num_high_interior_nodes"],
        "num_high_interior_control_volumes": rows[0][
            "num_high_interior_control_volumes"
        ],
    }
    numeric_keys = [
        key for key, value in rows[0].items() if isinstance(value, (int, float))
    ]
    for key in numeric_keys:
        if key in summary:
            continue
        summary[key] = sum(float(row[key]) for row in rows) / max(len(rows), 1)
    return summary


def build_dataset(args: argparse.Namespace) -> HydroGraphDataset:
    eval_data_dir = args.test_data_dir or args.data_dir
    norm_stats_dir = args.train_data_dir or args.data_dir
    has_edge_targets = (
        args.hecras_edge_flow_npz is not None
        or args.hecras_conservative_edge_target_dir is not None
    )
    return HydroGraphDataset(
        data_dir=eval_data_dir,
        prefix=args.prefix,
        n_time_steps=args.n_time_steps,
        hydrograph_ids_file=args.ids_file,
        split="test",
        rollout_length=args.rollout_length,
        return_physics=False,
        use_fidelity_zones=True,
        zone_label_file=args.zone_label_file,
        zone_weight_file=args.zone_weight_file,
        norm_stats_dir=norm_stats_dir,
        return_hecras_face=True,
        hecras_face_graph_file=args.hecras_face_graph_file,
        return_hecras_edge_flow=has_edge_targets,
        hecras_edge_flow_npz=args.hecras_edge_flow_npz,
        hecras_conservative_edge_target_dir=(
            args.hecras_conservative_edge_target_dir
        ),
        hecras_edge_flow_mode=(
            "conservative_sharded"
            if args.hecras_conservative_edge_target_dir is not None
            else "internal_plus_boundary_source"
        ),
        hecras_edge_flow_face_stats_npz=args.hecras_edge_flow_face_stats_npz,
        hecras_edge_flow_scale_stats_npz=args.hecras_edge_flow_scale_stats_npz,
        precipitation_unit_conversion=args.precipitation_unit_conversion,
        local_source_runoff_mode=args.local_source_runoff_mode,
        require_node_precipitation=True,
        hecras_edge_flow_time_offset=args.hecras_edge_flow_time_offset,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--test-data-dir")
    parser.add_argument("--train-data-dir")
    parser.add_argument("--ids-file", required=True)
    parser.add_argument(
        "--split-role", choices=("validation", "test"), default="validation"
    )
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--test-authorization-file", type=Path)
    parser.add_argument("--enforce-formal-split-contract", action="store_true")
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--rollout-length", type=int, default=10)
    parser.add_argument("--n-time-steps", type=int, default=2)
    parser.add_argument("--delta-t", type=float, default=300.0)
    parser.add_argument("--wet-depth-threshold-m", type=float, default=0.01)
    parser.add_argument("--precipitation-unit-conversion", type=float, default=2.7778e-7)
    parser.add_argument(
        "--local-source-runoff-mode",
        choices=("ip_fraction", "full_area"),
        default="ip_fraction",
    )
    parser.add_argument("--hecras-edge-flow-time-offset", type=int)
    parser.add_argument(
        "--storage-delta-mode",
        choices=("dataset_volume", "capped_volume", "area_extrapolated_volume"),
        default="dataset_volume",
    )
    parser.add_argument("--zone-label-file", default="zone_label.txt")
    parser.add_argument("--zone-weight-file", default="zone_weight.txt")
    parser.add_argument("--num-input-features", type=int, default=16)
    parser.add_argument("--num-edge-features", type=int, default=3)
    parser.add_argument("--num-output-features", type=int, default=2)
    parser.add_argument("--edge-head-hidden-dim", type=int, default=256)
    parser.add_argument("--edge-head-num-hidden-layers", type=int, default=4)
    parser.add_argument("--edge-head-scale", type=float, default=1.0)
    parser.add_argument("--edge-head-use-face-normal", action="store_true")
    parser.add_argument("--edge-head-use-surface-features", action="store_true")
    parser.add_argument("--edge-head-use-physical-surface-features", action="store_true")
    parser.add_argument("--edge-head-use-previous-face-flow", action="store_true")
    parser.add_argument("--edge-head-use-edge-physical-features", action="store_true")
    parser.add_argument("--edge-head-use-node-prediction-features", action="store_true")
    parser.add_argument("--edge-head-output-mode", default="asinh_per_face_rms")
    parser.add_argument(
        "--edge-head-active-face-mode",
        choices=("all", "zone4_touch", "high_interior_touch"),
        default="all",
    )
    parser.add_argument("--hecras-face-graph-file", required=True)
    parser.add_argument("--hecras-edge-flow-npz")
    parser.add_argument("--hecras-conservative-edge-target-dir")
    parser.add_argument("--hecras-edge-flow-face-stats-npz")
    parser.add_argument("--hecras-edge-flow-scale-stats-npz")
    parser.add_argument("--projection-mode", default="all")
    parser.add_argument(
        "--projection-target",
        default="node_delta",
        choices=(
            "node_delta",
            "initial_boundary_source",
            "hecras_boundary_source",
            "known_local_source",
        ),
    )
    parser.add_argument("--projection-ridge", type=float, default=0.1)
    parser.add_argument("--projection-high-weight", type=float, default=1.0)
    parser.add_argument("--projection-low-weight", type=float, default=1.0)
    parser.add_argument(
        "--projection-solver",
        default="cg",
        choices=("cg", "factorized"),
        help="Sparse solver used for edge-flux projection.",
    )
    parser.add_argument(
        "--state-update",
        default="original",
        choices=("original", "projected_volume"),
    )
    parser.add_argument(
        "--projected-volume-alpha",
        type=float,
        default=1.0,
        help=(
            "Blend factor used when --state-update=projected_volume. "
            "The rollout volume is updated as original + alpha * "
            "(projected - original)."
        ),
    )
    parser.add_argument(
        "--feedback-mode",
        default="selected",
        choices=(
            "selected",
            "all",
            "high",
            "high_interior",
            "high_interior_control_volume",
        ),
        help=(
            "Nodes whose rollout volume state receives the projected-volume "
            "feedback. 'selected' preserves the historical behavior and uses "
            "the projection-mode mask."
        ),
    )
    parser.add_argument("--cg-rtol", type=float, default=1e-8)
    parser.add_argument("--cg-maxiter", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--node-checkpoint-path", required=True, type=Path)
    parser.add_argument("--node-checkpoint-epoch", type=int, default=0)
    parser.add_argument("--edge-checkpoint-path", required=True, type=Path)
    parser.add_argument("--edge-checkpoint-epoch", type=int, default=0)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    if args.enforce_formal_split_contract:
        verify_split_contract(args)
    args.device_obj = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if args.projection_mode != "high_interior_control_volume":
        raise ValueError(
            "Selective hard conservation requires "
            "projection-mode=high_interior_control_volume."
        )
    if args.projection_target != "known_local_source":
        raise ValueError(
            "Selective hard conservation requires "
            "projection-target=known_local_source."
        )
    if args.projection_target == "known_local_source" and args.projection_mode not in {
        "interior",
        "high_interior",
        "high_interior_control_volume",
    }:
        raise ValueError(
            "known_local_source excludes external-boundary Face Flow and therefore "
            "requires projection-mode=interior, high_interior, or "
            "high_interior_control_volume."
        )

    dataset = build_dataset(args)
    zone_path = Path(args.zone_label_file)
    if not zone_path.is_absolute():
        zone_path = Path(args.test_data_dir or args.data_dir) / zone_path
    args.zone_mask_sha256 = hashlib.sha256(zone_path.read_bytes()).hexdigest()
    args.num_high_zone_nodes = int((dataset.zone_label == 3).sum())
    if dataset.hecras_boundary_node_mask is None:
        args.num_high_interior_nodes = args.num_high_zone_nodes
    else:
        args.num_high_interior_nodes = int(
            ((dataset.zone_label == 3) & (~dataset.hecras_boundary_node_mask)).sum()
        )
    model = MeshGraphKAN(
        args.num_input_features,
        args.num_edge_features,
        args.num_output_features,
    ).to(args.device_obj)
    edge_head = HecRasEdgeFluxHead(
        args.num_input_features,
        hidden_dim=args.edge_head_hidden_dim,
        num_hidden_layers=args.edge_head_num_hidden_layers,
        scale=args.edge_head_scale,
        use_face_normal=args.edge_head_use_face_normal,
        use_surface_features=args.edge_head_use_surface_features,
        use_physical_surface_features=args.edge_head_use_physical_surface_features,
        use_previous_face_flow=args.edge_head_use_previous_face_flow,
        use_edge_physical_features=args.edge_head_use_edge_physical_features,
        use_node_prediction_features=args.edge_head_use_node_prediction_features,
        output_mode=args.edge_head_output_mode,
        active_face_mode=args.edge_head_active_face_mode,
    ).to(args.device_obj)
    loaded_node_epoch = load_checkpoint(
        args.node_checkpoint_path,
        models=[model],
        epoch=args.node_checkpoint_epoch,
        device=args.device_obj,
    )
    loaded_edge_epoch = load_checkpoint(
        args.edge_checkpoint_path,
        models=[edge_head],
        epoch=args.edge_checkpoint_epoch,
        device=args.device_obj,
    )

    rows = evaluate_rollout(model, edge_head, dataset, args)
    summary = average_rows(rows)
    summary["hydrograph_id"] = "MEAN"
    summary["loaded_node_epoch"] = loaded_node_epoch
    summary["loaded_edge_epoch"] = loaded_edge_epoch
    rows.append(summary)

    fieldnames = sorted({key for row in rows for key in row})
    lead = [
        "hydrograph_id",
        "rollout_length",
        "projection_mode",
        "projection_target",
        "projection_ridge",
        "state_update",
        "projected_volume_alpha",
        "feedback_mode",
        "projection_high_weight",
        "projection_low_weight",
        "projection_solver",
    ]
    for key in lead:
        if key in fieldnames:
            fieldnames.remove(key)
    fieldnames = lead + fieldnames
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
