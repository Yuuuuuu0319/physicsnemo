# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Rollout diagnostic for projected HEC-RAS edge fluxes.

This script evaluates, at each autoregressive step, whether the edge head can be
projected to close the node-model volume delta.  It can either keep the
HydroGraphNet state rollout unchanged or feed a damped projected-volume update
back into selected conservation-projection nodes.
"""

import argparse
import csv
from pathlib import Path

import torch

from evaluate_pretrained_edge_flux_head import (
    add_metric,
    add_residual_metrics,
    compute_divergence,
    finalize_metrics,
    rmse,
)
from hecras_projection import (
    build_projection_solver,
    project_edge_flux,
    selected_node_mask,
)
from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.utils import load_checkpoint
from train import HecRasEdgeFluxHead


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


def attach_rollout_x(graph, x_iter: torch.Tensor) -> None:
    graph.x = x_iter


def projection_target(
    graph,
    pred_delta: torch.Tensor,
    mode: str,
    device: torch.device,
    boundary_source_delta: torch.Tensor | None = None,
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
    raise ValueError(f"Unknown projection target: {mode!r}")


def rollout_boundary_source_delta(dataset, hydrograph_id: str, transition_index: int):
    edge_flow = getattr(dataset, "hecras_edge_flow_delta_by_hydrograph", {}).get(
        hydrograph_id
    )
    if not isinstance(edge_flow, dict) or "boundary" not in edge_flow:
        raise AttributeError(
            "projection-target=hecras_boundary_source requires an edge-flow NPZ "
            "with split internal/boundary arrays."
        )
    if transition_index < 0 or transition_index >= edge_flow["boundary"].shape[0]:
        raise IndexError(
            f"Boundary/source transition {transition_index} is outside "
            f"0..{edge_flow['boundary'].shape[0] - 1} for {hydrograph_id}."
        )
    return torch.tensor(edge_flow["boundary"][transition_index], dtype=torch.float)


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
            wd_gt_seq = rollout_data["water_depth_gt"].to(args.device_obj)
            volume_gt_seq = rollout_data["volume_gt"].to(args.device_obj)
            projection_solver = build_projection_solver(
                graph,
                args.projection_mode,
                args.device_obj,
                args.projection_ridge,
                high_weight=args.projection_high_weight,
                low_weight=args.projection_low_weight,
                solver=args.projection_solver,
            )

            sums: dict[str, float] = {}
            cg_fallbacks = 0
            hydrograph_id = dataset.hydrograph_ids[idx]
            for step in range(args.rollout_length):
                water_depth_window = x_iter[:, 12 : 12 + args.n_time_steps]
                volume_window = x_iter[
                    :, 12 + args.n_time_steps : 12 + 2 * args.n_time_steps
                ]
                attach_rollout_x(graph, x_iter)
                pred = model(x_iter, edge_features, graph)
                pred_delta = pred[:, 1] * volume_std
                new_wd = water_depth_window[:, -1] + pred[:, 0]
                original_new_volume = volume_window[:, -1] + pred[:, 1]

                edge_flux = edge_head(graph).reshape(-1)
                boundary_source_delta = None
                if args.projection_target == "hecras_boundary_source":
                    boundary_source_delta = rollout_boundary_source_delta(
                        dataset,
                        hydrograph_id,
                        args.n_time_steps - 1 + step,
                    ).to(args.device_obj)
                target = projection_target(
                    graph,
                    pred_delta,
                    args.projection_target,
                    args.device_obj,
                    boundary_source_delta,
                )
                projected_flux, cg_info, correction_rms = project_edge_flux(
                    graph,
                    edge_flux,
                    target,
                    mode=args.projection_mode,
                    ridge=args.projection_ridge,
                    cg_rtol=args.cg_rtol,
                    cg_maxiter=args.cg_maxiter,
                    high_weight=args.projection_high_weight,
                    low_weight=args.projection_low_weight,
                    projection_solver=projection_solver,
                )
                if cg_info != 0:
                    cg_fallbacks += 1

                original_divergence = compute_divergence(graph, edge_flux)
                projected_divergence = compute_divergence(graph, projected_flux)
                if boundary_source_delta is None:
                    projected_total_delta = projected_divergence
                else:
                    projected_total_delta = projected_divergence + boundary_source_delta
                projected_volume_delta_norm = projected_total_delta / volume_std
                projected_new_volume = volume_window[:, -1] + projected_volume_delta_norm
                blended_projected_new_volume = (
                    original_new_volume
                    + args.projected_volume_alpha
                    * (projected_new_volume - original_new_volume)
                )
                feedback_node_mask = selected_node_mask(
                    graph, args.projection_mode, args.device_obj
                )
                blended_projected_new_volume = torch.where(
                    feedback_node_mask,
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
                add_residual_metrics(
                    sums,
                    "projection_delta",
                    projected_flux - edge_flux,
                    edge_flux,
                )
                add_metric(sums, "edge_flux_rms_ft3", rmse(edge_flux))
                add_metric(sums, "projected_edge_flux_rms_ft3", rmse(projected_flux))
                sums["projection_correction_rms_ft3"] = (
                    sums.get("projection_correction_rms_ft3", 0.0) + correction_rms
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

            row = finalize_metrics(sums)
            for key in list(row):
                if not key.endswith(
                    (
                        "_relative_rmse",
                        "_rmse_ft3",
                        "_mae_ft3",
                        "_bias_ft3",
                        "_target_rms_ft3",
                    )
                ):
                    row[key] = row[key] / max(args.rollout_length, 1)
            row["hydrograph_id"] = hydrograph_id
            row["rollout_length"] = args.rollout_length
            row["projection_mode"] = args.projection_mode
            row["projection_target"] = args.projection_target
            row["projection_ridge"] = args.projection_ridge
            row["state_update"] = args.state_update
            row["projected_volume_alpha"] = args.projected_volume_alpha
            row["projection_high_weight"] = args.projection_high_weight
            row["projection_low_weight"] = args.projection_low_weight
            row["projection_solver"] = args.projection_solver
            row["cg_fallbacks"] = cg_fallbacks
            rows.append(row)
            print(
                f"{row['hydrograph_id']}: rollout_rmse={row['rollout_rmse']:.6f} "
                f"closure_rel={row['projected_projection_closure_relative_rmse']:.6f}"
            )
    return rows


def average_rows(rows: list[dict[str, float | str | int]]) -> dict[str, float | str | int]:
    summary: dict[str, float | str | int] = {
        "hydrograph_id": "MEAN",
        "rollout_length": rows[0]["rollout_length"],
        "projection_mode": rows[0]["projection_mode"],
        "projection_target": rows[0]["projection_target"],
        "projection_ridge": rows[0]["projection_ridge"],
        "state_update": rows[0]["state_update"],
        "projected_volume_alpha": rows[0]["projected_volume_alpha"],
        "projection_high_weight": rows[0]["projection_high_weight"],
        "projection_low_weight": rows[0]["projection_low_weight"],
        "projection_solver": rows[0]["projection_solver"],
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
        return_hecras_edge_flow=args.hecras_edge_flow_npz is not None,
        hecras_edge_flow_npz=args.hecras_edge_flow_npz,
        hecras_edge_flow_mode="internal_plus_boundary_source",
        hecras_edge_flow_face_stats_npz=args.hecras_edge_flow_face_stats_npz,
        hecras_edge_flow_scale_stats_npz=args.hecras_edge_flow_scale_stats_npz,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--test-data-dir")
    parser.add_argument("--train-data-dir")
    parser.add_argument("--ids-file", required=True)
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--rollout-length", type=int, default=10)
    parser.add_argument("--n-time-steps", type=int, default=2)
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
    parser.add_argument("--edge-head-output-mode", default="asinh_face_transition_rms")
    parser.add_argument("--hecras-face-graph-file", required=True)
    parser.add_argument("--hecras-edge-flow-npz")
    parser.add_argument("--hecras-edge-flow-face-stats-npz")
    parser.add_argument("--hecras-edge-flow-scale-stats-npz")
    parser.add_argument("--projection-mode", default="all")
    parser.add_argument(
        "--projection-target",
        default="node_delta",
        choices=("node_delta", "initial_boundary_source", "hecras_boundary_source"),
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
    parser.add_argument("--cg-rtol", type=float, default=1e-8)
    parser.add_argument("--cg-maxiter", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--node-checkpoint-path", required=True, type=Path)
    parser.add_argument("--node-checkpoint-epoch", type=int, default=0)
    parser.add_argument("--edge-checkpoint-path", required=True, type=Path)
    parser.add_argument("--edge-checkpoint-epoch", type=int, default=0)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    args.device_obj = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(args)
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
        output_mode=args.edge_head_output_mode,
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
