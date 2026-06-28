# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Plot spatial local-conservation residual maps for Minxiong rollouts."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

from evaluate_pretrained_edge_flux_head import compute_divergence
from evaluate_rollout_projected_edge_flux import (
    attach_rollout_x,
    build_dataset,
    projection_target,
    rollout_boundary_source_delta,
    update_rollout_state,
)
from hecras_projection import (
    build_projection_solver,
    project_edge_flux,
    selected_node_mask,
)
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.utils import load_checkpoint
from train import HecRasEdgeFluxHead


def build_models(args):
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
    load_checkpoint(
        args.node_checkpoint_path,
        models=[model],
        epoch=args.node_checkpoint_epoch,
        device=args.device_obj,
    )
    load_checkpoint(
        args.edge_checkpoint_path,
        models=[edge_head],
        epoch=args.edge_checkpoint_epoch,
        device=args.device_obj,
    )
    model.eval()
    edge_head.eval()
    return model, edge_head


def find_event_index(dataset, event_id: str) -> int:
    for idx, hydrograph_id in enumerate(dataset.hydrograph_ids):
        if hydrograph_id == event_id:
            return idx
    raise ValueError(f"{event_id!r} not found in {dataset.hydrograph_ids}")


def compute_spatial_residuals(model, edge_head, dataset, args):
    idx = find_event_index(dataset, args.event_id)
    graph, rollout_data = dataset[idx]
    graph = graph.to(args.device_obj)
    x_iter = graph.x.to(args.device_obj)
    initial_x = x_iter.detach().cpu()
    edge_features = graph.edge_attr.to(args.device_obj)
    zone_label = graph.zone_label.to(args.device_obj)
    volume_std = graph.volume_std.reshape(-1)[0].to(args.device_obj)
    inflow_seq = rollout_data["inflow"].to(args.device_obj)
    precip_seq = rollout_data["precipitation"].to(args.device_obj)

    projection_solver = build_projection_solver(
        graph,
        args.projection_mode,
        args.device_obj,
        args.projection_ridge,
        high_weight=args.projection_high_weight,
        low_weight=args.projection_low_weight,
        solver=args.projection_solver,
    )

    original_sq = torch.zeros(graph.x.shape[0], device=args.device_obj)
    projected_sq = torch.zeros_like(original_sq)
    target_sq = torch.zeros_like(original_sq)
    original_abs_sum = torch.zeros_like(original_sq)
    projected_abs_sum = torch.zeros_like(original_sq)

    with torch.no_grad():
        for step in range(args.rollout_length):
            water_depth_window = x_iter[:, 12 : 12 + args.n_time_steps]
            volume_window = x_iter[
                :, 12 + args.n_time_steps : 12 + 2 * args.n_time_steps
            ]
            attach_rollout_x(graph, x_iter)
            pred = model(x_iter, edge_features, graph)
            pred_delta = pred[:, 1] * volume_std
            original_new_volume = volume_window[:, -1] + pred[:, 1]
            edge_flux = edge_head(graph).reshape(-1)
            boundary_source_delta = None
            if args.projection_target == "hecras_boundary_source":
                boundary_source_delta = rollout_boundary_source_delta(
                    dataset,
                    args.event_id,
                    args.n_time_steps - 1 + step,
                ).to(args.device_obj)
            target = projection_target(
                graph,
                pred_delta,
                args.projection_target,
                args.device_obj,
                boundary_source_delta,
            )
            projected_flux, _, _ = project_edge_flux(
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
            original_residual = compute_divergence(graph, edge_flux) - target
            projected_residual = compute_divergence(graph, projected_flux) - target
            original_sq += original_residual.square()
            projected_sq += projected_residual.square()
            target_sq += target.square()
            original_abs_sum += original_residual.abs()
            projected_abs_sum += projected_residual.abs()

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

    steps = max(args.rollout_length, 1)
    return {
        "xy": initial_x[:, :2].numpy(),
        "zone_label": zone_label.detach().cpu().numpy(),
        "original_rms": torch.sqrt(original_sq / steps).detach().cpu().numpy(),
        "projected_rms": torch.sqrt(projected_sq / steps).detach().cpu().numpy(),
        "target_rms": torch.sqrt(target_sq / steps).detach().cpu().numpy(),
        "original_mean_abs": (original_abs_sum / steps).detach().cpu().numpy(),
        "projected_mean_abs": (projected_abs_sum / steps).detach().cpu().numpy(),
    }


def save_spatial_plot(data: dict[str, np.ndarray], args) -> None:
    xy = data["xy"]
    zone_label = data["zone_label"]
    original = data["original_rms"]
    projected = data["projected_rms"]
    eps = 1e-6
    original_log = np.log10(original + eps)
    projected_log = np.log10(projected + eps)
    improvement = np.log10((original + eps) / (projected + eps))
    vmin = float(np.nanpercentile(np.concatenate([original_log, projected_log]), 2))
    vmax = float(np.nanpercentile(np.concatenate([original_log, projected_log]), 98))
    zone3 = zone_label == 3

    fig, axes = plt.subplots(1, 3, figsize=(18, 5.8), constrained_layout=True)
    panels = [
        (original_log, "Original closure residual", "log10 RMS ft^3", "magma", vmin, vmax),
        (
            projected_log,
            "Zone-weighted projected residual",
            "log10 RMS ft^3",
            "magma",
            vmin,
            vmax,
        ),
        (
            improvement,
            "Improvement: original / projected",
            "log10 ratio",
            "coolwarm",
            -2.0,
            2.0,
        ),
    ]
    for ax, (values, title, cbar_label, cmap, local_vmin, local_vmax) in zip(
        axes, panels
    ):
        scatter = ax.scatter(
            xy[:, 0],
            xy[:, 1],
            c=values,
            s=8,
            cmap=cmap,
            vmin=local_vmin,
            vmax=local_vmax,
            linewidths=0,
        )
        ax.scatter(
            xy[zone3, 0],
            xy[zone3, 1],
            facecolors="none",
            edgecolors="cyan",
            s=16,
            linewidths=0.45,
            label="Zone 3",
        )
        ax.set_title(title)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.legend(loc="upper right", fontsize=8, frameon=True)
        cbar = fig.colorbar(scatter, ax=ax, fraction=0.045, pad=0.02)
        cbar.set_label(cbar_label)

    output_path = args.output_dir / (
        f"minxiong_{args.event_id}_zone_weighted_spatial_residual.png"
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200)
    plt.close(fig)
    print(f"Wrote {output_path}")


def write_node_summary(data: dict[str, np.ndarray], args) -> None:
    output_path = args.output_dir / (
        f"minxiong_{args.event_id}_zone_weighted_spatial_residual_summary.csv"
    )
    xy = data["xy"]
    zone_label = data["zone_label"]
    original = data["original_rms"]
    projected = data["projected_rms"]
    improvement = (original + 1e-6) / (projected + 1e-6)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        handle.write(
            "node_id,x,y,zone,original_rms_ft3,projected_rms_ft3,improvement_ratio\n"
        )
        for idx in range(xy.shape[0]):
            handle.write(
                f"{idx},{xy[idx,0]},{xy[idx,1]},{int(zone_label[idx])},"
                f"{original[idx]},{projected[idx]},{improvement[idx]}\n"
            )
    print(f"Wrote {output_path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--test-data-dir")
    parser.add_argument("--train-data-dir")
    parser.add_argument("--ids-file", required=True)
    parser.add_argument("--event-id", default="H25")
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--rollout-length", type=int, default=23)
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
    parser.add_argument("--projection-mode", default="zone_weighted")
    parser.add_argument(
        "--projection-target",
        default="hecras_boundary_source",
        choices=("node_delta", "initial_boundary_source", "hecras_boundary_source"),
    )
    parser.add_argument("--projection-ridge", type=float, default=0.1)
    parser.add_argument("--projection-high-weight", type=float, default=100.0)
    parser.add_argument("--projection-low-weight", type=float, default=1.0)
    parser.add_argument("--projection-solver", default="factorized", choices=("cg", "factorized"))
    parser.add_argument("--state-update", default="projected_volume")
    parser.add_argument("--projected-volume-alpha", type=float, default=0.05)
    parser.add_argument("--cg-rtol", type=float, default=1e-8)
    parser.add_argument("--cg-maxiter", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--node-checkpoint-path", required=True, type=Path)
    parser.add_argument("--node-checkpoint-epoch", type=int, default=19)
    parser.add_argument("--edge-checkpoint-path", required=True, type=Path)
    parser.add_argument("--edge-checkpoint-epoch", type=int, default=19)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/8tb_hdd2/joyce/Report/assets"),
    )
    args = parser.parse_args()
    args.device_obj = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(args)
    model, edge_head = build_models(args)
    data = compute_spatial_residuals(model, edge_head, dataset, args)
    save_spatial_plot(data, args)
    write_node_summary(data, args)


if __name__ == "__main__":
    main()
