# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate projection targets available from a HydroGraphNet node model.

Unlike ``evaluate_projected_edge_flux_head.py``, this script can project edge
fluxes to the node-model closure target:

``predicted volume delta - boundary/source delta``.

That target is inference-available if forcings are known, so it is the next
step beyond the oracle HEC-RAS internal-delta projection diagnostic.
"""

import argparse
import csv
from pathlib import Path

import torch

from evaluate_pretrained_edge_flux_head import (
    add_metric,
    add_residual_metrics,
    build_dataset,
    compute_divergence,
    finalize_metrics,
    rmse,
)
from hecras_projection import project_edge_flux
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.utils import load_checkpoint
from train import HecRasEdgeFluxHead


def projection_target(graph, pred_delta: torch.Tensor, mode: str, device) -> torch.Tensor:
    if mode == "closure":
        boundary_source = graph.hecras_edge_boundary_source_delta.to(device).reshape(-1)
        return pred_delta.reshape(-1) - boundary_source
    if mode == "hecras_internal":
        return graph.hecras_edge_internal_delta.to(device).reshape(-1)
    raise ValueError(f"Unknown projection target mode: {mode!r}")


def evaluate(model: MeshGraphKAN, edge_head: HecRasEdgeFluxHead, dataset, args) -> dict:
    sums: dict[str, float] = {}
    count = 0
    cg_fallbacks = 0
    model.eval()
    edge_head.eval()
    with torch.no_grad():
        for graph in dataset:
            graph = graph.to(args.device_obj)
            pred = model(graph.x, graph.edge_attr, graph)
            volume_std = graph.volume_std.reshape(-1)[0].to(args.device_obj)
            pred_delta = pred[:, 1] * volume_std

            edge_flux = edge_head(graph).reshape(-1)
            face_target = graph.hecras_internal_face_flow_delta.to(
                args.device_obj
            ).reshape(-1)
            internal_target = graph.hecras_edge_internal_delta.to(
                args.device_obj
            ).reshape(-1)
            boundary_source = graph.hecras_edge_boundary_source_delta.to(
                args.device_obj
            ).reshape(-1)
            total_target = internal_target + boundary_source
            project_target = projection_target(
                graph, pred_delta, args.projection_target, args.device_obj
            )
            projected_flux, cg_info, correction_rms = project_edge_flux(
                graph,
                edge_flux,
                project_target,
                mode=args.projection_mode,
                ridge=args.projection_ridge,
                cg_rtol=args.cg_rtol,
                cg_maxiter=args.cg_maxiter,
            )
            if cg_info != 0:
                cg_fallbacks += 1

            original_divergence = compute_divergence(graph, edge_flux)
            projected_divergence = compute_divergence(graph, projected_flux)
            closure_target = pred_delta.reshape(-1) - boundary_source

            add_residual_metrics(sums, "face", projected_flux - face_target, face_target)
            add_residual_metrics(
                sums, "original_face", edge_flux - face_target, face_target
            )
            add_residual_metrics(
                sums,
                "internal_divergence",
                projected_divergence - internal_target,
                internal_target,
            )
            add_residual_metrics(
                sums,
                "original_internal_divergence",
                original_divergence - internal_target,
                internal_target,
            )
            add_residual_metrics(
                sums,
                "closure",
                projected_divergence - closure_target,
                closure_target,
            )
            add_residual_metrics(
                sums,
                "original_closure",
                original_divergence - closure_target,
                closure_target,
            )
            add_residual_metrics(sums, "node_total", pred_delta - total_target, total_target)
            add_residual_metrics(
                sums, "projection_delta", projected_flux - edge_flux, edge_flux
            )
            add_metric(sums, "edge_flux_rms_ft3", rmse(projected_flux))
            add_metric(sums, "original_edge_flux_rms_ft3", rmse(edge_flux))
            sums["projection_correction_rms_ft3"] = (
                sums.get("projection_correction_rms_ft3", 0.0) + correction_rms
            )

            if hasattr(graph, "zone_label"):
                zone_label = graph.zone_label.to(args.device_obj)
                src, dst = graph.hecras_face_index.to(args.device_obj)
                for zone in range(4):
                    node_mask = zone_label == zone
                    if torch.any(node_mask):
                        add_residual_metrics(
                            sums,
                            f"zone_{zone}_internal_divergence",
                            (projected_divergence - internal_target)[node_mask],
                            internal_target[node_mask],
                        )
                        add_residual_metrics(
                            sums,
                            f"zone_{zone}_original_internal_divergence",
                            (original_divergence - internal_target)[node_mask],
                            internal_target[node_mask],
                        )
                        add_residual_metrics(
                            sums,
                            f"zone_{zone}_closure",
                            (projected_divergence - closure_target)[node_mask],
                            closure_target[node_mask],
                        )
                        add_residual_metrics(
                            sums,
                            f"zone_{zone}_original_closure",
                            (original_divergence - closure_target)[node_mask],
                            closure_target[node_mask],
                        )
                    face_mask = (zone_label[src] == zone) | (zone_label[dst] == zone)
                    if torch.any(face_mask):
                        add_residual_metrics(
                            sums,
                            f"zone_{zone}_face",
                            (projected_flux - face_target)[face_mask],
                            face_target[face_mask],
                        )
                        add_residual_metrics(
                            sums,
                            f"zone_{zone}_original_face",
                            (edge_flux - face_target)[face_mask],
                            face_target[face_mask],
                        )
            count += 1

    rows = finalize_metrics(sums)
    for key in (
        "edge_flux_rms_ft3",
        "original_edge_flux_rms_ft3",
        "projection_correction_rms_ft3",
    ):
        if key in rows:
            rows[key] = rows[key] / max(count, 1)
    rows["cg_fallbacks"] = cg_fallbacks
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--ids-file", required=True)
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--num-samples", type=int, default=120)
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
    parser.add_argument("--hecras-edge-flow-npz", required=True)
    parser.add_argument("--hecras-edge-flow-face-stats-npz")
    parser.add_argument("--hecras-edge-flow-scale-stats-npz")
    parser.add_argument("--projection-mode", default="high")
    parser.add_argument(
        "--projection-target",
        default="closure",
        choices=("closure", "hecras_internal"),
    )
    parser.add_argument("--projection-ridge", type=float, default=0.1)
    parser.add_argument("--cg-rtol", type=float, default=1e-8)
    parser.add_argument("--cg-maxiter", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--node-checkpoint-path", required=True, type=Path)
    parser.add_argument("--node-checkpoint-epoch", type=int, default=0)
    parser.add_argument("--edge-checkpoint-path", required=True, type=Path)
    parser.add_argument("--edge-checkpoint-epoch", type=int, default=0)
    parser.add_argument("--label", default="nonoracle_projection")
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
    row = {
        "label": args.label,
        "node_checkpoint": str(args.node_checkpoint_path),
        "requested_node_epoch": args.node_checkpoint_epoch,
        "loaded_node_epoch": loaded_node_epoch,
        "edge_checkpoint": str(args.edge_checkpoint_path),
        "requested_edge_epoch": args.edge_checkpoint_epoch,
        "loaded_edge_epoch": loaded_edge_epoch,
        "projection_mode": args.projection_mode,
        "projection_target": args.projection_target,
        "projection_ridge": args.projection_ridge,
        "num_samples": len(dataset),
        **evaluate(model, edge_head, dataset, args),
    }

    fieldnames = list(row)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerow(row)
    print(
        f"{args.label}: face_rel={row['face_relative_rmse']:.6f} "
        f"closure_rel={row['closure_relative_rmse']:.6f} "
        f"div_rel={row['internal_divergence_relative_rmse']:.6f}"
    )
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
