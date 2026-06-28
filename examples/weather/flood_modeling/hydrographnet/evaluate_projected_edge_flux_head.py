# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate sparse incidence projection for a pretrained HEC-RAS edge head.

This is a diagnostic for edge-local conservation.  It keeps the learned face
fluxes as close as possible to the edge-head prediction while reducing the
node-level divergence residual on a selected node set.
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
from physicsnemo.utils import load_checkpoint
from train import HecRasEdgeFluxHead


def evaluate(edge_head: HecRasEdgeFluxHead, dataset, args, mode: str, ridge: float) -> dict:
    sums: dict[str, float] = {}
    count = 0
    cg_fallbacks = 0
    edge_head.eval()
    with torch.no_grad():
        for graph in dataset:
            graph = graph.to(args.device_obj)
            edge_flux = edge_head(graph).reshape(-1)
            face_target = graph.hecras_internal_face_flow_delta.to(
                args.device_obj
            ).reshape(-1)
            internal_target = graph.hecras_edge_internal_delta.to(args.device_obj).reshape(-1)
            projected_flux, cg_info, correction_rms = project_edge_flux(
                graph,
                edge_flux,
                internal_target,
                mode=mode,
                ridge=ridge,
                cg_rtol=args.cg_rtol,
                cg_maxiter=args.cg_maxiter,
            )
            if cg_info != 0:
                cg_fallbacks += 1

            face_residual = projected_flux - face_target
            original_face_residual = edge_flux - face_target
            divergence = compute_divergence(graph, projected_flux)
            original_divergence = compute_divergence(graph, edge_flux)
            divergence_residual = divergence - internal_target
            original_divergence_residual = original_divergence - internal_target

            add_residual_metrics(sums, "face", face_residual, face_target)
            add_residual_metrics(
                sums, "original_face", original_face_residual, face_target
            )
            add_residual_metrics(
                sums, "internal_divergence", divergence_residual, internal_target
            )
            add_residual_metrics(
                sums,
                "original_internal_divergence",
                original_divergence_residual,
                internal_target,
            )
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
                            divergence_residual[node_mask],
                            internal_target[node_mask],
                        )
                        add_residual_metrics(
                            sums,
                            f"zone_{zone}_original_internal_divergence",
                            original_divergence_residual[node_mask],
                            internal_target[node_mask],
                        )
                    face_mask = (zone_label[src] == zone) | (zone_label[dst] == zone)
                    if torch.any(face_mask):
                        add_residual_metrics(
                            sums,
                            f"zone_{zone}_face",
                            face_residual[face_mask],
                            face_target[face_mask],
                        )
                        add_residual_metrics(
                            sums,
                            f"zone_{zone}_original_face",
                            original_face_residual[face_mask],
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
    rows["projection_mode"] = mode
    rows["projection_ridge"] = ridge
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
    parser.add_argument("--edge-head-hidden-dim", type=int, default=128)
    parser.add_argument("--edge-head-num-hidden-layers", type=int, default=2)
    parser.add_argument("--edge-head-scale", type=float, default=1.0)
    parser.add_argument("--edge-head-use-face-normal", action="store_true")
    parser.add_argument("--edge-head-use-surface-features", action="store_true")
    parser.add_argument("--edge-head-use-physical-surface-features", action="store_true")
    parser.add_argument("--edge-head-use-previous-face-flow", action="store_true")
    parser.add_argument("--edge-head-use-edge-physical-features", action="store_true")
    parser.add_argument("--edge-head-output-mode", default="raw")
    parser.add_argument("--hecras-face-graph-file", required=True)
    parser.add_argument("--hecras-edge-flow-npz", required=True)
    parser.add_argument("--hecras-edge-flow-face-stats-npz")
    parser.add_argument("--hecras-edge-flow-scale-stats-npz")
    parser.add_argument("--projection-mode", action="append")
    parser.add_argument("--projection-ridge", action="append", type=float)
    parser.add_argument("--cg-rtol", type=float, default=1e-8)
    parser.add_argument("--cg-maxiter", type=int, default=2000)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=3,
        metavar=("NAME", "PATH", "EPOCH"),
        required=True,
    )
    args = parser.parse_args()
    if args.projection_mode is None:
        args.projection_mode = ["high"]
    if args.projection_ridge is None:
        args.projection_ridge = [1.0]
    args.device_obj = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(args)
    rows = []
    for name, path, epoch_raw in args.checkpoint:
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
        epoch = int(epoch_raw)
        loaded_epoch = load_checkpoint(
            Path(path), models=[edge_head], epoch=epoch, device=args.device_obj
        )
        for mode in args.projection_mode:
            for ridge in args.projection_ridge:
                row = {
                    "checkpoint": name,
                    "requested_epoch": epoch,
                    "loaded_epoch": loaded_epoch,
                    "num_samples": len(dataset),
                    **evaluate(edge_head, dataset, args, mode, ridge),
                }
                rows.append(row)
                print(
                    f"{name} epoch {loaded_epoch} {mode} ridge={ridge:g}: "
                    f"face_rel={row['face_relative_rmse']:.6f} "
                    f"div_rel={row['internal_divergence_relative_rmse']:.6f} "
                    f"zone3_div_rel={row.get('zone_3_internal_divergence_relative_rmse', float('nan')):.6f}"
                )

    fieldnames = sorted({key for row in rows for key in row})
    lead = [
        "checkpoint",
        "requested_epoch",
        "loaded_epoch",
        "num_samples",
        "projection_mode",
        "projection_ridge",
    ]
    for key in lead:
        if key in fieldnames:
            fieldnames.remove(key)
    fieldnames = [key for key in lead if key in rows[0]] + fieldnames
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
