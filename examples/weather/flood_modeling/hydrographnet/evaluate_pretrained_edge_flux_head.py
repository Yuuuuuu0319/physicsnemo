# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate an edge-flux head pretrained without the HydroGraphNet node model."""

import argparse
import csv
from pathlib import Path

import torch

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.utils import load_checkpoint
from train import HecRasEdgeFluxHead


def rmse(values: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(values**2))


def mae(values: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(values))


def add_metric(sums: dict[str, float], key: str, value: torch.Tensor) -> None:
    sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())


def add_residual_metrics(
    sums: dict[str, float],
    prefix: str,
    residual: torch.Tensor,
    target: torch.Tensor,
) -> None:
    residual = residual.reshape(-1)
    target = target.reshape(-1)
    count = residual.numel()
    sums[f"{prefix}_sse"] = sums.get(f"{prefix}_sse", 0.0) + float(
        torch.sum(residual**2).detach().cpu()
    )
    sums[f"{prefix}_target_sse"] = sums.get(f"{prefix}_target_sse", 0.0) + float(
        torch.sum(target**2).detach().cpu()
    )
    sums[f"{prefix}_abs_sum"] = sums.get(f"{prefix}_abs_sum", 0.0) + float(
        torch.sum(torch.abs(residual)).detach().cpu()
    )
    sums[f"{prefix}_sum"] = sums.get(f"{prefix}_sum", 0.0) + float(
        torch.sum(residual).detach().cpu()
    )
    sums[f"{prefix}_count"] = sums.get(f"{prefix}_count", 0.0) + float(count)


def finalize_metrics(
    sums: dict[str, float], volume_unit_suffix: str = "ft3"
) -> dict[str, float]:
    rows = {}
    prefixes = sorted(
        key[: -len("_sse")]
        for key in sums
        if key.endswith("_sse") and not key.endswith("_target_sse")
    )
    for prefix in prefixes:
        count = max(sums.get(f"{prefix}_count", 0.0), 1.0)
        sse = sums.get(f"{prefix}_sse", 0.0)
        target_sse = sums.get(f"{prefix}_target_sse", 0.0)
        rmse_value = (sse / count) ** 0.5
        target_rms_value = (target_sse / count) ** 0.5
        rows[f"{prefix}_rmse_{volume_unit_suffix}"] = rmse_value
        rows[f"{prefix}_mae_{volume_unit_suffix}"] = (
            sums.get(f"{prefix}_abs_sum", 0.0) / count
        )
        rows[f"{prefix}_bias_{volume_unit_suffix}"] = (
            sums.get(f"{prefix}_sum", 0.0) / count
        )
        rows[f"{prefix}_target_rms_{volume_unit_suffix}"] = target_rms_value
        rows[f"{prefix}_relative_rmse"] = rmse_value / max(target_rms_value, 1e-12)
    for key, value in sums.items():
        if not key.endswith(("_sse", "_target_sse", "_abs_sum", "_sum", "_count")):
            rows[key] = value
    return rows


def compute_divergence(graph, edge_flux_delta: torch.Tensor) -> torch.Tensor:
    face_index = graph.hecras_face_index.to(edge_flux_delta.device)
    src, dst = face_index
    divergence = torch.zeros(
        graph.x.shape[0], dtype=edge_flux_delta.dtype, device=edge_flux_delta.device
    )
    divergence.index_add_(0, src, -edge_flux_delta.reshape(-1))
    divergence.index_add_(0, dst, edge_flux_delta.reshape(-1))
    return divergence


def build_dataset(args: argparse.Namespace) -> HydroGraphDataset:
    return HydroGraphDataset(
        data_dir=args.data_dir,
        prefix=args.prefix,
        num_samples=args.num_samples,
        n_time_steps=args.n_time_steps,
        hydrograph_ids_file=args.ids_file,
        split="train",
        return_physics=False,
        use_fidelity_zones=True,
        zone_label_file=args.zone_label_file,
        zone_weight_file=args.zone_weight_file,
        return_hecras_face=True,
        hecras_face_graph_file=args.hecras_face_graph_file,
        return_hecras_edge_flow=True,
        hecras_edge_flow_npz=args.hecras_edge_flow_npz,
        hecras_edge_flow_mode="internal_plus_boundary_source",
        hecras_edge_flow_face_stats_npz=args.hecras_edge_flow_face_stats_npz,
        hecras_edge_flow_scale_stats_npz=args.hecras_edge_flow_scale_stats_npz,
    )


def evaluate(edge_head: HecRasEdgeFluxHead, dataset: HydroGraphDataset, args) -> dict:
    sums: dict[str, float] = {}
    count = 0
    edge_head.eval()
    with torch.no_grad():
        for graph in dataset:
            graph = graph.to(args.device_obj)
            edge_flux = edge_head(graph).reshape(-1)
            face_target = graph.hecras_internal_face_flow_delta.to(
                args.device_obj
            ).reshape(-1)
            face_residual = edge_flux - face_target
            divergence = compute_divergence(graph, edge_flux)
            internal_target = graph.hecras_edge_internal_delta.to(
                args.device_obj
            ).reshape(-1)
            divergence_residual = divergence - internal_target

            add_residual_metrics(sums, "face", face_residual, face_target)
            add_metric(sums, "edge_flux_rms_ft3", rmse(edge_flux))
            add_residual_metrics(
                sums, "internal_divergence", divergence_residual, internal_target
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
                    face_mask = (zone_label[src] == zone) | (zone_label[dst] == zone)
                    if torch.any(face_mask):
                        add_residual_metrics(
                            sums,
                            f"zone_{zone}_face",
                            face_residual[face_mask],
                            face_target[face_mask],
                        )
            count += 1
    rows = finalize_metrics(sums)
    for key in list(rows):
        if key == "edge_flux_rms_ft3":
            rows[key] = rows[key] / max(count, 1)
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
        row = {
            "checkpoint": name,
            "requested_epoch": epoch,
            "loaded_epoch": loaded_epoch,
            "num_samples": len(dataset),
            **evaluate(edge_head, dataset, args),
        }
        rows.append(row)
        print(
            f"{name} epoch {loaded_epoch}: face_rel={row['face_relative_rmse']:.6f} "
            f"div_rel={row['internal_divergence_relative_rmse']:.6f}"
        )

    fieldnames = sorted({key for row in rows for key in row})
    for key in ("checkpoint", "requested_epoch", "loaded_epoch", "num_samples"):
        if key in fieldnames:
            fieldnames.remove(key)
    fieldnames = ["checkpoint", "requested_epoch", "loaded_epoch", "num_samples"] + fieldnames
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
