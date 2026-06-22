# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate the experiment-local HEC-RAS edge-flux head.

This diagnostic loads both the main HydroGraphNet node model and the optional
``HecRasEdgeFluxHead``. It reports whether the learned edge flux creates a
reasonable node-wise divergence and whether the node storage prediction closes
with the explicit boundary/source correction.
"""

import argparse
import csv
from pathlib import Path

import torch

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.utils import load_checkpoint
from train import HecRasEdgeFluxHead
from utils import compute_hecras_edge_flux_head_loss


def rmse(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value**2))


def mae(value: torch.Tensor) -> torch.Tensor:
    return torch.mean(torch.abs(value))


def add_value(sums: dict[str, float], key: str, value: torch.Tensor) -> None:
    sums[key] = sums.get(key, 0.0) + float(value.detach().cpu())


def compute_divergence(graph, edge_flux_delta: torch.Tensor) -> torch.Tensor:
    face_index = graph.hecras_face_index.to(edge_flux_delta.device)
    src, dst = face_index
    divergence = torch.zeros(
        graph.x.shape[0], dtype=edge_flux_delta.dtype, device=edge_flux_delta.device
    )
    divergence.index_add_(0, src, -edge_flux_delta.reshape(-1))
    divergence.index_add_(0, dst, edge_flux_delta.reshape(-1))
    return divergence


def add_residual_metrics(
    sums: dict[str, float],
    prefix: str,
    residual: torch.Tensor,
    target: torch.Tensor,
) -> None:
    add_value(sums, f"{prefix}_rmse_ft3", rmse(residual))
    add_value(sums, f"{prefix}_mae_ft3", mae(residual))
    add_value(sums, f"{prefix}_bias_ft3", torch.mean(residual))
    add_value(sums, f"{prefix}_target_rms_ft3", rmse(target))
    target_rms = torch.clamp(rmse(target), min=1e-12)
    add_value(sums, f"{prefix}_relative_rmse", rmse(residual) / target_rms)


def evaluate_checkpoint(
    model: MeshGraphKAN,
    edge_head: HecRasEdgeFluxHead,
    dataset: HydroGraphDataset,
    args: argparse.Namespace,
) -> dict[str, float]:
    sums: dict[str, float] = {}
    count = 0
    with torch.no_grad():
        for graph in dataset:
            graph = graph.to(args.device_obj)
            pred = model(graph.x, graph.edge_attr, graph)
            edge_flux_delta = edge_head(graph)
            divergence = compute_divergence(graph, edge_flux_delta)

            volume_std = graph.volume_std.reshape(-1)[0].to(args.device_obj)
            pred_delta = pred[:, 1] * volume_std
            internal_target = graph.hecras_edge_internal_delta.to(args.device_obj).reshape(-1)
            boundary_target = graph.hecras_edge_boundary_source_delta.to(args.device_obj).reshape(-1)
            total_target = internal_target + boundary_target

            divergence_residual = divergence - internal_target
            closure_residual = pred_delta - boundary_target - divergence
            total_residual = pred_delta - total_target

            add_residual_metrics(
                sums, "divergence", divergence_residual, internal_target
            )
            add_residual_metrics(sums, "closure", closure_residual, pred_delta)
            add_residual_metrics(sums, "total_node", total_residual, total_target)
            add_value(sums, "edge_flux_rms_ft3", rmse(edge_flux_delta))
            add_value(sums, "edge_flux_mae_ft3", mae(edge_flux_delta))
            if hasattr(graph, "hecras_internal_face_flow_delta"):
                face_target = graph.hecras_internal_face_flow_delta.to(
                    args.device_obj
                ).reshape(-1)
                face_residual = edge_flux_delta.reshape(-1) - face_target
                add_residual_metrics(
                    sums, "internal_face", face_residual, face_target
                )

            loss, loss_metrics = compute_hecras_edge_flux_head_loss(
                pred,
                edge_flux_delta,
                graph,
                zone_mode=args.zone_mode,
                closure_target_weight=args.closure_target_weight,
                divergence_target_weight=args.divergence_target_weight,
                face_target_weight=args.face_target_weight,
                face_loss_normalization=args.face_loss_normalization,
            )
            add_value(sums, "selected_edge_flux_head_loss", loss)
            for key, value in loss_metrics.items():
                add_value(sums, f"selected_{key}", value)

            if hasattr(graph, "zone_label"):
                zone_label = graph.zone_label.to(args.device_obj)
                for zone in range(4):
                    mask = zone_label == zone
                    if torch.any(mask):
                        add_residual_metrics(
                            sums,
                            f"zone_{zone}_divergence",
                            divergence_residual[mask],
                            internal_target[mask],
                        )
                        add_residual_metrics(
                            sums,
                            f"zone_{zone}_closure",
                            closure_residual[mask],
                            pred_delta[mask],
                        )
                        add_residual_metrics(
                            sums,
                            f"zone_{zone}_total_node",
                            total_residual[mask],
                            total_target[mask],
                        )
            count += 1

    return {key: value / max(count, 1) for key, value in sums.items()}


def build_dataset(args: argparse.Namespace) -> HydroGraphDataset:
    return HydroGraphDataset(
        data_dir=args.data_dir,
        prefix=args.prefix,
        num_samples=args.num_samples,
        n_time_steps=args.n_time_steps,
        hydrograph_ids_file=args.eval_ids_file,
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
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--eval-ids-file", default="test.txt")
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--num-samples", type=int, default=120)
    parser.add_argument("--n-time-steps", type=int, default=2)
    parser.add_argument("--zone-label-file", default="zone_label.txt")
    parser.add_argument("--zone-weight-file", default="zone_weight.txt")
    parser.add_argument("--num-input-features", type=int, default=16)
    parser.add_argument("--num-edge-features", type=int, default=3)
    parser.add_argument("--num-output-features", type=int, default=2)
    parser.add_argument("--edge-head-hidden-dim", type=int, default=128)
    parser.add_argument("--edge-head-num-hidden-layers", type=int, default=2)
    parser.add_argument("--edge-head-use-face-normal", action="store_true")
    parser.add_argument("--edge-head-use-surface-features", action="store_true")
    parser.add_argument("--edge-head-use-physical-surface-features", action="store_true")
    parser.add_argument("--edge-head-use-previous-face-flow", action="store_true")
    parser.add_argument("--edge-head-use-edge-physical-features", action="store_true")
    parser.add_argument("--edge-head-output-mode", default="raw")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hecras-face-graph-file", required=True)
    parser.add_argument("--hecras-edge-flow-npz", required=True)
    parser.add_argument("--hecras-edge-flow-face-stats-npz")
    parser.add_argument("--zone-mode", default="high")
    parser.add_argument("--closure-target-weight", type=float, default=1.0)
    parser.add_argument("--divergence-target-weight", type=float, default=1.0)
    parser.add_argument("--face-target-weight", type=float, default=0.0)
    parser.add_argument(
        "--face-loss-normalization",
        default="none",
        choices=(
            "none",
            "target_rms",
            "asinh_target_rms",
            "signed_log1p_target_rms",
            "per_face_rms",
            "asinh_per_face_rms",
            "signed_log1p_per_face_rms",
        ),
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=4,
        metavar=("NAME", "PATH", "EPOCH", "EDGE_HEAD_SCALE"),
        required=True,
        help="Checkpoint label, directory, epoch, and edge-head scale.",
    )
    args = parser.parse_args()
    args.device_obj = torch.device(args.device if torch.cuda.is_available() else "cpu")

    dataset = build_dataset(args)
    rows = []
    for name, checkpoint_path_raw, epoch_raw, scale_raw in args.checkpoint:
        epoch = int(epoch_raw)
        scale = float(scale_raw)
        model = MeshGraphKAN(
            args.num_input_features,
            args.num_edge_features,
            args.num_output_features,
        ).to(args.device_obj)
        edge_head = HecRasEdgeFluxHead(
            args.num_input_features,
            hidden_dim=args.edge_head_hidden_dim,
            num_hidden_layers=args.edge_head_num_hidden_layers,
            scale=scale,
            use_face_normal=args.edge_head_use_face_normal,
            use_surface_features=args.edge_head_use_surface_features,
            use_physical_surface_features=(
                args.edge_head_use_physical_surface_features
            ),
            use_previous_face_flow=args.edge_head_use_previous_face_flow,
            use_edge_physical_features=(
                args.edge_head_use_edge_physical_features
            ),
            output_mode=args.edge_head_output_mode,
        ).to(args.device_obj)
        loaded_epoch = load_checkpoint(
            Path(checkpoint_path_raw),
            models=[model, edge_head],
            epoch=epoch,
            device=args.device_obj,
        )
        model.eval()
        edge_head.eval()
        row = {
            "checkpoint": name,
            "requested_epoch": epoch,
            "loaded_epoch": loaded_epoch,
            "edge_head_scale": scale,
            "num_samples": len(dataset),
            "zone_mode": args.zone_mode,
            "divergence_target_weight": args.divergence_target_weight,
        }
        row.update(evaluate_checkpoint(model, edge_head, dataset, args))
        rows.append(row)
        print(
            f"{name} epoch {epoch}: "
            f"closure_rmse={row['closure_rmse_ft3']:.3f}, "
            f"divergence_rmse={row['divergence_rmse_ft3']:.3f}, "
            f"zone3_closure_rmse={row.get('zone_3_closure_rmse_ft3', float('nan')):.3f}, "
            f"zone3_divergence_rmse={row.get('zone_3_divergence_rmse_ft3', float('nan')):.3f}"
        )

    fieldnames = sorted({key for row in rows for key in row})
    for key in ("checkpoint", "requested_epoch", "loaded_epoch", "edge_head_scale"):
        fieldnames.remove(key)
    fieldnames = [
        "checkpoint",
        "requested_epoch",
        "loaded_epoch",
        "edge_head_scale",
        *fieldnames,
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
