# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate source-aware true-face transport-gradient diagnostics.

This diagnostic uses HydroGraphNet `Pr` and `IP` files, HEC-RAS true internal
face connectivity, and HGN targets. It does not use HDF face velocity, so it is
safe when the HDF result event is not matched to the evaluated hydrographs.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.utils import load_checkpoint


def rmse(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value**2))


def add_metric(metrics: dict, key: str, value: torch.Tensor) -> None:
    metrics[key] = metrics.get(key, 0.0) + value.detach().item()


def load_face_graph(path: Path, device: torch.device) -> dict[str, torch.Tensor]:
    data = np.load(path)
    return {
        "face_index": torch.tensor(
            data["internal_face_index"], dtype=torch.long, device=device
        ),
        "face_length": torch.tensor(
            data["internal_face_length"], dtype=torch.float32, device=device
        ),
    }


def denormalize(value, mean: float, std: float):
    return value * std + mean


def source_delta_from_precip(
    precip_norm: torch.Tensor,
    area: torch.Tensor,
    infiltration: torch.Tensor,
    precip_mean: float,
    precip_std: float,
    delta_t: float,
) -> torch.Tensor:
    precip = denormalize(precip_norm, precip_mean, precip_std)
    return precip * area * (infiltration / 100.0) * delta_t


def face_gradient(
    node_values: torch.Tensor,
    face_graph: dict[str, torch.Tensor],
    area: torch.Tensor,
) -> torch.Tensor:
    src, dst = face_graph["face_index"]
    values_per_area = node_values / area.clamp_min(1e-6)
    return values_per_area[src] - values_per_area[dst]


def high_adjacent_face_mask(
    zone_label: torch.Tensor, face_graph: dict[str, torch.Tensor]
) -> torch.Tensor:
    src, dst = face_graph["face_index"]
    return (zone_label[src] == 3) | (zone_label[dst] == 3)


def update_rollout_state(
    x_iter: torch.Tensor,
    pred: torch.Tensor,
    inflow_value: torch.Tensor,
    precip_value: torch.Tensor,
    n_time_steps: int,
) -> torch.Tensor:
    static_part = x_iter[:, :12]
    water_depth_window = x_iter[:, 12 : 12 + n_time_steps]
    volume_window = x_iter[:, 12 + n_time_steps : 12 + 2 * n_time_steps]
    new_wd = water_depth_window[:, -1] + pred[:, 0]
    new_volume = volume_window[:, -1] + pred[:, 1]
    water_depth_updated = torch.cat(
        [water_depth_window[:, 1:], new_wd[:, None]], dim=1
    )
    volume_updated = torch.cat([volume_window[:, 1:], new_volume[:, None]], dim=1)
    static_updated = static_part.clone()
    static_updated[:, 10:11] = inflow_value.expand(x_iter.shape[0], 1)
    static_updated[:, 11:12] = precip_value.expand(x_iter.shape[0], 1)
    return torch.cat([static_updated, water_depth_updated, volume_updated], dim=1)


def evaluate_checkpoint(args, checkpoint_name: str, checkpoint_path: Path) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    eval_data_dir = args.test_data_dir or args.data_dir
    norm_stats_dir = args.train_data_dir or args.data_dir
    dataset = HydroGraphDataset(
        data_dir=eval_data_dir,
        prefix=args.prefix,
        n_time_steps=args.n_time_steps,
        hydrograph_ids_file=args.eval_ids_file,
        split="test",
        rollout_length=args.rollout_length,
        return_physics=False,
        use_fidelity_zones=True,
        zone_label_file=args.zone_label_file,
        zone_weight_file=args.zone_weight_file,
        norm_stats_dir=norm_stats_dir,
    )
    face_graph = load_face_graph(args.face_graph_file, device)

    model = MeshGraphKAN(
        args.num_input_features,
        args.num_edge_features,
        args.num_output_features,
    ).to(device)
    load_checkpoint(checkpoint_path, models=model, device=device)
    model.eval()

    volume_std = float(dataset.dynamic_stats["volume"]["std"])
    precip_mean = float(dataset.dynamic_stats["precipitation"]["mean"])
    precip_std = float(dataset.dynamic_stats["precipitation"]["std"])
    area = torch.tensor(
        dataset.static_data["area_denorm"].reshape(-1),
        dtype=torch.float32,
        device=device,
    )
    infiltration = torch.tensor(
        dataset.denormalize(
            dataset.static_data["infiltration"],
            dataset.static_stats["infiltration"]["mean"],
            dataset.static_stats["infiltration"]["std"],
        ).reshape(-1),
        dtype=torch.float32,
        device=device,
    )

    metric_sums = {}
    metric_count = 0
    with torch.no_grad():
        for idx in range(len(dataset)):
            graph, rollout_data = dataset[idx]
            graph = graph.to(device)
            x_iter = graph.x.to(device)
            edge_features = graph.edge_attr.to(device)
            zone_label = graph.zone_label.to(device)
            high_face_mask = high_adjacent_face_mask(zone_label, face_graph)

            volume_gt_seq = rollout_data["volume_gt"].to(device)
            inflow_seq = rollout_data["inflow"].to(device)
            precip_seq = rollout_data["precipitation"].to(device)

            for step in range(args.rollout_length):
                volume_window = x_iter[
                    :, 12 + args.n_time_steps : 12 + 2 * args.n_time_steps
                ]
                pred = model(x_iter, edge_features, graph)
                pred_delta = pred[:, 1] * volume_std
                gt_delta = (volume_gt_seq[step] - volume_window[:, -1]) * volume_std
                source_delta = source_delta_from_precip(
                    precip_seq[step],
                    area,
                    infiltration,
                    precip_mean,
                    precip_std,
                    args.delta_t,
                )
                pred_transport_grad = face_gradient(
                    pred_delta - source_delta, face_graph, area
                )
                gt_transport_grad = face_gradient(
                    gt_delta - source_delta, face_graph, area
                )
                residual = pred_transport_grad - gt_transport_grad

                weights = face_graph["face_length"]
                weighted_pred = torch.sqrt(
                    torch.sum(weights * pred_transport_grad**2)
                    / torch.sum(weights).clamp_min(1e-12)
                )
                weighted_gt = torch.sqrt(
                    torch.sum(weights * gt_transport_grad**2)
                    / torch.sum(weights).clamp_min(1e-12)
                )
                weighted_residual = torch.sqrt(
                    torch.sum(weights * residual**2)
                    / torch.sum(weights).clamp_min(1e-12)
                )
                add_metric(metric_sums, "pred_transport_gradient_rmse", weighted_pred)
                add_metric(metric_sums, "gt_transport_gradient_rmse", weighted_gt)
                add_metric(
                    metric_sums,
                    "pred_vs_gt_transport_gradient_rmse",
                    weighted_residual,
                )
                add_metric(
                    metric_sums,
                    "pred_transport_gradient_unweighted_rmse",
                    rmse(pred_transport_grad),
                )
                add_metric(
                    metric_sums,
                    "gt_transport_gradient_unweighted_rmse",
                    rmse(gt_transport_grad),
                )

                if torch.any(high_face_mask):
                    high_weights = weights[high_face_mask]
                    high_pred = pred_transport_grad[high_face_mask]
                    high_gt = gt_transport_grad[high_face_mask]
                    high_residual = residual[high_face_mask]
                    add_metric(
                        metric_sums,
                        "high_adjacent_pred_transport_gradient_rmse",
                        torch.sqrt(
                            torch.sum(high_weights * high_pred**2)
                            / torch.sum(high_weights).clamp_min(1e-12)
                        ),
                    )
                    add_metric(
                        metric_sums,
                        "high_adjacent_gt_transport_gradient_rmse",
                        torch.sqrt(
                            torch.sum(high_weights * high_gt**2)
                            / torch.sum(high_weights).clamp_min(1e-12)
                        ),
                    )
                    add_metric(
                        metric_sums,
                        "high_adjacent_pred_vs_gt_transport_gradient_rmse",
                        torch.sqrt(
                            torch.sum(high_weights * high_residual**2)
                            / torch.sum(high_weights).clamp_min(1e-12)
                        ),
                    )

                x_iter = update_rollout_state(
                    x_iter,
                    pred,
                    inflow_seq[step],
                    precip_seq[step],
                    args.n_time_steps,
                )
                metric_count += 1

    row = {
        "checkpoint": checkpoint_name,
        "num_hydrographs": len(dataset),
        "rollout_length": args.rollout_length,
    }
    for key in sorted(metric_sums):
        row[key] = metric_sums[key] / max(metric_count, 1)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-data-dir")
    parser.add_argument("--test-data-dir")
    parser.add_argument("--eval-ids-file", default="test.txt")
    parser.add_argument("--face-graph-file", required=True, type=Path)
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--n-time-steps", type=int, default=2)
    parser.add_argument("--rollout-length", type=int, default=10)
    parser.add_argument("--delta-t", type=float, default=1200.0)
    parser.add_argument("--zone-label-file", default="zone_label.txt")
    parser.add_argument("--zone-weight-file", default="zone_weight.txt")
    parser.add_argument("--num-input-features", type=int, default=16)
    parser.add_argument("--num-edge-features", type=int, default=3)
    parser.add_argument("--num-output-features", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--checkpoint", action="append", nargs=2, required=True)
    args = parser.parse_args()

    rows = [
        evaluate_checkpoint(args, name, Path(path))
        for name, path in args.checkpoint
    ]
    fieldnames = sorted({key for row in rows for key in row.keys()})
    fieldnames.remove("checkpoint")
    fieldnames.insert(0, "checkpoint")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
