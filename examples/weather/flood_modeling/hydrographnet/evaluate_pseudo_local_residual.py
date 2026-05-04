# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate velocity-proxy local residuals by fidelity zone.

This is an evaluation-only bridge toward selective local conservation. It uses
ground-truth VX/VY to build a kNN transport proxy, then compares model-predicted
volume changes against that proxy. It does not train with velocity data and does
not add an edge head.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import KDTree

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.utils import load_checkpoint


def rmse_tensor(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value**2))


def add_metric(metrics: dict, key: str, value: torch.Tensor) -> None:
    metrics[key] = metrics.get(key, 0.0) + value.detach().item()


def build_knn_edges(xy: np.ndarray, k: int) -> tuple[torch.Tensor, torch.Tensor]:
    _, neighbors = KDTree(xy).query(xy, k=k + 1)
    src = []
    dst = []
    dirs = []
    for i, nbrs in enumerate(neighbors):
        for j in nbrs:
            if i == j:
                continue
            vec = xy[j] - xy[i]
            norm = np.linalg.norm(vec)
            if norm <= 0:
                continue
            src.append(i)
            dst.append(j)
            dirs.append(vec / norm)
    edge_index = torch.tensor([src, dst], dtype=torch.long)
    directions = torch.tensor(np.asarray(dirs), dtype=torch.float32)
    return edge_index, directions


def load_velocity(data_dir: Path, prefix: str, hydro_id: str, skip: int = 72):
    vx = np.loadtxt(data_dir / f"{prefix}_VX_{hydro_id}.txt", delimiter="\t")[skip:]
    vy = np.loadtxt(data_dir / f"{prefix}_VY_{hydro_id}.txt", delimiter="\t")[skip:]
    inflow = np.loadtxt(data_dir / f"{prefix}_US_InF_{hydro_id}.txt", delimiter="\t")[
        skip:, 1
    ]
    peak_time_idx = int(np.argmax(inflow))
    end = peak_time_idx + 25
    return vx[:end], vy[:end]


def transport_proxy(
    vx: torch.Tensor,
    vy: torch.Tensor,
    edge_index: torch.Tensor,
    directions: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    src, dst = edge_index
    velocity = torch.stack([vx, vy], dim=1)
    edge_velocity = 0.5 * (velocity[src] + velocity[dst])
    edge_flux = torch.sum(edge_velocity * directions, dim=1)
    divergence = torch.zeros(num_nodes, device=vx.device)
    divergence.index_add_(0, src, edge_flux)
    divergence.index_add_(0, dst, -edge_flux)
    return -divergence


def evaluate_checkpoint(args, checkpoint_name: str, checkpoint_path: Path) -> dict:
    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = HydroGraphDataset(
        data_dir=data_dir,
        prefix=args.prefix,
        n_time_steps=args.n_time_steps,
        hydrograph_ids_file=args.eval_ids_file,
        split="test",
        rollout_length=args.rollout_length,
        return_physics=False,
        use_fidelity_zones=True,
        zone_label_file=args.zone_label_file,
        zone_weight_file=args.zone_weight_file,
    )

    raw_xy = np.loadtxt(data_dir / f"{args.prefix}_XY.txt", delimiter="\t")
    edge_index, directions = build_knn_edges(raw_xy, args.k)
    edge_index = edge_index.to(device)
    directions = directions.to(device)

    model = MeshGraphKAN(
        args.num_input_features,
        args.num_edge_features,
        args.num_output_features,
    ).to(device)
    load_checkpoint(checkpoint_path, models=model, device=device)
    model.eval()

    volume_std = float(dataset.dynamic_stats["volume"]["std"])
    metric_sums = {}
    metric_count = 0
    with torch.no_grad():
        for idx in range(len(dataset)):
            graph, rollout_data = dataset[idx]
            hydro_id = dataset.dynamic_data[idx]["hydro_id"]
            vx_np, vy_np = load_velocity(data_dir, args.prefix, hydro_id)

            graph = graph.to(device)
            x_iter = graph.x.to(device)
            zone_label = graph.zone_label.to(device)
            edge_features = graph.edge_attr.to(device)
            volume_gt_seq = rollout_data["volume_gt"].to(device)
            vx_all = torch.tensor(vx_np, dtype=torch.float32, device=device)
            vy_all = torch.tensor(vy_np, dtype=torch.float32, device=device)

            for step in range(args.rollout_length):
                volume_window = x_iter[
                    :, 12 + args.n_time_steps : 12 + 2 * args.n_time_steps
                ]
                pred = model(x_iter, edge_features, graph)
                pred_delta_denorm = pred[:, 1] * volume_std
                gt_delta_denorm = (
                    volume_gt_seq[step] - volume_window[:, -1]
                ) * volume_std

                velocity_t = args.n_time_steps + step
                proxy = transport_proxy(
                    vx_all[velocity_t],
                    vy_all[velocity_t],
                    edge_index,
                    directions,
                    x_iter.shape[0],
                )
                denom = torch.sum(proxy * proxy).clamp_min(1e-12)
                scale = torch.sum(gt_delta_denorm * proxy) / denom
                proxy_scaled = scale * proxy

                pred_residual = pred_delta_denorm - proxy_scaled
                gt_residual = gt_delta_denorm - proxy_scaled
                add_metric(metric_sums, "pred_proxy_residual_rmse", rmse_tensor(pred_residual))
                add_metric(metric_sums, "gt_proxy_residual_rmse", rmse_tensor(gt_residual))
                add_metric(
                    metric_sums,
                    "pred_vs_gt_delta_rmse",
                    rmse_tensor(pred_delta_denorm - gt_delta_denorm),
                )

                for zone in range(4):
                    mask = zone_label == zone
                    if torch.any(mask):
                        add_metric(
                            metric_sums,
                            f"zone_{zone}_pred_proxy_residual_rmse",
                            rmse_tensor(pred_residual[mask]),
                        )
                        add_metric(
                            metric_sums,
                            f"zone_{zone}_gt_proxy_residual_rmse",
                            rmse_tensor(gt_residual[mask]),
                        )
                        add_metric(
                            metric_sums,
                            f"zone_{zone}_pred_vs_gt_delta_rmse",
                            rmse_tensor(
                                pred_delta_denorm[mask] - gt_delta_denorm[mask]
                            ),
                        )

                static_part = x_iter[:, :12]
                water_depth_window = x_iter[:, 12 : 12 + args.n_time_steps]
                new_wd = water_depth_window[:, -1] + pred[:, 0]
                new_volume = volume_window[:, -1] + pred[:, 1]
                water_depth_updated = torch.cat(
                    [water_depth_window[:, 1:], new_wd[:, None]], dim=1
                )
                volume_updated = torch.cat(
                    [volume_window[:, 1:], new_volume[:, None]], dim=1
                )
                static_updated = static_part.clone()
                x_iter = torch.cat(
                    [static_updated, water_depth_updated, volume_updated], dim=1
                )
                metric_count += 1

    row = {
        "checkpoint": checkpoint_name,
        "num_hydrographs": len(dataset),
        "rollout_length": args.rollout_length,
        "k": args.k,
    }
    for key in sorted(metric_sums):
        row[key] = metric_sums[key] / max(metric_count, 1)
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--eval-ids-file", default="eval_h4_h6.txt")
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--n-time-steps", type=int, default=2)
    parser.add_argument("--rollout-length", type=int, default=10)
    parser.add_argument("--k", type=int, default=4)
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
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
