# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate checkpoint rollouts with node-level fidelity-zone metrics.

This is a lightweight, non-animation rollout evaluator. It remains separate from
future edge-informed local-conservation work.
"""

import argparse
import csv
from pathlib import Path

import torch

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.utils import load_checkpoint


def rmse(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value**2))


def add_metric(metrics: dict, key: str, value: torch.Tensor) -> None:
    metrics[key] = metrics.get(key, 0.0) + value.detach().item()


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

    model = MeshGraphKAN(
        args.num_input_features,
        args.num_edge_features,
        args.num_output_features,
    ).to(device)
    load_checkpoint(checkpoint_path, models=model, device=device)
    model.eval()

    metric_sums = {}
    metric_count = 0
    with torch.no_grad():
        for idx in range(len(dataset)):
            graph, rollout_data = dataset[idx]
            graph = graph.to(device)
            zone_label = graph.zone_label.to(device)
            edge_features = graph.edge_attr.to(device)
            x_iter = graph.x.to(device)
            num_nodes = x_iter.shape[0]

            inflow_seq = rollout_data["inflow"].to(device)
            precip_seq = rollout_data["precipitation"].to(device)
            wd_gt_seq = rollout_data["water_depth_gt"].to(device)
            volume_gt_seq = rollout_data["volume_gt"].to(device)

            for step in range(args.rollout_length):
                static_part = x_iter[:, :12]
                water_depth_window = x_iter[:, 12 : 12 + args.n_time_steps]
                volume_window = x_iter[
                    :, 12 + args.n_time_steps : 12 + 2 * args.n_time_steps
                ]
                pred = model(x_iter, edge_features, graph)
                new_wd = water_depth_window[:, -1] + pred[:, 0]
                new_volume = volume_window[:, -1] + pred[:, 1]

                wd_error = new_wd - wd_gt_seq[step]
                volume_error = new_volume - volume_gt_seq[step]
                combined_error = torch.stack([wd_error, volume_error], dim=1)
                add_metric(metric_sums, "rollout_rmse", rmse(combined_error))
                add_metric(metric_sums, "rollout_wd_rmse", rmse(wd_error))
                add_metric(metric_sums, "rollout_volume_rmse", rmse(volume_error))

                for zone in range(4):
                    mask = zone_label == zone
                    if torch.any(mask):
                        add_metric(
                            metric_sums,
                            f"zone_{zone}_rollout_rmse",
                            rmse(combined_error[mask]),
                        )
                        add_metric(
                            metric_sums,
                            f"zone_{zone}_rollout_wd_rmse",
                            rmse(wd_error[mask]),
                        )
                        add_metric(
                            metric_sums,
                            f"zone_{zone}_rollout_volume_rmse",
                            rmse(volume_error[mask]),
                        )

                water_depth_updated = torch.cat(
                    [water_depth_window[:, 1:], new_wd[:, None]], dim=1
                )
                volume_updated = torch.cat(
                    [volume_window[:, 1:], new_volume[:, None]], dim=1
                )
                static_updated = static_part.clone()
                static_updated[:, 10:11] = inflow_seq[step].expand(num_nodes, 1)
                static_updated[:, 11:12] = precip_seq[step].expand(num_nodes, 1)
                x_iter = torch.cat(
                    [static_updated, water_depth_updated, volume_updated], dim=1
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
    parser.add_argument(
        "--data-dir",
        required=True,
        help="Training data directory, or evaluation data directory when --test-data-dir is omitted.",
    )
    parser.add_argument(
        "--train-data-dir",
        help="Directory containing training normalization stats. Defaults to --data-dir.",
    )
    parser.add_argument(
        "--test-data-dir",
        help="Evaluation data directory. Use this for HGN_test while keeping train stats from HGN_train.",
    )
    parser.add_argument("--eval-ids-file", default="test.txt")
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--n-time-steps", type=int, default=2)
    parser.add_argument("--rollout-length", type=int, default=10)
    parser.add_argument("--zone-label-file", default="zone_label.txt")
    parser.add_argument("--zone-weight-file", default="zone_weight.txt")
    parser.add_argument("--num-input-features", type=int, default=16)
    parser.add_argument("--num-edge-features", type=int, default=3)
    parser.add_argument("--num-output-features", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
    )
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
