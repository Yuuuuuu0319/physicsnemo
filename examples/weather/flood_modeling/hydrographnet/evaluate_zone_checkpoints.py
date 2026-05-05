# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate HydroGraphNet checkpoints with node-level fidelity-zone metrics.

This script evaluates one-step prediction metrics on held-out hydrographs. It is
kept separate from future edge-informed local-conservation work.
"""

import argparse
import csv
from pathlib import Path

import torch

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.utils import load_checkpoint
from utils import compute_zone_metrics, compute_zone_weighted_loss


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
        rollout_length=1,
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
    count = 0
    with torch.no_grad():
        for idx in range(len(dataset)):
            graph, rollout_data = dataset[idx]
            graph = graph.to(device)
            n_time_steps = args.n_time_steps
            pred = model(graph.x, graph.edge_attr, graph)

            water_depth_window = graph.x[:, 12 : 12 + n_time_steps]
            volume_window = graph.x[:, 12 + n_time_steps : 12 + 2 * n_time_steps]
            target_depth = rollout_data["water_depth_gt"][0].to(device) - water_depth_window[
                :, -1
            ]
            target_volume = rollout_data["volume_gt"][0].to(device) - volume_window[:, -1]
            target = torch.stack([target_depth, target_volume], dim=1)

            loss = torch.mean((pred - target) ** 2)
            zone_loss = compute_zone_weighted_loss(pred, target, graph)
            metrics = {
                "mse_loss": loss,
                "zone_loss": zone_loss,
                **compute_zone_metrics(pred, target, graph),
            }
            for key, value in metrics.items():
                metric_sums[key] = metric_sums.get(key, 0.0) + value.detach().item()
            count += 1

    row = {"checkpoint": checkpoint_name, "num_hydrographs": count}
    for key in sorted(metric_sums):
        row[key] = metric_sums[key] / max(count, 1)
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
        help="Checkpoint name and checkpoint directory.",
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
