# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate checkpoints against formal HEC-RAS cell-balance targets."""

import argparse
import csv
from pathlib import Path

import torch

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.utils import load_checkpoint


def evaluate_checkpoint(args, checkpoint_name: str, checkpoint_path: Path) -> dict:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = HydroGraphDataset(
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
        return_hecras_cell_balance=True,
        hecras_cell_balance_glob=args.hecras_cell_balance_glob,
    )

    model = MeshGraphKAN(
        args.num_input_features,
        args.num_edge_features,
        args.num_output_features,
    ).to(device)
    load_checkpoint(checkpoint_path, models=model, device=device)
    model.eval()

    sums = {
        "mse_loss": 0.0,
        "cell_balance_mse_ft6": 0.0,
        "cell_balance_mae_ft3": 0.0,
        "target_budget_rms_ft3": 0.0,
        "zone_3_cell_balance_mse_ft6": 0.0,
        "zone_3_cell_balance_mae_ft3": 0.0,
    }
    count = 0
    with torch.no_grad():
        for graph in dataset:
            graph = graph.to(device)
            pred = model(graph.x, graph.edge_attr, graph)
            volume_std = graph.volume_std.reshape(-1)[0].to(device)
            pred_delta = pred[:, 1] * volume_std
            budget_delta = graph.hecras_cell_balance_delta.to(device)
            residual = pred_delta - budget_delta
            sums["mse_loss"] += torch.mean((pred - graph.y) ** 2).item()
            sums["cell_balance_mse_ft6"] += torch.mean(residual**2).item()
            sums["cell_balance_mae_ft3"] += torch.mean(torch.abs(residual)).item()
            sums["target_budget_rms_ft3"] += torch.sqrt(
                torch.mean(budget_delta**2)
            ).item()
            if hasattr(graph, "zone_label"):
                high_mask = graph.zone_label.to(device) == 3
                if torch.any(high_mask):
                    high_residual = residual[high_mask]
                    sums["zone_3_cell_balance_mse_ft6"] += torch.mean(
                        high_residual**2
                    ).item()
                    sums["zone_3_cell_balance_mae_ft3"] += torch.mean(
                        torch.abs(high_residual)
                    ).item()
            count += 1

    row = {"checkpoint": checkpoint_name, "num_samples": count}
    for key, value in sums.items():
        row[key] = value / max(count, 1)
    row["cell_balance_rmse_ft3"] = row["cell_balance_mse_ft6"] ** 0.5
    row["zone_3_cell_balance_rmse_ft3"] = (
        row["zone_3_cell_balance_mse_ft6"] ** 0.5
    )
    row["relative_cell_balance_rmse"] = row["cell_balance_rmse_ft3"] / max(
        row["target_budget_rms_ft3"], 1e-12
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--eval-ids-file", default="train.txt")
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--n-time-steps", type=int, default=2)
    parser.add_argument("--zone-label-file", default="zone_label.txt")
    parser.add_argument("--zone-weight-file", default="zone_weight.txt")
    parser.add_argument("--num-input-features", type=int, default=16)
    parser.add_argument("--num-edge-features", type=int, default=3)
    parser.add_argument("--num-output-features", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hecras-cell-balance-glob", required=True)
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
    fieldnames = sorted({key for row in rows for key in row})
    fieldnames.remove("checkpoint")
    fieldnames.insert(0, "checkpoint")
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    for row in rows:
        print(row)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
