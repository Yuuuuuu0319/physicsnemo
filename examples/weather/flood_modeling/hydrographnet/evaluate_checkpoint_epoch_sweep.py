# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Sweep saved HydroGraphNet epochs across prediction and conservation metrics.

This evaluator is intended for checkpoint-selection diagnostics. It keeps the
three current report metrics together:

* one-step held-out zone metrics,
* formal HEC-RAS native Cell Flow Balance residuals,
* autoregressive rollout zone metrics.

The script loads specific epochs from checkpoint directories via
``load_checkpoint(..., epoch=N)`` and writes one CSV row per checkpoint/epoch.
"""

import argparse
import csv
import re
from pathlib import Path

import torch

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.utils import load_checkpoint
from utils import compute_zone_metrics, compute_zone_weighted_loss


def rmse(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value**2))


def add_metric(metrics: dict[str, float], key: str, value: torch.Tensor) -> None:
    metrics[key] = metrics.get(key, 0.0) + value.detach().item()


def available_epochs(checkpoint_path: Path) -> list[int]:
    epochs = []
    pattern = re.compile(r"checkpoint\.\d+\.(\d+)\.pt$")
    for path in checkpoint_path.glob("checkpoint.*.pt"):
        match = pattern.search(path.name)
        if match:
            epochs.append(int(match.group(1)))
    return sorted(set(epochs))


def make_model(args: argparse.Namespace, device: torch.device) -> MeshGraphKAN:
    return MeshGraphKAN(
        args.num_input_features,
        args.num_edge_features,
        args.num_output_features,
    ).to(device)


def evaluate_one_step(model: MeshGraphKAN, dataset: HydroGraphDataset, args) -> dict:
    sums = {}
    count = 0
    with torch.no_grad():
        for idx in range(len(dataset)):
            graph, rollout_data = dataset[idx]
            graph = graph.to(args.device_obj)
            pred = model(graph.x, graph.edge_attr, graph)

            water_depth_window = graph.x[:, 12 : 12 + args.n_time_steps]
            volume_window = graph.x[
                :, 12 + args.n_time_steps : 12 + 2 * args.n_time_steps
            ]
            target_depth = (
                rollout_data["water_depth_gt"][0].to(args.device_obj)
                - water_depth_window[:, -1]
            )
            target_volume = (
                rollout_data["volume_gt"][0].to(args.device_obj)
                - volume_window[:, -1]
            )
            target = torch.stack([target_depth, target_volume], dim=1)

            metrics = {
                "mse_loss": torch.mean((pred - target) ** 2),
                "zone_loss": compute_zone_weighted_loss(pred, target, graph),
                **compute_zone_metrics(pred, target, graph),
            }
            for key, value in metrics.items():
                add_metric(sums, f"one_step_{key}", value)
            count += 1

    return {key: value / max(count, 1) for key, value in sums.items()}


def evaluate_cell_balance(
    model: MeshGraphKAN, dataset: HydroGraphDataset, args
) -> dict:
    sums = {
        "cell_balance_mse_ft6": 0.0,
        "cell_balance_mae_ft3": 0.0,
        "cell_mse_loss": 0.0,
        "target_budget_rms_ft3": 0.0,
        "zone_3_cell_balance_mse_ft6": 0.0,
        "zone_3_cell_balance_mae_ft3": 0.0,
    }
    count = 0
    with torch.no_grad():
        for graph in dataset:
            graph = graph.to(args.device_obj)
            pred = model(graph.x, graph.edge_attr, graph)
            volume_std = graph.volume_std.reshape(-1)[0].to(args.device_obj)
            pred_delta = pred[:, 1] * volume_std
            budget_delta = graph.hecras_cell_balance_delta.to(args.device_obj)
            residual = pred_delta - budget_delta

            sums["cell_mse_loss"] += torch.mean((pred - graph.y) ** 2).item()
            sums["cell_balance_mse_ft6"] += torch.mean(residual**2).item()
            sums["cell_balance_mae_ft3"] += torch.mean(torch.abs(residual)).item()
            sums["target_budget_rms_ft3"] += torch.sqrt(
                torch.mean(budget_delta**2)
            ).item()
            if hasattr(graph, "zone_label"):
                high_mask = graph.zone_label.to(args.device_obj) == 3
                if torch.any(high_mask):
                    high_residual = residual[high_mask]
                    sums["zone_3_cell_balance_mse_ft6"] += torch.mean(
                        high_residual**2
                    ).item()
                    sums["zone_3_cell_balance_mae_ft3"] += torch.mean(
                        torch.abs(high_residual)
                    ).item()
            count += 1

    row = {key: value / max(count, 1) for key, value in sums.items()}
    row["cell_balance_rmse_ft3"] = row["cell_balance_mse_ft6"] ** 0.5
    row["zone_3_cell_balance_rmse_ft3"] = (
        row["zone_3_cell_balance_mse_ft6"] ** 0.5
    )
    row["relative_cell_balance_rmse"] = row["cell_balance_rmse_ft3"] / max(
        row["target_budget_rms_ft3"], 1e-12
    )
    return row


def evaluate_rollout(model: MeshGraphKAN, dataset: HydroGraphDataset, args) -> dict:
    sums = {}
    count = 0
    with torch.no_grad():
        for idx in range(len(dataset)):
            graph, rollout_data = dataset[idx]
            graph = graph.to(args.device_obj)
            zone_label = graph.zone_label.to(args.device_obj)
            edge_features = graph.edge_attr.to(args.device_obj)
            x_iter = graph.x.to(args.device_obj)
            num_nodes = x_iter.shape[0]

            inflow_seq = rollout_data["inflow"].to(args.device_obj)
            precip_seq = rollout_data["precipitation"].to(args.device_obj)
            wd_gt_seq = rollout_data["water_depth_gt"].to(args.device_obj)
            volume_gt_seq = rollout_data["volume_gt"].to(args.device_obj)

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
                add_metric(sums, "rollout_rmse", rmse(combined_error))
                add_metric(sums, "rollout_wd_rmse", rmse(wd_error))
                add_metric(sums, "rollout_volume_rmse", rmse(volume_error))

                for zone in range(4):
                    mask = zone_label == zone
                    if torch.any(mask):
                        add_metric(
                            sums,
                            f"zone_{zone}_rollout_rmse",
                            rmse(combined_error[mask]),
                        )
                        add_metric(
                            sums,
                            f"zone_{zone}_rollout_wd_rmse",
                            rmse(wd_error[mask]),
                        )
                        add_metric(
                            sums,
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
                count += 1

    return {key: value / max(count, 1) for key, value in sums.items()}


def build_datasets(args: argparse.Namespace) -> tuple:
    eval_data_dir = args.test_data_dir or args.data_dir
    norm_stats_dir = args.train_data_dir or args.data_dir
    one_step_dataset = HydroGraphDataset(
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
    cell_dataset = HydroGraphDataset(
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
    rollout_dataset = HydroGraphDataset(
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
    return one_step_dataset, cell_dataset, rollout_dataset


def resolve_epochs(args: argparse.Namespace, checkpoint_path: Path) -> list[int]:
    if args.epochs:
        return args.epochs
    epochs = available_epochs(checkpoint_path)
    if args.epoch_start is not None:
        epochs = [epoch for epoch in epochs if epoch >= args.epoch_start]
    if args.epoch_end is not None:
        epochs = [epoch for epoch in epochs if epoch <= args.epoch_end]
    if not epochs:
        raise ValueError(f"No checkpoint epochs found in {checkpoint_path}")
    return epochs


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-data-dir")
    parser.add_argument("--test-data-dir")
    parser.add_argument("--eval-ids-file", default="test.txt")
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--n-time-steps", type=int, default=2)
    parser.add_argument("--rollout-length", type=int, default=25)
    parser.add_argument("--zone-label-file", default="zone_label.txt")
    parser.add_argument("--zone-weight-file", default="zone_weight.txt")
    parser.add_argument("--num-input-features", type=int, default=16)
    parser.add_argument("--num-edge-features", type=int, default=3)
    parser.add_argument("--num-output-features", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--hecras-cell-balance-glob", required=True)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--epoch-start", type=int)
    parser.add_argument("--epoch-end", type=int)
    parser.add_argument("--epochs", nargs="*", type=int)
    parser.add_argument(
        "--checkpoint",
        action="append",
        nargs=2,
        metavar=("NAME", "PATH"),
        required=True,
    )
    args = parser.parse_args()
    args.device_obj = torch.device(args.device if torch.cuda.is_available() else "cpu")

    one_step_dataset, cell_dataset, rollout_dataset = build_datasets(args)

    rows = []
    for checkpoint_name, checkpoint_path_raw in args.checkpoint:
        checkpoint_path = Path(checkpoint_path_raw)
        for epoch in resolve_epochs(args, checkpoint_path):
            model = make_model(args, args.device_obj)
            loaded_epoch = load_checkpoint(
                checkpoint_path,
                models=model,
                epoch=epoch,
                device=args.device_obj,
            )
            model.eval()
            row = {
                "checkpoint": checkpoint_name,
                "requested_epoch": epoch,
                "loaded_epoch": loaded_epoch,
                "num_one_step_hydrographs": len(one_step_dataset),
                "num_cell_samples": len(cell_dataset),
                "num_rollout_hydrographs": len(rollout_dataset),
                "rollout_length": args.rollout_length,
            }
            row.update(evaluate_one_step(model, one_step_dataset, args))
            row.update(evaluate_cell_balance(model, cell_dataset, args))
            row.update(evaluate_rollout(model, rollout_dataset, args))
            rows.append(row)
            print(
                f"{checkpoint_name} epoch {epoch}: "
                f"one_step_mse={row['one_step_mse_loss']:.6e}, "
                f"cell_rmse={row['cell_balance_rmse_ft3']:.3f}, "
                f"zone3_cell_rmse={row['zone_3_cell_balance_rmse_ft3']:.3f}, "
                f"rollout_rmse={row['rollout_rmse']:.6f}"
            )

    fieldnames = sorted({key for row in rows for key in row})
    for key in ("checkpoint", "requested_epoch", "loaded_epoch"):
        fieldnames.remove(key)
    fieldnames = ["checkpoint", "requested_epoch", "loaded_epoch", *fieldnames]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
