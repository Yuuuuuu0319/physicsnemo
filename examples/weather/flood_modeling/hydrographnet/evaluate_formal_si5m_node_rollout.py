#!/usr/bin/env python3
"""Evaluate HydroGraphNet node rollouts with common SI publication metrics."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import torch

from formal_si5m_forecast_metrics import compute_formal_forecast_metrics
from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.utils import load_checkpoint


def verify_split_contract(args: argparse.Namespace) -> dict:
    manifest_path = args.split_manifest or Path(args.data_dir) / "event_split_manifest.json"
    manifest = json.loads(Path(manifest_path).read_text())
    expected_ids = manifest["split_ids"][args.split_role]
    ids_path = Path(args.ids_file)
    if not ids_path.is_absolute():
        ids_path = Path(args.data_dir) / ids_path
    actual_ids = [line.strip() for line in ids_path.read_text().splitlines() if line.strip()]
    if actual_ids != expected_ids:
        raise ValueError(
            f"{args.split_role} IDs do not exactly match the frozen split manifest."
        )
    if args.split_role == "test":
        if args.test_authorization_file is None:
            raise ValueError("Test evaluation requires --test-authorization-file.")
        authorization = json.loads(args.test_authorization_file.read_text())
        if authorization.get("authorized") is not True:
            raise ValueError("The sealed-test authorization is not active.")
        if authorization.get("split_manifest_sha256") != manifest.get(
            "manifest_sha256"
        ):
            raise ValueError("Test authorization is bound to another split manifest.")
    return manifest


def update_state(
    x_iter: torch.Tensor,
    prediction: torch.Tensor,
    inflow: torch.Tensor,
    precipitation: torch.Tensor,
    n_time_steps: int,
) -> torch.Tensor:
    static = x_iter[:, :12].clone()
    depth = x_iter[:, 12 : 12 + n_time_steps]
    volume = x_iter[:, 12 + n_time_steps : 12 + 2 * n_time_steps]
    next_depth = depth[:, -1] + prediction[:, 0]
    next_volume = volume[:, -1] + prediction[:, 1]
    static[:, 10:11] = inflow.reshape(1, 1).expand(x_iter.shape[0], 1)
    static[:, 11:12] = precipitation.reshape(1, 1).expand(x_iter.shape[0], 1)
    return torch.cat(
        [
            static,
            torch.cat([depth[:, 1:], next_depth[:, None]], dim=1),
            torch.cat([volume[:, 1:], next_volume[:, None]], dim=1),
        ],
        dim=1,
    )


def evaluate(args: argparse.Namespace) -> list[dict[str, float | int | str]]:
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    dataset = HydroGraphDataset(
        data_dir=args.data_dir,
        prefix=args.prefix,
        n_time_steps=args.n_time_steps,
        hydrograph_ids_file=args.ids_file,
        split="test",
        rollout_length=args.rollout_length,
        use_fidelity_zones=True,
        zone_label_file=args.zone_label_file,
        zone_weight_file=args.zone_weight_file,
        norm_stats_dir=args.train_data_dir or args.data_dir,
    )
    model = MeshGraphKAN(
        args.num_input_features,
        args.num_edge_features,
        args.num_output_features,
    ).to(device)
    loaded_epoch = load_checkpoint(
        args.checkpoint_path,
        models=model,
        epoch=args.checkpoint_epoch,
        device=device,
    )
    model.eval()
    rows = []
    with torch.no_grad():
        for index in range(len(dataset)):
            graph, rollout = dataset[index]
            graph = graph.to(device)
            x_iter = graph.x.to(device)
            edge_features = graph.edge_attr.to(device)
            predicted_depth = []
            predicted_volume = []
            for step in range(args.rollout_length):
                prediction = model(x_iter, edge_features, graph)
                predicted_depth.append(
                    x_iter[:, 12 + args.n_time_steps - 1] + prediction[:, 0]
                )
                predicted_volume.append(
                    x_iter[:, 12 + 2 * args.n_time_steps - 1]
                    + prediction[:, 1]
                )
                x_iter = update_state(
                    x_iter,
                    prediction,
                    rollout["inflow"][step].to(device),
                    rollout["precipitation"][step].to(device),
                    args.n_time_steps,
                )
            metrics = compute_formal_forecast_metrics(
                torch.stack(predicted_depth),
                rollout["water_depth_gt"].to(device),
                torch.stack(predicted_volume),
                rollout["volume_gt"].to(device),
                graph.zone_label,
                water_depth_mean=dataset.dynamic_stats["water_depth"]["mean"],
                water_depth_std=dataset.dynamic_stats["water_depth"]["std"],
                volume_mean=dataset.dynamic_stats["volume"]["mean"],
                volume_std=dataset.dynamic_stats["volume"]["std"],
                delta_t_seconds=args.delta_t,
                wet_depth_threshold_m=args.wet_depth_threshold_m,
            )
            rows.append(
                {
                    "hydrograph_id": dataset.hydrograph_ids[index],
                    "method": args.method_name,
                    "split_role": args.split_role,
                    "loaded_epoch": int(loaded_epoch),
                    "future_hecras_target_used_for_inference": False,
                    **metrics,
                }
            )
    return rows


def mean_row(rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    result: dict[str, float | int | str] = {
        "hydrograph_id": "MEAN_EVENT",
        "method": rows[0]["method"],
        "split_role": rows[0]["split_role"],
        "loaded_epoch": rows[0]["loaded_epoch"],
        "future_hecras_target_used_for_inference": False,
    }
    for key, value in rows[0].items():
        if key in result or isinstance(value, bool):
            continue
        if isinstance(value, (int, float)):
            result[key] = sum(float(row[key]) for row in rows) / len(rows)
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-data-dir")
    parser.add_argument("--ids-file", required=True)
    parser.add_argument("--split-role", choices=("validation", "test"), default="validation")
    parser.add_argument("--split-manifest", type=Path)
    parser.add_argument("--test-authorization-file", type=Path)
    parser.add_argument("--checkpoint-path", required=True, type=Path)
    parser.add_argument("--checkpoint-epoch", type=int)
    parser.add_argument("--method-name", required=True)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--n-time-steps", type=int, default=2)
    parser.add_argument("--rollout-length", type=int, default=96)
    parser.add_argument("--delta-t", type=float, default=300.0)
    parser.add_argument("--wet-depth-threshold-m", type=float, default=0.01)
    parser.add_argument("--zone-label-file", default="zone_label.txt")
    parser.add_argument("--zone-weight-file", default="zone_weight.txt")
    parser.add_argument("--num-input-features", type=int, default=16)
    parser.add_argument("--num-edge-features", type=int, default=3)
    parser.add_argument("--num-output-features", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    verify_split_contract(args)
    rows = evaluate(args)
    rows.append(mean_row(rows))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0])
    with args.output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
