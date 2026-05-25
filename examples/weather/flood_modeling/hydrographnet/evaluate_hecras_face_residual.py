# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Evaluate HEC-RAS true-face local residuals by fidelity zone.

This script is a diagnostic bridge toward a future face-based local conservation
loss. It is intentionally separate from:

- node-zone weighted prediction loss
- kNN + VX/VY pseudo-local residuals
- model training

The HEC-RAS HDF face velocity is event-specific. Use this diagnostic only when
the HDF event is known to correspond to the selected HydroGraphNet hydrograph,
or treat the result strictly as an alignment/scale diagnostic. By default the
face proxy is calibrated against ground-truth volume transitions, so its scale
is independent of the checkpoint being compared.
"""

import argparse
import csv
import glob
import re
from pathlib import Path

import h5py
import numpy as np
import torch

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from physicsnemo.utils import load_checkpoint


FACE_VELOCITY_PATH = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/"
    "2D Flow Areas/per2/Face Velocity"
)


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
        "hdf_face_index": torch.tensor(
            data["internal_hdf_face_index"], dtype=torch.long, device=device
        ),
        "face_length": torch.tensor(
            data["internal_face_length"], dtype=torch.float32, device=device
        ),
    }


def infer_hydrograph_id_from_path(path: Path) -> str:
    """Infer H1/T1 style HydroGraphNet event ID from a HEC-RAS output path."""
    for part in reversed(path.parts):
        match = re.fullmatch(r"plan([HT]\d+)", part, flags=re.IGNORECASE)
        if match is not None:
            return match.group(1).upper()
    for part in reversed(path.parts):
        match = re.fullmatch(r"([HT]\d+)", part, flags=re.IGNORECASE)
        if match is not None:
            return match.group(1).upper()
    raise ValueError(
        "Could not infer hydrograph ID from HDF path. Expected a path containing "
        f"planH1/H1/T1, got: {path}"
    )


def load_face_velocity(path: Path, device: torch.device) -> torch.Tensor:
    with h5py.File(path, "r") as hdf:
        face_velocity_np = hdf[FACE_VELOCITY_PATH][:]
    return torch.tensor(face_velocity_np, dtype=torch.float32, device=device)


def load_face_velocity_by_hydrograph(
    hdf_glob: str, device: torch.device
) -> dict[str, torch.Tensor]:
    hdf_paths = sorted(Path(path) for path in glob.glob(hdf_glob))
    if not hdf_paths:
        raise FileNotFoundError(f"No HEC-RAS HDFs matched --hdf-glob: {hdf_glob}")

    face_velocity_by_hydrograph = {}
    for hdf_path in hdf_paths:
        hydrograph_id = infer_hydrograph_id_from_path(hdf_path)
        if hydrograph_id in face_velocity_by_hydrograph:
            raise ValueError(
                f"Multiple HEC-RAS HDF files map to hydrograph {hydrograph_id}."
            )
        face_velocity_by_hydrograph[hydrograph_id] = load_face_velocity(
            hdf_path, device
        )
    return face_velocity_by_hydrograph


def face_delta_proxy(
    face_velocity_all: torch.Tensor,
    face_graph: dict[str, torch.Tensor],
    num_nodes: int,
    delta_t: float,
) -> torch.Tensor:
    """Convert HEC-RAS internal face velocity to node volume-delta proxy."""
    src, dst = face_graph["face_index"]
    face_velocity = face_velocity_all[face_graph["hdf_face_index"]]
    face_flow = face_velocity * face_graph["face_length"]
    delta = torch.zeros(num_nodes, dtype=torch.float32, device=face_velocity.device)

    # HEC-RAS stores one signed velocity per face. We treat positive velocity as
    # flow from the first face cell to the second face cell. This is diagnostic
    # until validated against an event-specific HEC-RAS volume budget.
    delta.index_add_(0, src, -face_flow * delta_t)
    delta.index_add_(0, dst, face_flow * delta_t)
    return delta


def scale_proxy_to_target(proxy: torch.Tensor, target_delta: torch.Tensor) -> torch.Tensor:
    denom = torch.sum(proxy * proxy).clamp_min(1e-12)
    return (torch.sum(target_delta * proxy) / denom).detach()


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
    face_velocity = None
    face_velocity_by_hydrograph = {}
    if args.hdf_glob:
        face_velocity_by_hydrograph = load_face_velocity_by_hydrograph(
            args.hdf_glob, device
        )
    else:
        face_velocity = load_face_velocity(args.hdf_file, device)

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
            hydrograph_id = dataset.hydrograph_ids[idx]
            if face_velocity_by_hydrograph:
                if hydrograph_id not in face_velocity_by_hydrograph:
                    available = ", ".join(sorted(face_velocity_by_hydrograph))
                    raise ValueError(
                        f"No HEC-RAS HDF loaded for hydrograph {hydrograph_id}. "
                        f"Available hydrographs: {available}"
                    )
                face_velocity_for_hydrograph = face_velocity_by_hydrograph[hydrograph_id]
            else:
                face_velocity_for_hydrograph = face_velocity
            graph, rollout_data = dataset[idx]
            graph = graph.to(device)
            x_iter = graph.x.to(device)
            edge_features = graph.edge_attr.to(device)
            zone_label = graph.zone_label.to(device)
            volume_gt_seq = rollout_data["volume_gt"].to(device)
            inflow_seq = rollout_data["inflow"].to(device)
            precip_seq = rollout_data["precipitation"].to(device)
            initial_volume = x_iter[
                :, 12 + args.n_time_steps : 12 + 2 * args.n_time_steps
            ][:, -1].clone()

            for step in range(args.rollout_length):
                hdf_step = args.hdf_start_index + step
                if hdf_step >= face_velocity_for_hydrograph.shape[0]:
                    break

                volume_window = x_iter[
                    :, 12 + args.n_time_steps : 12 + 2 * args.n_time_steps
                ]
                pred = model(x_iter, edge_features, graph)
                pred_delta = pred[:, 1] * volume_std
                autoregressive_gt_delta = (
                    volume_gt_seq[step] - volume_window[:, -1]
                ) * volume_std
                gt_previous_volume = (
                    initial_volume if step == 0 else volume_gt_seq[step - 1]
                )
                ground_truth_delta = (
                    volume_gt_seq[step] - gt_previous_volume
                ) * volume_std
                proxy_target_delta = (
                    ground_truth_delta
                    if args.proxy_scale_reference == "ground_truth"
                    else autoregressive_gt_delta
                )
                proxy = face_delta_proxy(
                    face_velocity_for_hydrograph[hdf_step],
                    face_graph,
                    x_iter.shape[0],
                    args.delta_t,
                )
                scale = scale_proxy_to_target(proxy, proxy_target_delta)
                proxy_scaled = proxy * scale

                pred_residual = pred_delta - proxy_scaled
                gt_residual = proxy_target_delta - proxy_scaled
                add_metric(metric_sums, "pred_face_residual_rmse", rmse(pred_residual))
                add_metric(metric_sums, "gt_face_residual_rmse", rmse(gt_residual))
                add_metric(
                    metric_sums,
                    "pred_vs_gt_delta_rmse",
                    rmse(pred_delta - proxy_target_delta),
                )
                add_metric(metric_sums, "face_proxy_scale", scale)

                for zone in range(4):
                    mask = zone_label == zone
                    if torch.any(mask):
                        add_metric(
                            metric_sums,
                            f"zone_{zone}_pred_face_residual_rmse",
                            rmse(pred_residual[mask]),
                        )
                        add_metric(
                            metric_sums,
                            f"zone_{zone}_gt_face_residual_rmse",
                            rmse(gt_residual[mask]),
                        )
                        add_metric(
                            metric_sums,
                            f"zone_{zone}_pred_vs_gt_delta_rmse",
                            rmse(pred_delta[mask] - proxy_target_delta[mask]),
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
        "hdf_start_index": args.hdf_start_index,
        "matched_hdf_event": args.matched_hdf_event,
        "proxy_scale_reference": args.proxy_scale_reference,
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
    parser.add_argument("--hdf-file", type=Path)
    parser.add_argument(
        "--hdf-glob",
        help="Event-specific HEC-RAS HDF glob. Event IDs are inferred from planH1/H1/T1 path parts.",
    )
    parser.add_argument("--face-graph-file", required=True, type=Path)
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--n-time-steps", type=int, default=2)
    parser.add_argument("--rollout-length", type=int, default=10)
    parser.add_argument("--hdf-start-index", type=int, default=0)
    parser.add_argument("--delta-t", type=float, default=1200.0)
    parser.add_argument(
        "--proxy-scale-reference",
        choices=("ground_truth", "autoregressive"),
        default="ground_truth",
        help=(
            "Transition used to calibrate the HEC-RAS face proxy. ground_truth "
            "uses observed previous volumes and gives checkpoint-independent "
            "scales; autoregressive reproduces the legacy rollout diagnostic."
        ),
    )
    parser.add_argument("--zone-label-file", default="zone_label.txt")
    parser.add_argument("--zone-weight-file", default="zone_weight.txt")
    parser.add_argument("--num-input-features", type=int, default=16)
    parser.add_argument("--num-edge-features", type=int, default=3)
    parser.add_argument("--num-output-features", type=int, default=2)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument(
        "--matched-hdf-event",
        action="store_true",
        help="Set only when the HDF face velocity event matches the evaluated hydrographs.",
    )
    parser.add_argument("--checkpoint", action="append", nargs=2, required=True)
    args = parser.parse_args()

    if (args.hdf_file is None) == (args.hdf_glob is None):
        parser.error("Provide exactly one of --hdf-file or --hdf-glob.")

    if not args.matched_hdf_event:
        print(
            "WARNING: --matched-hdf-event was not set. Treat face residuals as "
            "alignment/scale diagnostics, not final physics metrics."
        )

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
