# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Fit an inference-available proxy for HEC-RAS Face Flow transition scale.

The target-aware transition RMS from ``compute_hecras_face_target_scale_stats.py``
is useful as a diagnostic, but it cannot be used at inference time because it is
computed from the true HEC-RAS Face Flow target.  This script fits a simple
ridge model that predicts the transition RMS from quantities available in the
HydroGraphNet input stream, such as current water depth, current volume,
upstream inflow, downstream outflow, and precipitation.
"""

import argparse
import csv
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "bias",
    "time_fraction",
    "upstream_inflow",
    "downstream_outflow",
    "precipitation",
    "total_volume",
    "mean_volume",
    "max_volume",
    "mean_water_depth",
    "max_water_depth",
    "p90_water_depth",
    "wet_fraction_1e_3",
    "wet_fraction_1e_2",
    "delta_upstream_inflow",
    "delta_precipitation",
    "delta_total_volume",
)


def read_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def load_1d_or_last_column(path: Path) -> np.ndarray:
    values = np.loadtxt(path, dtype=np.float64)
    if values.ndim == 1:
        return values
    return values[:, -1]


def hydrograph_sort_key(hydrograph_id: str) -> tuple[str, int]:
    prefix = "".join(ch for ch in hydrograph_id if not ch.isdigit())
    number = "".join(ch for ch in hydrograph_id if ch.isdigit())
    return prefix, int(number) if number else -1


def build_feature_rows(
    data_dir: Path,
    scale_stats_npz: Path,
    hydrograph_ids: list[str],
    prefix: str,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, float | int | str]]]:
    scale_data = np.load(scale_stats_npz)
    rows = []
    features = []
    targets = []
    for hydrograph_id in sorted(hydrograph_ids, key=hydrograph_sort_key):
        transition_key = f"{hydrograph_id}_transition_rms"
        if transition_key not in scale_data.files:
            raise KeyError(f"Missing {transition_key!r} in {scale_stats_npz}.")
        transition_rms = np.asarray(scale_data[transition_key], dtype=np.float64)
        wd = np.loadtxt(data_dir / f"{prefix}_WD_{hydrograph_id}.txt", dtype=np.float64)
        volume = np.loadtxt(data_dir / f"{prefix}_V_{hydrograph_id}.txt", dtype=np.float64)
        precip = load_1d_or_last_column(data_dir / f"{prefix}_Pr_{hydrograph_id}.txt")
        upstream = load_1d_or_last_column(data_dir / f"{prefix}_US_InF_{hydrograph_id}.txt")
        downstream = load_1d_or_last_column(data_dir / f"{prefix}_DS_OuF_{hydrograph_id}.txt")
        if wd.shape[0] < transition_rms.shape[0] or volume.shape[0] < transition_rms.shape[0]:
            raise ValueError(
                f"{hydrograph_id} has fewer HGN timesteps than transition RMS entries."
            )
        total_volume = np.sum(volume, axis=1)
        num_nodes = wd.shape[1]
        for transition_index, target in enumerate(transition_rms):
            t = transition_index
            previous_t = max(t - 1, 0)
            time_fraction = t / max(len(transition_rms) - 1, 1)
            current_wd = wd[t]
            current_volume = volume[t]
            feature = np.asarray(
                [
                    1.0,
                    time_fraction,
                    upstream[t],
                    downstream[t],
                    precip[t],
                    total_volume[t],
                    np.mean(current_volume),
                    np.max(current_volume),
                    np.mean(current_wd),
                    np.max(current_wd),
                    np.quantile(current_wd, 0.90),
                    np.mean(current_wd > 1e-3),
                    np.mean(current_wd > 1e-2),
                    upstream[t] - upstream[previous_t],
                    precip[t] - precip[previous_t],
                    total_volume[t] - total_volume[previous_t],
                ],
                dtype=np.float64,
            )
            features.append(feature)
            targets.append(target)
            rows.append(
                {
                    "hydrograph_id": hydrograph_id,
                    "transition_index": transition_index,
                    "num_nodes": num_nodes,
                    "target_transition_rms": float(target),
                    **{
                        name: float(value)
                        for name, value in zip(FEATURE_NAMES, feature)
                        if name != "bias"
                    },
                }
            )
    return np.vstack(features), np.asarray(targets, dtype=np.float64), rows


def fit_ridge_log_model(features: np.ndarray, targets: np.ndarray, alpha: float):
    x = features.copy()
    mean = x[:, 1:].mean(axis=0)
    std = np.maximum(x[:, 1:].std(axis=0), 1e-12)
    x[:, 1:] = (x[:, 1:] - mean) / std
    y = np.log1p(np.maximum(targets, 0.0))
    penalty = np.eye(x.shape[1], dtype=np.float64) * alpha
    penalty[0, 0] = 0.0
    coef = np.linalg.solve(x.T @ x + penalty, x.T @ y)
    return coef, mean, std


def predict_log_model(features: np.ndarray, coef: np.ndarray, mean: np.ndarray, std: np.ndarray):
    x = features.copy()
    x[:, 1:] = (x[:, 1:] - mean) / std
    return np.expm1(x @ coef)


def metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    pred = np.maximum(np.asarray(pred, dtype=np.float64), 0.0)
    target = np.asarray(target, dtype=np.float64)
    residual = pred - target
    target_rms = np.sqrt(np.mean(target * target))
    rmse = np.sqrt(np.mean(residual * residual))
    rel_rmse = rmse / max(target_rms, 1e-12)
    mae = np.mean(np.abs(residual))
    corr = np.corrcoef(pred, target)[0, 1] if pred.size > 1 else np.nan
    return {
        "target_rms": float(target_rms),
        "pred_rms": float(np.sqrt(np.mean(pred * pred))),
        "rmse": float(rmse),
        "relative_rmse": float(rel_rmse),
        "mae": float(mae),
        "correlation": float(corr),
        "bias": float(np.mean(residual)),
        "target_p50": float(np.quantile(target, 0.50)),
        "target_p90": float(np.quantile(target, 0.90)),
        "pred_p50": float(np.quantile(pred, 0.50)),
        "pred_p90": float(np.quantile(pred, 0.90)),
    }


def attach_predictions(rows, predictions):
    return [
        {
            **row,
            "pred_transition_rms": float(pred),
            "scale_residual": float(pred - row["target_transition_rms"]),
            "scale_ratio": float(
                pred / max(float(row["target_transition_rms"]), 1e-12)
            ),
        }
        for row, pred in zip(rows, predictions)
    ]


def build_predicted_scale_payload(rows: list[dict[str, float | int | str]]) -> dict:
    event_ids = []
    transition_values = {}
    for row in rows:
        hydrograph_id = str(row["hydrograph_id"])
        event_ids.append(hydrograph_id)
    event_ids = sorted(set(event_ids), key=hydrograph_sort_key)
    all_values = []
    event_rms = []
    payload = {"event_ids": np.asarray(event_ids)}
    for hydrograph_id in event_ids:
        event_rows = [
            row for row in rows if str(row["hydrograph_id"]) == hydrograph_id
        ]
        event_rows.sort(key=lambda row: int(row["transition_index"]))
        values = np.asarray(
            [float(row["pred_transition_rms"]) for row in event_rows],
            dtype=np.float32,
        )
        transition_values[hydrograph_id] = values
        all_values.append(values)
        event_rms.append(float(np.sqrt(np.mean(values.astype(np.float64) ** 2))))
        payload[f"{hydrograph_id}_transition_rms"] = values
    all_values_flat = np.concatenate(all_values).astype(np.float64)
    payload["event_rms"] = np.asarray(event_rms, dtype=np.float32)
    payload["global_rms"] = np.asarray(
        [np.sqrt(np.mean(all_values_flat * all_values_flat))], dtype=np.float32
    )
    payload["num_events"] = np.asarray([len(event_ids)], dtype=np.int64)
    payload["num_transitions"] = np.asarray([all_values_flat.shape[0]], dtype=np.int64)
    return payload


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    for key in ("split", "hydrograph_id", "transition_index"):
        if key in fieldnames:
            fieldnames.remove(key)
    ordered = [key for key in ("split", "hydrograph_id", "transition_index") if key in rows[0]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered + fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--train-ids-file", required=True, type=Path)
    parser.add_argument("--test-ids-file", required=True, type=Path)
    parser.add_argument("--train-scale-stats-npz", required=True, type=Path)
    parser.add_argument("--test-scale-stats-npz", required=True, type=Path)
    parser.add_argument("--alpha", type=float, default=1e-2)
    parser.add_argument("--output-model-npz", required=True, type=Path)
    parser.add_argument("--output-predicted-scale-npz", type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    args = parser.parse_args()

    train_ids = read_ids(args.data_dir / args.train_ids_file)
    test_ids = read_ids(args.data_dir / args.test_ids_file)
    train_x, train_y, train_rows = build_feature_rows(
        args.data_dir, args.train_scale_stats_npz, train_ids, args.prefix
    )
    test_x, test_y, test_rows = build_feature_rows(
        args.data_dir, args.test_scale_stats_npz, test_ids, args.prefix
    )

    coef, mean, std = fit_ridge_log_model(train_x, train_y, args.alpha)
    train_pred = predict_log_model(train_x, coef, mean, std)
    test_pred = predict_log_model(test_x, coef, mean, std)
    train_metrics = metrics(train_pred, train_y)
    test_metrics = metrics(test_pred, test_y)

    args.output_model_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_model_npz,
        feature_names=np.asarray(FEATURE_NAMES),
        coefficient=coef.astype(np.float64),
        feature_mean=mean.astype(np.float64),
        feature_std=std.astype(np.float64),
        alpha=np.asarray([args.alpha], dtype=np.float64),
        train_ids=np.asarray(train_ids),
        test_ids=np.asarray(test_ids),
    )

    output_rows = [
        {"split": "train", **row}
        for row in attach_predictions(train_rows, train_pred)
    ] + [
        {"split": "test", **row}
        for row in attach_predictions(test_rows, test_pred)
    ]
    write_csv(args.output_csv, output_rows)
    if args.output_predicted_scale_npz is not None:
        args.output_predicted_scale_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            args.output_predicted_scale_npz,
            **build_predicted_scale_payload(output_rows),
        )

    with args.summary_md.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# HEC-RAS Face Scale Proxy Fit\n\n")
        handle.write(
            "This diagnostic predicts Face Flow transition RMS from inference-available "
            "HGN quantities. It is intended to replace target-aware transition scale "
            "in future edge-head experiments.\n\n"
        )
        handle.write(f"- train ids: `{args.train_ids_file}` ({len(train_ids)} events)\n")
        handle.write(f"- test ids: `{args.test_ids_file}` ({len(test_ids)} events)\n")
        handle.write(f"- alpha: `{args.alpha}`\n")
        handle.write("\n## Metrics\n\n")
        handle.write("| Split | Target RMS | Pred RMS | RMSE | Relative RMSE | MAE | Corr | Bias |\n")
        handle.write("|---|---:|---:|---:|---:|---:|---:|---:|\n")
        for split, item in (("train", train_metrics), ("test", test_metrics)):
            handle.write(
                f"| {split} | `{item['target_rms']:.6f}` | "
                f"`{item['pred_rms']:.6f}` | `{item['rmse']:.6f}` | "
                f"`{item['relative_rmse']:.6f}` | `{item['mae']:.6f}` | "
                f"`{item['correlation']:.6f}` | `{item['bias']:.6f}` |\n"
            )
        handle.write("\n## Largest Coefficients\n\n")
        scaled_coef = list(zip(FEATURE_NAMES, coef))
        scaled_coef.sort(key=lambda item: abs(item[1]), reverse=True)
        for name, value in scaled_coef[:8]:
            handle.write(f"- `{name}`: `{value:.6f}`\n")
        handle.write("\n## Outputs\n\n")
        handle.write(f"- model NPZ: `{args.output_model_npz}`\n")
        if args.output_predicted_scale_npz is not None:
            handle.write(
                f"- predicted scale NPZ: `{args.output_predicted_scale_npz}`\n"
            )
        handle.write(f"- predictions CSV: `{args.output_csv}`\n")

    print(f"Wrote {args.output_model_npz}")
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.summary_md}")
    print(f"test_relative_rmse={test_metrics['relative_rmse']:.6f}")
    print(f"test_correlation={test_metrics['correlation']:.6f}")


if __name__ == "__main__":
    main()
