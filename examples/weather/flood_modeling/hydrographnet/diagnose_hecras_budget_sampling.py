#!/usr/bin/env python3
"""Diagnose how stored HEC-RAS cell-flow samples relate to storage change.

This script does not fit or apply a calibration factor.  It compares several
physically distinct interpretations of the stored m3/s samples so that a
sparse-output aliasing problem is not mistaken for a unit-conversion problem.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import numpy as np

from evaluate_hecras_physical_budget import (
    GEOMETRY_BASE,
    RESULT_BASE,
    RESULT_TIME_PATH,
    infer_native_volume_contract,
    map_hgn_nodes_to_hdf_cells,
    match_hgn_times_to_hdf,
    remap_face_cells,
)


def metrics(
    target: np.ndarray,
    budget: np.ndarray,
    node_indices: np.ndarray | None = None,
    observation_mask: np.ndarray | None = None,
) -> dict[str, object]:
    residual = target - budget
    if observation_mask is None:
        observation_mask = np.ones(target.shape, dtype=bool)
    if observation_mask.shape != target.shape or not np.any(observation_mask):
        raise ValueError("The diagnostic observation mask is empty or has a wrong shape.")
    target_flat = target[observation_mask]
    budget_flat = budget[observation_mask]
    residual_flat = residual[observation_mask]
    count_by_node = np.sum(observation_mask, axis=0)
    valid_local_nodes = np.flatnonzero(count_by_node > 0)
    per_node_rmse = np.sqrt(
        np.sum(np.where(observation_mask, residual**2, 0.0), axis=0)[valid_local_nodes]
        / count_by_node[valid_local_nodes]
    )
    target_rms = float(np.sqrt(np.mean(target_flat**2)))
    correlation = float("nan")
    if np.std(target_flat) > 0 and np.std(budget_flat) > 0:
        correlation = float(np.corrcoef(target_flat, budget_flat)[0, 1])
    denominator = float(np.dot(budget_flat, budget_flat))
    diagnostic_slope = (
        float(np.dot(target_flat, budget_flat) / denominator)
        if denominator > 0
        else float("nan")
    )
    top_node_positions = np.argsort(per_node_rmse)[-10:][::-1]
    if node_indices is None:
        node_indices = np.arange(target.shape[1])
    masked_absolute_residual = np.where(observation_mask, np.abs(residual), -np.inf)
    maximum_flat_index = int(np.argmax(masked_absolute_residual))
    maximum_transition, maximum_local_node = np.unravel_index(
        maximum_flat_index, residual.shape
    )
    return {
        "target_rms_volume": target_rms,
        "budget_rms_volume": float(np.sqrt(np.mean(budget_flat**2))),
        "residual_rmse_volume": float(np.sqrt(np.mean(residual_flat**2))),
        "relative_rmse": float(
            np.sqrt(np.mean(residual_flat**2)) / max(target_rms, 1e-12)
        ),
        "correlation": correlation,
        "diagnostic_slope_not_applied": diagnostic_slope,
        "domain_residual_rmse_volume": float(
            np.sqrt(
                np.mean(
                    np.sum(np.where(observation_mask, residual, 0.0), axis=1) ** 2
                )
            )
        ),
        "num_observations": int(np.sum(observation_mask)),
        "absolute_residual_p95_volume": float(
            np.percentile(np.abs(residual_flat), 95)
        ),
        "absolute_residual_p99_volume": float(
            np.percentile(np.abs(residual_flat), 99)
        ),
        "absolute_residual_max_volume": float(np.max(np.abs(residual_flat))),
        "per_node_rmse_p50_volume": float(np.percentile(per_node_rmse, 50)),
        "per_node_rmse_p95_volume": float(np.percentile(per_node_rmse, 95)),
        "per_node_rmse_p99_volume": float(np.percentile(per_node_rmse, 99)),
        "per_node_rmse_max_volume": float(np.max(per_node_rmse)),
        "maximum_residual_transition_offset": int(maximum_transition),
        "maximum_residual_node": int(node_indices[maximum_local_node]),
        "top_residual_nodes": [
            {
                "node": int(node_indices[valid_local_nodes[position]]),
                "rmse_volume": float(per_node_rmse[position]),
            }
            for position in top_node_positions
        ],
    }


def face_values_to_node_balance(
    face_values: np.ndarray, mapped_face_cells: np.ndarray, num_nodes: int
) -> np.ndarray:
    """Apply the HEC face orientation to obtain signed node inflow."""
    node_balance = np.zeros((face_values.shape[0], num_nodes), dtype=np.float64)
    first = (mapped_face_cells[:, 0] >= 0) & (mapped_face_cells[:, 0] < num_nodes)
    second = (mapped_face_cells[:, 1] >= 0) & (mapped_face_cells[:, 1] < num_nodes)
    for time_index, values in enumerate(face_values):
        np.add.at(node_balance[time_index], mapped_face_cells[first, 0], -values[first])
        np.add.at(node_balance[time_index], mapped_face_cells[second, 1], values[second])
    return node_balance


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--event-id", required=True)
    parser.add_argument("--hdf-path", type=Path, required=True)
    parser.add_argument("--zone-label-file", default="zone_label.txt")
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int)
    parser.add_argument("--wet-depth-threshold", type=float, default=0.01)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    hdf_path = args.hdf_path.resolve()
    prefix = args.prefix
    event_id = args.event_id

    volume = np.loadtxt(data_dir / f"{prefix}_V_{event_id}.txt")
    depth = np.loadtxt(data_dir / f"{prefix}_WD_{event_id}.txt")
    hgn_xy = np.loadtxt(data_dir / f"{prefix}_XY.txt")
    hgn_time_days = np.loadtxt(data_dir / f"{prefix}_US_InF_{event_id}.txt")[:, 0]
    zone_label = np.loadtxt(data_dir / args.zone_label_file).astype(np.int64)
    end_index = args.end_index if args.end_index is not None else volume.shape[0] - 1
    if not 1 <= args.start_index <= end_index < volume.shape[0]:
        raise ValueError("Expected 1 <= start-index <= end-index < number of frames.")

    with h5py.File(hdf_path, "r") as hdf:
        volume_unit, precip_to_length = infer_native_volume_contract(hdf)
        hdf_time_days = np.asarray(hdf[RESULT_TIME_PATH], dtype=np.float64)
        hdf_xy = np.asarray(
            hdf[GEOMETRY_BASE + "Cells Center Coordinate"], dtype=np.float64
        )
        original_indices = map_hgn_nodes_to_hdf_cells(hgn_xy, hdf_xy)
        face_cells = np.asarray(
            hdf[GEOMETRY_BASE + "Faces Cell Indexes"], dtype=np.int64
        )
        mapped_face_cells = remap_face_cells(face_cells, original_indices)
        surface_area = np.asarray(
            hdf[GEOMETRY_BASE + "Cells Surface Area"], dtype=np.float64
        )[original_indices]
        balance_dataset = hdf[RESULT_BASE + "Cell Flow Balance"]
        balance_unit = balance_dataset.attrs.get("Units", b"").decode(
            "utf-8", errors="replace"
        )
        target_time_indices = match_hgn_times_to_hdf(hgn_time_days, hdf_time_days)
        selected_hdf = target_time_indices[args.start_index - 1 : end_index + 1]
        balance = np.nan_to_num(
            np.asarray(balance_dataset[selected_hdf][:, original_indices], dtype=np.float64),
            nan=0.0,
        )
        cumulative_precip = np.asarray(
            hdf[RESULT_BASE + "Cell Cumulative Precipitation Depth"][selected_hdf][
                :, original_indices
            ],
            dtype=np.float64,
        )
        cell_volume_error = None
        cell_volume_error_unit = None
        cell_volume_error_path = RESULT_BASE + "Cell Volume Error"
        if cell_volume_error_path in hdf:
            dataset = hdf[cell_volume_error_path]
            cell_volume_error = np.nan_to_num(
                np.asarray(dataset[selected_hdf][:, original_indices], dtype=np.float64),
                nan=0.0,
            )
            cell_volume_error_unit = dataset.attrs.get("Units", b"").decode(
                "utf-8", errors="replace"
            )
        optional_face_paths = {
            "period_average": RESULT_BASE + "Face Flow Period Average",
            "cumulative_volume": RESULT_BASE + "Face Cumulative Volume",
        }
        optional_face_data = {}
        optional_face_units = {}
        for name, path in optional_face_paths.items():
            if path not in hdf:
                continue
            dataset = hdf[path]
            optional_face_data[name] = np.nan_to_num(
                np.asarray(dataset[selected_hdf], dtype=np.float64), nan=0.0
            )
            optional_face_units[name] = dataset.attrs.get("Units", b"").decode(
                "utf-8", errors="replace"
            )

    time_seconds = hdf_time_days[selected_hdf] * 86400.0
    dt = np.diff(time_seconds)[:, None]
    target_delta = volume[args.start_index : end_index + 1] - volume[
        args.start_index - 1 : end_index
    ]
    precip_delta = (
        np.diff(cumulative_precip, axis=0)
        * precip_to_length
        * surface_area[None, :]
    )
    transport_target = target_delta - precip_delta

    candidate_transport = {
        "left_endpoint_rate": balance[:-1] * dt,
        "right_endpoint_rate": balance[1:] * dt,
        "trapezoid_rate": 0.5 * (balance[:-1] + balance[1:]) * dt,
        "stored_value_as_volume": balance[1:],
    }
    if "period_average" in optional_face_data:
        period_average_node = face_values_to_node_balance(
            optional_face_data.pop("period_average"), mapped_face_cells, volume.shape[1]
        )
        candidate_transport["face_period_average_right"] = period_average_node[1:] * dt
        candidate_transport["face_period_average_left"] = period_average_node[:-1] * dt
        if cell_volume_error is not None:
            error_volume = cell_volume_error * surface_area[None, :]
            period_transport = period_average_node[1:] * dt
            candidate_transport["period_average_plus_right_cell_volume_error"] = (
                period_transport + error_volume[1:]
            )
            candidate_transport["period_average_minus_right_cell_volume_error"] = (
                period_transport - error_volume[1:]
            )
            candidate_transport["period_average_plus_cell_volume_error_difference"] = (
                period_transport + np.diff(error_volume, axis=0)
            )
            candidate_transport["period_average_minus_cell_volume_error_difference"] = (
                period_transport - np.diff(error_volume, axis=0)
            )
    if "cumulative_volume" in optional_face_data:
        cumulative_node = face_values_to_node_balance(
            optional_face_data.pop("cumulative_volume"), mapped_face_cells, volume.shape[1]
        )
        candidate_transport["face_cumulative_volume_difference"] = np.diff(
            cumulative_node, axis=0
        )
    wet_previous = (
        depth[args.start_index - 1 : end_index] > args.wet_depth_threshold
    )
    wet_current = depth[args.start_index : end_index + 1] > args.wet_depth_threshold
    wet_either = wet_previous | wet_current
    wet_both = wet_previous & wet_current
    scopes = [
        ("all", np.ones(volume.shape[1], dtype=bool), None),
        ("zone_3", zone_label == 3, None),
        ("zone_3_wet_either", zone_label == 3, wet_either),
        ("zone_3_wet_both", zone_label == 3, wet_both),
    ]
    rows = []
    for candidate, transport in candidate_transport.items():
        for scope, mask, dynamic_mask in scopes:
            node_indices = np.flatnonzero(mask)
            scoped_observation_mask = (
                None if dynamic_mask is None else dynamic_mask[:, mask]
            )
            row = {
                "candidate": candidate,
                "scope": scope,
                "num_nodes": int(mask.sum()),
                "num_transitions": int(target_delta.shape[0]),
            }
            row.update(
                metrics(
                    target_delta[:, mask],
                    (transport + precip_delta)[:, mask],
                    node_indices,
                    scoped_observation_mask,
                )
            )
            transport_metrics = metrics(
                transport_target[:, mask],
                transport[:, mask],
                node_indices,
                scoped_observation_mask,
            )
            row.update(
                {f"transport_{key}": value for key, value in transport_metrics.items()}
            )
            rows.append(row)

    report = {
        "event_id": event_id,
        "hdf_path": str(hdf_path),
        "volume_unit": volume_unit,
        "cell_flow_balance_unit": balance_unit,
        "optional_face_units": optional_face_units,
        "cell_volume_error_unit": cell_volume_error_unit,
        "start_index": args.start_index,
        "end_index": end_index,
        "wet_depth_threshold_m": args.wet_depth_threshold,
        "median_output_interval_seconds": float(np.median(dt)),
        "uses_fitted_scale": False,
        "diagnostic_slope_note": (
            "Reported only to diagnose units/semantics; it is never applied to a budget."
        ),
        "rows": rows,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# HEC-RAS 5-minute Cell-Balance Sampling Diagnostic",
        "",
        f"- Event: `{event_id}`",
        f"- HDF: `{hdf_path}`",
        f"- Stored balance unit: `{balance_unit}`; integrated volume unit: `{volume_unit}`",
        f"- Median stored interval: `{float(np.median(dt)):.6f} s`",
        "- Fitted scale applied: **No**",
        "- `diagnostic slope` is reported only to distinguish a unit error from temporal aliasing.",
        "",
        "| Candidate | Scope | Relative RMSE | Residual RMSE | Correlation | Diagnostic slope (not applied) | Domain residual RMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['candidate']} | {row['scope']} | {row['relative_rmse']:.6f} "
            f"| {row['residual_rmse_volume']:.6f} | {row['correlation']:.6f} "
            f"| {row['diagnostic_slope_not_applied']:.6f} "
            f"| {row['domain_residual_rmse_volume']:.6f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation rule",
            "",
            "- A candidate is usable as an uncalibrated target only when its slope is near 1 and its residual is independently small.",
            "- If left/right/trapezoid candidates all remain poor, 5-minute instantaneous samples do not identify the solver-step-integrated flux.",
            "- The diagnostic does not alter targets, train a model, or authorize a local-conservation claim.",
        ]
    )
    args.output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {args.output_json}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
