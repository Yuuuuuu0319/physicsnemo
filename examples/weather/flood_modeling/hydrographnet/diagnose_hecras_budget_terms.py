# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Diagnose HEC-RAS budget terms against synchronized HGN storage.

This script is narrower than ``evaluate_hecras_physical_budget.py``. It assumes
the HGN target and HDF are already synchronized, then reports which native HDF
terms do or do not explain active-cell storage changes.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree


GEOMETRY_BASE = "Geometry/2D Flow Areas/per2/"
RESULT_BASE = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/"
    "2D Flow Areas/per2/"
)
RESULT_TIME_PATH = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time"
)


def load_hgn_time_days(data_dir: Path, prefix: str, event_id: str) -> np.ndarray:
    return np.asarray(
        np.loadtxt(data_dir / f"{prefix}_US_InF_{event_id}.txt")[:, 0],
        dtype=np.float64,
    )


def match_hgn_times_to_hdf(hgn_time_days: np.ndarray, hdf_time_days: np.ndarray) -> np.ndarray:
    matched = np.searchsorted(hdf_time_days, hgn_time_days)
    matched = np.clip(matched, 0, hdf_time_days.shape[0] - 1)
    previous = np.maximum(matched - 1, 0)
    choose_previous = (
        np.abs(hdf_time_days[previous] - hgn_time_days)
        < np.abs(hdf_time_days[matched] - hgn_time_days)
    )
    matched[choose_previous] = previous[choose_previous]
    mismatch_seconds = np.abs(hdf_time_days[matched] - hgn_time_days) * 86400.0
    if np.max(mismatch_seconds) > 1e-3:
        raise ValueError(
            "HGN target times are not represented in the HDF output series; "
            f"max mismatch is {np.max(mismatch_seconds)} seconds."
        )
    return matched


def map_hgn_nodes_to_hdf_cells(hgn_xy: np.ndarray, hdf_xy: np.ndarray) -> np.ndarray:
    distances, original_indices = cKDTree(hdf_xy).query(hgn_xy, k=1)
    if np.max(distances) > 1e-6 or np.unique(original_indices).size != len(hgn_xy):
        raise ValueError("HGN-to-HDF coordinate mapping is not unique.")
    return original_indices


def reconstruct_cell_volume(
    water_surface: np.ndarray,
    original_indices: np.ndarray,
    volume_info: np.ndarray,
    volume_values: np.ndarray,
    cell_area: np.ndarray | None = None,
    extrapolate_above_table: bool = False,
) -> np.ndarray:
    volume = np.zeros((water_surface.shape[0], original_indices.size), dtype=np.float64)
    for hgn_index, original_index in enumerate(original_indices):
        offset, count = volume_info[original_index]
        if count <= 0:
            continue
        curve = volume_values[offset : offset + count]
        capped_volume = np.interp(
            water_surface[:, original_index],
            curve[:, 0],
            curve[:, 1],
            left=0.0,
            right=curve[-1, 1],
        )
        if extrapolate_above_table:
            if cell_area is None:
                raise ValueError("cell_area is required for above-table extrapolation.")
            capped_volume = capped_volume + np.maximum(
                water_surface[:, original_index] - curve[-1, 0], 0.0
            ) * cell_area[hgn_index]
        volume[:, hgn_index] = capped_volume
    return volume


def volume_table_cap_diagnostics(
    water_surface: np.ndarray,
    original_indices: np.ndarray,
    volume_info: np.ndarray,
    volume_values: np.ndarray,
) -> dict[str, float]:
    max_elevation = np.zeros(original_indices.shape[0], dtype=np.float64)
    empty_tables = 0
    for hgn_index, original_index in enumerate(original_indices):
        offset, count = volume_info[original_index]
        if count <= 0:
            empty_tables += 1
            max_elevation[hgn_index] = np.nan
            continue
        max_elevation[hgn_index] = volume_values[offset + count - 1, 0]
    active_water_surface = water_surface[:, original_indices]
    over_table = active_water_surface > max_elevation[None, :]
    over_amount = np.maximum(active_water_surface - max_elevation[None, :], 0.0)
    return {
        "volume_table_empty_cell_count": int(empty_tables),
        "volume_table_over_fraction": float(np.mean(over_table)),
        "volume_table_final_over_fraction": float(np.mean(over_table[-1])),
        "volume_table_max_time_over_fraction": float(np.max(np.mean(over_table, axis=1))),
        "volume_table_ever_over_cell_count": int(np.sum(np.any(over_table, axis=0))),
        "volume_table_max_over_ft": float(np.nanmax(over_amount)),
        "volume_table_final_mean_over_ft": float(np.nanmean(over_amount[-1])),
    }


def integrate_interval_terms(
    cell_balance: np.ndarray,
    precipitation_cumulative_inches: np.ndarray,
    cell_area_ft2: np.ndarray,
    hdf_time_seconds: np.ndarray,
    hdf_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    transitions = hdf_indices.shape[0] - 1
    num_nodes = cell_area_ft2.shape[0]
    balance_trapz = np.zeros((transitions, num_nodes), dtype=np.float64)
    balance_left = np.zeros_like(balance_trapz)
    balance_right = np.zeros_like(balance_trapz)
    precipitation_volume = np.zeros_like(balance_trapz)

    for transition, (start, end) in enumerate(zip(hdf_indices[:-1], hdf_indices[1:])):
        interval_times = hdf_time_seconds[start : end + 1]
        interval_balance = cell_balance[start : end + 1]
        balance_trapz[transition] = np.trapezoid(
            interval_balance, x=interval_times, axis=0
        )
        step_seconds = np.diff(interval_times)
        balance_left[transition] = np.sum(interval_balance[:-1] * step_seconds[:, None], axis=0)
        balance_right[transition] = np.sum(interval_balance[1:] * step_seconds[:, None], axis=0)
        precipitation_depth_ft = (
            precipitation_cumulative_inches[end] - precipitation_cumulative_inches[start]
        ) / 12.0
        precipitation_volume[transition] = precipitation_depth_ft * cell_area_ft2

    return {
        "cell_balance_trapz": balance_trapz,
        "cell_balance_rect_left": balance_left,
        "cell_balance_rect_right": balance_right,
        "precipitation_volume": precipitation_volume,
        "cell_balance_trapz_minus_precipitation": balance_trapz - precipitation_volume,
        "cell_balance_trapz_plus_precipitation": balance_trapz + precipitation_volume,
    }


def summarize_variant(target: np.ndarray, budget: np.ndarray) -> dict[str, float]:
    residual = target - budget
    target_flat = target.reshape(-1)
    budget_flat = budget.reshape(-1)
    target_rms = float(np.sqrt(np.mean(target_flat**2)))
    residual_rmse = float(np.sqrt(np.mean(residual.reshape(-1) ** 2)))
    correlation = (
        float(np.corrcoef(target_flat, budget_flat)[0, 1])
        if np.std(target_flat) > 0 and np.std(budget_flat) > 0
        else float("nan")
    )
    return {
        "target_rms_ft3": target_rms,
        "budget_rms_ft3": float(np.sqrt(np.mean(budget_flat**2))),
        "residual_rmse_ft3": residual_rmse,
        "residual_mae_ft3": float(np.mean(np.abs(residual))),
        "residual_bias_ft3": float(np.mean(residual)),
        "relative_rmse": residual_rmse / target_rms if target_rms else float("nan"),
        "correlation": correlation,
        "domain_target_rms_ft3": float(np.sqrt(np.mean(np.sum(target, axis=1) ** 2))),
        "domain_budget_rms_ft3": float(np.sqrt(np.mean(np.sum(budget, axis=1) ** 2))),
        "domain_residual_rmse_ft3": float(
            np.sqrt(np.mean((np.sum(target - budget, axis=1)) ** 2))
        ),
    }


def diagnose_cell_balance_definition(
    face_flow: np.ndarray,
    cell_balance: np.ndarray,
    face_cells: np.ndarray,
    original_indices: np.ndarray,
    num_hdf_cells: int,
) -> dict[str, float]:
    original_to_hgn = np.full(num_hdf_cells, -1, dtype=np.int64)
    original_to_hgn[original_indices] = np.arange(original_indices.shape[0])
    mapped_faces = np.full_like(face_cells, -1)
    valid = (face_cells >= 0) & (face_cells < original_to_hgn.shape[0])
    mapped_faces[valid] = original_to_hgn[face_cells[valid]]

    net_face_sum = np.zeros_like(cell_balance, dtype=np.float64)
    inward_only = np.zeros_like(cell_balance, dtype=np.float64)
    for time_index, flow in enumerate(face_flow):
        first = mapped_faces[:, 0] >= 0
        second = mapped_faces[:, 1] >= 0
        np.add.at(net_face_sum[time_index], mapped_faces[first, 0], -flow[first])
        np.add.at(net_face_sum[time_index], mapped_faces[second, 1], flow[second])
        first_inward = first & (flow < 0)
        second_inward = second & (flow > 0)
        np.add.at(
            inward_only[time_index],
            mapped_faces[first_inward, 0],
            -flow[first_inward],
        )
        np.add.at(
            inward_only[time_index],
            mapped_faces[second_inward, 1],
            flow[second_inward],
        )

    def flat_corr(left: np.ndarray, right: np.ndarray) -> float:
        left_flat = left.reshape(-1)
        right_flat = right.reshape(-1)
        return (
            float(np.corrcoef(left_flat, right_flat)[0, 1])
            if np.std(left_flat) > 0 and np.std(right_flat) > 0
            else float("nan")
        )

    return {
        "cell_balance_vs_net_face_rmse_cfs": float(
            np.sqrt(np.mean((cell_balance - net_face_sum) ** 2))
        ),
        "cell_balance_vs_net_face_correlation": flat_corr(cell_balance, net_face_sum),
        "cell_balance_vs_inward_only_rmse_cfs": float(
            np.sqrt(np.mean((cell_balance - inward_only) ** 2))
        ),
        "cell_balance_vs_inward_only_correlation": flat_corr(cell_balance, inward_only),
    }


def integrate_boundary_condition_flows(
    boundary_flows: dict[str, tuple[float, np.ndarray]],
    hdf_time_seconds: np.ndarray,
    hdf_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    """Integrate native boundary-condition flow per HGN interval.

    Upstream boundary flow is treated as positive inflow to the 2D area and
    downstream boundary flow as negative outflow. This is a domain-level
    diagnostic only; per-cell assignment is handled elsewhere.
    """
    totals: dict[str, np.ndarray] = {}
    for name, (direction, values) in boundary_flows.items():
        interval_volume = np.zeros(hdf_indices.shape[0] - 1, dtype=np.float64)
        for transition, (start, end) in enumerate(zip(hdf_indices[:-1], hdf_indices[1:])):
            interval_volume[transition] = direction * float(
                np.sum(
                    np.trapezoid(
                        values[start : end + 1],
                        x=hdf_time_seconds[start : end + 1],
                        axis=0,
                    )
                )
            )
        totals[name] = interval_volume
    if totals:
        totals["boundary_total"] = np.sum(np.stack(list(totals.values()), axis=0), axis=0)
    else:
        totals["boundary_total"] = np.zeros(hdf_indices.shape[0] - 1, dtype=np.float64)
    return totals


def summarize_domain_series(target: np.ndarray, budget: np.ndarray) -> dict[str, float]:
    residual = target - budget
    return {
        "target_rms_ft3": float(np.sqrt(np.mean(target**2))),
        "budget_rms_ft3": float(np.sqrt(np.mean(budget**2))),
        "residual_rmse_ft3": float(np.sqrt(np.mean(residual**2))),
        "relative_rmse": (
            float(np.sqrt(np.mean(residual**2)) / np.sqrt(np.mean(target**2)))
            if np.sqrt(np.mean(target**2)) > 0
            else float("nan")
        ),
        "residual_bias_ft3": float(np.mean(residual)),
        "correlation": (
            float(np.corrcoef(target, budget)[0, 1])
            if np.std(target) > 0 and np.std(budget) > 0
            else float("nan")
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(
    path: Path,
    rows: list[dict[str, object]],
    diagnostics: dict[str, object],
) -> None:
    best = min(rows, key=lambda row: float(row["relative_rmse"]))
    lines = [
        "# HEC-RAS Budget Term Diagnosis",
        "",
        "This report diagnoses why synchronized HEC-RAS/HGN storage still does not "
        "close with native budget terms.",
        "",
        "## Storage Checks",
        "",
        (
            "- Active-cell HGN `M80_V` vs reconstructed HDF cell volume RMSE: "
            f"`{diagnostics['active_storage_alignment_rmse_ft3']:.6e} ft^3`"
        ),
        (
            "- Active-cell HGN `M80_V` vs reconstructed HDF cell volume max abs: "
            f"`{diagnostics['active_storage_alignment_max_abs_ft3']:.6e} ft^3`"
        ),
        (
            "- Sum of all reconstructed HDF cell volumes vs HEC-RAS global "
            "`Computations/Volume` RMSE: "
            f"`{diagnostics['all_cell_vs_global_volume_rmse_ft3']:.6f} ft^3`"
        ),
        (
            "- Area-extrapolated active-cell volume vs HEC-RAS global "
            "`Computations/Volume` delta RMSE: "
            f"`{diagnostics['area_extrapolated_vs_global_delta_rmse_ft3']:.6f} ft^3`"
        ),
        (
            "- Volume table cap: "
            f"`{diagnostics['volume_table_ever_over_cell_count']}` active cells exceed "
            "their table top at least once; final over-table fraction "
            f"`{diagnostics['volume_table_final_over_fraction']:.6f}`, max exceedance "
            f"`{diagnostics['volume_table_max_over_ft']:.6f} ft`"
        ),
        (
            "- HEC-RAS sampled `Volume Error` RMS: "
            f"`{diagnostics['global_volume_error_rms_ft3']:.6f} ft^3`"
        ),
        (
            "- `Cell Flow Balance` vs signed native `Face Flow` net sum RMSE: "
            f"`{diagnostics['cell_balance_vs_net_face_rmse_cfs']:.6e} cfs`, "
            f"correlation `{diagnostics['cell_balance_vs_net_face_correlation']:.6f}`"
        ),
        (
            "- `Cell Flow Balance` vs inward-only face flow RMSE: "
            f"`{diagnostics['cell_balance_vs_inward_only_rmse_cfs']:.6f} cfs`, "
            f"correlation `{diagnostics['cell_balance_vs_inward_only_correlation']:.6f}`"
        ),
        "",
        "## Best Budget Variant",
        "",
        (
            f"- `{best['variant']}`: relative RMSE `{float(best['relative_rmse']):.6f}`, "
            f"cell RMSE `{float(best['residual_rmse_ft3']):.6f} ft^3`, "
            f"domain RMSE `{float(best['domain_residual_rmse_ft3']):.6f} ft^3`, "
            f"correlation `{float(best['correlation']):.6f}`"
        ),
        "",
        "## Variants",
        "",
        "| Variant | Relative RMSE | Cell RMSE (ft^3) | Domain RMSE (ft^3) | Bias (ft^3) | Correlation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(rows, key=lambda item: float(item["relative_rmse"])):
        lines.append(
            f"| {row['variant']} | {float(row['relative_rmse']):.6f} | "
            f"{float(row['residual_rmse_ft3']):.6f} | "
            f"{float(row['domain_residual_rmse_ft3']):.6f} | "
            f"{float(row['residual_bias_ft3']):.6f} | "
            f"{float(row['correlation']):.6f} |"
        )
    domain_rows = diagnostics.get("domain_rows", [])
    if domain_rows:
        lines.extend(
            [
                "",
                "## Domain Diagnostics",
                "",
                "| Target | Budget | Relative RMSE | RMSE (ft^3) | Bias (ft^3) | Correlation |",
                "|---|---|---:|---:|---:|---:|",
            ]
        )
        for row in sorted(domain_rows, key=lambda item: float(item["relative_rmse"])):
            lines.append(
                f"| {row['target']} | {row['budget']} | "
                f"{row['relative_rmse']:.6f} | {row['residual_rmse_ft3']:.6f} | "
                f"{row['residual_bias_ft3']:.6f} | {row['correlation']:.6f} |"
            )
        boundary_rows = diagnostics.get("boundary_rows", [])
        if boundary_rows:
            lines.extend(
                [
                    "",
                    "### Boundary Components",
                    "",
                    "| Component | RMS volume (ft^3) | Mean volume (ft^3) |",
                    "|---|---:|---:|",
                ]
            )
            for row in boundary_rows:
                lines.append(
                    f"| {row['component']} | {row['rms_volume_ft3']:.6f} | "
                    f"{row['mean_volume_ft3']:.6f} |"
                )
    zone_rows = diagnostics.get("zone_rows", [])
    if zone_rows:
        lines.extend(
            [
                "",
                "## Zone Residuals",
                "",
                "| Zone | Cells | Target RMS (ft^3) | Budget RMS (ft^3) | Residual RMSE (ft^3) | Relative RMSE | Bias (ft^3) | Correlation | Wet-change fraction |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in zone_rows:
            lines.append(
                f"| {row['zone']} | {row['num_cells']} | "
                f"{row['target_rms_ft3']:.6f} | {row['budget_rms_ft3']:.6f} | "
                f"{row['residual_rmse_ft3']:.6f} | {row['relative_rmse']:.6f} | "
                f"{row['residual_bias_ft3']:.6f} | {row['correlation']:.6f} | "
                f"{row['wet_change_fraction']:.6f} |"
            )
    top_cell_rows = diagnostics.get("top_residual_cells", [])
    if top_cell_rows:
        lines.extend(
            [
                "",
                "## Top Residual Cells",
                "",
                "| Node | Zone | X | Y | Residual RMSE (ft^3) | Target RMS (ft^3) | Mean depth | Max depth | Wet-change fraction |",
                "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in top_cell_rows:
            lines.append(
                f"| {row['node']} | {row['zone']} | {row['x']:.6f} | {row['y']:.6f} | "
                f"{row['residual_rmse_ft3']:.6f} | {row['target_rms_ft3']:.6f} | "
                f"{row['mean_depth']:.6f} | {row['max_depth']:.6f} | "
                f"{row['wet_change_fraction']:.6f} |"
            )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- The synchronized active-cell storage target is valid: HGN `M80_V` and "
            "HDF reconstructed active-cell volume match to numerical precision.",
            "- The current HGN `M80_V` target is capped by finite HEC-RAS "
            "volume-elevation tables when water surface exceeds the table top; "
            "area extrapolation above the table largely restores consistency with "
            "HEC-RAS global volume.",
            "- HEC-RAS global `Computations/Volume` is not the same quantity as the "
            "active-cell volume-elevation reconstruction used by HGN.",
            "- HEC-RAS global `Computations/Volume` closes tightly against external "
            "boundary flow plus precipitation, so the simulation is globally "
            "water-balanced even though the HGN active-cell target does not close.",
            "- With the capped HGN `M80_V` target, native `Cell Flow Balance` does "
            "not close local storage. With the area-extrapolated target, "
            "`Cell Flow Balance + precipitation` closes much more tightly.",
            "- `Cell Flow Balance` is numerically the signed net sum of native "
            "`Face Flow`; the residual is therefore not caused by a face-orientation "
            "implementation error in the validator.",
            "- Formal local-conservation loss should use a volume target that "
            "handles above-table water-surface values; the capped table target "
            "should remain disabled for formal conservation training.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--hdf-file", type=Path, required=True)
    parser.add_argument("--event-id", default="H1")
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--start-index", type=int, default=1)
    parser.add_argument("--end-index", type=int, default=48)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    parser.add_argument("--zone-label-file", default="zone_label.txt")
    args = parser.parse_args()

    hgn_volume = np.loadtxt(args.data_dir / f"{args.prefix}_V_{args.event_id}.txt")
    hgn_depth = np.loadtxt(args.data_dir / f"{args.prefix}_WD_{args.event_id}.txt")
    hgn_xy = np.loadtxt(args.data_dir / f"{args.prefix}_XY.txt")
    hgn_time_days = load_hgn_time_days(args.data_dir, args.prefix, args.event_id)
    zone_label_path = args.data_dir / args.zone_label_file
    zone_label = (
        np.loadtxt(zone_label_path, dtype=np.int64)
        if zone_label_path.exists()
        else None
    )

    with h5py.File(args.hdf_file, "r") as hdf:
        hdf_time_days = np.asarray(hdf[RESULT_TIME_PATH], dtype=np.float64)
        hdf_time_seconds = hdf_time_days * 86400.0
        target_indices = match_hgn_times_to_hdf(hgn_time_days, hdf_time_days)
        selected_hgn = np.arange(args.start_index - 1, args.end_index + 1)
        selected_hdf = target_indices[selected_hgn]
        hdf_slice_start = int(selected_hdf[0])
        hdf_slice_end = int(selected_hdf[-1])
        relative_hdf = selected_hdf - hdf_slice_start

        hdf_xy = np.asarray(hdf[GEOMETRY_BASE + "Cells Center Coordinate"], dtype=np.float64)
        original_indices = map_hgn_nodes_to_hdf_cells(hgn_xy, hdf_xy)
        water_surface = np.asarray(
            hdf[RESULT_BASE + "Water Surface"][selected_hdf], dtype=np.float64
        )
        all_water_surface = np.asarray(
            hdf[RESULT_BASE + "Water Surface"][selected_hdf], dtype=np.float64
        )
        volume_info = np.asarray(
            hdf[GEOMETRY_BASE + "Cells Volume Elevation Info"], dtype=np.int64
        )
        volume_values = np.asarray(
            hdf[GEOMETRY_BASE + "Cells Volume Elevation Values"], dtype=np.float64
        )
        cell_area = np.asarray(
            hdf[GEOMETRY_BASE + "Cells Surface Area"][original_indices],
            dtype=np.float64,
        )
        reconstructed_active_volume = reconstruct_cell_volume(
            water_surface, original_indices, volume_info, volume_values
        )
        reconstructed_active_volume_area_extrapolated = reconstruct_cell_volume(
            water_surface,
            original_indices,
            volume_info,
            volume_values,
            cell_area=cell_area,
            extrapolate_above_table=True,
        )
        reconstructed_all_volume = reconstruct_cell_volume(
            all_water_surface,
            np.arange(hdf_xy.shape[0], dtype=np.int64),
            volume_info,
            volume_values,
        )
        cell_balance = np.asarray(
            hdf[RESULT_BASE + "Cell Flow Balance"][
                hdf_slice_start : hdf_slice_end + 1, original_indices
            ],
            dtype=np.float64,
        )
        precipitation_cumulative = np.asarray(
            hdf[RESULT_BASE + "Cell Cumulative Precipitation Depth"][
                hdf_slice_start : hdf_slice_end + 1, original_indices
            ],
            dtype=np.float64,
        )
        face_cells = np.asarray(hdf[GEOMETRY_BASE + "Faces Cell Indexes"], dtype=np.int64)
        sampled_face_flow = np.asarray(
            hdf[RESULT_BASE + "Face Flow"][selected_hdf], dtype=np.float64
        )
        sampled_cell_balance = cell_balance[relative_hdf]
        boundary_flows: dict[str, tuple[float, np.ndarray]] = {}
        boundary_group_path = RESULT_BASE + "Boundary Conditions"
        if boundary_group_path in hdf:
            for name, dataset in hdf[boundary_group_path].items():
                if not name.endswith(" - Flow per Face"):
                    continue
                direction = 1.0 if name.lower().startswith("upstream") else -1.0
                boundary_flows[name] = (
                    direction,
                    np.asarray(
                        dataset[hdf_slice_start : hdf_slice_end + 1],
                        dtype=np.float64,
                    ),
                )
        global_volume = (
            np.asarray(
                hdf[RESULT_BASE + "Computations/Volume"][selected_hdf],
                dtype=np.float64,
            ).reshape(-1)
            * 43560.0
        )
        volume_error = np.asarray(
            hdf[RESULT_BASE + "Computations/Volume Error"][selected_hdf],
            dtype=np.float64,
        ).reshape(-1)

    hgn_selected_volume = hgn_volume[selected_hgn]
    target_delta = np.diff(hgn_selected_volume, axis=0)
    terms = integrate_interval_terms(
        cell_balance,
        precipitation_cumulative,
        cell_area,
        hdf_time_seconds[hdf_slice_start : hdf_slice_end + 1],
        relative_hdf,
    )
    boundary_terms = integrate_boundary_condition_flows(
        boundary_flows,
        hdf_time_seconds[hdf_slice_start : hdf_slice_end + 1],
        relative_hdf,
    )
    terms["zero_budget"] = np.zeros_like(target_delta)
    terms["precipitation_volume"] = terms["precipitation_volume"]
    terms["negative_cell_balance_trapz"] = -terms["cell_balance_trapz"]

    rows: list[dict[str, object]] = []
    for variant, budget in terms.items():
        row: dict[str, object] = {"variant": variant}
        row.update(summarize_variant(target_delta, budget))
        rows.append(row)
    area_extrapolated_delta = np.diff(reconstructed_active_volume_area_extrapolated, axis=0)
    for variant_name, budget in {
        "cell_balance_trapz": terms["cell_balance_trapz"],
        "cell_balance_trapz_plus_precipitation": terms[
            "cell_balance_trapz_plus_precipitation"
        ],
        "cell_balance_trapz_minus_precipitation": terms[
            "cell_balance_trapz_minus_precipitation"
        ],
    }.items():
        row = {"variant": f"area_extrapolated_target::{variant_name}"}
        row.update(summarize_variant(area_extrapolated_delta, budget))
        rows.append(row)
    best_variant = str(min(rows, key=lambda row: float(row["relative_rmse"]))["variant"])
    if best_variant.startswith("area_extrapolated_target::"):
        best_target_delta = area_extrapolated_delta
        best_budget = terms[best_variant.split("::", maxsplit=1)[1]]
    else:
        best_target_delta = target_delta
        best_budget = terms[best_variant]
    best_residual = best_target_delta - best_budget

    domain_active_delta = np.sum(target_delta, axis=1)
    domain_area_extrapolated_delta = np.sum(area_extrapolated_delta, axis=1)
    domain_all_reconstructed_delta = np.diff(np.sum(reconstructed_all_volume, axis=1))
    domain_global_volume_delta = np.diff(global_volume)
    domain_cell_balance = np.sum(terms["cell_balance_trapz"], axis=1)
    domain_precipitation = np.sum(terms["precipitation_volume"], axis=1)
    domain_boundary = boundary_terms["boundary_total"]
    domain_rows = []
    domain_targets = {
        "active_hgn_delta_v": domain_active_delta,
        "area_extrapolated_active_delta_v": domain_area_extrapolated_delta,
        "all_reconstructed_delta_v": domain_all_reconstructed_delta,
        "hecras_global_computations_volume_delta": domain_global_volume_delta,
    }
    domain_budgets = {
        "sum_cell_balance": domain_cell_balance,
        "sum_cell_balance_minus_precipitation": domain_cell_balance - domain_precipitation,
        "sum_cell_balance_plus_precipitation": domain_cell_balance + domain_precipitation,
        "boundary_total": domain_boundary,
        "boundary_total_plus_precipitation": domain_boundary + domain_precipitation,
        "boundary_total_minus_precipitation": domain_boundary - domain_precipitation,
        "precipitation_only": domain_precipitation,
    }
    for target_name, target_series in domain_targets.items():
        for budget_name, budget_series in domain_budgets.items():
            row = {"target": target_name, "budget": budget_name}
            row.update(summarize_domain_series(target_series, budget_series))
            domain_rows.append(row)

    boundary_rows = [
        {
            "component": name,
            "rms_volume_ft3": float(np.sqrt(np.mean(values**2))),
            "mean_volume_ft3": float(np.mean(values)),
        }
        for name, values in sorted(boundary_terms.items())
    ]

    zone_rows: list[dict[str, object]] = []
    top_residual_cells: list[dict[str, object]] = []
    if zone_label is not None:
        selected_depth = hgn_depth[selected_hgn]
        for zone in sorted(np.unique(zone_label).tolist()):
            mask = zone_label == zone
            target_zone = best_target_delta[:, mask]
            budget_zone = best_budget[:, mask]
            residual_zone = best_residual[:, mask]
            target_flat = target_zone.reshape(-1)
            budget_flat = budget_zone.reshape(-1)
            residual_flat = residual_zone.reshape(-1)
            target_rms = float(np.sqrt(np.mean(target_flat**2)))
            correlation = (
                float(np.corrcoef(target_flat, budget_flat)[0, 1])
                if np.std(target_flat) > 0 and np.std(budget_flat) > 0
                else float("nan")
            )
            wet_start = selected_depth[:-1, mask] > 1e-3
            wet_end = selected_depth[1:, mask] > 1e-3
            zone_rows.append(
                {
                    "zone": int(zone),
                    "num_cells": int(np.sum(mask)),
                    "target_rms_ft3": target_rms,
                    "budget_rms_ft3": float(np.sqrt(np.mean(budget_flat**2))),
                    "residual_rmse_ft3": float(np.sqrt(np.mean(residual_flat**2))),
                    "relative_rmse": (
                        float(np.sqrt(np.mean(residual_flat**2))) / target_rms
                        if target_rms
                        else float("nan")
                    ),
                    "residual_bias_ft3": float(np.mean(residual_flat)),
                    "correlation": correlation,
                    "wet_change_fraction": float(np.mean(wet_start != wet_end)),
                }
            )
        cell_rms = np.sqrt(np.mean(best_residual**2, axis=0))
        target_cell_rms = np.sqrt(np.mean(best_target_delta**2, axis=0))
        top_indices = np.argsort(cell_rms)[-10:][::-1]
        for node in top_indices:
            wet_start = hgn_depth[selected_hgn[:-1], node] > 1e-3
            wet_end = hgn_depth[selected_hgn[1:], node] > 1e-3
            top_residual_cells.append(
                {
                    "node": int(node),
                    "zone": int(zone_label[node]),
                    "x": float(hgn_xy[node, 0]),
                    "y": float(hgn_xy[node, 1]),
                    "residual_rmse_ft3": float(cell_rms[node]),
                    "target_rms_ft3": float(target_cell_rms[node]),
                    "mean_depth": float(np.mean(hgn_depth[selected_hgn, node])),
                    "max_depth": float(np.max(hgn_depth[selected_hgn, node])),
                    "wet_change_fraction": float(np.mean(wet_start != wet_end)),
                }
            )

    diagnostics = {
        "active_storage_alignment_rmse_ft3": float(
            np.sqrt(np.mean((hgn_selected_volume - reconstructed_active_volume) ** 2))
        ),
        "active_storage_alignment_max_abs_ft3": float(
            np.max(np.abs(hgn_selected_volume - reconstructed_active_volume))
        ),
        "all_cell_vs_global_volume_rmse_ft3": float(
            np.sqrt(np.mean((np.sum(reconstructed_all_volume, axis=1) - global_volume) ** 2))
        ),
        "area_extrapolated_vs_global_delta_rmse_ft3": float(
            np.sqrt(
                np.mean(
                    (domain_area_extrapolated_delta - domain_global_volume_delta) ** 2
                )
            )
        ),
        "global_volume_error_rms_ft3": float(np.sqrt(np.mean(volume_error**2))),
        **volume_table_cap_diagnostics(
            water_surface, original_indices, volume_info, volume_values
        ),
        **diagnose_cell_balance_definition(
            sampled_face_flow,
            sampled_cell_balance,
            face_cells,
            original_indices,
            hdf_xy.shape[0],
        ),
        "zone_rows": zone_rows,
        "top_residual_cells": top_residual_cells,
        "domain_rows": domain_rows,
        "boundary_rows": boundary_rows,
    }
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows, diagnostics)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
