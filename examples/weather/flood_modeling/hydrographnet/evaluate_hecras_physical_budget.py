# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Validate an uncalibrated HEC-RAS local storage budget for HGN events.

This diagnostic is intentionally separate from model inference and training.
For each HGN cell and output interval it compares the observed volume change
against a reconstructed HEC-RAS budget in native volume units:

    delta_volume = sum(signed face velocity * wetted face area * delta_t)
                   + optional precipitation volume.

No fitted scale factor is used. A low residual here is a prerequisite for
using HDF face terms as a formal local-conservation training target.
"""

import argparse
import csv
import glob
import re
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
FACE_STAGE_RULES = ("mean", "minimum", "maximum")
SIGN_MODES = ("cell_order", "negated")
SOURCE_MODES = ("none", "hdf_precipitation")
TARGET_VOLUME_SOURCES = ("hgn_target", "hdf_reconstructed")
BOUNDARY_MODES = ("reconstructed_all_touching_faces", "native_bc_only")
RECONSTRUCTED_FLUX_SOURCE = "reconstructed_velocity_area"
NATIVE_FLUX_SOURCE = "native_face_flow"


def infer_hydrograph_id_from_path(path: Path) -> str:
    """Infer an H1/T1 style event identifier from a HEC-RAS HDF path."""
    for part in reversed(path.parts):
        match = re.fullmatch(r"plan([HT]\d+)", part, flags=re.IGNORECASE)
        if match is not None:
            return match.group(1).upper()
    for part in reversed(path.parts):
        match = re.fullmatch(r"([HT]\d+)", part, flags=re.IGNORECASE)
        if match is not None:
            return match.group(1).upper()
    raise ValueError(f"Could not infer hydrograph ID from HDF path: {path}")


def load_event_ids(path: Path) -> list[str]:
    with path.open() as event_file:
        return [line.strip() for line in event_file if line.strip()]


def load_hdf_paths(hdf_glob: str) -> dict[str, Path]:
    hdf_paths = sorted(Path(path) for path in glob.glob(hdf_glob))
    if not hdf_paths:
        raise FileNotFoundError(f"No HEC-RAS HDF files matched: {hdf_glob}")
    event_paths = {}
    for path in hdf_paths:
        event_id = infer_hydrograph_id_from_path(path)
        if event_id in event_paths:
            raise ValueError(f"Multiple HDF files map to event {event_id}.")
        event_paths[event_id] = path
    return event_paths


def infer_output_delta_t_seconds(inflow_path: Path) -> float:
    inflow = np.loadtxt(inflow_path)
    time_days = np.asarray(inflow[:, 0], dtype=np.float64)
    intervals = np.diff(time_days) * 86400.0
    if intervals.size == 0 or not np.allclose(intervals, intervals[0], atol=1e-3):
        raise ValueError(f"Non-constant or empty event time axis in {inflow_path}")
    return float(intervals[0])


def match_hgn_times_to_hdf(hgn_time_days: np.ndarray, hdf_time_days: np.ndarray) -> np.ndarray:
    """Return HDF output indices corresponding to each HGN target time."""
    matched = np.searchsorted(hdf_time_days, hgn_time_days)
    matched = np.clip(matched, 0, hdf_time_days.shape[0] - 1)
    previous = np.maximum(matched - 1, 0)
    choose_previous = (
        np.abs(hdf_time_days[previous] - hgn_time_days)
        < np.abs(hdf_time_days[matched] - hgn_time_days)
    )
    matched[choose_previous] = previous[choose_previous]
    differences_seconds = np.abs(hdf_time_days[matched] - hgn_time_days) * 86400.0
    if np.max(differences_seconds) > 1e-3:
        raise ValueError(
            "HGN target times are not represented in the HDF output series; "
            f"max mismatch is {np.max(differences_seconds)} seconds."
        )
    return matched


def map_hgn_nodes_to_hdf_cells(
    hgn_xy: np.ndarray, hdf_xy: np.ndarray, tolerance: float = 1e-6
) -> np.ndarray:
    """Map active HGN node order back to original HDF cell indices by XY."""
    distances, original_indices = cKDTree(hdf_xy).query(hgn_xy, k=1)
    if np.max(distances) > tolerance or np.unique(original_indices).size != len(hgn_xy):
        raise ValueError("HGN-to-HDF coordinate mapping is not unique or exceeds tolerance.")
    return original_indices


def remap_face_cells(face_cells: np.ndarray, original_indices: np.ndarray) -> np.ndarray:
    original_to_hgn = np.full(
        max(int(np.max(face_cells)), int(np.max(original_indices))) + 1,
        -1,
        dtype=np.int64,
    )
    original_to_hgn[original_indices] = np.arange(original_indices.shape[0])
    mapped = np.full_like(face_cells, -1)
    valid = (face_cells >= 0) & (face_cells < original_to_hgn.shape[0])
    mapped[valid] = original_to_hgn[face_cells[valid]]
    return mapped


def interpolate_face_area(
    face_stage: np.ndarray, area_info: np.ndarray, area_values: np.ndarray
) -> np.ndarray:
    """Interpolate HEC-RAS wetted hydraulic area at each face stage."""
    wetted_area = np.zeros_like(face_stage, dtype=np.float64)
    for face_index, (offset, count) in enumerate(area_info):
        curve = area_values[offset : offset + count]
        stage = np.nan_to_num(face_stage[:, face_index], nan=curve[0, 0])
        wetted_area[:, face_index] = np.interp(
            stage, curve[:, 0], curve[:, 1], left=0.0, right=curve[-1, 1]
        )
    return wetted_area


def reconstruct_cell_volume(
    water_surface: np.ndarray,
    original_indices: np.ndarray,
    volume_info: np.ndarray,
    volume_values: np.ndarray,
) -> np.ndarray:
    reconstructed = np.zeros((water_surface.shape[0], original_indices.size))
    for hgn_index, original_index in enumerate(original_indices):
        offset, count = volume_info[original_index]
        curve = volume_values[offset : offset + count]
        reconstructed[:, hgn_index] = np.interp(
            water_surface[:, original_index],
            curve[:, 0],
            curve[:, 1],
            left=0.0,
            right=curve[-1, 1],
        )
    return reconstructed


def face_stage_from_cells(
    water_surface: np.ndarray, face_cells: np.ndarray, stage_rule: str
) -> np.ndarray:
    num_times = water_surface.shape[0]
    stage = np.full((num_times, face_cells.shape[0]), np.nan, dtype=np.float64)
    valid_0 = (face_cells[:, 0] >= 0) & (face_cells[:, 0] < water_surface.shape[1])
    valid_1 = (face_cells[:, 1] >= 0) & (face_cells[:, 1] < water_surface.shape[1])
    first = np.full_like(stage, np.nan)
    second = np.full_like(stage, np.nan)
    first[:, valid_0] = water_surface[:, face_cells[valid_0, 0]]
    second[:, valid_1] = water_surface[:, face_cells[valid_1, 1]]
    pair = np.stack([first, second], axis=0)
    with np.errstate(all="ignore"):
        if stage_rule == "mean":
            stage = np.nanmean(pair, axis=0)
        elif stage_rule == "minimum":
            stage = np.nanmin(pair, axis=0)
        elif stage_rule == "maximum":
            stage = np.nanmax(pair, axis=0)
        else:
            raise ValueError(f"Unsupported face stage rule: {stage_rule}")
    return stage


def integrate_node_transport_delta(
    discharge: np.ndarray,
    face_cells: np.ndarray,
    num_nodes: int,
    hdf_time_days: np.ndarray,
    target_time_indices: np.ndarray,
    sign_mode: str,
) -> np.ndarray:
    """Integrate face discharge between consecutive HGN target time points."""
    signed_discharge = discharge if sign_mode == "cell_order" else -discharge
    transport_delta = np.zeros(
        (target_time_indices.shape[0] - 1, num_nodes), dtype=np.float64
    )
    first_mask = (face_cells[:, 0] >= 0) & (face_cells[:, 0] < num_nodes)
    second_mask = (face_cells[:, 1] >= 0) & (face_cells[:, 1] < num_nodes)
    hdf_time_seconds = hdf_time_days * 86400.0
    for transition_index, (start, end) in enumerate(
        zip(target_time_indices[:-1], target_time_indices[1:])
    ):
        if end <= start:
            raise ValueError("Target times must map to increasing HDF output indices.")
        face_volume = np.trapezoid(
            signed_discharge[start : end + 1],
            x=hdf_time_seconds[start : end + 1],
            axis=0,
        )
        np.add.at(
            transport_delta[transition_index],
            face_cells[first_mask, 0],
            -face_volume[first_mask],
        )
        np.add.at(
            transport_delta[transition_index],
            face_cells[second_mask, 1],
            face_volume[second_mask],
        )
    return transport_delta


def integrate_transport_from_hdf(
    hdf_path: Path,
    hdf_face_cells: np.ndarray,
    mapped_face_cells: np.ndarray,
    num_nodes: int,
    hdf_time_days: np.ndarray,
    target_time_indices: np.ndarray,
    area_info: np.ndarray,
    area_values: np.ndarray,
    stage_rule: str,
    sign_mode: str,
    boundary_mode: str,
    flux_source: str,
) -> np.ndarray:
    """Stream HDF face samples and integrate transport over HGN intervals."""
    transport_delta = np.zeros(
        (target_time_indices.shape[0] - 1, num_nodes), dtype=np.float64
    )
    valid_first = (mapped_face_cells[:, 0] >= 0) & (
        mapped_face_cells[:, 0] < num_nodes
    )
    valid_second = (mapped_face_cells[:, 1] >= 0) & (
        mapped_face_cells[:, 1] < num_nodes
    )
    if boundary_mode == "native_bc_only":
        first_mask = valid_first & valid_second
        second_mask = first_mask
    elif boundary_mode == "reconstructed_all_touching_faces":
        first_mask = valid_first
        second_mask = valid_second
    else:
        raise ValueError(f"Unsupported boundary mode: {boundary_mode}")
    hdf_time_seconds = hdf_time_days * 86400.0
    sign = 1.0 if sign_mode == "cell_order" else -1.0

    def accumulate_interval(transition_index: int, discharge: np.ndarray, times: np.ndarray):
        face_volume = np.trapezoid(discharge, x=times, axis=0)
        np.add.at(
            transport_delta[transition_index],
            mapped_face_cells[first_mask, 0],
            -face_volume[first_mask],
        )
        np.add.at(
            transport_delta[transition_index],
            mapped_face_cells[second_mask, 1],
            face_volume[second_mask],
        )

    with h5py.File(hdf_path, "r") as hdf:
        if flux_source == NATIVE_FLUX_SOURCE:
            flow_dataset = hdf[RESULT_BASE + "Face Flow"]
        else:
            velocity_dataset = hdf[RESULT_BASE + "Face Velocity"]
            water_surface_dataset = hdf[RESULT_BASE + "Water Surface"]
        total_selected_steps = target_time_indices[-1] - target_time_indices[0] + 1
        if total_selected_steps <= 1000:
            start = target_time_indices[0]
            end = target_time_indices[-1]
            if flux_source == NATIVE_FLUX_SOURCE:
                discharge = sign * np.asarray(
                    flow_dataset[start : end + 1], dtype=np.float64
                )
            else:
                water_surface = np.asarray(
                    water_surface_dataset[start : end + 1], dtype=np.float64
                )
                face_velocity = np.asarray(
                    velocity_dataset[start : end + 1], dtype=np.float64
                )
                face_stage = face_stage_from_cells(
                    water_surface, hdf_face_cells, stage_rule
                )
                wetted_area = interpolate_face_area(face_stage, area_info, area_values)
                discharge = sign * np.nan_to_num(
                    face_velocity * wetted_area, nan=0.0
                )
            for transition_index, (left, right) in enumerate(
                zip(target_time_indices[:-1], target_time_indices[1:])
            ):
                left -= start
                right -= start
                accumulate_interval(
                    transition_index,
                    discharge[left : right + 1],
                    hdf_time_seconds[left + start : right + start + 1],
                )
        else:
            for transition_index, (start, end) in enumerate(
                zip(target_time_indices[:-1], target_time_indices[1:])
            ):
                if flux_source == NATIVE_FLUX_SOURCE:
                    discharge = sign * np.asarray(
                        flow_dataset[start : end + 1], dtype=np.float64
                    )
                else:
                    water_surface = np.asarray(
                        water_surface_dataset[start : end + 1], dtype=np.float64
                    )
                    face_velocity = np.asarray(
                        velocity_dataset[start : end + 1], dtype=np.float64
                    )
                    face_stage = face_stage_from_cells(
                        water_surface, hdf_face_cells, stage_rule
                    )
                    wetted_area = interpolate_face_area(
                        face_stage, area_info, area_values
                    )
                    discharge = sign * np.nan_to_num(
                        face_velocity * wetted_area, nan=0.0
                    )
                accumulate_interval(
                    transition_index,
                    discharge,
                    hdf_time_seconds[start : end + 1],
                )
        if boundary_mode == "native_bc_only":
            boundary_group = hdf[RESULT_BASE + "Boundary Conditions"]
            for name in boundary_group:
                if not name.endswith(" - Flow per Face"):
                    continue
                dataset = boundary_group[name]
                faces = np.asarray(dataset.attrs["Faces"], dtype=np.int64)
                direction = 1.0 if name.lower().startswith("upstream") else -1.0
                native_flow = np.asarray(dataset, dtype=np.float64)
                for transition_index, (start, end) in enumerate(
                    zip(target_time_indices[:-1], target_time_indices[1:])
                ):
                    face_volume = direction * np.trapezoid(
                        native_flow[start : end + 1],
                        x=hdf_time_seconds[start : end + 1],
                        axis=0,
                    )
                    for column, face_index in enumerate(faces):
                        nodes = mapped_face_cells[face_index]
                        valid_nodes = nodes[(nodes >= 0) & (nodes < num_nodes)]
                        if valid_nodes.size == 1:
                            transport_delta[transition_index, valid_nodes[0]] += (
                                face_volume[column]
                            )
    return transport_delta


def metric_dict(target: np.ndarray, budget: np.ndarray) -> dict[str, float]:
    residual = target - budget
    target_flat = target.reshape(-1)
    budget_flat = budget.reshape(-1)
    residual_flat = residual.reshape(-1)
    target_rms = np.sqrt(np.mean(target_flat**2))
    if np.std(target_flat) > 0.0 and np.std(budget_flat) > 0.0:
        correlation = float(np.corrcoef(target_flat, budget_flat)[0, 1])
    else:
        correlation = float("nan")
    return {
        "target_rms_ft3": float(target_rms),
        "budget_rms_ft3": float(np.sqrt(np.mean(budget_flat**2))),
        "residual_rmse_ft3": float(np.sqrt(np.mean(residual_flat**2))),
        "residual_mae_ft3": float(np.mean(np.abs(residual_flat))),
        "residual_bias_ft3": float(np.mean(residual_flat)),
        "relative_rmse": float(
            np.sqrt(np.mean(residual_flat**2)) / max(target_rms, 1e-12)
        ),
        "correlation": correlation,
    }


def evaluate_event(
    event_id: str,
    hdf_path: Path,
    data_dir: Path,
    prefix: str,
    zone_label: np.ndarray,
    start_index: int,
    end_index: int | None,
    storage_alignment_atol: float,
    target_volume_sources: tuple[str, ...],
) -> list[dict]:
    volume = np.loadtxt(data_dir / f"{prefix}_V_{event_id}.txt")
    hgn_xy = np.loadtxt(data_dir / f"{prefix}_XY.txt")
    num_nodes = volume.shape[1]
    if zone_label.shape[0] != num_nodes:
        raise ValueError("Zone-label count does not match HGN volume columns.")
    inflow = np.loadtxt(data_dir / f"{prefix}_US_InF_{event_id}.txt")
    if end_index is not None:
        volume = volume[: end_index + 1]
        inflow = inflow[: end_index + 1]
    hgn_time_days = np.asarray(inflow[:, 0], dtype=np.float64)
    delta_t = infer_output_delta_t_seconds(data_dir / f"{prefix}_US_InF_{event_id}.txt")
    with h5py.File(hdf_path, "r") as hdf:
        face_cells = np.asarray(
            hdf[GEOMETRY_BASE + "Faces Cell Indexes"], dtype=np.int64
        )
        hdf_xy = np.asarray(
            hdf[GEOMETRY_BASE + "Cells Center Coordinate"], dtype=np.float64
        )
        area_info = np.asarray(
            hdf[GEOMETRY_BASE + "Faces Area Elevation Info"], dtype=np.int64
        )
        area_values = np.asarray(
            hdf[GEOMETRY_BASE + "Faces Area Elevation Values"], dtype=np.float64
        )
        volume_info = np.asarray(
            hdf[GEOMETRY_BASE + "Cells Volume Elevation Info"], dtype=np.int64
        )
        volume_values = np.asarray(
            hdf[GEOMETRY_BASE + "Cells Volume Elevation Values"], dtype=np.float64
        )
        all_cell_surface_area = np.asarray(
            hdf[GEOMETRY_BASE + "Cells Surface Area"], dtype=np.float64
        )
        hdf_time_days = np.asarray(hdf[RESULT_TIME_PATH], dtype=np.float64)
        has_native_face_flow = RESULT_BASE + "Face Flow" in hdf
        has_native_cell_flow_balance = RESULT_BASE + "Cell Flow Balance" in hdf
        computation_delta_t = float(
            np.nanmedian(np.asarray(hdf[RESULT_BASE + "Computations/Time Step"]))
        )
        hdf_output_delta_t = float(np.nanmedian(np.diff(hdf_time_days)) * 86400.0)

    target_time_indices = match_hgn_times_to_hdf(hgn_time_days, hdf_time_days)
    original_indices = map_hgn_nodes_to_hdf_cells(hgn_xy, hdf_xy)
    mapped_face_cells = remap_face_cells(face_cells, original_indices)
    cell_surface_area = all_cell_surface_area[original_indices]
    with h5py.File(hdf_path, "r") as hdf:
        water_surface = np.asarray(
            hdf[RESULT_BASE + "Water Surface"][target_time_indices], dtype=np.float64
        )
        precip = np.asarray(
            hdf[RESULT_BASE + "Cell Cumulative Precipitation Depth"][
                target_time_indices
            ],
            dtype=np.float64,
        )
    hdf_volume_all = reconstruct_cell_volume(
        water_surface, original_indices, volume_info, volume_values
    )
    hdf_volume = hdf_volume_all
    storage_difference = hdf_volume - volume
    storage_alignment_rmse = float(np.sqrt(np.mean(storage_difference**2)))
    storage_alignment_max_abs = float(np.max(np.abs(storage_difference)))
    storage_aligned = storage_alignment_max_abs <= storage_alignment_atol
    first_transition = max(1, start_index)
    selected = slice(first_transition, volume.shape[0])
    target_delta_by_source = {
        "hgn_target": volume[selected] - volume[first_transition - 1 : -1],
        "hdf_reconstructed": (
            hdf_volume[selected] - hdf_volume[first_transition - 1 : -1]
        ),
    }
    precip_delta = (
        (
            precip[first_transition:, original_indices]
            - precip[first_transition - 1 : -1, original_indices]
        )
        / 12.0
        * cell_surface_area[None, :]
    )
    rows = []
    flux_sources = [RECONSTRUCTED_FLUX_SOURCE]
    if has_native_face_flow:
        flux_sources.append(NATIVE_FLUX_SOURCE)
    for flux_source in flux_sources:
        stage_rules = FACE_STAGE_RULES if flux_source == RECONSTRUCTED_FLUX_SOURCE else ("native",)
        for stage_rule in stage_rules:
            for sign_mode in SIGN_MODES:
                for boundary_mode in BOUNDARY_MODES:
                    transport_delta = integrate_transport_from_hdf(
                        hdf_path,
                        face_cells,
                        mapped_face_cells,
                        num_nodes,
                        hdf_time_days,
                        target_time_indices[first_transition - 1 :],
                        area_info,
                        area_values,
                        stage_rule,
                        sign_mode,
                        boundary_mode,
                        flux_source,
                    )
                    for source_mode in SOURCE_MODES:
                        budget = transport_delta.copy()
                        if source_mode == "hdf_precipitation":
                            budget = budget + precip_delta
                        for target_volume_source in target_volume_sources:
                            target_delta = target_delta_by_source[target_volume_source]
                            for scope, mask in (
                                ("all", np.ones(num_nodes, dtype=bool)),
                                ("zone_3", zone_label == 3),
                            ):
                                metrics = metric_dict(target_delta[:, mask], budget[:, mask])
                                domain_residual = np.sum(
                                    target_delta[:, mask] - budget[:, mask], axis=1
                                )
                                row = {
                                    "event_id": event_id,
                                    "target_volume_source": target_volume_source,
                                    "scope": scope,
                                    "flux_source": flux_source,
                                    "face_stage_rule": stage_rule,
                                    "sign_mode": sign_mode,
                                    "boundary_mode": boundary_mode,
                                    "source_mode": source_mode,
                                    "has_native_face_flow": has_native_face_flow,
                                    "has_native_cell_flow_balance": (
                                        has_native_cell_flow_balance
                                    ),
                                    "num_nodes": int(np.sum(mask)),
                                    "num_transitions": int(target_delta.shape[0]),
                                    "delta_t_seconds": delta_t,
                                    "hecras_computation_delta_t_seconds": computation_delta_t,
                                    "hdf_output_delta_t_seconds": hdf_output_delta_t,
                                    "hdf_output_matches_computation": bool(
                                        np.isclose(
                                            hdf_output_delta_t,
                                            computation_delta_t,
                                            atol=1e-3,
                                        )
                                    ),
                                    "computation_steps_per_budget_interval": (
                                        delta_t / computation_delta_t
                                    ),
                                    "storage_aligned": storage_aligned,
                                    "storage_alignment_rmse_ft3": storage_alignment_rmse,
                                    "storage_alignment_max_abs_ft3": storage_alignment_max_abs,
                                    "domain_residual_rmse_ft3": float(
                                        np.sqrt(np.mean(domain_residual**2))
                                    ),
                                    "hdf_path": str(hdf_path),
                                }
                                row.update(metrics)
                                rows.append(row)
    return rows


def add_aggregate_rows(rows: list[dict]) -> list[dict]:
    """Average per-event diagnostic metrics for each physical variant."""
    metric_keys = [
        "target_rms_ft3",
        "budget_rms_ft3",
        "residual_rmse_ft3",
        "residual_mae_ft3",
        "residual_bias_ft3",
        "relative_rmse",
        "correlation",
        "domain_residual_rmse_ft3",
    ]
    groups = {}
    for row in rows:
        key = (
            row["target_volume_source"],
            row["scope"],
            row["flux_source"],
            row["face_stage_rule"],
            row["sign_mode"],
            row["boundary_mode"],
            row["source_mode"],
        )
        groups.setdefault(key, []).append(row)
    aggregate_rows = []
    for key, group in groups.items():
        row = {
            "event_id": "MEAN",
            "target_volume_source": key[0],
            "scope": key[1],
            "flux_source": key[2],
            "face_stage_rule": key[3],
            "sign_mode": key[4],
            "boundary_mode": key[5],
            "source_mode": key[6],
            "has_native_face_flow": all(
                item["has_native_face_flow"] for item in group
            ),
            "has_native_cell_flow_balance": all(
                item["has_native_cell_flow_balance"] for item in group
            ),
            "num_nodes": group[0]["num_nodes"],
            "num_transitions": int(sum(item["num_transitions"] for item in group)),
            "delta_t_seconds": group[0]["delta_t_seconds"],
            "hecras_computation_delta_t_seconds": group[0][
                "hecras_computation_delta_t_seconds"
            ],
            "hdf_output_delta_t_seconds": group[0]["hdf_output_delta_t_seconds"],
            "hdf_output_matches_computation": all(
                item["hdf_output_matches_computation"] for item in group
            ),
            "computation_steps_per_budget_interval": group[0][
                "computation_steps_per_budget_interval"
            ],
            "storage_aligned": all(item["storage_aligned"] for item in group),
            "storage_alignment_rmse_ft3": float(
                np.mean([item["storage_alignment_rmse_ft3"] for item in group])
            ),
            "storage_alignment_max_abs_ft3": float(
                np.max([item["storage_alignment_max_abs_ft3"] for item in group])
            ),
            "hdf_path": "",
        }
        for metric_key in metric_keys:
            row[metric_key] = float(np.nanmean([item[metric_key] for item in group]))
        aggregate_rows.append(row)
    return aggregate_rows


def write_markdown_summary(
    output_path: Path,
    rows: list[dict],
    data_dir: Path,
    ids_file: str,
    hdf_glob: str,
) -> None:
    aggregate = [
        row for row in rows if row["event_id"] == "MEAN" and row["scope"] == "all"
    ]
    hgn_best = min(
        [row for row in aggregate if row["target_volume_source"] == "hgn_target"],
        key=lambda row: row["relative_rmse"],
    )
    hdf_best = min(
        [
            row
            for row in aggregate
            if row["target_volume_source"] == "hdf_reconstructed"
        ],
        key=lambda row: row["relative_rmse"],
    )
    status = (
        "BLOCKED"
        if (
            not hgn_best["storage_aligned"]
            or not hgn_best["has_native_face_flow"]
            or not hgn_best["hdf_output_matches_computation"]
        )
        else "REQUIRES RESIDUAL REVIEW"
    )
    lines = [
        "# HEC-RAS / HydroGraphNet Physical Budget Validation",
        "",
        f"Formal local conservation status: **{status}**",
        "",
        "## Inputs",
        "",
        f"- HGN data: `{data_dir}` (`{ids_file}`)",
        f"- Event-specific HDF glob: `{hdf_glob}`",
        f"- Evaluated event transitions: `{hgn_best['num_transitions']}`",
        f"- HGN budget interval: `{hgn_best['delta_t_seconds']:.6f} s`",
        (
            "- HEC-RAS computation interval: "
            f"`{hgn_best['hecras_computation_delta_t_seconds']:.6f} s` "
            f"(`{hgn_best['computation_steps_per_budget_interval']:.1f}` "
            "solver steps per HGN budget interval)"
        ),
        f"- HDF stored face-output interval: `{hgn_best['hdf_output_delta_t_seconds']:.6f} s`",
        "",
        "## Formal Gates",
        "",
        "| Gate | Result | Evidence |",
        "|---|---|---|",
        (
            "| HDF reconstructed cell storage matches HGN `M80_V` target | "
            f"{'PASS' if hgn_best['storage_aligned'] else 'FAIL'} | "
            f"RMSE `{hgn_best['storage_alignment_rmse_ft3']:.6f} ft^3`, "
            f"max abs `{hgn_best['storage_alignment_max_abs_ft3']:.6f} ft^3` |"
        ),
        (
            "| HDF stores face output at solver-computation frequency | "
            f"{'PASS' if hgn_best['hdf_output_matches_computation'] else 'FAIL'} | "
            f"HDF `{hgn_best['hdf_output_delta_t_seconds']:.6f} s`, "
            f"computation `{hgn_best['hecras_computation_delta_t_seconds']:.6f} s` |"
        ),
        (
            "| Native HEC-RAS internal `Face Flow` is available | "
            f"{'PASS' if hgn_best['has_native_face_flow'] else 'FAIL'} | "
            "Required for formal internal-face flux evidence |"
        ),
        (
            "| Evaluation uses no fitted scale factor | PASS | "
            "Budget is evaluated directly in native volume units |"
        ),
        (
            "| Uncalibrated residual is acceptable for a training target | FAIL | "
            f"Best HGN-target relative RMSE `{hgn_best['relative_rmse']:.6f}` |"
        ),
        (
            "| HDF self-storage can be closed by reconstructed face transport | FAIL | "
            f"Best HDF-self relative RMSE `{hdf_best['relative_rmse']:.6f}` |"
        ),
        "",
        "## Best Uncalibrated Variants",
        "",
        "| Target | Flux source | Face stage rule | Sign mode | Boundary mode | Source mode | Residual RMSE (ft^3) | Relative RMSE | Correlation |",
        "|---|---|---|---|---|---|---:|---:|---:|",
        (
            f"| HGN `M80_V` | {hgn_best['flux_source']} | "
            f"{hgn_best['face_stage_rule']} | "
            f"{hgn_best['sign_mode']} | {hgn_best['boundary_mode']} | "
            f"{hgn_best['source_mode']} | "
            f"{hgn_best['residual_rmse_ft3']:.6f} | "
            f"{hgn_best['relative_rmse']:.6f} | {hgn_best['correlation']:.6f} |"
        ),
        (
            f"| HDF reconstructed storage | {hdf_best['flux_source']} | "
            f"{hdf_best['face_stage_rule']} | "
            f"{hdf_best['sign_mode']} | {hdf_best['boundary_mode']} | "
            f"{hdf_best['source_mode']} | "
            f"{hdf_best['residual_rmse_ft3']:.6f} | "
            f"{hdf_best['relative_rmse']:.6f} | {hdf_best['correlation']:.6f} |"
        ),
        "",
        "## Interpretation",
        "",
        "- The result is uncalibrated: no regression scale factor is fitted to target volume changes.",
        "- Current HDF forcing matching is insufficient because its reconstructed dynamic storage does not reproduce the HGN target storage.",
        "- HEC-RAS documentation identifies optional `Face flow` and `Cell flow balance` HDF variables; they are not present in the evaluated HDF and should be enabled for formal budget evidence.",
        (
            "- Temporal sampling check: the HDF stores face samples every "
            f"`{hgn_best['hdf_output_delta_t_seconds']:.6f} s`, while the "
            f"solver computation interval is `{hgn_best['hecras_computation_delta_t_seconds']:.6f} s`. "
            + (
                "The high-frequency preflight output therefore addresses the prior sampling blocker."
                if hgn_best["hdf_output_matches_computation"]
                else "The stored output is still too sparse to integrate every solver-step face transfer."
            )
        ),
        "- Do not activate a formal HEC-RAS local-conservation training loss until a synchronized dataset/HDF pair and an interval-integrated or sufficiently sampled internal-face flux budget pass these gates.",
        "",
    ]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ids-file", default="test.txt")
    parser.add_argument("--hdf-glob", required=True)
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--zone-label-file", default="zone_label.txt")
    parser.add_argument(
        "--start-index",
        type=int,
        default=1,
        help="First HDF/HGN output index to evaluate as a transition endpoint.",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        help="Last HGN target output index to evaluate, inclusive.",
    )
    parser.add_argument(
        "--storage-alignment-atol",
        type=float,
        default=1e-3,
        help="Maximum absolute HDF-vs-HGN reconstructed cell-volume difference in ft^3.",
    )
    parser.add_argument(
        "--target-volume-source",
        action="append",
        choices=TARGET_VOLUME_SOURCES,
        help="Volume transition target to evaluate. Defaults to both sources.",
    )
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--output-md", type=Path)
    args = parser.parse_args()

    event_ids = load_event_ids(args.data_dir / args.ids_file)
    event_paths = load_hdf_paths(args.hdf_glob)
    zone_label = np.loadtxt(args.data_dir / args.zone_label_file, dtype=np.int64)
    rows = []
    target_volume_sources = tuple(args.target_volume_source or TARGET_VOLUME_SOURCES)
    for event_id in event_ids:
        if event_id not in event_paths:
            raise FileNotFoundError(f"No matched HDF loaded for event {event_id}.")
        rows.extend(
            evaluate_event(
                event_id,
                event_paths[event_id],
                args.data_dir,
                args.prefix,
                zone_label,
                args.start_index,
                args.end_index,
                args.storage_alignment_atol,
                target_volume_sources,
            )
        )
    rows.extend(add_aggregate_rows(rows))
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys())
    with args.output_csv.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    if args.output_md is not None:
        write_markdown_summary(
            args.output_md, rows, args.data_dir, args.ids_file, args.hdf_glob
        )
    aggregate = [
        row for row in rows if row["event_id"] == "MEAN" and row["scope"] == "all"
    ]
    print(f"Wrote {args.output_csv}")
    if args.output_md is not None:
        print(f"Wrote {args.output_md}")
    if not all(row["storage_aligned"] for row in aggregate):
        print(
            "FORMAL BUDGET BLOCKED: HDF reconstructed storage does not match "
            "the HGN event volume target."
        )
    for target_volume_source in target_volume_sources:
        source_rows = [
            row
            for row in aggregate
            if row["target_volume_source"] == target_volume_source
        ]
        best = min(source_rows, key=lambda row: row["relative_rmse"])
        print(f"Best uncalibrated all-node mean variant for {target_volume_source}:")
        print(best)


if __name__ == "__main__":
    main()
