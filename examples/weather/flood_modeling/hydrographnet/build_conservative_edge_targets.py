# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Build storage-consistent HEC-RAS edge targets for selected HGN zones.

The raw HEC-RAS period-average face flows are retained as the hydraulic prior.
A minimum-L2-change projection then enforces the finite-volume balance on the
selected nodes without fitting or applying an empirical scale factor.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


RESULT_BASE = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/"
    "2D Flow Areas/per2/"
)
RESULT_TIME_PATH = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time"
)
GEOMETRY_BASE = "Geometry/2D Flow Areas/per2/"
PERIOD_AVERAGE_FACE_FLOW_PATH = RESULT_BASE + "Face Flow Period Average"
CUMULATIVE_PRECIPITATION_PATH = (
    RESULT_BASE + "Cell Cumulative Precipitation Depth"
)
PRECIPITATION_BASE = "Event Conditions/Meteorology/Precipitation/"
PRECIPITATION_CELL_BASE = PRECIPITATION_BASE + "2D Flow Areas/per2/"
TARGET_BUILDER_SCHEMA_VERSION = 4


def decode_attr(value) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def save_npy_atomic(path: Path, values: np.ndarray) -> None:
    """Write a NumPy array atomically so interrupted runs remain resumable."""

    temporary_path = path.with_suffix(".tmp.npy")
    np.save(temporary_path, values)
    temporary_path.replace(path)


def match_times(reference_days: np.ndarray, candidate_days: np.ndarray) -> np.ndarray:
    """Return exact candidate indexes for each reference time in days."""

    indexes = np.searchsorted(candidate_days, reference_days)
    indexes = np.clip(indexes, 0, candidate_days.size - 1)
    previous = np.maximum(indexes - 1, 0)
    use_previous = (
        np.abs(candidate_days[previous] - reference_days)
        < np.abs(candidate_days[indexes] - reference_days)
    )
    indexes[use_previous] = previous[use_previous]
    mismatch_seconds = np.abs(candidate_days[indexes] - reference_days) * 86400.0
    if np.max(mismatch_seconds) > 1.0e-3:
        raise ValueError(
            "HGN and HEC-RAS time axes do not match; maximum error is "
            f"{np.max(mismatch_seconds):.6g} seconds."
        )
    return indexes.astype(np.int64)


def build_selected_incidence(
    face_index: np.ndarray, selected_nodes: np.ndarray, num_nodes: int
) -> sp.csr_matrix:
    """Build incidence A with -1 at face source and +1 at face destination."""

    node_to_row = np.full(num_nodes, -1, dtype=np.int64)
    node_to_row[selected_nodes] = np.arange(selected_nodes.size, dtype=np.int64)
    source_rows = np.full(face_index.shape[1], -1, dtype=np.int64)
    destination_rows = np.full(face_index.shape[1], -1, dtype=np.int64)
    valid_source_nodes = (face_index[0] >= 0) & (face_index[0] < num_nodes)
    valid_destination_nodes = (face_index[1] >= 0) & (
        face_index[1] < num_nodes
    )
    source_rows[valid_source_nodes] = node_to_row[
        face_index[0, valid_source_nodes]
    ]
    destination_rows[valid_destination_nodes] = node_to_row[
        face_index[1, valid_destination_nodes]
    ]
    source_valid = source_rows >= 0
    destination_valid = destination_rows >= 0
    rows = np.concatenate(
        [source_rows[source_valid], destination_rows[destination_valid]]
    )
    columns = np.concatenate(
        [np.flatnonzero(source_valid), np.flatnonzero(destination_valid)]
    )
    values = np.concatenate(
        [
            -np.ones(np.count_nonzero(source_valid), dtype=np.float64),
            np.ones(np.count_nonzero(destination_valid), dtype=np.float64),
        ]
    )
    return sp.csr_matrix(
        (values, (rows, columns)),
        shape=(selected_nodes.size, face_index.shape[1]),
    )


def reconstruct_selected_input_precipitation_rate(
    hdf: h5py.File,
    selected_hdf_cells: np.ndarray,
    transition_end_hours: np.ndarray,
) -> np.ndarray:
    """Reconstruct cell rain rates from pre-simulation raster inputs."""

    values_path = PRECIPITATION_BASE + "Values"
    info_path = PRECIPITATION_CELL_BASE + "Cell Info"
    indexes_path = PRECIPITATION_CELL_BASE + "Cell Indexes"
    weights_path = PRECIPITATION_CELL_BASE + "Cell Weights"
    missing = [
        path
        for path in (values_path, info_path, indexes_path, weights_path)
        if path not in hdf
    ]
    if missing:
        raise KeyError(f"Missing input precipitation mapping datasets: {missing}")

    raster_values = np.asarray(hdf[values_path], dtype=np.float64)
    cell_info = np.asarray(hdf[info_path], dtype=np.int64)
    cell_indexes = np.asarray(hdf[indexes_path], dtype=np.int64)
    cell_weights = np.asarray(hdf[weights_path], dtype=np.float64)
    if np.max(selected_hdf_cells) >= cell_info.shape[0]:
        raise ValueError(
            "Selected HGN nodes include cells outside the precipitation mapping."
        )
    rows = []
    columns = []
    weights = []
    for local_row, hdf_cell in enumerate(selected_hdf_cells):
        offset, count = cell_info[hdf_cell]
        rows.extend([local_row] * int(count))
        columns.extend(cell_indexes[offset : offset + count].tolist())
        weights.extend(cell_weights[offset : offset + count].tolist())
    mapping = sp.csr_matrix(
        (weights, (rows, columns)),
        shape=(selected_hdf_cells.size, raster_values.shape[1]),
    )
    weight_sums = np.asarray(mapping.sum(axis=1)).reshape(-1)
    if not np.allclose(weight_sums, 1.0, rtol=0.0, atol=1.0e-6):
        raise ValueError("HEC-RAS precipitation cell weights do not sum to one.")
    hourly_cell_rate = (mapping @ raster_values.T).T

    # HEC-RAS stores the first timestamp as initialization and applies row k
    # over the interval ending at hour k. This indexing is checked against the
    # result-side cumulative depth below, but the target uses only these inputs.
    # Text HGN timestamps can place an exact hour a few micro-hours above the
    # integer (for example 94.000000008). Keep a 0.0036-second tolerance so an
    # interval ending at hour 94 cannot be misclassified as hour 95.
    input_rows = np.ceil(transition_end_hours - 1.0e-6).astype(np.int64)
    if np.min(input_rows) < 0 or np.max(input_rows) >= hourly_cell_rate.shape[0]:
        raise ValueError("Result time axis exceeds the precipitation input table.")
    return hourly_cell_rate[input_rows]


def project_minimum_change(
    incidence: sp.csr_matrix,
    raw_edge_delta: np.ndarray,
    target_divergence: np.ndarray,
    solve_fn,
) -> tuple[np.ndarray, np.ndarray]:
    """Project q by q' = q - A.T (A A.T)^-1 (A q - d)."""

    residual = incidence @ raw_edge_delta.T - target_divergence.T
    lagrange = solve_fn(residual)
    correction = (incidence.T @ lagrange).T
    return raw_edge_delta - correction, correction


def error_metrics(target: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    residual = prediction - target
    absolute = np.abs(residual)
    target_rms = float(np.sqrt(np.mean(target**2)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    return {
        "target_rms_m3": target_rms,
        "residual_rmse_m3": rmse,
        "relative_rmse": rmse / max(target_rms, 1.0e-12),
        "absolute_residual_p50_m3": float(np.percentile(absolute, 50)),
        "absolute_residual_p95_m3": float(np.percentile(absolute, 95)),
        "absolute_residual_p99_m3": float(np.percentile(absolute, 99)),
        "absolute_residual_max_m3": float(np.max(absolute)),
        "selected_domain_residual_rmse_m3": float(
            np.sqrt(np.mean(np.sum(residual, axis=1) ** 2))
        ),
    }


def resolve_hdf_path(template: str, event_id: str) -> Path:
    return Path(template.format(event_id=event_id)).expanduser().resolve()


def build_event(
    event_id: str,
    data_dir: Path,
    hdf_path: Path,
    face_graph: np.lib.npyio.NpzFile,
    selected_nodes: np.ndarray,
    incidence: sp.csr_matrix,
    solve_fn,
    output_path: Path,
    prefix: str,
    chunk_size: int,
    selected_zone: int,
    face_graph_sha256: str,
    zone_label_sha256: str,
) -> dict:
    volume_path = data_dir / f"{prefix}_V_{event_id}.txt"
    inflow_path = data_dir / f"{prefix}_US_InF_{event_id}.txt"
    volume = np.loadtxt(volume_path)
    hgn_time_days = np.loadtxt(inflow_path)[:, 0]
    if volume.ndim != 2 or volume.shape[0] != hgn_time_days.size:
        raise ValueError(f"Invalid HGN volume/time shape for {event_id}.")

    hgn_to_hdf = np.asarray(face_graph["hgn_to_hdf_cell_index"], dtype=np.int64)
    hdf_to_hgn = np.asarray(face_graph["hdf_to_hgn_node_index"], dtype=np.int64)
    internal_hdf_faces = np.asarray(
        face_graph["internal_hdf_face_index"], dtype=np.int64
    )
    boundary_hdf_faces = np.asarray(
        face_graph["boundary_hdf_face_index"], dtype=np.int64
    )
    boundary_cells_hdf = np.asarray(
        face_graph["boundary_face_cell_index"], dtype=np.int64
    )
    mapped_boundary_cells = np.full_like(boundary_cells_hdf, -1)
    valid_boundary_cells = (
        (boundary_cells_hdf >= 0) & (boundary_cells_hdf < hdf_to_hgn.size)
    )
    mapped_boundary_cells[valid_boundary_cells] = hdf_to_hgn[
        boundary_cells_hdf[valid_boundary_cells]
    ]
    boundary_incidence = build_selected_incidence(
        mapped_boundary_cells.T,
        selected_nodes,
        volume.shape[1],
    )

    native_area = np.asarray(
        face_graph["hgn_node_surface_area"], dtype=np.float64
    )
    hgn_area = np.loadtxt(data_dir / f"{prefix}_CA.txt").reshape(-1)
    area_error = float(np.max(np.abs(native_area - hgn_area)))
    if area_error > 1.0e-3:
        raise ValueError(
            "M80_CA must use HEC-RAS Cells Surface Area; maximum mismatch is "
            f"{area_error:.6g} m2."
        )

    with h5py.File(hdf_path, "r") as hdf:
        unit_system = decode_attr(hdf.attrs.get("Units System", ""))
        if "si" not in unit_system.lower():
            raise ValueError(
                f"Conservative targets require native SI HDF data, got {unit_system!r}."
            )
        required = [
            RESULT_TIME_PATH,
            PERIOD_AVERAGE_FACE_FLOW_PATH,
            CUMULATIVE_PRECIPITATION_PATH,
        ]
        missing = [path for path in required if path not in hdf]
        if missing:
            raise KeyError(f"Missing publication edge-target datasets: {missing}")
        flow_dataset = hdf[PERIOD_AVERAGE_FACE_FLOW_PATH]
        flow_units = decode_attr(flow_dataset.attrs.get("Units", ""))
        precipitation_dataset = hdf[CUMULATIVE_PRECIPITATION_PATH]
        precipitation_units = decode_attr(
            precipitation_dataset.attrs.get("Units", "")
        )
        if flow_units.replace("^", "").lower() not in {"m3/s", "m3/sec"}:
            raise ValueError(f"Unexpected period-average flow units: {flow_units!r}")
        if precipitation_units.lower() != "mm":
            raise ValueError(
                f"Unexpected cumulative precipitation units: {precipitation_units!r}"
            )
        hdf_time_days = np.asarray(hdf[RESULT_TIME_PATH], dtype=np.float64)
        matched_time = match_times(hgn_time_days, hdf_time_days)
        time_seconds = hdf_time_days[matched_time] * 86400.0
        delta_t = np.diff(time_seconds)
        if not np.allclose(delta_t, 300.0, rtol=0.0, atol=1.0e-3):
            raise ValueError("Conservative targets require exact five-minute outputs.")
        selected_hdf_cells = hgn_to_hdf[selected_nodes]
        transition_end_hours = (time_seconds[1:] - time_seconds[0]) / 3600.0
        selected_input_precipitation_rate = (
            reconstruct_selected_input_precipitation_rate(
                hdf, selected_hdf_cells, transition_end_hours
            )
        )
        precipitation_delta_selected = (
            selected_input_precipitation_rate
            * (delta_t[:, None] / 3600.0)
            * 1.0e-3
            * native_area[selected_nodes][None, :]
        )
        cumulative_precipitation_selected = np.asarray(
            precipitation_dataset[matched_time][:, selected_hdf_cells],
            dtype=np.float64,
        )
        result_precipitation_rate = (
            np.diff(cumulative_precipitation_selected, axis=0)
            / (delta_t[:, None] / 3600.0)
        )
        precipitation_rate_error = (
            result_precipitation_rate - selected_input_precipitation_rate
        )
        storage_delta_selected = np.diff(
            volume[:, selected_nodes].astype(np.float64), axis=0
        )
        transport_target_selected = (
            storage_delta_selected - precipitation_delta_selected
        )
        num_transitions = volume.shape[0] - 1
        num_internal_faces = internal_hdf_faces.size
        raw_internal = np.empty(
            (num_transitions, num_internal_faces), dtype=np.float32
        )
        projected_internal = np.empty_like(raw_internal)
        target_internal_selected = np.empty(
            (num_transitions, selected_nodes.size), dtype=np.float32
        )
        precipitation_volume_selected = np.empty_like(target_internal_selected)
        boundary_source_selected = np.empty_like(target_internal_selected)
        period_average_cell_budget_selected = np.empty_like(
            target_internal_selected
        )

        raw_residual_parts = []
        projected_residual_parts = []
        correction_parts = []
        saved_projected_residual_parts = []
        raw_edge_square_sum = 0.0
        raw_edge_count = 0
        for start in range(0, num_transitions, chunk_size):
            stop = min(start + chunk_size, num_transitions)
            result_rows = matched_time[start + 1 : stop + 1]
            dt = delta_t[start:stop, None]
            raw_chunk = np.asarray(
                flow_dataset[result_rows][:, internal_hdf_faces], dtype=np.float64
            ) * dt
            boundary_chunk = np.asarray(
                flow_dataset[result_rows][:, boundary_hdf_faces], dtype=np.float64
            ) * dt
            boundary_source = (boundary_incidence @ boundary_chunk.T).T
            target_chunk = (
                transport_target_selected[start:stop] - boundary_source
            )
            projected_chunk, correction = project_minimum_change(
                incidence, raw_chunk, target_chunk, solve_fn
            )
            raw_divergence = (incidence @ raw_chunk.T).T
            projected_divergence = (incidence @ projected_chunk.T).T

            raw_internal[start:stop] = raw_chunk.astype(np.float32)
            projected_internal[start:stop] = projected_chunk.astype(np.float32)
            target_internal_selected[start:stop] = target_chunk.astype(np.float32)
            precipitation_volume_selected[start:stop] = (
                precipitation_delta_selected[start:stop].astype(np.float32)
            )
            boundary_source_selected[start:stop] = boundary_source.astype(np.float32)
            period_average_cell_budget_selected[start:stop] = (
                raw_divergence
                + boundary_source
                + precipitation_delta_selected[start:stop]
            ).astype(np.float32)
            raw_residual_parts.append(raw_divergence - target_chunk)
            projected_residual_parts.append(projected_divergence - target_chunk)
            saved_projected_residual_parts.append(
                (
                    incidence
                    @ projected_internal[start:stop].astype(np.float64).T
                ).T
                - target_internal_selected[start:stop].astype(np.float64)
            )
            correction_parts.append(correction)
            raw_edge_square_sum += float(np.sum(raw_chunk**2))
            raw_edge_count += raw_chunk.size

    raw_residual = np.concatenate(raw_residual_parts, axis=0)
    projected_residual = np.concatenate(projected_residual_parts, axis=0)
    saved_projected_residual = np.concatenate(
        saved_projected_residual_parts, axis=0
    )
    correction = np.concatenate(correction_parts, axis=0)
    selected_target = target_internal_selected.astype(np.float64)
    raw_metrics = error_metrics(selected_target, selected_target + raw_residual)
    projected_metrics = error_metrics(
        selected_target, selected_target + projected_residual
    )
    raw_edge_rms = float(np.sqrt(raw_edge_square_sum / max(raw_edge_count, 1)))
    correction_rms = float(np.sqrt(np.mean(correction**2)))
    summary = {
        "target_builder_schema_version": TARGET_BUILDER_SCHEMA_VERSION,
        "event_id": event_id,
        "hdf_path": str(hdf_path),
        "hdf_sha256": file_sha256(hdf_path),
        "hgn_volume_sha256": file_sha256(volume_path),
        "hgn_inflow_time_sha256": file_sha256(inflow_path),
        "face_graph_sha256": face_graph_sha256,
        "zone_label_sha256": zone_label_sha256,
        "output_path": str(output_path),
        "num_frames": int(volume.shape[0]),
        "num_transitions": int(volume.shape[0] - 1),
        "num_nodes": int(volume.shape[1]),
        "num_selected_nodes": int(selected_nodes.size),
        "num_internal_faces": int(internal_hdf_faces.size),
        "num_boundary_or_omitted_faces": int(boundary_hdf_faces.size),
        "delta_t_seconds": float(np.median(delta_t)),
        "raw_closure": raw_metrics,
        "projected_closure": projected_metrics,
        "saved_float32_projected_closure": error_metrics(
            selected_target, selected_target + saved_projected_residual
        ),
        "raw_edge_rms_m3": raw_edge_rms,
        "correction_rms_m3": correction_rms,
        "correction_to_raw_edge_rms_ratio": correction_rms
        / max(raw_edge_rms, 1.0e-12),
        "absolute_correction_p50_m3": float(np.percentile(np.abs(correction), 50)),
        "absolute_correction_p95_m3": float(np.percentile(np.abs(correction), 95)),
        "absolute_correction_p99_m3": float(np.percentile(np.abs(correction), 99)),
        "absolute_correction_max_m3": float(np.max(np.abs(correction))),
        "area_max_abs_error_m2": area_error,
        "flow_units": flow_units,
        "precipitation_units": precipitation_units,
        "local_precipitation_source": "pre_simulation_raster_values_and_cell_weights",
        "input_vs_result_precipitation_rate_rmse_mmh": float(
            np.sqrt(np.mean(precipitation_rate_error**2))
        ),
        "input_vs_result_precipitation_rate_max_abs_mmh": float(
            np.max(np.abs(precipitation_rate_error))
        ),
        "projection": "minimum_l2_change_exact_selected_node_divergence",
        "empirical_scale_factor": None,
    }
    training_arrays = {
        "raw_internal_face_delta_m3": (
            output_path.parent / f"{event_id}_raw_internal_face_delta_m3.npy"
        ),
        "projected_internal_face_delta_m3": (
            output_path.parent
            / f"{event_id}_projected_internal_face_delta_m3.npy"
        ),
        "target_internal_divergence_selected_m3": (
            output_path.parent
            / f"{event_id}_target_internal_divergence_selected_m3.npy"
        ),
        "omitted_boundary_source_selected_m3": (
            output_path.parent
            / f"{event_id}_omitted_boundary_source_selected_m3.npy"
        ),
        "precipitation_volume_selected_m3": (
            output_path.parent
            / f"{event_id}_precipitation_volume_selected_m3.npy"
        ),
        "period_average_cell_budget_delta_selected_m3": (
            output_path.parent
            / f"{event_id}_period_average_cell_budget_delta_selected_m3.npy"
        ),
    }
    training_values = {
        "raw_internal_face_delta_m3": raw_internal,
        "projected_internal_face_delta_m3": projected_internal,
        "target_internal_divergence_selected_m3": target_internal_selected,
        "omitted_boundary_source_selected_m3": boundary_source_selected,
        "precipitation_volume_selected_m3": precipitation_volume_selected,
        "period_average_cell_budget_delta_selected_m3": (
            period_average_cell_budget_selected
        ),
    }
    for name, path in training_arrays.items():
        save_npy_atomic(path, training_values[name])
    summary["training_arrays"] = {
        name: {
            "path": str(path),
            "sha256": file_sha256(path),
            "shape": list(training_values[name].shape),
            "dtype": str(training_values[name].dtype),
        }
        for name, path in training_arrays.items()
    }
    metadata = {
        "schema_version": TARGET_BUILDER_SCHEMA_VERSION,
        "event_id": event_id,
        "selected_zone_label": selected_zone,
        "edge_orientation": "positive_hdf_cell0_to_hdf_cell1",
        "edge_delta_units": "m3_per_5min_transition",
        "target_equation": (
            "A_internal q_projected = delta_cell_volume - precipitation_volume "
            "- omitted_boundary_face_divergence"
        ),
        "projection": "q_projected=q_raw-A.T*(A*A.T)^-1*(A*q_raw-d)",
        "uses_empirical_scale_factor": False,
        "source_face_dataset": PERIOD_AVERAGE_FACE_FLOW_PATH,
        "source_precipitation_datasets": [
            PRECIPITATION_BASE + "Values",
            PRECIPITATION_CELL_BASE + "Cell Info",
            PRECIPITATION_CELL_BASE + "Cell Indexes",
            PRECIPITATION_CELL_BASE + "Cell Weights",
        ],
        "result_cumulative_precipitation_role": "validation_only_not_target_input",
        "cell_only_baseline_target": (
            "period_average_face_flow_divergence_plus_input_precipitation; "
            "the model receives only the aggregated per-cell budget"
        ),
    }
    temporary_path = output_path.with_suffix(".tmp.npz")
    np.savez_compressed(
        temporary_path,
        event_id=np.asarray(event_id),
        selected_node_index=selected_nodes.astype(np.int64),
        delta_t_seconds=delta_t.astype(np.float64),
        raw_internal_face_delta_m3=raw_internal,
        projected_internal_face_delta_m3=projected_internal,
        target_internal_divergence_selected_m3=target_internal_selected,
        precipitation_volume_selected_m3=precipitation_volume_selected,
        omitted_boundary_source_selected_m3=boundary_source_selected,
        period_average_cell_budget_delta_selected_m3=(
            period_average_cell_budget_selected
        ),
        metadata_json=np.asarray(json.dumps(metadata, sort_keys=True)),
        summary_json=np.asarray(json.dumps(summary, sort_keys=True)),
    )
    temporary_path.replace(output_path)
    summary["output_sha256"] = file_sha256(output_path)
    return summary


def load_current_resumable_summary(
    output_path: Path,
    event_id: str,
    hdf_path: Path,
    data_dir: Path,
    prefix: str,
    face_graph_sha256: str,
    zone_label_sha256: str,
) -> dict | None:
    """Return an existing summary only when every source contract still matches."""

    try:
        with np.load(output_path, allow_pickle=False) as existing:
            metadata = json.loads(str(existing["metadata_json"]))
            summary = json.loads(str(existing["summary_json"]))
        expected = {
            "target_builder_schema_version": TARGET_BUILDER_SCHEMA_VERSION,
            "event_id": event_id,
            "hdf_sha256": file_sha256(hdf_path),
            "hgn_volume_sha256": file_sha256(
                data_dir / f"{prefix}_V_{event_id}.txt"
            ),
            "hgn_inflow_time_sha256": file_sha256(
                data_dir / f"{prefix}_US_InF_{event_id}.txt"
            ),
            "face_graph_sha256": face_graph_sha256,
            "zone_label_sha256": zone_label_sha256,
        }
        if metadata.get("schema_version") != TARGET_BUILDER_SCHEMA_VERSION:
            return None
        if any(summary.get(key) != value for key, value in expected.items()):
            return None
        for array_metadata in summary.get("training_arrays", {}).values():
            path = Path(array_metadata["path"])
            if not path.is_file() or file_sha256(path) != array_metadata.get("sha256"):
                return None
        summary["output_sha256"] = file_sha256(output_path)
        return summary
    except (KeyError, OSError, TypeError, ValueError, json.JSONDecodeError):
        return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--ids-file", type=Path, required=True)
    parser.add_argument("--hdf-template", required=True)
    parser.add_argument("--face-graph-file", type=Path, required=True)
    parser.add_argument("--zone-label-file", type=Path)
    parser.add_argument("--selected-zone", type=int, default=3)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_dir = args.data_dir.resolve()
    ids_file = args.ids_file if args.ids_file.is_absolute() else data_dir / args.ids_file
    event_ids = [line.strip() for line in ids_file.read_text().splitlines() if line.strip()]
    zone_path = args.zone_label_file or data_dir / "zone_label.txt"
    zone_labels = np.loadtxt(zone_path, dtype=np.int64).reshape(-1)
    selected_nodes = np.flatnonzero(zone_labels == args.selected_zone).astype(np.int64)
    if selected_nodes.size == 0:
        raise ValueError(f"No nodes found for selected zone {args.selected_zone}.")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    selected_node_path = args.output_dir / "selected_node_index.npy"
    save_npy_atomic(selected_node_path, selected_nodes)
    face_graph_sha256 = file_sha256(args.face_graph_file)
    zone_label_sha256 = file_sha256(zone_path)

    with np.load(args.face_graph_file, allow_pickle=False) as face_graph:
        face_index = np.asarray(face_graph["internal_face_index"], dtype=np.int64)
        incidence = build_selected_incidence(
            face_index, selected_nodes, zone_labels.size
        )
        lhs = incidence @ incidence.T
        solve_fn = spla.factorized(lhs.tocsc())
        summaries = []
        for position, event_id in enumerate(event_ids, start=1):
            output_path = args.output_dir / f"{event_id}_conservative_edge_target.npz"
            if args.resume and output_path.exists():
                hdf_path = resolve_hdf_path(args.hdf_template, event_id)
                existing_summary = load_current_resumable_summary(
                    output_path,
                    event_id,
                    hdf_path,
                    data_dir,
                    args.prefix,
                    face_graph_sha256,
                    zone_label_sha256,
                )
                if existing_summary is not None:
                    summaries.append(existing_summary)
                    print(
                        f"[{position}/{len(event_ids)}] {event_id}: "
                        "verified existing shard kept"
                    )
                    continue
                print(
                    f"[{position}/{len(event_ids)}] {event_id}: "
                    "existing shard is stale or incomplete; rebuilding"
                )
            print(f"[{position}/{len(event_ids)}] {event_id}: building targets")
            summaries.append(
                build_event(
                    event_id=event_id,
                    data_dir=data_dir,
                    hdf_path=resolve_hdf_path(args.hdf_template, event_id),
                    face_graph=face_graph,
                    selected_nodes=selected_nodes,
                    incidence=incidence,
                    solve_fn=solve_fn,
                    output_path=output_path,
                    prefix=args.prefix,
                    chunk_size=args.chunk_size,
                    selected_zone=args.selected_zone,
                    face_graph_sha256=face_graph_sha256,
                    zone_label_sha256=zone_label_sha256,
                )
            )
            print(
                f"  raw relative RMSE={summaries[-1]['raw_closure']['relative_rmse']:.6g}; "
                "projected relative RMSE="
                f"{summaries[-1]['projected_closure']['relative_rmse']:.6g}"
            )

    manifest = {
        "schema_version": TARGET_BUILDER_SCHEMA_VERSION,
        "transition_index_origin": 0,
        "transition_index_definition": (
            "row i represents the interval from original HGN frame i to i+1; "
            "training loaders must add dynamic_skip_steps"
        ),
        "data_dir": str(data_dir),
        "ids_file": str(ids_file.resolve()),
        "face_graph_file": str(args.face_graph_file.resolve()),
        "face_graph_sha256": face_graph_sha256,
        "zone_label_file": str(zone_path.resolve()),
        "zone_label_sha256": zone_label_sha256,
        "selected_zone": args.selected_zone,
        "selected_node_count": int(selected_nodes.size),
        "selected_node_index_file": str(selected_node_path),
        "selected_node_index_sha256": file_sha256(selected_node_path),
        "event_count_requested": len(event_ids),
        "events": summaries,
    }
    manifest_path = args.output_dir / "conservative_edge_target_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
