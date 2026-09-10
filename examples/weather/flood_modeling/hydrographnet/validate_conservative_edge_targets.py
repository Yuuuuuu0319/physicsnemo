# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Independently validate sharded conservative HEC-RAS edge targets."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import scipy.sparse as sp


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def selected_incidence(
    face_index: np.ndarray, selected_nodes: np.ndarray, node_count: int
) -> sp.csr_matrix:
    node_to_row = np.full(node_count, -1, dtype=np.int64)
    node_to_row[selected_nodes] = np.arange(selected_nodes.size, dtype=np.int64)
    source_rows = node_to_row[face_index[0]]
    destination_rows = node_to_row[face_index[1]]
    source_valid = source_rows >= 0
    destination_valid = destination_rows >= 0
    return sp.csr_matrix(
        (
            np.concatenate(
                [
                    -np.ones(np.count_nonzero(source_valid)),
                    np.ones(np.count_nonzero(destination_valid)),
                ]
            ),
            (
                np.concatenate(
                    [source_rows[source_valid], destination_rows[destination_valid]]
                ),
                np.concatenate(
                    [np.flatnonzero(source_valid), np.flatnonzero(destination_valid)]
                ),
            ),
        ),
        shape=(selected_nodes.size, face_index.shape[1]),
    )


def rms_ratio(squared_error: float, squared_reference: float) -> float:
    return float(np.sqrt(squared_error / max(squared_reference, 1.0e-24)))


def check(name: str, passed: bool, evidence, severity: str = "error") -> dict:
    return {
        "name": name,
        "passed": bool(passed),
        "severity": severity,
        "evidence": evidence,
    }


def validate_event(
    event: dict,
    data_dir: Path,
    target_dir: Path,
    incidence: sp.csr_matrix,
    selected_nodes: np.ndarray,
    args: argparse.Namespace,
) -> tuple[list[dict], dict]:
    event_id = str(event["event_id"])
    checks = []
    array_names = (
        "raw_internal_face_delta_m3",
        "projected_internal_face_delta_m3",
        "target_internal_divergence_selected_m3",
        "omitted_boundary_source_selected_m3",
        "precipitation_volume_selected_m3",
        "period_average_cell_budget_delta_selected_m3",
    )
    arrays = {}
    for name in array_names:
        metadata = event.get("training_arrays", {}).get(name, {})
        path = target_dir / f"{event_id}_{name}.npy"
        hash_ok = path.is_file() and file_sha256(path) == metadata.get("sha256")
        checks.append(
            check(
                f"{event_id}_{name}_hash",
                hash_ok,
                {"path": str(path), "expected_sha256": metadata.get("sha256")},
            )
        )
        if not path.is_file():
            continue
        arrays[name] = np.load(path, mmap_mode="r")

    if len(arrays) != len(array_names):
        return checks, {"event_id": event_id, "validated": False}
    raw_internal = arrays["raw_internal_face_delta_m3"]
    q = arrays["projected_internal_face_delta_m3"]
    target = arrays["target_internal_divergence_selected_m3"]
    boundary = arrays["omitted_boundary_source_selected_m3"]
    precipitation = arrays["precipitation_volume_selected_m3"]
    cell_budget = arrays["period_average_cell_budget_delta_selected_m3"]
    shape_ok = (
        q.ndim == 2
        and raw_internal.shape == q.shape
        and q.shape[1] == incidence.shape[1]
        and target.shape == boundary.shape == precipitation.shape == cell_budget.shape
        and target.shape == (q.shape[0], selected_nodes.size)
    )
    checks.append(
        check(
            f"{event_id}_array_shapes",
            shape_ok,
            {name: list(values.shape) for name, values in arrays.items()},
        )
    )
    if not shape_ok:
        return checks, {"event_id": event_id, "validated": False}

    closure_error_sq = 0.0
    closure_reference_sq = 0.0
    closure_max = 0.0
    for start in range(0, q.shape[0], args.chunk_size):
        stop = min(start + args.chunk_size, q.shape[0])
        predicted = (incidence @ np.asarray(q[start:stop], dtype=np.float64).T).T
        reference = np.asarray(target[start:stop], dtype=np.float64)
        residual = predicted - reference
        closure_error_sq += float(np.sum(residual**2))
        closure_reference_sq += float(np.sum(reference**2))
        closure_max = max(closure_max, float(np.max(np.abs(residual))))
    closure_relative_rmse = rms_ratio(closure_error_sq, closure_reference_sq)
    checks.append(
        check(
            f"{event_id}_recomputed_float32_closure",
            closure_relative_rmse <= args.max_closure_relative_rmse,
            {
                "relative_rmse": closure_relative_rmse,
                "maximum_absolute_residual_m3": closure_max,
                "threshold": args.max_closure_relative_rmse,
            },
        )
    )

    volume_path = data_dir / f"M80_V_{event_id}.txt"
    volume = np.loadtxt(volume_path, delimiter="\t", usecols=selected_nodes.tolist())
    storage_delta = np.diff(np.asarray(volume, dtype=np.float64), axis=0)
    reconstructed_storage = (
        np.asarray(target, dtype=np.float64)
        + np.asarray(boundary, dtype=np.float64)
        + np.asarray(precipitation, dtype=np.float64)
    )
    source_residual = reconstructed_storage - storage_delta
    source_relative_rmse = rms_ratio(
        float(np.sum(source_residual**2)), float(np.sum(storage_delta**2))
    )
    checks.append(
        check(
            f"{event_id}_source_equation",
            source_relative_rmse <= args.max_source_relative_rmse,
            {
                "relative_rmse": source_relative_rmse,
                "maximum_absolute_residual_m3": float(
                    np.max(np.abs(source_residual))
                ),
                "equation": "target_internal + boundary + precipitation = delta_volume",
                "threshold": args.max_source_relative_rmse,
            },
        )
    )

    cell_budget_error_sq = 0.0
    cell_budget_reference_sq = 0.0
    cell_budget_max = 0.0
    for start in range(0, raw_internal.shape[0], args.chunk_size):
        stop = min(start + args.chunk_size, raw_internal.shape[0])
        raw_divergence = (
            incidence @ np.asarray(raw_internal[start:stop], dtype=np.float64).T
        ).T
        expected = (
            raw_divergence
            + np.asarray(boundary[start:stop], dtype=np.float64)
            + np.asarray(precipitation[start:stop], dtype=np.float64)
        )
        residual = np.asarray(cell_budget[start:stop], dtype=np.float64) - expected
        cell_budget_error_sq += float(np.sum(residual**2))
        cell_budget_reference_sq += float(np.sum(expected**2))
        cell_budget_max = max(cell_budget_max, float(np.max(np.abs(residual))))
    cell_budget_relative_rmse = rms_ratio(
        cell_budget_error_sq, cell_budget_reference_sq
    )
    checks.append(
        check(
            f"{event_id}_period_average_cell_budget_equation",
            cell_budget_relative_rmse <= args.max_cell_budget_relative_rmse,
            {
                "relative_rmse": cell_budget_relative_rmse,
                "maximum_absolute_residual_m3": cell_budget_max,
                "equation": (
                    "cell_budget = raw_internal_face_divergence + "
                    "omitted_boundary_source + input_precipitation"
                ),
                "threshold": args.max_cell_budget_relative_rmse,
            },
        )
    )

    node_precipitation_path = data_dir / f"M80_PrNode_{event_id}.npz"
    node_precipitation_ok = node_precipitation_path.is_file()
    node_precipitation_evidence = {"path": str(node_precipitation_path)}
    if node_precipitation_ok:
        with np.load(node_precipitation_path, allow_pickle=False) as node_precipitation:
            required_keys = {
                "input_rate_mm_per_hour",
                "hgn_time_origin_days",
                "input_interval_seconds",
                "units",
                "source",
            }
            missing_keys = required_keys.difference(node_precipitation.files)
            if missing_keys:
                node_precipitation_ok = False
                node_precipitation_evidence["missing_keys"] = sorted(missing_keys)
            else:
                input_rate = np.asarray(
                    node_precipitation["input_rate_mm_per_hour"],
                    dtype=np.float64,
                )
                origin_days = float(node_precipitation["hgn_time_origin_days"])
                input_interval_seconds = float(
                    node_precipitation["input_interval_seconds"]
                )
                units = str(node_precipitation["units"])
                source = str(node_precipitation["source"])
        if node_precipitation_ok:
            hgn_time_days = np.loadtxt(
                data_dir / f"M80_US_InF_{event_id}.txt", usecols=0
            )
            area = np.loadtxt(data_dir / "M80_CA.txt").reshape(-1)
            elapsed_seconds = (hgn_time_days - origin_days) * 86400.0
            node_precipitation_ok = input_interval_seconds > 0.0
            if node_precipitation_ok:
                input_rows = np.ceil(
                    elapsed_seconds / input_interval_seconds - 1.0e-6
                ).astype(np.int64)
            else:
                input_rows = np.asarray([-1], dtype=np.int64)
            node_precipitation_ok = node_precipitation_ok and (
                input_rate.ndim == 2
                and input_rate.shape[1] == area.size
                and units == "mm/h"
                and source == "pre_simulation_raster_values_and_cell_weights"
                and np.min(input_rows) >= 0
                and np.max(input_rows) < input_rate.shape[0]
            )
            if node_precipitation_ok:
                delta_hours = np.diff(hgn_time_days) * 24.0
                reconstructed_precipitation = (
                    input_rate[input_rows[1:]][:, selected_nodes]
                    * delta_hours[:, None]
                    * 1.0e-3
                    * area[selected_nodes][None, :]
                )
                rain_residual = reconstructed_precipitation - np.asarray(
                    precipitation, dtype=np.float64
                )
                node_rain_relative_rmse = rms_ratio(
                    float(np.sum(rain_residual**2)),
                    float(np.sum(np.asarray(precipitation, dtype=np.float64) ** 2)),
                )
                node_rain_max_abs = float(np.max(np.abs(rain_residual)))
                node_precipitation_ok = (
                    node_rain_relative_rmse
                    <= args.max_node_rain_relative_rmse
                )
                node_precipitation_evidence.update(
                    {
                        "relative_rmse": node_rain_relative_rmse,
                        "maximum_absolute_residual_m3": node_rain_max_abs,
                        "threshold": args.max_node_rain_relative_rmse,
                        "units": units,
                        "source": source,
                    }
                )
            else:
                node_precipitation_evidence.update(
                    {
                        "shape": list(input_rate.shape),
                        "node_count": int(area.size),
                        "units": units,
                        "source": source,
                        "input_interval_seconds": input_interval_seconds,
                    }
                )
    checks.append(
        check(
            f"{event_id}_inference_node_precipitation_matches_target",
            node_precipitation_ok,
            node_precipitation_evidence,
        )
    )

    hdf_path = Path(event["hdf_path"])
    archive_path = target_dir / f"{event_id}_conservative_edge_target.npz"
    provenance_ok = (
        event.get("empirical_scale_factor") is None
        and event.get("local_precipitation_source")
        == "pre_simulation_raster_values_and_cell_weights"
        and hdf_path.is_file()
        and file_sha256(hdf_path) == event.get("hdf_sha256")
        and archive_path.is_file()
        and file_sha256(archive_path) == event.get("output_sha256")
    )
    checks.append(
        check(
            f"{event_id}_physical_provenance",
            provenance_ok,
            {
                "empirical_scale_factor": event.get("empirical_scale_factor"),
                "local_precipitation_source": event.get(
                    "local_precipitation_source"
                ),
                "hdf": str(hdf_path),
                "archive": str(archive_path),
            },
        )
    )
    correction_ratio = float(event.get("correction_to_raw_edge_rms_ratio", np.inf))
    rain_error = float(
        event.get("input_vs_result_precipitation_rate_max_abs_mmh", np.inf)
    )
    checks.append(
        check(
            f"{event_id}_projection_correction_warning_threshold",
            correction_ratio <= args.warn_correction_rms_ratio,
            {
                "ratio": correction_ratio,
                "threshold": args.warn_correction_rms_ratio,
                "interpretation": (
                    "Diagnostic only; no conservation law imposes a 1% bound."
                ),
            },
            severity="warning",
        )
    )
    checks.append(
        check(
            f"{event_id}_projection_correction_hard_limit",
            correction_ratio <= args.max_correction_rms_ratio,
            {"ratio": correction_ratio, "threshold": args.max_correction_rms_ratio},
        )
    )
    checks.append(
        check(
            f"{event_id}_input_rain_reconstruction",
            rain_error <= args.max_rain_error_mmh,
            {"maximum_error_mmh": rain_error, "threshold": args.max_rain_error_mmh},
        )
    )
    return checks, {
        "event_id": event_id,
        "validated": not any(
            not item["passed"] and item["severity"] == "error"
            for item in checks
        ),
        "closure_relative_rmse": closure_relative_rmse,
        "source_equation_relative_rmse": source_relative_rmse,
        "period_average_cell_budget_relative_rmse": (
            cell_budget_relative_rmse
        ),
        "correction_to_raw_edge_rms_ratio": correction_ratio,
        "input_rain_max_error_mmh": rain_error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--output-json", type=Path)
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--max-closure-relative-rmse", type=float, default=1.0e-4)
    parser.add_argument("--max-source-relative-rmse", type=float, default=1.0e-4)
    parser.add_argument(
        "--max-cell-budget-relative-rmse", type=float, default=1.0e-5
    )
    parser.add_argument("--warn-correction-rms-ratio", type=float, default=0.01)
    parser.add_argument("--max-correction-rms-ratio", type=float, default=0.05)
    parser.add_argument("--max-rain-error-mmh", type=float, default=1.0e-3)
    parser.add_argument(
        "--max-node-rain-relative-rmse", type=float, default=1.0e-6
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_dir = args.target_dir.resolve()
    manifest_path = target_dir / "conservative_edge_target_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    data_dir = Path(manifest["data_dir"])
    face_graph_path = Path(manifest["face_graph_file"])
    zone_path = Path(manifest["zone_label_file"])
    selected_path = target_dir / "selected_node_index.npy"
    selected_nodes = np.load(selected_path)
    with np.load(face_graph_path, allow_pickle=False) as face_graph:
        face_index = np.asarray(face_graph["internal_face_index"], dtype=np.int64)
        node_count = int(np.asarray(face_graph["hgn_to_hdf_cell_index"]).size)
    incidence = selected_incidence(face_index, selected_nodes, node_count)

    checks = [
        check(
            "manifest_transition_origin",
            manifest.get("transition_index_origin") == 0,
            manifest.get("transition_index_definition"),
        ),
        check(
            "face_graph_hash",
            file_sha256(face_graph_path) == manifest.get("face_graph_sha256"),
            str(face_graph_path),
        ),
        check(
            "zone_label_hash",
            file_sha256(zone_path) == manifest.get("zone_label_sha256"),
            str(zone_path),
        ),
        check(
            "selected_node_hash",
            file_sha256(selected_path) == manifest.get("selected_node_index_sha256"),
            str(selected_path),
        ),
    ]
    event_summaries = []
    for position, event in enumerate(manifest["events"], start=1):
        print(f"[{position}/{len(manifest['events'])}] validating {event['event_id']}")
        event_checks, summary = validate_event(
            event, data_dir, target_dir, incidence, selected_nodes, args
        )
        checks.extend(event_checks)
        event_summaries.append(summary)

    failures = [
        item
        for item in checks
        if not item["passed"] and item["severity"] == "error"
    ]
    warnings = [
        item
        for item in checks
        if not item["passed"] and item["severity"] == "warning"
    ]
    report = {
        "schema_version": 1,
        "target_dir": str(target_dir),
        "manifest": str(manifest_path),
        "ready_for_edge_training": not failures,
        "event_count": len(event_summaries),
        "selected_node_count": int(selected_nodes.size),
        "failure_count": len(failures),
        "warning_count": len(warnings),
        "events": event_summaries,
        "checks": checks,
    }
    report_text = json.dumps(report, indent=2) + "\n"
    report["report_sha256"] = hashlib.sha256(report_text.encode()).hexdigest()
    output_path = args.output_json or target_dir / "conservative_edge_target_validation.json"
    output_path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"ready_for_edge_training={report['ready_for_edge_training']}; "
        f"events={report['event_count']}; failures={report['failure_count']}; "
        f"warnings={report['warning_count']}"
    )
    print(f"Validation report: {output_path}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
