# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Validate interval-integrated HEC-RAS Face Flow against Cell Flow Balance.

This is a pre-training diagnostic for a future edge-informed local
mass-conservation loss.  It reconstructs node-wise transport deltas from native
HEC-RAS ``Face Flow`` and compares them with the existing formal
``Cell Flow Balance`` target over the exact HGN time intervals.
"""

from __future__ import annotations

import argparse
import csv
import glob
import re
from pathlib import Path

import h5py
import numpy as np
from scipy import sparse


RESULT_BASE = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/"
    "2D Flow Areas/per2/"
)
RESULT_TIME_PATH = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time"
)
GEOMETRY_BASE = "Geometry/2D Flow Areas/per2/"
FACE_FLOW_PATH = RESULT_BASE + "Face Flow"
CELL_BALANCE_PATH = RESULT_BASE + "Cell Flow Balance"
PRECIPITATION_PATH = RESULT_BASE + "Cell Cumulative Precipitation Depth"


def hydrograph_sort_key(value: str) -> tuple[str, int | str]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", value)
    if match:
        return match.group(1), int(match.group(2))
    return value, value


def load_event_ids(data_dir: Path, event_ids_file: str | None, prefix: str) -> list[str]:
    if event_ids_file:
        path = data_dir / event_ids_file
        return [line.strip() for line in path.read_text().splitlines() if line.strip()]
    ids = []
    pattern = re.compile(rf"{re.escape(prefix)}_V_(H\d+)\.txt$")
    for path in data_dir.glob(f"{prefix}_V_H*.txt"):
        match = pattern.match(path.name)
        if match:
            ids.append(match.group(1))
    return sorted(ids, key=hydrograph_sort_key)


def load_hdf_paths(hdf_glob: str) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for raw_path in glob.glob(hdf_glob):
        path = Path(raw_path)
        match = re.search(r"plan(H\d+)", str(path))
        if match:
            paths[match.group(1)] = path
    return paths


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
    if np.any(np.diff(matched) <= 0):
        raise ValueError("Matched HDF indices must be strictly increasing.")
    return matched


def signed_face_sum(
    face_flow: np.ndarray,
    face_cells: np.ndarray,
    hdf_to_hgn: np.ndarray,
    num_nodes: int,
    mode: str,
) -> np.ndarray:
    """Convert HEC-RAS face flow cfs to node-wise signed net flow cfs."""
    mapped_faces = np.full_like(face_cells, -1)
    valid = (face_cells >= 0) & (face_cells < hdf_to_hgn.shape[0])
    mapped_faces[valid] = hdf_to_hgn[face_cells[valid]]

    first = mapped_faces[:, 0] >= 0
    second = mapped_faces[:, 1] >= 0
    if mode == "internal":
        use_face = first & second
    elif mode == "all_touching":
        use_face = first | second
    else:
        raise ValueError(f"Unknown mode: {mode}")

    first_use = use_face & first
    second_use = use_face & second
    row_index = np.concatenate(
        [np.nonzero(first_use)[0], np.nonzero(second_use)[0]]
    )
    col_index = np.concatenate(
        [mapped_faces[first_use, 0], mapped_faces[second_use, 1]]
    )
    sign = np.concatenate(
        [
            -np.ones(np.count_nonzero(first_use), dtype=np.float64),
            np.ones(np.count_nonzero(second_use), dtype=np.float64),
        ]
    )
    incidence = sparse.csr_matrix(
        (sign, (row_index, col_index)),
        shape=(face_cells.shape[0], num_nodes),
        dtype=np.float64,
    )
    return face_flow @ incidence


def integrate_intervals(
    values: np.ndarray, hdf_time_seconds: np.ndarray, hdf_indices: np.ndarray
) -> np.ndarray:
    out = np.zeros((hdf_indices.shape[0] - 1, values.shape[1]), dtype=np.float64)
    for idx, (start, end) in enumerate(zip(hdf_indices[:-1], hdf_indices[1:])):
        out[idx] = np.trapezoid(
            values[start : end + 1], x=hdf_time_seconds[start : end + 1], axis=0
        )
    return out


def summarize_residual(
    event_id: str,
    mode: str,
    edge_delta: np.ndarray,
    balance_delta: np.ndarray,
    zone_label: np.ndarray | None,
) -> dict[str, object]:
    residual = edge_delta - balance_delta
    target_rms = float(np.sqrt(np.mean(balance_delta**2)))
    rmse = float(np.sqrt(np.mean(residual**2)))
    row: dict[str, object] = {
        "event_id": event_id,
        "mode": mode,
        "num_intervals": int(edge_delta.shape[0]),
        "num_nodes": int(edge_delta.shape[1]),
        "target_rms_ft3": target_rms,
        "edge_rms_ft3": float(np.sqrt(np.mean(edge_delta**2))),
        "residual_rmse_ft3": rmse,
        "residual_mae_ft3": float(np.mean(np.abs(residual))),
        "residual_bias_ft3": float(np.mean(residual)),
        "relative_rmse": rmse / target_rms if target_rms else float("nan"),
        "correlation": float(
            np.corrcoef(edge_delta.reshape(-1), balance_delta.reshape(-1))[0, 1]
        ),
        "domain_residual_rmse_ft3": float(
            np.sqrt(np.mean(np.sum(residual, axis=1) ** 2))
        ),
    }
    if zone_label is not None:
        for zone in range(4):
            mask = zone_label == zone
            if np.any(mask):
                zone_residual = residual[:, mask]
                zone_target = balance_delta[:, mask]
                zone_target_rms = float(np.sqrt(np.mean(zone_target**2)))
                zone_rmse = float(np.sqrt(np.mean(zone_residual**2)))
                row[f"zone_{zone}_residual_rmse_ft3"] = zone_rmse
                row[f"zone_{zone}_relative_rmse"] = (
                    zone_rmse / zone_target_rms if zone_target_rms else float("nan")
                )
    return row


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]], args: argparse.Namespace) -> None:
    def fmt(value: object) -> str:
        return f"{float(value):.6e}" if isinstance(value, float) else str(value)

    lines = [
        "# HEC-RAS Edge Flow Delta Validation",
        "",
        "This diagnostic compares interval-integrated native HEC-RAS `Face Flow` "
        "against the existing `Cell Flow Balance` transport delta over HGN time "
        "intervals. It is a gate before training an edge-informed loss.",
        "",
        "## Inputs",
        "",
        f"- HGN data dir: `{args.data_dir}`",
        f"- HDF glob: `{args.hdf_glob}`",
        f"- Face graph: `{args.face_graph_npz}`",
        f"- Event ids file: `{args.event_ids_file or 'auto-discovered'}`",
        "",
        "## Summary",
        "",
        "| Event | Mode | Relative RMSE | RMSE (ft^3) | Correlation | Zone 3 RMSE (ft^3) | Zone 3 Relative RMSE |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['event_id']} | {row['mode']} | "
            f"{float(row['relative_rmse']):.6e} | "
            f"{float(row['residual_rmse_ft3']):.6e} | "
            f"{float(row['correlation']):.6f} | "
            f"{float(row.get('zone_3_residual_rmse_ft3', float('nan'))):.6e} | "
            f"{float(row.get('zone_3_relative_rmse', float('nan'))):.6e} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation Guide",
            "",
            "- `all_touching` includes any HEC-RAS face that touches an HGN node, "
            "including boundary/ghost faces.",
            "- `internal` includes only faces where both adjacent cells are HGN nodes.",
            "- If `all_touching` matches Cell Flow Balance but `internal` does not, "
            "the boundary-face contribution is required for exact local closure.",
            "- A future high-zone edge loss can still use an internal physical edge "
            "graph, but boundary nodes need explicit treatment or masking.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--hdf-glob", required=True)
    parser.add_argument("--face-graph-npz", required=True, type=Path)
    parser.add_argument("--event-ids-file")
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    parser.add_argument("--output-npz", type=Path)
    parser.add_argument(
        "--store-mode",
        choices=(
            "all",
            "internal",
            "all_touching",
            "boundary_source",
            "internal_plus_boundary_source",
        ),
        default="all",
        help="Which edge-flow delta arrays to store in --output-npz.",
    )
    args = parser.parse_args()

    event_ids = load_event_ids(args.data_dir, args.event_ids_file, args.prefix)
    hdf_paths = load_hdf_paths(args.hdf_glob)
    face_graph = np.load(args.face_graph_npz)
    hdf_to_hgn = face_graph["hdf_to_hgn_node_index"].astype(np.int64)
    internal_hdf_face_index = face_graph["internal_hdf_face_index"].astype(np.int64)
    num_nodes = int(np.sum(hdf_to_hgn >= 0))
    zone_path = args.data_dir / "zone_label.txt"
    zone_label = (
        np.loadtxt(zone_path, dtype=np.int64)
        if zone_path.exists()
        else None
    )

    rows: list[dict[str, object]] = []
    edge_deltas: dict[str, np.ndarray] = {}
    for event_id in event_ids:
        hdf_path = hdf_paths.get(event_id)
        if hdf_path is None:
            raise KeyError(f"No HDF path found for {event_id}")
        hgn_time_days = np.loadtxt(args.data_dir / f"{args.prefix}_US_InF_{event_id}.txt")[:, 0]
        with h5py.File(hdf_path, "r") as hdf:
            hdf_time_days = np.asarray(hdf[RESULT_TIME_PATH], dtype=np.float64)
            hdf_indices = match_hgn_times_to_hdf(hgn_time_days, hdf_time_days)
            hdf_time_seconds = hdf_time_days * 86400.0
            face_cells = np.asarray(hdf[GEOMETRY_BASE + "Faces Cell Indexes"], dtype=np.int64)
            face_flow = np.nan_to_num(np.asarray(hdf[FACE_FLOW_PATH], dtype=np.float64), nan=0.0)
            balance = np.nan_to_num(
                np.asarray(hdf[CELL_BALANCE_PATH][:, hdf_to_hgn >= 0], dtype=np.float64),
                nan=0.0,
            )
            balance_delta = integrate_intervals(balance, hdf_time_seconds, hdf_indices)

            event_edge_deltas = {}
            for mode in ("internal", "all_touching"):
                net_flow = signed_face_sum(face_flow, face_cells, hdf_to_hgn, num_nodes, mode)
                edge_delta = integrate_intervals(net_flow, hdf_time_seconds, hdf_indices)
                event_edge_deltas[mode] = edge_delta
                rows.append(
                    summarize_residual(event_id, mode, edge_delta, balance_delta, zone_label)
                )
                if args.output_npz:
                    if args.store_mode in {"all", mode}:
                        edge_deltas[f"{event_id}_{mode}_edge_delta"] = (
                            edge_delta.astype(np.float32)
                        )
            if args.output_npz and args.store_mode in {
                "all",
                "boundary_source",
                "internal_plus_boundary_source",
            }:
                boundary_source = (
                    event_edge_deltas["all_touching"] - event_edge_deltas["internal"]
                )
                edge_deltas[f"{event_id}_boundary_source_delta"] = (
                    boundary_source.astype(np.float32)
                )
            if args.output_npz and args.store_mode in {
                "all",
                "internal_plus_boundary_source",
            }:
                edge_deltas[f"{event_id}_internal_edge_delta"] = (
                    event_edge_deltas["internal"].astype(np.float32)
                )
                internal_face_delta = integrate_intervals(
                    face_flow[:, internal_hdf_face_index],
                    hdf_time_seconds,
                    hdf_indices,
                )
                edge_deltas[f"{event_id}_internal_face_delta"] = (
                    internal_face_delta.astype(np.float32)
                )
            if args.output_npz:
                edge_deltas[f"{event_id}_cell_balance_delta"] = balance_delta.astype(
                    np.float32
                )

    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows, args)
    if args.output_npz:
        args.output_npz.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(args.output_npz, **edge_deltas)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")
    if args.output_npz:
        print(f"Wrote {args.output_npz}")


if __name__ == "__main__":
    main()
