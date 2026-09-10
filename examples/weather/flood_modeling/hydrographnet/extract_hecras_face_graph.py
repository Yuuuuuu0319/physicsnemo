# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Extract a HEC-RAS true-face graph aligned to HydroGraphNet nodes.

This preprocessing script is deliberately separate from the existing kNN
HydroGraphNet graph and from the rough VX/VY proxy branch. It extracts the face
connectivity and face geometry that will be needed for a future face-based local
mass-conservation loss.
"""

import argparse
import hashlib
import json
from pathlib import Path

import h5py
import numpy as np
from scipy.spatial import cKDTree


GEOM_AREA_PATH = "Geometry/2D Flow Areas/per2"
FACE_VELOCITY_PATH = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/"
    "2D Flow Areas/per2/Face Velocity"
)
FACE_FLOW_PATH = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/"
    "2D Flow Areas/per2/Face Flow"
)


def decode_attr(value) -> str:
    """Decode a scalar HDF attribute for portable metadata."""

    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, np.ndarray) and value.ndim == 0:
        return decode_attr(value.item())
    return str(value)


def read_hdf_face_data(hdf_path: Path) -> dict[str, np.ndarray]:
    """Read HEC-RAS cell and face arrays from an HDF result file."""
    with h5py.File(hdf_path, "r") as hdf:
        area_group = hdf[GEOM_AREA_PATH]
        data = {
            "cell_centers": area_group["Cells Center Coordinate"][:],
            "cell_surface_area": area_group["Cells Surface Area"][:],
            "cells_facepoint_indexes": area_group["Cells FacePoint Indexes"][:],
            "cell_volume_elevation_info": area_group[
                "Cells Volume Elevation Info"
            ][:],
            "cell_volume_elevation_values": area_group[
                "Cells Volume Elevation Values"
            ][:],
            "faces_cell_indexes": area_group["Faces Cell Indexes"][:],
            "faces_facepoint_indexes": area_group["Faces FacePoint Indexes"][:],
            "faces_normal_length": area_group[
                "Faces NormalUnitVector and Length"
            ][:],
            "facepoints_coordinate": area_group["FacePoints Coordinate"][:],
            "unit_system": np.asarray(hdf.attrs.get("Units System", b"")),
        }
        data["has_face_velocity"] = np.array(FACE_VELOCITY_PATH in hdf, dtype=bool)
        if FACE_VELOCITY_PATH in hdf:
            data["face_velocity_shape"] = np.asarray(hdf[FACE_VELOCITY_PATH].shape)
        data["has_face_flow"] = np.array(FACE_FLOW_PATH in hdf, dtype=bool)
        if FACE_FLOW_PATH in hdf:
            data["face_flow_shape"] = np.asarray(hdf[FACE_FLOW_PATH].shape)
    return data


def polygon_areas(
    facepoint_coordinates: np.ndarray, cells_facepoint_indexes: np.ndarray
) -> np.ndarray:
    """Calculate horizontal polygon area as an independent geometry check."""

    areas = np.zeros(cells_facepoint_indexes.shape[0], dtype=np.float64)
    for cell_index, indexes in enumerate(cells_facepoint_indexes):
        indexes = indexes[indexes >= 0]
        points = facepoint_coordinates[indexes]
        x = points[:, 0]
        y = points[:, 1]
        areas[cell_index] = 0.5 * abs(
            np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))
        )
    return areas


def validate_hgn_alignment(
    hgn_xy: np.ndarray,
    hgn_area: np.ndarray,
    hdf_cell_centers: np.ndarray,
    hdf_cell_area: np.ndarray,
    hdf_polygon_area: np.ndarray,
    atol: float,
) -> tuple[dict, np.ndarray]:
    """Map HGN nodes to HDF cells by coordinates and validate area definitions."""

    num_nodes = hgn_xy.shape[0]
    if hdf_cell_centers.shape[0] < num_nodes:
        raise ValueError(
            f"HDF has {hdf_cell_centers.shape[0]} cells but HGN has {num_nodes} nodes."
        )
    coordinate_distance, hgn_to_hdf = cKDTree(hdf_cell_centers).query(
        hgn_xy, k=1
    )
    hgn_to_hdf = np.asarray(hgn_to_hdf, dtype=np.int64)
    max_coordinate_distance = float(np.max(coordinate_distance))
    if max_coordinate_distance > atol:
        raise ValueError(
            "HGN XY does not align with HDF cell centers. "
            f"Max distance: {max_coordinate_distance}"
        )
    if np.unique(hgn_to_hdf).size != num_nodes:
        raise ValueError("HGN-to-HDF coordinate mapping is not one-to-one.")

    native_area_delta = hdf_cell_area[hgn_to_hdf] - hgn_area
    polygon_area_delta = hdf_polygon_area[hgn_to_hdf] - hgn_area
    alignment = {
        "num_hgn_nodes": int(num_nodes),
        "num_hdf_cells": int(hdf_cell_centers.shape[0]),
        "mapping_mode": "nearest_coordinate_one_to_one",
        "max_coordinate_distance": max_coordinate_distance,
        "mean_coordinate_distance": float(np.mean(coordinate_distance)),
        "native_area_rmse": float(np.sqrt(np.mean(native_area_delta**2))),
        "native_area_max_abs_diff": float(np.max(np.abs(native_area_delta))),
        "native_area_exact_count_1e-3": int(
            np.sum(np.abs(native_area_delta) < 1e-3)
        ),
        "polygon_area_rmse": float(np.sqrt(np.mean(polygon_area_delta**2))),
        "polygon_area_max_abs_diff": float(np.max(np.abs(polygon_area_delta))),
        "polygon_area_exact_count_1e-3": int(
            np.sum(np.abs(polygon_area_delta) < 1e-3)
        ),
    }
    return alignment, hgn_to_hdf


def build_face_graph(
    face_cells: np.ndarray, hdf_to_hgn: np.ndarray
) -> dict[str, np.ndarray]:
    """Split HEC-RAS faces into internal HGN faces and boundary/ghost faces."""

    mapped_cells = np.full_like(face_cells, -1)
    valid = (face_cells >= 0) & (face_cells < hdf_to_hgn.size)
    mapped_cells[valid] = hdf_to_hgn[face_cells[valid]]
    internal_mask = (mapped_cells[:, 0] >= 0) & (mapped_cells[:, 1] >= 0)
    boundary_mask = (mapped_cells[:, 0] >= 0) ^ (mapped_cells[:, 1] >= 0)

    internal_hdf_face_index = np.nonzero(internal_mask)[0].astype(np.int64)
    boundary_hdf_face_index = np.nonzero(boundary_mask)[0].astype(np.int64)
    internal_faces = mapped_cells[internal_mask].astype(np.int64)
    boundary_faces = face_cells[boundary_mask].astype(np.int64)

    return {
        "internal_face_index": internal_faces.T,
        "internal_hdf_face_index": internal_hdf_face_index,
        "boundary_face_cell_index": boundary_faces,
        "boundary_hdf_face_index": boundary_hdf_face_index,
        "internal_mask": internal_mask,
        "boundary_mask": boundary_mask,
    }


def extract_volume_table_top(
    volume_info: np.ndarray,
    volume_values: np.ndarray,
    hgn_to_hdf: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the highest tabulated elevation and volume for each HGN cell."""

    num_nodes = hgn_to_hdf.size
    top_elevation = np.full(num_nodes, np.nan, dtype=np.float64)
    top_volume = np.full(num_nodes, np.nan, dtype=np.float64)
    for node_index, hdf_cell_index in enumerate(hgn_to_hdf):
        offset, count = volume_info[hdf_cell_index]
        if count <= 0:
            continue
        top_elevation[node_index] = volume_values[offset + count - 1, 0]
        top_volume[node_index] = volume_values[offset + count - 1, 1]
    if not np.all(np.isfinite(top_elevation)):
        missing = np.flatnonzero(~np.isfinite(top_elevation))
        raise ValueError(
            "Every active HGN cell requires a HEC-RAS volume-elevation table; "
            f"missing nodes: {missing[:10].tolist()} (total={missing.size})."
        )
    return top_elevation, top_volume


def build_selective_scope(
    face_graph: dict[str, np.ndarray],
    hdf_to_hgn: np.ndarray,
    zone_labels: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict]:
    """Build the fixed high-zone interior control-volume partition."""

    num_nodes = zone_labels.size
    boundary_hdf_cells = face_graph["boundary_face_cell_index"].reshape(-1)
    valid = (boundary_hdf_cells >= 0) & (
        boundary_hdf_cells < hdf_to_hgn.size
    )
    mapped_boundary_nodes = hdf_to_hgn[boundary_hdf_cells[valid]]
    mapped_boundary_nodes = mapped_boundary_nodes[
        (mapped_boundary_nodes >= 0) & (mapped_boundary_nodes < num_nodes)
    ]
    boundary_node_mask = np.zeros(num_nodes, dtype=np.bool_)
    boundary_node_mask[mapped_boundary_nodes] = True
    high_interior_mask = (zone_labels == 3) & (~boundary_node_mask)

    parent = np.arange(num_nodes, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    src, dst = face_graph["internal_face_index"]
    interior_faces = high_interior_mask[src] & high_interior_mask[dst]
    for first, second in zip(src[interior_faces], dst[interior_faces]):
        union(int(first), int(second))

    labels = np.full(num_nodes, -1, dtype=np.int64)
    selected_nodes = np.flatnonzero(high_interior_mask)
    roots = np.asarray([find(int(node)) for node in selected_nodes], dtype=np.int64)
    _, compact = np.unique(roots, return_inverse=True)
    labels[selected_nodes] = compact
    component_sizes = np.bincount(compact) if compact.size else np.asarray([], dtype=int)
    active_face_mask = (labels[src] >= 0) | (labels[dst] >= 0)
    component_boundary_face_mask = (labels[src] >= 0) ^ (labels[dst] >= 0)

    arrays = {
        "boundary_node_mask": boundary_node_mask,
        "high_interior_control_volume_label": labels,
    }
    summary = {
        "zone_label_code": 3,
        "num_high_zone_nodes": int(np.sum(zone_labels == 3)),
        "num_boundary_nodes": int(np.sum(boundary_node_mask)),
        "num_high_zone_boundary_nodes": int(
            np.sum((zone_labels == 3) & boundary_node_mask)
        ),
        "num_high_interior_nodes": int(np.sum(high_interior_mask)),
        "num_high_interior_control_volumes": int(component_sizes.size),
        "control_volume_size_min": (
            int(component_sizes.min()) if component_sizes.size else 0
        ),
        "control_volume_size_median": (
            float(np.median(component_sizes)) if component_sizes.size else 0.0
        ),
        "control_volume_size_max": (
            int(component_sizes.max()) if component_sizes.size else 0
        ),
        "num_active_high_interior_touch_faces": int(np.sum(active_face_mask)),
        "active_face_fraction": float(np.mean(active_face_mask)),
        "num_control_volume_boundary_faces": int(
            np.sum(component_boundary_face_mask)
        ),
    }
    return arrays, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf-file", required=True, type=Path)
    parser.add_argument("--hgn-data-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--summary-file", required=True, type=Path)
    parser.add_argument("--alignment-atol", type=float, default=1e-6)
    args = parser.parse_args()

    hgn_xy = np.loadtxt(args.hgn_data_dir / f"{args.prefix}_XY.txt", delimiter="\t")
    hgn_area = np.loadtxt(args.hgn_data_dir / f"{args.prefix}_CA.txt").reshape(-1)
    hdf_data = read_hdf_face_data(args.hdf_file)
    hdf_polygon_area = polygon_areas(
        hdf_data["facepoints_coordinate"], hdf_data["cells_facepoint_indexes"]
    )
    alignment, hgn_to_hdf_cell_index = validate_hgn_alignment(
        hgn_xy,
        hgn_area,
        hdf_data["cell_centers"],
        hdf_data["cell_surface_area"],
        hdf_polygon_area,
        args.alignment_atol,
    )
    num_hgn_nodes = hgn_xy.shape[0]
    hdf_to_hgn_node_index = np.full(hdf_data["cell_centers"].shape[0], -1, dtype=np.int64)
    hdf_to_hgn_node_index[hgn_to_hdf_cell_index] = np.arange(
        num_hgn_nodes, dtype=np.int64
    )
    face_graph = build_face_graph(
        hdf_data["faces_cell_indexes"], hdf_to_hgn_node_index
    )
    zone_path = args.hgn_data_dir / "zone_label.txt"
    selective_arrays = {}
    selective_summary = None
    zone_label_sha256 = None
    if zone_path.is_file():
        zone_labels = np.loadtxt(zone_path, dtype=np.int64).reshape(-1)
        if zone_labels.size != num_hgn_nodes:
            raise ValueError(
                f"zone_label.txt has {zone_labels.size} nodes, expected "
                f"{num_hgn_nodes}."
            )
        selective_arrays, selective_summary = build_selective_scope(
            face_graph, hdf_to_hgn_node_index, zone_labels
        )
        zone_label_sha256 = hashlib.sha256(zone_path.read_bytes()).hexdigest()

    internal_hdf_face_index = face_graph["internal_hdf_face_index"]
    boundary_hdf_face_index = face_graph["boundary_hdf_face_index"]
    normals = hdf_data["faces_normal_length"]
    node_volume_table_top_elevation, node_volume_table_top_volume = (
        extract_volume_table_top(
            hdf_data["cell_volume_elevation_info"],
            hdf_data["cell_volume_elevation_values"],
            hgn_to_hdf_cell_index,
        )
    )

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        hgn_to_hdf_cell_index=hgn_to_hdf_cell_index,
        hdf_to_hgn_node_index=hdf_to_hgn_node_index,
        hgn_node_surface_area=hdf_data["cell_surface_area"][hgn_to_hdf_cell_index],
        hgn_node_polygon_area=hdf_polygon_area[hgn_to_hdf_cell_index],
        hgn_node_volume_table_top_elevation=node_volume_table_top_elevation,
        hgn_node_volume_table_top_volume=node_volume_table_top_volume,
        internal_face_index=face_graph["internal_face_index"],
        internal_hdf_face_index=internal_hdf_face_index,
        internal_normal_unit=normals[internal_hdf_face_index, :2],
        internal_face_length=normals[internal_hdf_face_index, 2],
        boundary_face_cell_index=face_graph["boundary_face_cell_index"],
        boundary_hdf_face_index=boundary_hdf_face_index,
        boundary_normal_unit=normals[boundary_hdf_face_index, :2],
        boundary_face_length=normals[boundary_hdf_face_index, 2],
        faces_facepoint_indexes=hdf_data["faces_facepoint_indexes"],
        facepoints_coordinate=hdf_data["facepoints_coordinate"],
        hdf_face_velocity_path=np.asarray(FACE_VELOCITY_PATH),
        hdf_face_flow_path=np.asarray(FACE_FLOW_PATH),
        zone_label_sha256=np.asarray(zone_label_sha256 or ""),
        **selective_arrays,
    )

    summary = {
        "hdf_file": str(args.hdf_file),
        "hgn_data_dir": str(args.hgn_data_dir),
        "output_npz": str(args.output_npz),
        "alignment": alignment,
        "num_hdf_faces": int(hdf_data["faces_cell_indexes"].shape[0]),
        "num_internal_hgn_faces": int(internal_hdf_face_index.shape[0]),
        "num_boundary_or_ghost_faces": int(boundary_hdf_face_index.shape[0]),
        "hdf_units_system": decode_attr(hdf_data["unit_system"]),
        "node_volume_table_top_elevation_min_native": float(
            np.min(node_volume_table_top_elevation)
        ),
        "node_volume_table_top_elevation_max_native": float(
            np.max(node_volume_table_top_elevation)
        ),
        "face_velocity_path": FACE_VELOCITY_PATH,
        "face_flow_path": FACE_FLOW_PATH,
        "has_face_velocity": bool(hdf_data["has_face_velocity"]),
        "face_velocity_shape": (
            hdf_data["face_velocity_shape"].astype(int).tolist()
            if bool(hdf_data["has_face_velocity"])
            else None
        ),
        "has_face_flow": bool(hdf_data["has_face_flow"]),
        "zone_label_file": str(zone_path) if zone_path.is_file() else None,
        "zone_label_sha256": zone_label_sha256,
        "selective_high_zone_scope": selective_summary,
        "face_flow_shape": (
            hdf_data["face_flow_shape"].astype(int).tolist()
            if bool(hdf_data["has_face_flow"])
            else None
        ),
        "notes": [
            "HGN nodes are mapped one-to-one to HDF cells by center coordinates; internal faces require both endpoints in that mapping.",
            "Boundary/ghost faces are saved separately and are not mixed into the internal face graph.",
            "hgn_node_surface_area is the native HEC-RAS area used by storage and precipitation-volume accounting.",
            "hgn_node_polygon_area is retained only as an independent horizontal-geometry diagnostic.",
            "Native Face Flow is the preferred physical edge-flux source; Face Velocity should only be used as a fallback with geometry.",
            "The HDF face velocity dataset is event-specific; use its face indexes only when the HDF event matches the evaluated hydrograph.",
        ],
    }
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
