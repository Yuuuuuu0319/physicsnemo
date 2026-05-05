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
import json
from pathlib import Path

import h5py
import numpy as np


GEOM_AREA_PATH = "Geometry/2D Flow Areas/per2"
FACE_VELOCITY_PATH = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/"
    "2D Flow Areas/per2/Face Velocity"
)


def read_hdf_face_data(hdf_path: Path) -> dict[str, np.ndarray]:
    """Read HEC-RAS cell and face arrays from an HDF result file."""
    with h5py.File(hdf_path, "r") as hdf:
        area_group = hdf[GEOM_AREA_PATH]
        data = {
            "cell_centers": area_group["Cells Center Coordinate"][:],
            "cell_surface_area": area_group["Cells Surface Area"][:],
            "faces_cell_indexes": area_group["Faces Cell Indexes"][:],
            "faces_facepoint_indexes": area_group["Faces FacePoint Indexes"][:],
            "faces_normal_length": area_group[
                "Faces NormalUnitVector and Length"
            ][:],
            "facepoints_coordinate": area_group["FacePoints Coordinate"][:],
        }
        data["has_face_velocity"] = np.array(FACE_VELOCITY_PATH in hdf, dtype=bool)
        if FACE_VELOCITY_PATH in hdf:
            data["face_velocity_shape"] = np.asarray(hdf[FACE_VELOCITY_PATH].shape)
    return data


def validate_hgn_alignment(
    hgn_xy: np.ndarray,
    hgn_area: np.ndarray,
    hdf_cell_centers: np.ndarray,
    hdf_cell_area: np.ndarray,
    atol: float,
) -> dict:
    """Validate that HGN node order matches the first HDF cells."""
    num_nodes = hgn_xy.shape[0]
    if hdf_cell_centers.shape[0] < num_nodes:
        raise ValueError(
            f"HDF has {hdf_cell_centers.shape[0]} cells but HGN has {num_nodes} nodes."
        )
    coordinate_delta = hdf_cell_centers[:num_nodes] - hgn_xy
    coordinate_distance = np.linalg.norm(coordinate_delta, axis=1)
    max_coordinate_distance = float(np.max(coordinate_distance))
    if max_coordinate_distance > atol:
        raise ValueError(
            "HGN XY does not align with the first HDF cell centers. "
            f"Max distance: {max_coordinate_distance}"
        )

    area_delta = hdf_cell_area[:num_nodes] - hgn_area
    return {
        "num_hgn_nodes": int(num_nodes),
        "num_hdf_cells": int(hdf_cell_centers.shape[0]),
        "max_coordinate_distance": max_coordinate_distance,
        "mean_coordinate_distance": float(np.mean(coordinate_distance)),
        "area_rmse": float(np.sqrt(np.mean(area_delta**2))),
        "area_max_abs_diff": float(np.max(np.abs(area_delta))),
        "area_exact_count_1e-3": int(np.sum(np.abs(area_delta) < 1e-3)),
    }


def build_face_graph(face_cells: np.ndarray, num_nodes: int) -> dict[str, np.ndarray]:
    """Split HEC-RAS faces into internal HGN faces and boundary/ghost faces."""
    internal_mask = (face_cells[:, 0] < num_nodes) & (face_cells[:, 1] < num_nodes)
    boundary_mask = (face_cells[:, 0] < num_nodes) ^ (face_cells[:, 1] < num_nodes)

    internal_hdf_face_index = np.nonzero(internal_mask)[0].astype(np.int64)
    boundary_hdf_face_index = np.nonzero(boundary_mask)[0].astype(np.int64)
    internal_faces = face_cells[internal_mask].astype(np.int64)
    boundary_faces = face_cells[boundary_mask].astype(np.int64)

    return {
        "internal_face_index": internal_faces.T,
        "internal_hdf_face_index": internal_hdf_face_index,
        "boundary_face_cell_index": boundary_faces,
        "boundary_hdf_face_index": boundary_hdf_face_index,
        "internal_mask": internal_mask,
        "boundary_mask": boundary_mask,
    }


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
    alignment = validate_hgn_alignment(
        hgn_xy,
        hgn_area,
        hdf_data["cell_centers"],
        hdf_data["cell_surface_area"],
        args.alignment_atol,
    )
    face_graph = build_face_graph(hdf_data["faces_cell_indexes"], hgn_xy.shape[0])

    internal_hdf_face_index = face_graph["internal_hdf_face_index"]
    boundary_hdf_face_index = face_graph["boundary_hdf_face_index"]
    normals = hdf_data["faces_normal_length"]

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
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
    )

    summary = {
        "hdf_file": str(args.hdf_file),
        "hgn_data_dir": str(args.hgn_data_dir),
        "output_npz": str(args.output_npz),
        "alignment": alignment,
        "num_hdf_faces": int(hdf_data["faces_cell_indexes"].shape[0]),
        "num_internal_hgn_faces": int(internal_hdf_face_index.shape[0]),
        "num_boundary_or_ghost_faces": int(boundary_hdf_face_index.shape[0]),
        "face_velocity_path": FACE_VELOCITY_PATH,
        "has_face_velocity": bool(hdf_data["has_face_velocity"]),
        "face_velocity_shape": (
            hdf_data["face_velocity_shape"].astype(int).tolist()
            if bool(hdf_data["has_face_velocity"])
            else None
        ),
        "notes": [
            "Only faces whose two cell indexes are both < num_hgn_nodes are included as internal HGN faces.",
            "Boundary/ghost faces are saved separately and are not mixed into the internal face graph.",
            "The HDF face velocity dataset is event-specific; use its face indexes only when the HDF event matches the evaluated hydrograph.",
        ],
    }
    args.summary_file.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
