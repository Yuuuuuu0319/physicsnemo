# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create node-level fidelity zones from HydroGraphNet cell areas.

This script is deliberately separate from edge-informed local-conservation work.
It only turns per-node cell area into a zone label and a soft local-loss weight.
"""

import argparse
import json
from pathlib import Path

import numpy as np


DEFAULT_NOMINAL_RESOLUTIONS = (100.0, 80.0, 60.0, 40.0)
DEFAULT_WEIGHTS = (0.0, 0.25, 0.5, 1.0)


def midpoint_thresholds(nominal_resolutions: tuple[float, ...]) -> np.ndarray:
    """Return midpoint thresholds between nominal square-cell areas."""
    nominal_areas = np.asarray(nominal_resolutions, dtype=np.float64) ** 2
    if not np.all(nominal_areas[:-1] > nominal_areas[1:]):
        raise ValueError("Nominal resolutions must be ordered from coarse to fine.")
    return 0.5 * (nominal_areas[:-1] + nominal_areas[1:])


def create_zones(
    area: np.ndarray,
    nominal_resolutions: tuple[float, ...] = DEFAULT_NOMINAL_RESOLUTIONS,
    weights_by_zone: tuple[float, ...] = DEFAULT_WEIGHTS,
) -> tuple[np.ndarray, np.ndarray]:
    """Map cell area to fidelity zone labels and soft weights."""
    thresholds = midpoint_thresholds(nominal_resolutions)
    zone_label = np.full(area.shape[0], len(nominal_resolutions) - 1, dtype=np.int64)
    for zone, threshold in enumerate(thresholds):
        zone_label[area >= threshold] = zone
        area = np.where(area >= threshold, -np.inf, area)

    weights = np.asarray(weights_by_zone, dtype=np.float32)
    if weights.shape[0] != len(nominal_resolutions):
        raise ValueError("Number of zone weights must match nominal resolutions.")
    zone_weight = weights[zone_label]
    return zone_label, zone_weight


def summarize(
    area: np.ndarray,
    zone_label: np.ndarray,
    zone_weight: np.ndarray,
    nominal_resolutions: tuple[float, ...],
) -> dict:
    """Build a JSON-serializable zone summary."""
    nominal_areas = np.asarray(nominal_resolutions, dtype=np.float64) ** 2
    thresholds = midpoint_thresholds(nominal_resolutions)
    zones = {}
    for zone in range(len(nominal_resolutions)):
        mask = zone_label == zone
        zones[str(zone)] = {
            "count": int(mask.sum()),
            "fraction": float(mask.mean()),
            "weight": float(zone_weight[mask][0]) if mask.any() else 0.0,
            "nominal_resolution": float(nominal_resolutions[zone]),
            "nominal_area": float(nominal_areas[zone]),
            "area_min": float(area[mask].min()) if mask.any() else None,
            "area_max": float(area[mask].max()) if mask.any() else None,
            "area_mean": float(area[mask].mean()) if mask.any() else None,
        }
    return {
        "num_nodes": int(area.shape[0]),
        "nominal_resolutions": [float(value) for value in nominal_resolutions],
        "nominal_areas": [float(value) for value in nominal_areas],
        "thresholds": [float(value) for value in thresholds],
        "threshold_rule": "zone boundaries are midpoints between nominal square-cell areas",
        "zones": zones,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--label-file", default="zone_label.txt")
    parser.add_argument("--weight-file", default="zone_weight.txt")
    parser.add_argument("--summary-file", default="zone_summary.json")
    parser.add_argument(
        "--nominal-resolutions",
        nargs=4,
        type=float,
        default=DEFAULT_NOMINAL_RESOLUTIONS,
        metavar=("ZONE0", "ZONE1", "ZONE2", "ZONE3"),
        help="Coarse-to-fine nominal mesh resolutions. Defaults to 100 80 60 40.",
    )
    parser.add_argument(
        "--zone-weights",
        nargs=4,
        type=float,
        default=DEFAULT_WEIGHTS,
        metavar=("ZONE0", "ZONE1", "ZONE2", "ZONE3"),
        help="Soft local-loss weights for zones. Defaults to 0 0.25 0.5 1.",
    )
    args = parser.parse_args()

    area_path = args.data_dir / f"{args.prefix}_CA.txt"
    area = np.loadtxt(area_path, delimiter="\t").reshape(-1)
    nominal_resolutions = tuple(args.nominal_resolutions)
    zone_weights = tuple(args.zone_weights)
    zone_label, zone_weight = create_zones(area, nominal_resolutions, zone_weights)
    summary = summarize(area, zone_label, zone_weight, nominal_resolutions)

    np.savetxt(args.data_dir / args.label_file, zone_label, fmt="%d")
    np.savetxt(args.data_dir / args.weight_file, zone_weight, fmt="%.6f")
    with open(args.data_dir / args.summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
