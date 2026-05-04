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


def create_zones(area: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Map cell area to fidelity zone labels and soft weights."""
    zone_label = np.empty(area.shape[0], dtype=np.int64)
    zone_label[area >= 8000.0] = 0
    zone_label[(area >= 5000.0) & (area < 8000.0)] = 1
    zone_label[(area >= 2500.0) & (area < 5000.0)] = 2
    zone_label[area < 2500.0] = 3

    weights_by_zone = np.array([0.0, 0.25, 0.5, 1.0], dtype=np.float32)
    zone_weight = weights_by_zone[zone_label]
    return zone_label, zone_weight


def summarize(area: np.ndarray, zone_label: np.ndarray, zone_weight: np.ndarray) -> dict:
    """Build a JSON-serializable zone summary."""
    zones = {}
    for zone in range(4):
        mask = zone_label == zone
        zones[str(zone)] = {
            "count": int(mask.sum()),
            "fraction": float(mask.mean()),
            "weight": float(zone_weight[mask][0]) if mask.any() else 0.0,
            "area_min": float(area[mask].min()) if mask.any() else None,
            "area_max": float(area[mask].max()) if mask.any() else None,
            "area_mean": float(area[mask].mean()) if mask.any() else None,
        }
    return {
        "num_nodes": int(area.shape[0]),
        "thresholds": {
            "zone_0_low_fidelity": "area >= 8000",
            "zone_1": "5000 <= area < 8000",
            "zone_2": "2500 <= area < 5000",
            "zone_3_high_fidelity": "area < 2500",
        },
        "zones": zones,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--label-file", default="zone_label.txt")
    parser.add_argument("--weight-file", default="zone_weight.txt")
    parser.add_argument("--summary-file", default="zone_summary.json")
    args = parser.parse_args()

    area_path = args.data_dir / f"{args.prefix}_CA.txt"
    area = np.loadtxt(area_path, delimiter="\t").reshape(-1)
    zone_label, zone_weight = create_zones(area)
    summary = summarize(area, zone_label, zone_weight)

    np.savetxt(args.data_dir / args.label_file, zone_label, fmt="%d")
    np.savetxt(args.data_dir / args.weight_file, zone_weight, fmt="%.6f")
    with open(args.data_dir / args.summary_file, "w") as f:
        json.dump(summary, f, indent=2)

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
