# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Create fidelity-zone ablation weight files.

These controls stay in the node-zone branch. They do not introduce edge
features, edge heads, or velocity-based local flux.
"""

import argparse
import json
from pathlib import Path

import numpy as np


def summarize(weights: np.ndarray) -> dict:
    unique, counts = np.unique(weights, return_counts=True)
    return {
        "num_nodes": int(weights.shape[0]),
        "weights": {
            f"{float(weight):.6g}": int(count)
            for weight, count in zip(unique, counts)
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--zone-weight-file", default="zone_weight.txt")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--full-weight-file", default="zone_weight_full.txt")
    parser.add_argument("--random-weight-file", default=None)
    parser.add_argument("--summary-file", default=None)
    args = parser.parse_args()

    zone_weight = np.loadtxt(args.data_dir / args.zone_weight_file, dtype=np.float32)
    full_weight = np.ones_like(zone_weight, dtype=np.float32)

    rng = np.random.default_rng(args.seed)
    random_weight = zone_weight.copy()
    rng.shuffle(random_weight)

    random_weight_file = (
        args.random_weight_file or f"zone_weight_random_seed{args.seed}.txt"
    )
    summary_file = args.summary_file or f"zone_ablation_summary_seed{args.seed}.json"

    np.savetxt(args.data_dir / args.full_weight_file, full_weight, fmt="%.6f")
    np.savetxt(args.data_dir / random_weight_file, random_weight, fmt="%.6f")

    summary = {
        "source_weight_file": args.zone_weight_file,
        "seed": args.seed,
        "full_weight_file": args.full_weight_file,
        "random_weight_file": random_weight_file,
        "full": summarize(full_weight),
        "random": summarize(random_weight),
    }
    with open(args.data_dir / summary_file, "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
