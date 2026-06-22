# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Compute per-internal-face statistics for HEC-RAS Face Flow targets."""

import argparse
import re
from pathlib import Path

import numpy as np


def natural_event_key(key: str) -> tuple[str, int]:
    hydrograph_id = key[: -len("_internal_face_delta")]
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", hydrograph_id)
    if match:
        return match.group(1), int(match.group(2))
    return hydrograph_id, -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-flow-npz", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--max-transitions", type=int)
    args = parser.parse_args()

    data = np.load(args.edge_flow_npz)
    keys = sorted(
        (key for key in data.files if key.endswith("_internal_face_delta")),
        key=natural_event_key,
    )
    if args.max_events is not None:
        keys = keys[: args.max_events]
    if not keys:
        raise KeyError(
            f"No '*_internal_face_delta' arrays found in {args.edge_flow_npz}."
        )

    count = 0
    sum_values = None
    sum_squared = None
    for key in keys:
        values = np.asarray(data[key], dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"{key} must be [transitions, faces], got {values.shape}.")
        if args.max_transitions is not None:
            values = values[: args.max_transitions]
        if sum_values is None:
            sum_values = np.zeros(values.shape[1], dtype=np.float64)
            sum_squared = np.zeros(values.shape[1], dtype=np.float64)
        elif values.shape[1] != sum_values.shape[0]:
            raise ValueError(
                f"{key} has {values.shape[1]} faces, expected {sum_values.shape[0]}."
            )
        sum_values += np.sum(values, axis=0)
        sum_squared += np.sum(values * values, axis=0)
        count += values.shape[0]

    if count == 0:
        raise ValueError("No transitions selected for face target statistics.")
    face_mean = sum_values / count
    face_rms = np.sqrt(sum_squared / count)
    variance = np.maximum(sum_squared / count - face_mean * face_mean, 0.0)
    face_std = np.sqrt(variance)
    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output_npz,
        face_mean=face_mean.astype(np.float32),
        face_std=face_std.astype(np.float32),
        face_rms=face_rms.astype(np.float32),
        num_events=np.array([len(keys)], dtype=np.int64),
        num_transitions=np.array([count], dtype=np.int64),
    )
    print(f"Wrote {args.output_npz}")
    print(f"events={len(keys)} transitions={count} faces={face_mean.shape[0]}")
    print(f"mean(face_rms)={float(np.mean(face_rms)):.6f}")
    print(f"median(face_rms)={float(np.median(face_rms)):.6f}")
    print(f"max(face_rms)={float(np.max(face_rms)):.6f}")


if __name__ == "__main__":
    main()
