# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Compute train-only per-face statistics for HEC-RAS Face Flow targets.

The formal conservative targets are stored as one memory-mapped array per
event. This tool requires an explicit event-ID file for that format so
validation and sealed-test targets cannot contribute to normalization.
"""

import argparse
import hashlib
import json
import re
from pathlib import Path

import numpy as np


def natural_event_key(key: str) -> tuple[str, int]:
    hydrograph_id = key[: -len("_internal_face_delta")]
    return natural_event_id(hydrograph_id)


def natural_event_id(event_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", event_id)
    if match:
        return match.group(1), int(match.group(2))
    return event_id, -1


def read_event_ids(path: Path) -> list[str]:
    event_ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not event_ids:
        raise ValueError(f"No event IDs found in {path}.")
    if len(event_ids) != len(set(event_ids)):
        raise ValueError(f"Duplicate event IDs found in {path}.")
    return event_ids


def load_time_contract(hgn_data_dir: Path) -> tuple[int, int]:
    validation_path = hgn_data_dir / "hgn_dataset_validation.json"
    if not validation_path.is_file():
        raise FileNotFoundError(validation_path)
    validation = json.loads(validation_path.read_text())
    contract = validation.get("model_time_contract", {})
    return int(contract["dynamic_skip_steps"]), int(contract["post_peak_steps"])


def usable_transition_slice(
    hgn_data_dir: Path,
    event_id: str,
    target_transition_count: int,
    n_time_steps: int,
) -> slice:
    """Return the exact raw-target intervals used by one-step training."""
    dynamic_skip_steps, post_peak_steps = load_time_contract(hgn_data_dir)
    inflow_path = hgn_data_dir / f"M80_US_InF_{event_id}.txt"
    inflow = np.loadtxt(inflow_path, delimiter="\t", ndmin=2)[:, 1]
    trimmed_inflow = inflow[dynamic_skip_steps:]
    peak_index = int(np.argmax(trimmed_inflow))
    trimmed_length = min(trimmed_inflow.shape[0], peak_index + post_peak_steps)
    if trimmed_length <= n_time_steps:
        raise ValueError(
            f"{event_id} has only {trimmed_length} retained frames for "
            f"n_time_steps={n_time_steps}."
        )

    # Dataset samples predict frame t+n from frame t+n-1. Therefore the first
    # target interval is skip+n-1 and the last is skip+trimmed_length-2.
    start = dynamic_skip_steps + n_time_steps - 1
    stop = dynamic_skip_steps + trimmed_length - 1
    if stop > target_transition_count:
        raise ValueError(
            f"{event_id} requires target transitions through {stop - 1}, but "
            f"only {target_transition_count} exist."
        )
    return slice(start, stop)


def load_target_arrays(args) -> list[tuple[str, np.ndarray]]:
    requested_ids = (
        read_event_ids(args.event_ids_file)
        if args.event_ids_file is not None
        else None
    )
    if args.target_dir is not None:
        if requested_ids is None:
            raise ValueError("--target-dir requires --event-ids-file.")
        arrays = []
        for event_id in requested_ids:
            path = (
                args.target_dir
                / f"{event_id}_projected_internal_face_delta_m3.npy"
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            arrays.append((event_id, np.load(path, mmap_mode="r")))
        return arrays

    data = np.load(args.edge_flow_npz)
    keys = sorted(
        (key for key in data.files if key.endswith("_internal_face_delta")),
        key=natural_event_key,
    )
    if not keys:
        raise KeyError(
            f"No '*_internal_face_delta' arrays found in {args.edge_flow_npz}."
        )
    arrays_by_id = {
        key[: -len("_internal_face_delta")]: data[key] for key in keys
    }
    event_ids = requested_ids or sorted(arrays_by_id, key=natural_event_id)
    missing = [event_id for event_id in event_ids if event_id not in arrays_by_id]
    if missing:
        raise KeyError(f"Missing requested edge-flow events: {missing}.")
    return [(event_id, arrays_by_id[event_id]) for event_id in event_ids]


def main() -> None:
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--edge-flow-npz", type=Path)
    source.add_argument("--target-dir", type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--max-events", type=int)
    parser.add_argument("--event-ids-file", type=Path)
    parser.add_argument("--hgn-data-dir", type=Path)
    parser.add_argument("--n-time-steps", type=int, default=2)
    parser.add_argument("--transition-start-index", type=int, default=0)
    parser.add_argument("--max-transitions", type=int)
    args = parser.parse_args()

    event_arrays = load_target_arrays(args)
    if args.max_events is not None:
        event_arrays = event_arrays[: args.max_events]
    event_ids = [event_id for event_id, _ in event_arrays]

    count = 0
    sum_values = None
    sum_squared = None
    used_transition_ranges = []
    for event_id, raw_values in event_arrays:
        values = np.asarray(raw_values)
        if values.ndim != 2:
            raise ValueError(
                f"{event_id} must be [transitions, faces], got {values.shape}."
            )
        if args.hgn_data_dir is not None:
            selected_slice = usable_transition_slice(
                args.hgn_data_dir,
                event_id,
                values.shape[0],
                args.n_time_steps,
            )
        else:
            selected_slice = slice(args.transition_start_index, None)
        values = np.asarray(values[selected_slice], dtype=np.float64)
        if args.max_transitions is not None:
            values = values[: args.max_transitions]
        selected_start = int(selected_slice.start or 0)
        used_transition_ranges.append(
            (event_id, selected_start, selected_start + values.shape[0])
        )
        if sum_values is None:
            sum_values = np.zeros(values.shape[1], dtype=np.float64)
            sum_squared = np.zeros(values.shape[1], dtype=np.float64)
        elif values.shape[1] != sum_values.shape[0]:
            raise ValueError(
                f"{event_id} has {values.shape[1]} faces, expected "
                f"{sum_values.shape[0]}."
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
        num_events=np.array([len(event_arrays)], dtype=np.int64),
        num_transitions=np.array([count], dtype=np.int64),
        transition_start_index=np.array(
            [args.transition_start_index], dtype=np.int64
        ),
        event_ids=np.asarray(event_ids),
        target_units=np.asarray(["m^3" if args.target_dir is not None else "legacy"]),
        source_type=np.asarray(
            ["conservative_sharded" if args.target_dir is not None else "legacy_npz"]
        ),
        source_path=np.asarray(
            [str((args.target_dir or args.edge_flow_npz).resolve())]
        ),
        event_ids_file=np.asarray(
            [str(args.event_ids_file.resolve()) if args.event_ids_file else ""]
        ),
        event_ids_sha256=np.asarray(
            [
                hashlib.sha256(
                    ("\n".join(event_ids) + "\n").encode("utf-8")
                ).hexdigest()
            ]
        ),
        used_transition_event_ids=np.asarray(
            [event_id for event_id, _, _ in used_transition_ranges]
        ),
        used_transition_start=np.asarray(
            [start for _, start, _ in used_transition_ranges], dtype=np.int64
        ),
        used_transition_stop=np.asarray(
            [stop for _, _, stop in used_transition_ranges], dtype=np.int64
        ),
    )
    print(f"Wrote {args.output_npz}")
    print(
        f"events={len(event_arrays)} transitions={count} "
        f"faces={face_mean.shape[0]}"
    )
    print(f"mean(face_rms)={float(np.mean(face_rms)):.6f}")
    print(f"median(face_rms)={float(np.median(face_rms)):.6f}")
    print(f"max(face_rms)={float(np.max(face_rms)):.6f}")


if __name__ == "__main__":
    main()
