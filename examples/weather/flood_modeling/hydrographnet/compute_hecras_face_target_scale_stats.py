# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Compute scale statistics for HEC-RAS internal Face Flow targets.

This script is a diagnostic bridge for edge-informed local conservation.  The
raw HEC-RAS Face Flow interval deltas have a very wide magnitude range, so an
edge head that directly predicts ft^3 often collapses near zero.  These scale
statistics let later experiments separate three effects:

* persistent per-face magnitude differences,
* event-level hydrograph magnitude differences,
* transition-level flood-stage magnitude differences.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def natural_event_key(key: str) -> tuple[str, int]:
    hydrograph_id = key[: -len("_internal_face_delta")]
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", hydrograph_id)
    if match:
        return match.group(1), int(match.group(2))
    return hydrograph_id, -1


def rms(values: np.ndarray, axis=None) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.sqrt(np.mean(values * values, axis=axis))


def summarize_abs(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    abs_values = np.abs(values)
    return {
        f"{prefix}_rms": float(rms(values)),
        f"{prefix}_abs_p50": float(np.quantile(abs_values, 0.50)),
        f"{prefix}_abs_p90": float(np.quantile(abs_values, 0.90)),
        f"{prefix}_abs_p99": float(np.quantile(abs_values, 0.99)),
        f"{prefix}_abs_max": float(np.max(abs_values)),
    }


def write_csv(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    for key in ("hydrograph_id", "transition_index"):
        if key in fieldnames:
            fieldnames.remove(key)
    ordered = [key for key in ("hydrograph_id", "transition_index") if key in rows[0]]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ordered + fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-flow-npz", required=True, type=Path)
    parser.add_argument("--output-npz", required=True, type=Path)
    parser.add_argument("--event-csv", required=True, type=Path)
    parser.add_argument("--transition-csv", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
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
        raise KeyError(f"No '*_internal_face_delta' arrays found in {args.edge_flow_npz}.")

    event_ids = []
    event_rms = []
    transition_rms_by_event = {}
    event_rows = []
    transition_rows = []
    face_sum_squared = None
    face_sum = None
    total_count = 0
    global_values = []

    for key in keys:
        hydrograph_id = key[: -len("_internal_face_delta")]
        values = np.asarray(data[key], dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"{key} must be [transitions, faces], got {values.shape}.")
        if args.max_transitions is not None:
            values = values[: args.max_transitions]
        if face_sum_squared is None:
            face_sum_squared = np.zeros(values.shape[1], dtype=np.float64)
            face_sum = np.zeros(values.shape[1], dtype=np.float64)
        elif values.shape[1] != face_sum_squared.shape[0]:
            raise ValueError(
                f"{key} has {values.shape[1]} faces, expected {face_sum_squared.shape[0]}."
            )

        transition_rms = rms(values, axis=1)
        event_ids.append(hydrograph_id)
        event_rms_value = float(rms(values))
        event_rms.append(event_rms_value)
        transition_rms_by_event[hydrograph_id] = transition_rms.astype(np.float32)
        face_sum += np.sum(values, axis=0)
        face_sum_squared += np.sum(values * values, axis=0)
        total_count += values.shape[0]
        global_values.append(values.reshape(-1))

        event_rows.append(
            {
                "hydrograph_id": hydrograph_id,
                "num_transitions": values.shape[0],
                "num_faces": values.shape[1],
                **summarize_abs(values, "face_delta"),
                "transition_rms_min": float(np.min(transition_rms)),
                "transition_rms_p50": float(np.quantile(transition_rms, 0.50)),
                "transition_rms_p90": float(np.quantile(transition_rms, 0.90)),
                "transition_rms_max": float(np.max(transition_rms)),
            }
        )
        for transition_index, transition_value in enumerate(transition_rms):
            transition_rows.append(
                {
                    "hydrograph_id": hydrograph_id,
                    "transition_index": transition_index,
                    "transition_rms": float(transition_value),
                    **summarize_abs(values[transition_index], "face_delta"),
                }
            )

    if total_count == 0:
        raise ValueError("No selected transitions.")
    global_values = np.concatenate(global_values)
    global_rms = float(rms(global_values))
    face_mean = face_sum / total_count
    face_rms = np.sqrt(face_sum_squared / total_count)
    face_std = np.sqrt(np.maximum(face_sum_squared / total_count - face_mean * face_mean, 0.0))

    npz_payload = {
        "event_ids": np.asarray(event_ids),
        "event_rms": np.asarray(event_rms, dtype=np.float32),
        "global_rms": np.asarray([global_rms], dtype=np.float32),
        "face_mean": face_mean.astype(np.float32),
        "face_std": face_std.astype(np.float32),
        "face_rms": face_rms.astype(np.float32),
        "num_events": np.asarray([len(event_ids)], dtype=np.int64),
        "num_transitions": np.asarray([total_count], dtype=np.int64),
    }
    for hydrograph_id, transition_rms in transition_rms_by_event.items():
        npz_payload[f"{hydrograph_id}_transition_rms"] = transition_rms

    args.output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_npz, **npz_payload)
    write_csv(args.event_csv, event_rows)
    write_csv(args.transition_csv, transition_rows)

    top_event = max(event_rows, key=lambda row: float(row["face_delta_rms"]))
    top_transition = max(transition_rows, key=lambda row: float(row["transition_rms"]))
    with args.summary_md.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# HEC-RAS Face Target Scale Statistics\n\n")
        handle.write(f"- edge flow npz: `{args.edge_flow_npz}`\n")
        handle.write(f"- events: `{len(event_ids)}`\n")
        handle.write(f"- transitions: `{total_count}`\n")
        handle.write(f"- internal faces: `{face_rms.shape[0]}`\n")
        handle.write(f"- global RMS: `{global_rms:.6f}` ft^3\n")
        handle.write(f"- median per-face RMS: `{float(np.median(face_rms)):.6f}` ft^3\n")
        handle.write(f"- mean per-face RMS: `{float(np.mean(face_rms)):.6f}` ft^3\n")
        handle.write(f"- max per-face RMS: `{float(np.max(face_rms)):.6f}` ft^3\n")
        handle.write(
            "- largest event RMS: "
            f"`{top_event['hydrograph_id']}` `{float(top_event['face_delta_rms']):.6f}` ft^3\n"
        )
        handle.write(
            "- largest transition RMS: "
            f"`{top_transition['hydrograph_id']}` transition "
            f"`{top_transition['transition_index']}` "
            f"`{float(top_transition['transition_rms']):.6f}` ft^3\n"
        )
        handle.write("\n## Outputs\n\n")
        handle.write(f"- scale NPZ: `{args.output_npz}`\n")
        handle.write(f"- event CSV: `{args.event_csv}`\n")
        handle.write(f"- transition CSV: `{args.transition_csv}`\n")

    print(f"Wrote {args.output_npz}")
    print(f"Wrote {args.event_csv}")
    print(f"Wrote {args.transition_csv}")
    print(f"Wrote {args.summary_md}")


if __name__ == "__main__":
    main()
