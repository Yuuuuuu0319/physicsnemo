# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Summarize HEC-RAS internal Face Flow delta targets.

This is a diagnostic for the model-side edge-flux branch. It reads the
precomputed ``*_internal_face_delta`` arrays and reports whether the target
magnitude is dominated by a few events, timesteps, zones, or signs.
"""

import argparse
import csv
import re
from pathlib import Path

import numpy as np


def rms(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return float("nan")
    return float(np.sqrt(np.mean(values * values)))


def summarize(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_rms": float("nan"),
            f"{prefix}_mae": float("nan"),
            f"{prefix}_abs_p50": float("nan"),
            f"{prefix}_abs_p90": float("nan"),
            f"{prefix}_abs_p99": float("nan"),
            f"{prefix}_abs_max": float("nan"),
            f"{prefix}_positive_fraction": float("nan"),
            f"{prefix}_negative_fraction": float("nan"),
            f"{prefix}_near_zero_fraction": float("nan"),
        }
    abs_values = np.abs(values)
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_rms": rms(values),
        f"{prefix}_mae": float(np.mean(abs_values)),
        f"{prefix}_abs_p50": float(np.quantile(abs_values, 0.50)),
        f"{prefix}_abs_p90": float(np.quantile(abs_values, 0.90)),
        f"{prefix}_abs_p99": float(np.quantile(abs_values, 0.99)),
        f"{prefix}_abs_max": float(np.max(abs_values)),
        f"{prefix}_positive_fraction": float(np.mean(values > 0.0)),
        f"{prefix}_negative_fraction": float(np.mean(values < 0.0)),
        f"{prefix}_near_zero_fraction": float(np.mean(abs_values < 1e-6)),
    }


def summarize_fast(values: np.ndarray, prefix: str) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {
            f"{prefix}_count": 0,
            f"{prefix}_rms": float("nan"),
            f"{prefix}_mae": float("nan"),
            f"{prefix}_abs_max": float("nan"),
            f"{prefix}_positive_fraction": float("nan"),
            f"{prefix}_negative_fraction": float("nan"),
            f"{prefix}_near_zero_fraction": float("nan"),
        }
    abs_values = np.abs(values)
    return {
        f"{prefix}_count": int(values.size),
        f"{prefix}_rms": rms(values),
        f"{prefix}_mae": float(np.mean(abs_values)),
        f"{prefix}_abs_max": float(np.max(abs_values)),
        f"{prefix}_positive_fraction": float(np.mean(values > 0.0)),
        f"{prefix}_negative_fraction": float(np.mean(values < 0.0)),
        f"{prefix}_near_zero_fraction": float(np.mean(abs_values < 1e-6)),
    }


def load_zone_masks(face_graph_file: Path, zone_label_file: Path | None) -> dict[str, np.ndarray]:
    if zone_label_file is None:
        return {}
    face_graph = np.load(face_graph_file)
    face_index = np.asarray(face_graph["internal_face_index"], dtype=np.int64)
    zone_label = np.loadtxt(zone_label_file, dtype=np.int64).reshape(-1)
    src, dst = face_index
    if int(np.max(face_index)) >= zone_label.shape[0]:
        raise ValueError(
            f"Face graph references node {int(np.max(face_index))}, but "
            f"zone_label has only {zone_label.shape[0]} rows."
        )
    return {
        "zone3_touch": (zone_label[src] == 3) | (zone_label[dst] == 3),
        "zone3_internal": (zone_label[src] == 3) & (zone_label[dst] == 3),
    }


def write_csv(path: Path, rows: list[dict[str, float | str | int]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    for key in ("hydrograph_id", "transition_index"):
        if key in fieldnames:
            fieldnames.remove(key)
    ordered = [key for key in ("hydrograph_id", "transition_index") if key in rows[0]]
    fieldnames = ordered + fieldnames
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def natural_event_key(key: str) -> tuple[str, int]:
    hydrograph_id = key[: -len("_internal_face_delta")]
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", hydrograph_id)
    if match:
        return match.group(1), int(match.group(2))
    return hydrograph_id, -1


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-flow-npz", required=True, type=Path)
    parser.add_argument("--face-graph-file", required=True, type=Path)
    parser.add_argument("--zone-label-file", type=Path)
    parser.add_argument("--event-csv", required=True, type=Path)
    parser.add_argument("--transition-csv", required=True, type=Path)
    parser.add_argument("--summary-md", required=True, type=Path)
    parser.add_argument("--max-events", type=int, default=None)
    parser.add_argument("--max-transitions", type=int, default=None)
    parser.add_argument(
        "--event-quantiles",
        action="store_true",
        help="Also compute p50/p90/p99 for every event and the global summary.",
    )
    parser.add_argument(
        "--transition-quantiles",
        action="store_true",
        help="Also compute p50/p90/p99 for every transition. This is slower.",
    )
    args = parser.parse_args()

    data = np.load(args.edge_flow_npz)
    face_keys = sorted(
        (key for key in data.files if key.endswith("_internal_face_delta")),
        key=natural_event_key,
    )
    if args.max_events is not None:
        face_keys = face_keys[: args.max_events]
    if not face_keys:
        raise KeyError(
            f"No '*_internal_face_delta' arrays found in {args.edge_flow_npz}. "
            f"Available arrays: {', '.join(data.files)}"
        )

    zone_masks = load_zone_masks(args.face_graph_file, args.zone_label_file)
    event_rows = []
    transition_rows = []
    all_values = []
    selected_values_by_key = {}
    top_transition = None
    top_transition_rms = -np.inf

    for key in face_keys:
        hydrograph_id = key[: -len("_internal_face_delta")]
        values = np.asarray(data[key], dtype=np.float64)
        if values.ndim != 2:
            raise ValueError(f"{key} must be [transitions, internal_faces], got {values.shape}")
        if args.max_transitions is not None:
            values = values[: args.max_transitions]
        selected_values_by_key[key] = values
        all_values.append(values.reshape(-1))
        event_row = {
            "hydrograph_id": hydrograph_id,
            **(
                summarize(values, "all_face")
                if args.event_quantiles
                else summarize_fast(values, "all_face")
            ),
        }
        for name, mask in zone_masks.items():
            event_row.update(
                summarize(values[:, mask], name)
                if args.event_quantiles
                else summarize_fast(values[:, mask], name)
            )
        event_rows.append(event_row)
        for transition_index in range(values.shape[0]):
            transition = values[transition_index]
            transition_summary = (
                summarize(transition, "all_face")
                if args.transition_quantiles
                else summarize_fast(transition, "all_face")
            )
            row = {
                "hydrograph_id": hydrograph_id,
                "transition_index": transition_index,
                **transition_summary,
            }
            for name, mask in zone_masks.items():
                row.update(
                    summarize(transition[mask], name)
                    if args.transition_quantiles
                    else summarize_fast(transition[mask], name)
                )
            transition_rows.append(row)
            transition_rms = rms(transition)
            if transition_rms > top_transition_rms:
                top_transition_rms = transition_rms
                top_transition = (hydrograph_id, transition_index, transition_rms)

    write_csv(args.event_csv, event_rows)
    write_csv(args.transition_csv, transition_rows)

    global_values = np.concatenate(all_values)
    global_summary = (
        summarize(global_values, "all_face")
        if args.event_quantiles
        else summarize_fast(global_values, "all_face")
    )
    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    with args.summary_md.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write("# HEC-RAS Internal Face Target Distribution\n\n")
        handle.write(f"- edge flow npz: `{args.edge_flow_npz}`\n")
        handle.write(f"- face graph: `{args.face_graph_file}`\n")
        if args.zone_label_file is not None:
            handle.write(f"- zone labels: `{args.zone_label_file}`\n")
        handle.write(f"- events: `{len(face_keys)}`\n")
        handle.write(f"- transitions: `{len(transition_rows)}`\n")
        handle.write(f"- global face target RMS: `{global_summary['all_face_rms']:.6f}` ft^3\n")
        if args.event_quantiles:
            handle.write(
                "- global face target |p50|: "
                f"`{global_summary['all_face_abs_p50']:.6f}` ft^3\n"
            )
            handle.write(
                "- global face target |p90|: "
                f"`{global_summary['all_face_abs_p90']:.6f}` ft^3\n"
            )
            handle.write(
                "- global face target |p99|: "
                f"`{global_summary['all_face_abs_p99']:.6f}` ft^3\n"
            )
        handle.write(f"- global face target |max|: `{global_summary['all_face_abs_max']:.6f}` ft^3\n")
        if top_transition is not None:
            handle.write(
                "- largest transition RMS: "
                f"`{top_transition[0]}` transition `{top_transition[1]}` "
                f"RMS `{top_transition[2]:.6f}` ft^3\n"
            )
        if zone_masks:
            zone3_touch = np.concatenate(
                [
                    selected_values_by_key[key][:, zone_masks["zone3_touch"]].reshape(-1)
                    for key in face_keys
                ]
            )
            zone3_internal = np.concatenate(
                [
                    selected_values_by_key[key][
                        :, zone_masks["zone3_internal"]
                    ].reshape(-1)
                    for key in face_keys
                ]
            )
            handle.write(f"- Zone 3 touching face RMS: `{rms(zone3_touch):.6f}` ft^3\n")
            handle.write(f"- Zone 3 internal face RMS: `{rms(zone3_internal):.6f}` ft^3\n")
        handle.write("\n## Outputs\n\n")
        handle.write(f"- event CSV: `{args.event_csv}`\n")
        handle.write(f"- transition CSV: `{args.transition_csv}`\n")
    print(f"Wrote {args.event_csv}")
    print(f"Wrote {args.transition_csv}")
    print(f"Wrote {args.summary_md}")


if __name__ == "__main__":
    main()
