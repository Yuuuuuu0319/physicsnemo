# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Audit training-only provenance for HEC-RAS face-target statistics."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

import numpy as np


TARGET_SUFFIX = "_internal_face_delta"


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest without loading the file into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def natural_event_key(hydrograph_id: str) -> tuple[str, int]:
    match = re.fullmatch(r"([A-Za-z]+)(\d+)", hydrograph_id)
    if match:
        return match.group(1), int(match.group(2))
    return hydrograph_id, -1


def load_split_ids(path: Path) -> list[str]:
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids:
        raise ValueError(f"No hydrograph IDs found in {path}.")
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate hydrograph IDs found in {path}.")
    return ids


def read_npy_shape(stream) -> tuple[int, ...]:
    version = np.lib.format.read_magic(stream)
    if version == (1, 0):
        shape, _, _ = np.lib.format.read_array_header_1_0(stream)
    elif version in {(2, 0), (3, 0)}:
        shape, _, _ = np.lib.format.read_array_header_2_0(stream)
    else:
        raise ValueError(f"Unsupported NPY version {version}.")
    return tuple(int(value) for value in shape)


def edge_target_shapes(path: Path) -> dict[str, tuple[int, ...]]:
    """Read target array shapes directly from NPZ member headers."""
    shapes = {}
    with zipfile.ZipFile(path) as archive:
        members = {
            name[: -len(".npy")]: name
            for name in archive.namelist()
            if name.endswith(f"{TARGET_SUFFIX}.npy")
        }
        for key, member in members.items():
            hydrograph_id = key[: -len(TARGET_SUFFIX)]
            with archive.open(member) as stream:
                shapes[hydrograph_id] = read_npy_shape(stream)
    return shapes


def scalar_int(data: np.lib.npyio.NpzFile, key: str) -> int:
    if key not in data.files:
        raise KeyError(f"Missing {key!r} in target-statistics NPZ.")
    return int(np.asarray(data[key]).reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--edge-flow-npz", required=True, type=Path)
    parser.add_argument("--face-stats-npz", required=True, type=Path)
    parser.add_argument("--event-ids-file", required=True, type=Path)
    parser.add_argument("--expected-transition-start-index", required=True, type=int)
    parser.add_argument("--expected-max-transitions", type=int)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--hash-edge-flow-npz", action="store_true")
    args = parser.parse_args()

    split_ids = load_split_ids(args.event_ids_file)
    shapes = edge_target_shapes(args.edge_flow_npz)
    edge_ids = sorted(shapes, key=natural_event_key)
    invalid_shapes = {
        hydrograph_id: list(shape)
        for hydrograph_id, shape in shapes.items()
        if len(shape) != 2
    }
    transition_counts = {
        hydrograph_id: min(
            shape[0] - args.expected_transition_start_index,
            args.expected_max_transitions
            if args.expected_max_transitions is not None
            else shape[0],
        )
        for hydrograph_id, shape in shapes.items()
        if len(shape) == 2
    }
    face_counts = {
        shape[1] for shape in shapes.values() if len(shape) == 2
    }

    with np.load(args.face_stats_npz, allow_pickle=False) as stats:
        stats_num_events = scalar_int(stats, "num_events")
        stats_num_transitions = scalar_int(stats, "num_transitions")
        stats_transition_start = scalar_int(stats, "transition_start_index")
        stats_num_faces = int(np.asarray(stats["face_rms"]).reshape(-1).shape[0])

    expected_total_transitions = sum(transition_counts.values())
    checks = {
        "event_id_order_exact_match": edge_ids == split_ids,
        "event_id_set_exact_match": set(edge_ids) == set(split_ids),
        "all_target_arrays_are_2d": not invalid_shapes,
        "all_transition_counts_positive": bool(transition_counts)
        and min(transition_counts.values()) > 0,
        "single_face_count": len(face_counts) == 1,
        "stats_event_count_matches": stats_num_events == len(split_ids),
        "stats_transition_count_matches": (
            stats_num_transitions == expected_total_transitions
        ),
        "stats_transition_start_matches": (
            stats_transition_start == args.expected_transition_start_index
        ),
        "stats_face_count_matches": (
            len(face_counts) == 1 and stats_num_faces == next(iter(face_counts))
        ),
    }
    passed = all(checks.values())
    event_ids_payload = "\n".join(split_ids) + "\n"
    manifest = {
        "status": "passed" if passed else "failed",
        "purpose": "prove_face_normalization_statistics_use_training_events_only",
        "edge_flow_npz": str(args.edge_flow_npz.resolve()),
        "edge_flow_npz_size_bytes": args.edge_flow_npz.stat().st_size,
        "edge_flow_npz_sha256": (
            sha256_file(args.edge_flow_npz) if args.hash_edge_flow_npz else None
        ),
        "face_stats_npz": str(args.face_stats_npz.resolve()),
        "face_stats_npz_sha256": sha256_file(args.face_stats_npz),
        "event_ids_file": str(args.event_ids_file.resolve()),
        "event_ids_file_sha256": sha256_file(args.event_ids_file),
        "event_ids_payload_sha256": hashlib.sha256(
            event_ids_payload.encode("utf-8")
        ).hexdigest(),
        "first_event": edge_ids[0] if edge_ids else None,
        "last_event": edge_ids[-1] if edge_ids else None,
        "num_events": len(edge_ids),
        "event_ids": edge_ids,
        "transition_start_index": stats_transition_start,
        "max_transitions_per_event": args.expected_max_transitions,
        "num_transitions": stats_num_transitions,
        "transitions_per_event_min": (
            min(transition_counts.values()) if transition_counts else None
        ),
        "transitions_per_event_max": (
            max(transition_counts.values()) if transition_counts else None
        ),
        "num_faces": stats_num_faces,
        "checks": checks,
        "invalid_target_shapes": invalid_shapes,
        "unexpected_event_ids": sorted(set(edge_ids) - set(split_ids)),
        "missing_event_ids": sorted(set(split_ids) - set(edge_ids)),
        "inference_policy": "statistics_are_frozen_after_H1_H350_and_never_recomputed_from_validation_or_test",
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"status={manifest['status']}")
    print(
        f"events={len(edge_ids)} transitions={stats_num_transitions} "
        f"faces={stats_num_faces}"
    )
    print(f"Wrote {args.output_json}")
    if not passed:
        failed = [name for name, value in checks.items() if not value]
        raise SystemExit(f"Face-target provenance audit failed: {failed}")


if __name__ == "__main__":
    main()
