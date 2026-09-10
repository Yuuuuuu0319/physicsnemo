# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Audit no-leakage provenance for formal conservative Face Flow statistics."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from compute_hecras_face_target_stats import (
    load_time_contract,
    read_event_ids,
    usable_transition_slice,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def scalar_text(data: np.lib.npyio.NpzFile, key: str) -> str:
    if key not in data.files:
        raise KeyError(f"Missing {key!r} in face-statistics NPZ.")
    return str(np.asarray(data[key]).reshape(-1)[0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-dir", required=True, type=Path)
    parser.add_argument("--face-stats-npz", required=True, type=Path)
    parser.add_argument("--event-ids-file", required=True, type=Path)
    parser.add_argument("--hgn-data-dir", required=True, type=Path)
    parser.add_argument("--n-time-steps", type=int, default=2)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()

    train_ids = read_event_ids(args.event_ids_file)
    validation_ids = read_event_ids(args.hgn_data_dir / "validation_ids.txt")
    test_ids = read_event_ids(args.hgn_data_dir / "test_ids.txt")
    expected_ranges = []
    target_shapes = {}
    missing_target_files = []
    for event_id in train_ids:
        path = args.target_dir / f"{event_id}_projected_internal_face_delta_m3.npy"
        if not path.is_file():
            missing_target_files.append(str(path))
            continue
        target = np.load(path, mmap_mode="r")
        target_shapes[event_id] = list(target.shape)
        selected = usable_transition_slice(
            args.hgn_data_dir,
            event_id,
            target.shape[0],
            args.n_time_steps,
        )
        expected_ranges.append((event_id, selected.start, selected.stop))

    with np.load(args.face_stats_npz, allow_pickle=False) as stats:
        stats_event_ids = [str(value) for value in stats["event_ids"].tolist()]
        stats_starts = np.asarray(stats["used_transition_start"], dtype=np.int64)
        stats_stops = np.asarray(stats["used_transition_stop"], dtype=np.int64)
        stats_range_ids = [
            str(value) for value in stats["used_transition_event_ids"].tolist()
        ]
        face_mean = np.asarray(stats["face_mean"])
        face_std = np.asarray(stats["face_std"])
        face_rms = np.asarray(stats["face_rms"])
        stats_num_events = int(np.asarray(stats["num_events"]).reshape(-1)[0])
        stats_num_transitions = int(
            np.asarray(stats["num_transitions"]).reshape(-1)[0]
        )
        embedded_payload_sha256 = scalar_text(stats, "event_ids_sha256")
        source_type = scalar_text(stats, "source_type")
        target_units = scalar_text(stats, "target_units")
        source_path = scalar_text(stats, "source_path")

    expected_ids = [event_id for event_id, _, _ in expected_ranges]
    expected_starts = np.asarray(
        [start for _, start, _ in expected_ranges], dtype=np.int64
    )
    expected_stops = np.asarray(
        [stop for _, _, stop in expected_ranges], dtype=np.int64
    )
    expected_transitions = int(np.sum(expected_stops - expected_starts))
    payload_sha256 = hashlib.sha256(
        ("\n".join(train_ids) + "\n").encode("utf-8")
    ).hexdigest()
    face_counts = {
        shape[1] for shape in target_shapes.values() if len(shape) == 2
    }
    split_disjoint = not (
        set(train_ids) & set(validation_ids)
        or set(train_ids) & set(test_ids)
        or set(validation_ids) & set(test_ids)
    )
    checks = {
        "all_train_target_files_exist": not missing_target_files,
        "all_target_arrays_are_2d": bool(target_shapes)
        and all(len(shape) == 2 for shape in target_shapes.values()),
        "single_face_count": len(face_counts) == 1,
        "train_validation_test_disjoint": split_disjoint,
        "stats_event_ids_exact_train_order": stats_event_ids == train_ids,
        "stats_range_ids_exact_train_order": stats_range_ids == train_ids,
        "stats_event_count_matches": stats_num_events == len(train_ids),
        "stats_transition_ranges_match_hgn_sampling": (
            expected_ids == train_ids
            and np.array_equal(stats_starts, expected_starts)
            and np.array_equal(stats_stops, expected_stops)
        ),
        "stats_transition_count_matches": (
            stats_num_transitions == expected_transitions
        ),
        "stats_face_count_matches": (
            len(face_counts) == 1 and face_rms.size == next(iter(face_counts))
        ),
        "stats_arrays_align": (
            face_mean.shape == face_std.shape == face_rms.shape
        ),
        "stats_arrays_are_finite": bool(
            np.all(np.isfinite(face_mean))
            and np.all(np.isfinite(face_std))
            and np.all(np.isfinite(face_rms))
        ),
        "stats_scales_are_nonnegative": bool(
            np.all(face_std >= 0) and np.all(face_rms >= 0)
        ),
        "embedded_event_hash_matches": embedded_payload_sha256 == payload_sha256,
        "source_type_is_conservative_sharded": (
            source_type == "conservative_sharded"
        ),
        "target_units_are_m3": target_units == "m^3",
        "source_path_matches": (
            Path(source_path).resolve() == args.target_dir.resolve()
        ),
    }
    passed = all(checks.values())
    dynamic_skip_steps, post_peak_steps = load_time_contract(args.hgn_data_dir)
    report = {
        "schema_version": 1,
        "status": "passed" if passed else "failed",
        "purpose": "prove_face_normalization_uses_training_events_only",
        "target_dir": str(args.target_dir.resolve()),
        "face_stats_npz": str(args.face_stats_npz.resolve()),
        "face_stats_sha256": sha256_file(args.face_stats_npz),
        "event_ids_file": str(args.event_ids_file.resolve()),
        "event_ids_file_sha256": sha256_file(args.event_ids_file),
        "event_ids_payload_sha256": payload_sha256,
        "num_train_events": len(train_ids),
        "num_validation_events": len(validation_ids),
        "num_test_events": len(test_ids),
        "num_transitions": stats_num_transitions,
        "num_faces": int(face_rms.size),
        "dynamic_skip_steps": dynamic_skip_steps,
        "post_peak_steps": post_peak_steps,
        "n_time_steps": args.n_time_steps,
        "train_event_ids": train_ids,
        "validation_event_ids": validation_ids,
        "test_event_ids": test_ids,
        "checks": checks,
        "missing_target_files": missing_target_files,
        "inference_policy": (
            "face scales are frozen from the train split and reused unchanged "
            "for validation and sealed test"
        ),
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, indent=2) + "\n")
    print(f"status={report['status']}")
    print(
        f"train_events={len(train_ids)} transitions={stats_num_transitions} "
        f"faces={face_rms.size}"
    )
    print(f"Wrote {args.output_json}")
    if not passed:
        failed = [name for name, value in checks.items() if not value]
        raise SystemExit(f"Face-target statistics audit failed: {failed}")


if __name__ == "__main__":
    main()
