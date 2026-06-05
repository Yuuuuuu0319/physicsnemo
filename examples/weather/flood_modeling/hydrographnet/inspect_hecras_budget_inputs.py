# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Inspect HEC-RAS plan/HDF inputs before physical-budget validation.

This lightweight preflight is intentionally separate from training. It answers
whether a candidate HEC-RAS plan/result file has the basic ingredients required
before running the heavier uncalibrated budget validator:

* HDF5 output is enabled in the plan.
* Stored computation-level output is frequent enough for target intervals.
* Native ``Face Flow`` and ``Cell Flow Balance`` datasets are present in HDF.

The plan text cannot prove optional HDF variables were written; the resulting
HDF is the authority for native output availability.
"""

from __future__ import annotations

import argparse
import csv
import re
from pathlib import Path

import h5py
import numpy as np


RESULT_BASE = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/"
    "2D Flow Areas/per2/"
)
RESULT_TIME_PATH = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time"
)


TIME_FACTORS = {
    "SEC": 1.0,
    "SECOND": 1.0,
    "SECONDS": 1.0,
    "MIN": 60.0,
    "MINUTE": 60.0,
    "MINUTES": 60.0,
    "HOUR": 3600.0,
    "HOURS": 3600.0,
    "DAY": 86400.0,
    "DAYS": 86400.0,
}


def normalize_line_endings(text: str) -> list[str]:
    return text.replace("\r\n", "\n").replace("\r", "\n").splitlines()


def parse_interval_to_seconds(raw: str | None) -> float | None:
    if raw is None:
        return None
    compact = raw.strip().upper().replace(" ", "")
    match = re.fullmatch(r"([0-9]*\.?[0-9]+)([A-Z]+)", compact)
    if match is None:
        return None
    value = float(match.group(1))
    unit = match.group(2)
    return value * TIME_FACTORS[unit] if unit in TIME_FACTORS else None


def parse_named_interval(lines: list[str], name: str) -> tuple[str | None, float | None]:
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*=\s*(\S+)", flags=re.IGNORECASE)
    for line in lines:
        match = pattern.search(line)
        if match is not None:
            raw = match.group(1)
            return raw, parse_interval_to_seconds(raw)
    return None, None


def parse_named_value(lines: list[str], name: str) -> str | None:
    pattern = re.compile(rf"^\s*{re.escape(name)}\s*=\s*(.+?)\s*$", flags=re.IGNORECASE)
    for line in lines:
        match = pattern.search(line)
        if match is not None:
            return match.group(1).strip()
    return None


def parse_computation_level_output_interval(
    lines: list[str],
) -> tuple[str | None, float | None]:
    for index, line in enumerate(lines):
        if line.strip().lower() == "computation level output":
            if index + 1 >= len(lines):
                return None, None
            parts = lines[index + 1].split()
            raw = parts[-1] if parts else None
            return raw, parse_interval_to_seconds(raw)
    return None, None


def inspect_plan(plan_file: Path, required_output_seconds: float) -> dict[str, object]:
    lines = normalize_line_endings(plan_file.read_text(encoding="utf-8", errors="replace"))
    computation_raw, computation_seconds = parse_named_interval(
        lines, "Computation Interval"
    )
    hydrograph_raw, hydrograph_seconds = parse_named_interval(lines, "Hydrograph Interval")
    output_raw, output_seconds = parse_computation_level_output_interval(lines)
    write_hdf5 = parse_named_value(lines, "Write HDF5 File")
    start_time = parse_named_value(lines, "Start Date/Time")
    end_time = parse_named_value(lines, "End Date/Time")
    native_keywords = {
        "face_flow_keyword_in_plan": bool(
            re.search(r"\bFace\s+Flow\b", "\n".join(lines), flags=re.IGNORECASE)
        ),
        "cell_flow_balance_keyword_in_plan": bool(
            re.search(
                r"\bCell\s+Flow\s+Balance\b", "\n".join(lines), flags=re.IGNORECASE
            )
        ),
    }
    output_interval_ok = (
        output_seconds is not None and output_seconds <= required_output_seconds
    )
    return {
        "input_type": "plan",
        "path": str(plan_file),
        "write_hdf5_file": write_hdf5,
        "start_date_time": start_time,
        "end_date_time": end_time,
        "computation_interval_raw": computation_raw,
        "computation_interval_seconds": computation_seconds,
        "hydrograph_interval_raw": hydrograph_raw,
        "hydrograph_interval_seconds": hydrograph_seconds,
        "computation_level_output_raw": output_raw,
        "computation_level_output_seconds": output_seconds,
        "required_output_seconds": required_output_seconds,
        "output_interval_ok": output_interval_ok,
        "face_flow_available": None,
        "cell_flow_balance_available": None,
        "hdf_output_interval_seconds": None,
        "hdf_output_matches_computation": None,
        "status": "PASS" if output_interval_ok and write_hdf5 == "T" else "BLOCKED",
        "note": (
            "Plan text does not prove native Face Flow / Cell Flow Balance output; "
            "inspect the result HDF after simulation."
        ),
        **native_keywords,
    }


def inspect_hdf(hdf_file: Path) -> dict[str, object]:
    with h5py.File(hdf_file, "r") as hdf:
        has_face_velocity = RESULT_BASE + "Face Velocity" in hdf
        has_face_flow = RESULT_BASE + "Face Flow" in hdf
        has_cell_flow_balance = RESULT_BASE + "Cell Flow Balance" in hdf
        has_water_surface = RESULT_BASE + "Water Surface" in hdf
        hdf_times = np.asarray(hdf[RESULT_TIME_PATH], dtype=np.float64)
        output_seconds = (
            float(np.nanmedian(np.diff(hdf_times)) * 86400.0)
            if hdf_times.shape[0] > 1
            else None
        )
        computation_seconds = None
        computation_path = RESULT_BASE + "Computations/Time Step"
        if computation_path in hdf:
            computation_seconds = float(
                np.nanmedian(np.asarray(hdf[computation_path], dtype=np.float64))
            )
        matches_computation = (
            output_seconds is not None
            and computation_seconds is not None
            and np.isclose(output_seconds, computation_seconds, atol=1e-3)
        )
        face_velocity_shape = (
            "x".join(str(value) for value in hdf[RESULT_BASE + "Face Velocity"].shape)
            if has_face_velocity
            else None
        )
        face_flow_shape = (
            "x".join(str(value) for value in hdf[RESULT_BASE + "Face Flow"].shape)
            if has_face_flow
            else None
        )
    native_ready = has_face_flow and has_cell_flow_balance and matches_computation
    return {
        "input_type": "hdf",
        "path": str(hdf_file),
        "write_hdf5_file": None,
        "start_date_time": None,
        "end_date_time": None,
        "computation_interval_raw": None,
        "computation_interval_seconds": computation_seconds,
        "hydrograph_interval_raw": None,
        "hydrograph_interval_seconds": None,
        "computation_level_output_raw": None,
        "computation_level_output_seconds": None,
        "required_output_seconds": None,
        "output_interval_ok": matches_computation,
        "face_flow_keyword_in_plan": None,
        "cell_flow_balance_keyword_in_plan": None,
        "face_velocity_available": has_face_velocity,
        "face_velocity_shape": face_velocity_shape,
        "face_flow_available": has_face_flow,
        "face_flow_shape": face_flow_shape,
        "cell_flow_balance_available": has_cell_flow_balance,
        "water_surface_available": has_water_surface,
        "hdf_output_interval_seconds": output_seconds,
        "hdf_output_matches_computation": matches_computation,
        "status": "PASS" if native_ready else "BLOCKED",
        "note": (
            "Native budget-ready HDF requires Face Flow, Cell Flow Balance, and "
            "stored output matching the computation interval."
        ),
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "# HEC-RAS Budget Input Preflight",
        "",
        "This report checks whether candidate HEC-RAS plan/result inputs are ready "
        "for formal HydroGraphNet physical-budget validation.",
        "",
        "| Type | Path | Status | Output interval | Native Face Flow | Cell Flow Balance | Note |",
        "|---|---|---|---:|---|---|---|",
    ]
    for row in rows:
        interval = row.get("hdf_output_interval_seconds")
        if interval is None:
            interval = row.get("computation_level_output_seconds")
        interval_text = "" if interval is None else f"{float(interval):.3f} s"
        lines.append(
            "| "
            f"{row.get('input_type')} | `{row.get('path')}` | {row.get('status')} | "
            f"{interval_text} | {row.get('face_flow_available')} | "
            f"{row.get('cell_flow_balance_available')} | {row.get('note')} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "- `BLOCKED` means the file is not sufficient for formal local-conservation "
            "training yet.",
            "- A plan file can show the intended output interval, but native optional "
            "variables must be verified in the completed HDF.",
            "- The next heavy check remains `evaluate_hecras_physical_budget.py` after "
            "a candidate HDF passes this preflight.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan-file", action="append", default=[])
    parser.add_argument("--hdf-file", action="append", default=[])
    parser.add_argument("--required-output-seconds", type=float, default=30.0)
    parser.add_argument("--output-csv", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    for plan_file in args.plan_file:
        rows.append(inspect_plan(Path(plan_file), args.required_output_seconds))
    for hdf_file in args.hdf_file:
        rows.append(inspect_hdf(Path(hdf_file)))
    if not rows:
        raise ValueError("Provide at least one --plan-file or --hdf-file.")
    write_csv(Path(args.output_csv), rows)
    write_markdown(Path(args.output_md), rows)
    for row in rows:
        print(f"{row['status']}: {row['input_type']} {row['path']}")


if __name__ == "__main__":
    main()
