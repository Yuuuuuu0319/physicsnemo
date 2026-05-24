# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Build HEC-RAS batch-run events.json from HydroGraphNet forcing files.

The Linux HEC-RAS batch runner in this Minxiong workflow consumes an
``events.json`` file with ``inflow_b01`` and ``precipitation/RainGage1`` time
series. This utility writes that file directly from HGN ``M80_US_InF_H*.txt``
and ``M80_Pr_H*.txt`` files so event-specific HDF face velocity can be matched
back to the selected HGN hydrographs.
"""

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def load_hydrograph_ids(data_dirs: list[Path], ids: list[str], ids_file: str | None) -> list[str]:
    hydrograph_ids = list(ids)
    if ids_file is not None:
        for data_dir in data_dirs:
            path = data_dir / ids_file
            if path.exists():
                hydrograph_ids.extend(
                    line.strip() for line in path.read_text().splitlines() if line.strip()
                )
    if not hydrograph_ids:
        for data_dir in data_dirs:
            for path in sorted(data_dir.glob("M80_US_InF_H*.txt")):
                hydrograph_ids.append(path.stem.split("_")[-1])
    return sorted(set(hydrograph_ids), key=lambda hid: int(hid[1:]) if hid[1:].isdigit() else hid)


def find_event_file(data_dirs: list[Path], prefix: str, kind: str, hydrograph_id: str) -> Path:
    filename = f"{prefix}_{kind}_{hydrograph_id}.txt"
    for data_dir in data_dirs:
        path = data_dir / filename
        if path.exists():
            return path
    raise FileNotFoundError(f"Could not find {filename} in {[str(d) for d in data_dirs]}")


def load_inflow(path: Path) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(np.loadtxt(path, delimiter="\t"), dtype=np.float64)
    if data.ndim != 2 or data.shape[1] < 2:
        raise ValueError(f"Expected two-column inflow file: {path}")
    return data[:, 0], data[:, 1]


def load_precip(path: Path, fallback_time: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(np.loadtxt(path, delimiter="\t"), dtype=np.float64)
    if data.ndim == 2 and data.shape[1] >= 2:
        return data[:, 0], data[:, 1]
    values = data.reshape(-1)
    if values.shape[0] == fallback_time.shape[0]:
        return fallback_time, values
    return np.linspace(float(fallback_time[0]), float(fallback_time[-1]), values.shape[0]), values


def time_value_dict(times: np.ndarray, values: np.ndarray) -> dict[str, str]:
    return {
        f"{float(t):.9g}": f"{float(v):.9g}"
        for t, v in zip(times.reshape(-1), values.reshape(-1))
    }


def load_template(path: Path | None) -> dict[str, Any]:
    if path is None or not path.exists():
        return {}
    return json.loads(path.read_text())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", action="append", required=True, type=Path)
    parser.add_argument("--ids", nargs="*", default=[])
    parser.add_argument("--ids-file")
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--template-events-json", type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    parser.add_argument("--downstream-b01", default="0.0010")
    args = parser.parse_args()

    template = load_template(args.template_events_json)
    events = {}
    hydrograph_ids = load_hydrograph_ids(args.data_dir, args.ids, args.ids_file)
    for hydrograph_id in hydrograph_ids:
        inflow_path = find_event_file(args.data_dir, args.prefix, "US_InF", hydrograph_id)
        precip_path = find_event_file(args.data_dir, args.prefix, "Pr", hydrograph_id)
        inflow_time, inflow = load_inflow(inflow_path)
        precip_time, precip = load_precip(precip_path, inflow_time)
        events[hydrograph_id] = {
            "inflow_b01": time_value_dict(inflow_time, inflow),
            "downstream_b01": args.downstream_b01,
            "precipitation": {
                "RainGage1": time_value_dict(precip_time, precip),
            },
        }

    output = {
        "boundary_paths": template.get("boundary_paths", {}),
        "originals": template.get("originals", {}),
        "events": events,
    }
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(output, indent=2) + "\n")
    print(f"Wrote {len(events)} events to {args.output_json}")


if __name__ == "__main__":
    main()
