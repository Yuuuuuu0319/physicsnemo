# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Match HEC-RAS HDF event hydrographs against HydroGraphNet hydrographs.

This is an inspection tool for deciding whether an HEC-RAS result HDF can be
used as an event-specific face-velocity source for HydroGraphNet training.
It compares HDF boundary-condition time series with HGN `M80_US_InF_H*.txt`
and optional precipitation files.
"""

import argparse
import csv
from pathlib import Path
import re

import h5py
import numpy as np


BASE_TS = (
    "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series"
)
DEFAULT_SERIES = {
    "event_upstreamBC1": (
        "Event Conditions/Unsteady/Boundary Conditions/Flow Hydrographs/"
        "2D: per2 BCLine: upstreamBC1"
    ),
    "event_upstreamBC2": (
        "Event Conditions/Unsteady/Boundary Conditions/Flow Hydrographs/"
        "2D: per2 BCLine: upstreamBC2"
    ),
    "result_upstreamBC1": (
        f"{BASE_TS}/2D Flow Areas/per2/Boundary Conditions/upstreamBC1"
    ),
    "result_upstreamBC2": (
        f"{BASE_TS}/2D Flow Areas/per2/Boundary Conditions/upstreamBC2"
    ),
    "result_upstreamBC1_flow_per_face_sum": (
        f"{BASE_TS}/2D Flow Areas/per2/Boundary Conditions/"
        "upstreamBC1 - Flow per Face"
    ),
    "result_upstreamBC2_flow_per_face_sum": (
        f"{BASE_TS}/2D Flow Areas/per2/Boundary Conditions/"
        "upstreamBC2 - Flow per Face"
    ),
}
HDF_PRECIP_PATH = (
    "Event Conditions/Unsteady/Boundary Conditions/Precipitation Hydrographs/2D: per2"
)


def hydrograph_sort_key(hid: str) -> tuple[int, str]:
    match = re.fullmatch(r"H(\d+)", hid)
    return (int(match.group(1)) if match else 10**9, hid)


def zscore(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    std = float(np.std(values))
    if std <= 1e-12:
        return values - float(np.mean(values))
    return (values - float(np.mean(values))) / std


def resample_to_length(values: np.ndarray, length: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.shape[0] == length:
        return values
    if values.shape[0] == 1:
        return np.full(length, float(values[0]), dtype=np.float64)
    source_x = np.linspace(0.0, 1.0, values.shape[0])
    target_x = np.linspace(0.0, 1.0, length)
    return np.interp(target_x, source_x, values)


def compare_series(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    length = max(reference.shape[0], candidate.shape[0])
    ref = resample_to_length(reference, length)
    cand = resample_to_length(candidate, length)
    diff = cand - ref
    ref_z = zscore(ref)
    cand_z = zscore(cand)
    if np.std(ref_z) <= 1e-12 or np.std(cand_z) <= 1e-12:
        corr = 1.0 if np.allclose(ref_z, cand_z) else 0.0
    else:
        corr = float(np.corrcoef(ref_z, cand_z)[0, 1])
    nrmse = float(np.sqrt(np.mean((cand_z - ref_z) ** 2)))
    max_abs = float(np.max(np.abs(diff)))
    rmse = float(np.sqrt(np.mean(diff**2)))
    ref_peak_idx = int(np.argmax(ref))
    cand_peak_idx = int(np.argmax(cand))
    ref_peak = float(np.max(ref))
    cand_peak = float(np.max(cand))
    peak_ratio = cand_peak / ref_peak if abs(ref_peak) > 1e-12 else np.nan
    return {
        "corr": corr,
        "nrmse_z": nrmse,
        "rmse": rmse,
        "max_abs_diff": max_abs,
        "ref_peak_index": ref_peak_idx,
        "candidate_peak_index": cand_peak_idx,
        "peak_index_diff": cand_peak_idx - ref_peak_idx,
        "ref_peak": ref_peak,
        "candidate_peak": cand_peak,
        "peak_ratio": float(peak_ratio),
        "exact_or_close": float(np.allclose(ref, cand, rtol=1e-6, atol=1e-5)),
    }


def read_hdf_series(hdf: h5py.File, path: str) -> np.ndarray:
    data = np.asarray(hdf[path][:], dtype=np.float64)
    if data.ndim == 1:
        return data.reshape(-1)
    if "Flow per Face" in path:
        return np.sum(data, axis=1)
    if data.shape[1] >= 2:
        return data[:, 1]
    return data.reshape(data.shape[0], -1).sum(axis=1)


def load_hgn_series(path: Path) -> np.ndarray:
    data = np.loadtxt(path, delimiter="\t")
    data = np.asarray(data, dtype=np.float64)
    if data.ndim == 1:
        return data
    if data.shape[1] >= 2:
        return data[:, 1]
    return data.reshape(-1)


def discover_hydrographs(data_dirs: list[Path], prefix: str) -> dict[str, Path]:
    hydrographs = {}
    for data_dir in data_dirs:
        for path in data_dir.glob(f"{prefix}_US_InF_H*.txt"):
            hid = path.stem.split("_")[-1]
            hydrographs[hid] = path
    return dict(sorted(hydrographs.items(), key=lambda item: hydrograph_sort_key(item[0])))


def summarize_hgn_precipitation(data_dirs: list[Path], prefix: str) -> dict[str, float]:
    maxima = []
    nonzero_files = 0
    file_count = 0
    for data_dir in data_dirs:
        for path in data_dir.glob(f"{prefix}_Pr_H*.txt"):
            values = np.asarray(np.loadtxt(path, delimiter="\t"), dtype=np.float64)
            file_count += 1
            max_abs = float(np.max(np.abs(values))) if values.size else 0.0
            maxima.append(max_abs)
            if max_abs > 1e-12:
                nonzero_files += 1
    return {
        "file_count": file_count,
        "nonzero_files": nonzero_files,
        "max_abs": max(maxima) if maxima else 0.0,
    }


def write_markdown_summary(
    output_md: Path,
    hdf_file: Path,
    rows: list[dict[str, object]],
    top_n: int,
    hdf_precip_max: float | None,
    hgn_precip_summary: dict[str, float],
) -> None:
    best = sorted(
        rows,
        key=lambda row: (
            row["series_name"] != "result_upstreamBC1",
            -float(row["corr"]),
            float(row["nrmse_z"]),
            float(row["max_abs_diff"]),
        ),
    )[:top_n]
    exact_rows = [
        row for row in rows if row["series_name"] == "result_upstreamBC1"
        and float(row["exact_or_close"]) > 0.5
    ]
    lines = [
        "# HEC-RAS / HGN Event Match Report",
        "",
        f"HDF file: `{hdf_file}`",
        "",
        "## Best Matches",
        "",
        "| HGN ID | Split | HDF series | Corr | z-NRMSE | Max abs diff | Peak index diff | Exact/close |",
        "|---|---|---|---:|---:|---:|---:|---:|",
    ]
    for row in best:
        lines.append(
            "| {hydrograph_id} | {split} | {series_name} | {corr:.6f} | "
            "{nrmse_z:.6e} | {max_abs_diff:.6e} | {peak_index_diff} | "
            "{exact_or_close:.0f} |".format(**row)
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            f"- `{len(exact_rows)}` HGN hydrographs exactly/closely match "
            "`result_upstreamBC1` within numerical tolerance.",
            "- If all H1-H50 match the same HDF upstream hydrograph, then this HDF "
            "appears to use a shared inflow template rather than uniquely identifying "
            "one HGN event by upstream flow alone.",
            "- Event matching for face velocity is stronger when inflow, precipitation, "
            "and output state/volume timing all match; this report focuses on boundary "
            "hydrographs and should be treated as a necessary check, not the only check.",
            "",
            "## Precipitation Check",
            "",
            f"- HDF event precipitation max: `{hdf_precip_max}`",
            f"- HGN precipitation files checked: `{hgn_precip_summary['file_count']}`",
            f"- HGN precipitation nonzero files: `{hgn_precip_summary['nonzero_files']}`",
            f"- HGN precipitation max abs: `{hgn_precip_summary['max_abs']}`",
            "",
            "If HDF precipitation is nonzero while all HGN precipitation files are zero, "
            "the upstream boundary can still match exactly, but the HDF face velocity "
            "should not yet be treated as a fully matched event-specific target.",
        ]
    )
    output_md.write_text("\n".join(lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hdf-file", required=True, type=Path)
    parser.add_argument("--data-dir", action="append", required=True, type=Path)
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", type=Path)
    parser.add_argument("--top-n", type=int, default=20)
    args = parser.parse_args()

    hydrographs = discover_hydrographs(args.data_dir, args.prefix)
    if not hydrographs:
        raise FileNotFoundError("No HGN upstream inflow files were found.")

    hdf_series = {}
    hdf_precip_max = None
    with h5py.File(args.hdf_file, "r") as hdf:
        for name, path in DEFAULT_SERIES.items():
            if path in hdf:
                hdf_series[name] = read_hdf_series(hdf, path)
        if HDF_PRECIP_PATH in hdf:
            hdf_precip = read_hdf_series(hdf, HDF_PRECIP_PATH)
            hdf_precip_max = float(np.max(np.abs(hdf_precip)))

    rows = []
    for hid, hgn_path in hydrographs.items():
        hgn = load_hgn_series(hgn_path)
        split = hgn_path.parent.name
        for series_name, series_values in hdf_series.items():
            metrics = compare_series(series_values, hgn)
            rows.append(
                {
                    "hydrograph_id": hid,
                    "split": split,
                    "hgn_file": str(hgn_path),
                    "series_name": series_name,
                    "hgn_length": hgn.shape[0],
                    "hdf_length": series_values.shape[0],
                    **metrics,
                }
            )

    fieldnames = [
        "hydrograph_id",
        "split",
        "series_name",
        "hgn_length",
        "hdf_length",
        "corr",
        "nrmse_z",
        "rmse",
        "max_abs_diff",
        "ref_peak_index",
        "candidate_peak_index",
        "peak_index_diff",
        "ref_peak",
        "candidate_peak",
        "peak_ratio",
        "exact_or_close",
        "hgn_file",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if args.output_md is not None:
        args.output_md.parent.mkdir(parents=True, exist_ok=True)
        write_markdown_summary(
            args.output_md,
            args.hdf_file,
            rows,
            args.top_n,
            hdf_precip_max,
            summarize_hgn_precipitation(args.data_dir, args.prefix),
        )

    best = sorted(
        rows,
        key=lambda row: (-float(row["corr"]), float(row["nrmse_z"]), float(row["max_abs_diff"])),
    )[: args.top_n]
    for row in best:
        print(
            "{hydrograph_id} {split} {series_name} corr={corr:.6f} "
            "nrmse_z={nrmse_z:.3e} max_abs={max_abs_diff:.3e} exact={exact_or_close:.0f}".format(
                **row
            )
        )
    print(f"Wrote {args.output_csv}")
    if args.output_md is not None:
        print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
