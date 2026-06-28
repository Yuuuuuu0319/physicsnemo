# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Summarize edge-flux projection diagnostics into compact report tables."""

import argparse
import csv
from pathlib import Path


DEFAULT_COLUMNS = (
    "checkpoint",
    "loaded_epoch",
    "projection_mode",
    "projection_ridge",
    "original_face_relative_rmse",
    "face_relative_rmse",
    "original_internal_divergence_relative_rmse",
    "internal_divergence_relative_rmse",
    "zone_3_original_face_relative_rmse",
    "zone_3_face_relative_rmse",
    "zone_3_original_internal_divergence_relative_rmse",
    "zone_3_internal_divergence_relative_rmse",
    "projection_correction_rms_ft3",
    "original_edge_flux_rms_ft3",
    "edge_flux_rms_ft3",
)


def read_rows(paths: list[Path]) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in paths:
        with path.open("r", encoding="utf-8", newline="") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                row["source_csv"] = str(path)
                rows.append(row)
    return rows


def format_float(value: str, digits: int = 6) -> str:
    if value == "":
        return ""
    try:
        return f"{float(value):.{digits}f}"
    except ValueError:
        return value


def compact_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    compact = []
    for row in rows:
        item = {column: row.get(column, "") for column in DEFAULT_COLUMNS}
        for key, value in list(item.items()):
            if key not in ("checkpoint", "projection_mode"):
                item[key] = format_float(value)
        item["face_relative_rmse_delta"] = format_float(
            str(
                float(row["face_relative_rmse"])
                - float(row["original_face_relative_rmse"])
            )
        )
        item["internal_divergence_relative_rmse_delta"] = format_float(
            str(
                float(row["internal_divergence_relative_rmse"])
                - float(row["original_internal_divergence_relative_rmse"])
            )
        )
        item["zone_3_divergence_relative_rmse_delta"] = format_float(
            str(
                float(row["zone_3_internal_divergence_relative_rmse"])
                - float(row["zone_3_original_internal_divergence_relative_rmse"])
            )
        )
        compact.append(item)
    return compact


def write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else list(DEFAULT_COLUMNS)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = (
        "checkpoint",
        "projection_mode",
        "projection_ridge",
        "original_face_relative_rmse",
        "face_relative_rmse",
        "original_internal_divergence_relative_rmse",
        "internal_divergence_relative_rmse",
        "zone_3_original_internal_divergence_relative_rmse",
        "zone_3_internal_divergence_relative_rmse",
        "projection_correction_rms_ft3",
    )
    header = "| " + " | ".join(columns) + " |"
    separator = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [
        "# Minxiong H25-H30 Edge-Flux Projection Summary",
        "",
        "This compact table is generated from the projection diagnostic CSV files.",
        "It compares the edge head before and after sparse incidence projection.",
        "",
        header,
        separator,
    ]
    for row in rows:
        lines.append("| " + " | ".join(row.get(column, "") for column in columns) + " |")
    lines.extend(
        [
            "",
            "Lower relative RMSE is better.  Projection correction RMS is measured",
            "in the original HEC-RAS Face Flow delta units (`ft^3`).",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-csv", action="append", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--output-md", required=True, type=Path)
    args = parser.parse_args()

    rows = compact_rows(read_rows(args.input_csv))
    write_csv(args.output_csv, rows)
    write_markdown(args.output_md, rows)
    print(f"Wrote {args.output_csv}")
    print(f"Wrote {args.output_md}")


if __name__ == "__main__":
    main()
