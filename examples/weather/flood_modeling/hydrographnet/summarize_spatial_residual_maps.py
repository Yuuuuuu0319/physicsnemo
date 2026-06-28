# SPDX-License-Identifier: Apache-2.0

"""Summarize Minxiong spatial residual map CSV files by event and zone."""

import argparse
import csv
from pathlib import Path
from statistics import mean, median


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def summarize_event(path: Path, event_id: str) -> list[dict[str, float | int | str]]:
    rows = read_rows(path)
    summaries = []
    for zone in range(4):
        zone_rows = [row for row in rows if int(row["zone"]) == zone]
        if not zone_rows:
            continue
        original = [float(row["original_rms_ft3"]) for row in zone_rows]
        projected = [float(row["projected_rms_ft3"]) for row in zone_rows]
        improvement = [float(row["improvement_ratio"]) for row in zone_rows]
        summaries.append(
            {
                "event_id": event_id,
                "zone": zone,
                "nodes": len(zone_rows),
                "median_improvement_ratio": median(improvement),
                "mean_improvement_ratio": mean(improvement),
                "mean_original_rms_ft3": mean(original),
                "mean_projected_rms_ft3": mean(projected),
                "mean_residual_reduction_percent": (
                    100.0 * (1.0 - mean(projected) / mean(original))
                    if mean(original) != 0.0
                    else float("nan")
                ),
            }
        )
    return summaries


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("/mnt/8tb_hdd2/joyce/Report/assets"),
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=Path(
            "/mnt/8tb_hdd2/joyce/Report/assets/"
            "minxiong_H25_H30_zone_weighted_spatial_residual_zone_summary.csv"
        ),
    )
    parser.add_argument("--events", nargs="+", default=["H25", "H26", "H27", "H28", "H29", "H30"])
    args = parser.parse_args()

    rows = []
    for event_id in args.events:
        path = args.input_dir / f"minxiong_{event_id}_zone_weighted_spatial_residual_summary.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        rows.extend(summarize_event(path, event_id))

    fieldnames = [
        "event_id",
        "zone",
        "nodes",
        "median_improvement_ratio",
        "mean_improvement_ratio",
        "mean_original_rms_ft3",
        "mean_projected_rms_ft3",
        "mean_residual_reduction_percent",
    ]
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
