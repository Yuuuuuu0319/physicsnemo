# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Plot Minxiong zone-weighted edge-local conservation rollout results."""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt


CASES = [
    {
        "label": "Original rollout",
        "csv": "results_minxiong_h25h30_rollout_projected_edge_flux_baseline_node_e19_face_only_all_boundary_source_l23_original_state.csv",
        "mode": "all",
        "high_weight": "",
        "alpha": "0.0",
    },
    {
        "label": "All projection",
        "csv": "results_minxiong_h25h30_rollout_projected_edge_flux_baseline_node_e19_face_only_all_boundary_source_l23_projected_state_alpha01.csv",
        "mode": "all",
        "high_weight": "",
        "alpha": "0.1",
    },
    {
        "label": "High only",
        "csv": "results_minxiong_h25h30_rollout_projected_edge_flux_baseline_node_e19_face_only_high_boundary_source_l23_projected_state_alpha01.csv",
        "mode": "high",
        "high_weight": "",
        "alpha": "0.1",
    },
    {
        "label": "Weighted h10 a0.1",
        "csv": "results_minxiong_h25h30_rollout_projected_edge_flux_baseline_node_e19_face_only_zone_weighted_h10_l1_boundary_source_l23_projected_state_alpha01.csv",
        "mode": "zone_weighted",
        "high_weight": "10",
        "alpha": "0.1",
    },
    {
        "label": "Weighted h25 a0.1",
        "csv": "results_minxiong_h25h30_rollout_projected_edge_flux_baseline_node_e19_face_only_zone_weighted_h25_l1_boundary_source_l23_projected_state_alpha01_factorized.csv",
        "mode": "zone_weighted",
        "high_weight": "25",
        "alpha": "0.1",
    },
    {
        "label": "Weighted h50 a0.05",
        "csv": "results_minxiong_h25h30_rollout_projected_edge_flux_baseline_node_e19_face_only_zone_weighted_h50_l1_boundary_source_l23_projected_state_alpha005_factorized.csv",
        "mode": "zone_weighted",
        "high_weight": "50",
        "alpha": "0.05",
    },
    {
        "label": "Weighted h50 a0.1",
        "csv": "results_minxiong_h25h30_rollout_projected_edge_flux_baseline_node_e19_face_only_zone_weighted_h50_l1_boundary_source_l23_projected_state_alpha01_factorized.csv",
        "mode": "zone_weighted",
        "high_weight": "50",
        "alpha": "0.1",
    },
    {
        "label": "Weighted h100 a0.05",
        "csv": "results_minxiong_h25h30_rollout_projected_edge_flux_baseline_node_e19_face_only_zone_weighted_h100_l1_boundary_source_l23_projected_state_alpha005_factorized.csv",
        "mode": "zone_weighted",
        "high_weight": "100",
        "alpha": "0.05",
    },
    {
        "label": "Weighted h100 a0.1",
        "csv": "results_minxiong_h25h30_rollout_projected_edge_flux_baseline_node_e19_face_only_zone_weighted_h100_l1_boundary_source_l23_projected_state_alpha01_factorized.csv",
        "mode": "zone_weighted",
        "high_weight": "100",
        "alpha": "0.1",
    },
]


def read_mean_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows in {path}")
    return rows[-1]


def to_float(row: dict[str, str], key: str) -> float:
    value = row.get(key)
    if value in (None, ""):
        return float("nan")
    return float(value)


def collect_rows(results_dir: Path) -> list[dict[str, str | float]]:
    rows: list[dict[str, str | float]] = []
    for case in CASES:
        csv_path = results_dir / case["csv"]
        if not csv_path.exists():
            continue
        row = read_mean_row(csv_path)
        rows.append(
            {
                "label": case["label"],
                "mode": case["mode"],
                "high_weight": case["high_weight"],
                "alpha": case["alpha"],
                "rollout_rmse": to_float(row, "rollout_rmse"),
                "volume_rmse": to_float(row, "rollout_volume_rmse"),
                "global_closure": to_float(
                    row, "projected_projection_closure_relative_rmse"
                ),
                "zone3_closure": to_float(
                    row, "zone_3_projected_projection_closure_relative_rmse"
                ),
                "zone3_rollout_rmse": to_float(row, "zone_3_rollout_rmse"),
                "correction_rms_ft3": to_float(row, "projection_correction_rms_ft3"),
            }
        )
    return rows


def write_summary_csv(rows: list[dict[str, str | float]], path: Path) -> None:
    fieldnames = [
        "label",
        "mode",
        "high_weight",
        "alpha",
        "rollout_rmse",
        "volume_rmse",
        "global_closure",
        "zone3_closure",
        "zone3_rollout_rmse",
        "correction_rms_ft3",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_bar_chart(
    rows: list[dict[str, str | float]],
    output_path: Path,
    metrics: list[tuple[str, str]],
    ylabel: str,
    title: str,
) -> None:
    labels = [str(row["label"]) for row in rows]
    x = range(len(rows))
    width = 0.8 / len(metrics)
    fig, ax = plt.subplots(figsize=(12, 5.5))
    for idx, (key, name) in enumerate(metrics):
        offsets = [value + (idx - (len(metrics) - 1) / 2) * width for value in x]
        ax.bar(offsets, [float(row[key]) for row in rows], width=width, label=name)
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def save_tradeoff_plot(rows: list[dict[str, str | float]], output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 5.8))
    for row in rows:
        ax.scatter(float(row["rollout_rmse"]), float(row["zone3_closure"]), s=70)
        ax.annotate(
            str(row["label"]),
            (float(row["rollout_rmse"]), float(row["zone3_closure"])),
            xytext=(5, 4),
            textcoords="offset points",
            fontsize=8,
        )
    ax.set_xlabel("Rollout RMSE")
    ax.set_ylabel("Zone 3 projected closure relative RMSE")
    ax.set_title("Prediction vs Zone 3 Local Conservation Trade-off")
    ax.grid(linestyle="--", alpha=0.35)
    fig.tight_layout()
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("examples/weather/flood_modeling/hydrographnet"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("/mnt/8tb_hdd2/joyce/Report/assets"),
    )
    args = parser.parse_args()

    rows = collect_rows(args.results_dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_csv = args.output_dir / "minxiong_zone_weighted_projection_sweep_summary.csv"
    write_summary_csv(rows, summary_csv)

    save_bar_chart(
        rows,
        args.output_dir / "minxiong_zone_weighted_projection_rollout_rmse.png",
        [("rollout_rmse", "Rollout RMSE"), ("volume_rmse", "Volume RMSE")],
        "RMSE",
        "H25-H30 Rollout Prediction Error",
    )
    save_bar_chart(
        rows,
        args.output_dir / "minxiong_zone_weighted_projection_closure.png",
        [
            ("global_closure", "Global closure"),
            ("zone3_closure", "Zone 3 closure"),
        ],
        "Relative RMSE",
        "Global vs Zone 3 Local Conservation",
    )
    save_bar_chart(
        rows,
        args.output_dir / "minxiong_zone_weighted_projection_correction.png",
        [("correction_rms_ft3", "Correction RMS")],
        "ft^3",
        "Projection Correction Magnitude",
    )
    save_tradeoff_plot(
        rows,
        args.output_dir / "minxiong_zone_weighted_projection_tradeoff.png",
    )

    print(f"Wrote {summary_csv}")


if __name__ == "__main__":
    main()
