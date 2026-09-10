# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Audit whether HGN storage, precipitation, and HEC-RAS Face Flow targets close."""

import argparse
import csv
import hashlib
from pathlib import Path

import numpy as np


def load_ids(path: Path) -> list[str]:
    return [line.strip() for line in path.read_text().splitlines() if line.strip()]


def summarize(storage: np.ndarray, budget: np.ndarray) -> dict[str, float]:
    residual = storage - budget
    storage_flat = storage.reshape(-1)
    budget_flat = budget.reshape(-1)
    correlation = (
        float(np.corrcoef(storage_flat, budget_flat)[0, 1])
        if np.std(storage_flat) > 0 and np.std(budget_flat) > 0
        else float("nan")
    )
    residual_rmse = float(np.sqrt(np.mean(residual**2)))
    budget_rms = float(np.sqrt(np.mean(budget**2)))
    return {
        "storage_rms_ft3": float(np.sqrt(np.mean(storage**2))),
        "budget_rms_ft3": budget_rms,
        "residual_rmse_ft3": residual_rmse,
        "relative_rmse": residual_rmse / max(budget_rms, 1e-12),
        "residual_bias_ft3": float(np.mean(residual)),
        "correlation": correlation,
    }


def update_pooled(
    accumulator: dict[str, float], storage: np.ndarray, budget: np.ndarray
) -> None:
    storage = storage.reshape(-1)
    budget = budget.reshape(-1)
    residual = storage - budget
    accumulator["count"] += storage.size
    accumulator["storage_sse"] += float(np.sum(storage**2))
    accumulator["budget_sse"] += float(np.sum(budget**2))
    accumulator["residual_sse"] += float(np.sum(residual**2))
    accumulator["residual_sum"] += float(np.sum(residual))
    accumulator["storage_sum"] += float(np.sum(storage))
    accumulator["budget_sum"] += float(np.sum(budget))
    accumulator["cross_sum"] += float(np.sum(storage * budget))


def summarize_pooled(accumulator: dict[str, float]) -> dict[str, float]:
    count = max(accumulator["count"], 1.0)
    storage_rms = np.sqrt(accumulator["storage_sse"] / count)
    budget_rms = np.sqrt(accumulator["budget_sse"] / count)
    residual_rmse = np.sqrt(accumulator["residual_sse"] / count)
    storage_mean = accumulator["storage_sum"] / count
    budget_mean = accumulator["budget_sum"] / count
    covariance = accumulator["cross_sum"] / count - storage_mean * budget_mean
    storage_variance = accumulator["storage_sse"] / count - storage_mean**2
    budget_variance = accumulator["budget_sse"] / count - budget_mean**2
    correlation = covariance / max(
        np.sqrt(max(storage_variance, 0.0) * max(budget_variance, 0.0)), 1e-12
    )
    return {
        "storage_rms_ft3": float(storage_rms),
        "budget_rms_ft3": float(budget_rms),
        "residual_rmse_ft3": float(residual_rmse),
        "relative_rmse": float(residual_rmse / max(budget_rms, 1e-12)),
        "residual_bias_ft3": accumulator["residual_sum"] / count,
        "correlation": float(correlation),
    }


def connected_component_labels(
    face_index: np.ndarray, node_mask: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Label connected components induced by a fixed node mask."""

    parent = np.arange(node_mask.size, dtype=np.int64)

    def find(node: int) -> int:
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = int(parent[node])
        return node

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    src, dst = face_index
    for first, second in zip(src[node_mask[src] & node_mask[dst]], dst[node_mask[src] & node_mask[dst]]):
        union(int(first), int(second))
    labels = np.full(node_mask.size, -1, dtype=np.int64)
    selected_nodes = np.flatnonzero(node_mask)
    roots = np.asarray([find(int(node)) for node in selected_nodes], dtype=np.int64)
    _, compact_labels = np.unique(roots, return_inverse=True)
    labels[selected_nodes] = compact_labels
    return labels, np.bincount(compact_labels)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True, type=Path)
    parser.add_argument("--ids-file", required=True, type=Path)
    parser.add_argument("--edge-flow-npz", required=True, type=Path)
    parser.add_argument("--face-graph-npz", required=True, type=Path)
    parser.add_argument("--zone-label-file", required=True, type=Path)
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--transition-start-index", type=int, default=73)
    parser.add_argument("--delta-t", type=float, default=1800.0)
    parser.add_argument(
        "--precipitation-unit-conversion",
        type=float,
        default=9.11344415281417e-7,
        help="Conversion from dataset precipitation units to ft/s.",
    )
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()

    ids = load_ids(args.ids_file)
    edge_data = np.load(args.edge_flow_npz, allow_pickle=False)
    face_graph = np.load(args.face_graph_npz, allow_pickle=False)
    zone_label = np.loadtxt(args.zone_label_file).astype(np.int64).reshape(-1)
    area = np.asarray(face_graph["hgn_node_surface_area"], dtype=np.float64)
    boundary_node_mask = np.zeros(zone_label.size, dtype=bool)
    boundary_cells = np.asarray(face_graph["boundary_face_cell_index"])
    valid_boundary_cells = boundary_cells[
        (boundary_cells >= 0) & (boundary_cells < zone_label.size)
    ]
    boundary_node_mask[valid_boundary_cells] = True
    masks = {
        "all": np.ones(zone_label.size, dtype=bool),
        "zone4": zone_label == 3,
        "zone4_interior": (zone_label == 3) & (~boundary_node_mask),
    }
    component_labels, component_sizes = connected_component_labels(
        np.asarray(face_graph["internal_face_index"], dtype=np.int64),
        masks["zone4_interior"],
    )
    component_min_sizes = (1, 2, 4, 8, 16, 32, 64)

    rows = []
    pooled = {
        name: {
            "count": 0.0,
            "storage_sse": 0.0,
            "budget_sse": 0.0,
            "residual_sse": 0.0,
            "residual_sum": 0.0,
            "storage_sum": 0.0,
            "budget_sum": 0.0,
            "cross_sum": 0.0,
        }
        for name in masks
    }
    pooled_aggregate = {
        name: {key: value for key, value in accumulator.items()}
        for name, accumulator in pooled.items()
    }
    pooled_components = {
        minimum_size: {
            key: value for key, value in pooled["zone4_interior"].items()
        }
        for minimum_size in component_min_sizes
    }
    for event_id in ids:
        internal = np.asarray(
            edge_data[f"{event_id}_internal_edge_delta"], dtype=np.float64
        )
        boundary = np.asarray(
            edge_data[f"{event_id}_boundary_source_delta"], dtype=np.float64
        )
        cell_balance = np.asarray(
            edge_data[f"{event_id}_cell_balance_delta"], dtype=np.float64
        )
        volume = np.loadtxt(
            args.data_dir / f"{args.prefix}_V_{event_id}.txt", delimiter="\t"
        )
        precipitation = np.loadtxt(
            args.data_dir / f"{args.prefix}_Pr_{event_id}.txt", delimiter="\t"
        ).reshape(-1)
        stop = min(internal.shape[0], volume.shape[0] - 1)
        start = args.transition_start_index
        if start >= stop:
            raise ValueError(f"No transitions {start}:{stop} available for {event_id}.")
        storage_delta = volume[start + 1 : stop + 1] - volume[start:stop]
        average_precipitation = 0.5 * (
            precipitation[start:stop] + precipitation[start + 1 : stop + 1]
        )
        precipitation_delta = (
            average_precipitation[:, None]
            * args.precipitation_unit_conversion
            * area[None, :]
            * args.delta_t
        )
        face_divergence = internal[start:stop] + boundary[start:stop]
        budget_delta = face_divergence + precipitation_delta
        cell_balance_error = face_divergence - cell_balance[start:stop]
        row = {
            "event_id": event_id,
            "transition_start_index": start,
            "num_transitions": stop - start,
            "face_vs_cell_balance_rmse_ft3": float(
                np.sqrt(np.mean(cell_balance_error**2))
            ),
        }
        for name, mask in masks.items():
            metrics = summarize(storage_delta[:, mask], budget_delta[:, mask])
            row.update({f"{name}_{key}": value for key, value in metrics.items()})
            update_pooled(pooled[name], storage_delta[:, mask], budget_delta[:, mask])
            aggregate_storage = np.sum(storage_delta[:, mask], axis=1, keepdims=True)
            aggregate_budget = np.sum(budget_delta[:, mask], axis=1, keepdims=True)
            aggregate_metrics = summarize(aggregate_storage, aggregate_budget)
            row.update(
                {
                    f"{name}_aggregate_{key}": value
                    for key, value in aggregate_metrics.items()
                }
            )
            update_pooled(
                pooled_aggregate[name], aggregate_storage, aggregate_budget
            )
        component_storage = np.stack(
            [
                np.sum(storage_delta[:, component_labels == label], axis=1)
                for label in range(component_sizes.size)
            ],
            axis=1,
        )
        component_budget = np.stack(
            [
                np.sum(budget_delta[:, component_labels == label], axis=1)
                for label in range(component_sizes.size)
            ],
            axis=1,
        )
        for minimum_size in component_min_sizes:
            selected_components = component_sizes >= minimum_size
            component_metrics = summarize(
                component_storage[:, selected_components],
                component_budget[:, selected_components],
            )
            prefix = f"zone4_interior_components_min{minimum_size}"
            row.update(
                {f"{prefix}_{key}": value for key, value in component_metrics.items()}
            )
            row[f"{prefix}_num_components"] = int(np.sum(selected_components))
            row[f"{prefix}_num_nodes"] = int(
                np.sum(component_sizes[selected_components])
            )
            update_pooled(
                pooled_components[minimum_size],
                component_storage[:, selected_components],
                component_budget[:, selected_components],
            )
        rows.append(row)

    pooled_row = {
        "event_id": "POOLED",
        "transition_start_index": args.transition_start_index,
        "num_transitions": sum(int(row["num_transitions"]) for row in rows),
        "face_vs_cell_balance_rmse_ft3": float(
            np.sqrt(np.mean([row["face_vs_cell_balance_rmse_ft3"] ** 2 for row in rows]))
        ),
    }
    for name in masks:
        metrics = summarize_pooled(pooled[name])
        pooled_row.update({f"{name}_{key}": value for key, value in metrics.items()})
        aggregate_metrics = summarize_pooled(pooled_aggregate[name])
        pooled_row.update(
            {
                f"{name}_aggregate_{key}": value
                for key, value in aggregate_metrics.items()
            }
        )
    for minimum_size in component_min_sizes:
        prefix = f"zone4_interior_components_min{minimum_size}"
        component_metrics = summarize_pooled(pooled_components[minimum_size])
        pooled_row.update(
            {f"{prefix}_{key}": value for key, value in component_metrics.items()}
        )
        selected_components = component_sizes >= minimum_size
        pooled_row[f"{prefix}_num_components"] = int(np.sum(selected_components))
        pooled_row[f"{prefix}_num_nodes"] = int(
            np.sum(component_sizes[selected_components])
        )
    rows.append(pooled_row)

    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"events={len(ids)} output={args.output_csv}")
    print(
        "zone4_interior_relative_rmse="
        f"{pooled_row['zone4_interior_relative_rmse']:.6f} "
        "zone4_interior_correlation="
        f"{pooled_row['zone4_interior_correlation']:.6f}"
    )
    print(
        "zone_mask_sha256="
        f"{hashlib.sha256(args.zone_label_file.read_bytes()).hexdigest()}"
    )


if __name__ == "__main__":
    main()
