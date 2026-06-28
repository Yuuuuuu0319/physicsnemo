# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Pretrain the experiment-local HEC-RAS edge-flux head.

This isolates direct internal Face Flow target learning from the node rollout
model. It is meant as a diagnostic: if the edge head cannot learn face-flow
magnitude here, coupling it to HydroGraphNet will not fix the core issue.
"""

import argparse
import csv
from pathlib import Path

import torch
from torch_geometric.loader import DataLoader as PyGDataLoader

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.utils import load_checkpoint, save_checkpoint
from train import HecRasEdgeFluxHead


def rmse(value: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.mean(value**2))


def masked_mean(values: torch.Tensor, weights: torch.Tensor | None = None) -> torch.Tensor:
    if weights is None:
        return torch.mean(values)
    denom = torch.sum(weights)
    if denom <= 0:
        return torch.mean(values)
    return torch.sum(values * weights) / denom


def face_weights_for_zone(graph, zone_mode: str, device) -> torch.Tensor | None:
    if zone_mode == "all":
        return torch.ones(graph.hecras_face_index.shape[1], device=device)
    src, dst = graph.hecras_face_index.to(device)
    if zone_mode == "zone_weight" and hasattr(graph, "zone_weight"):
        node_weights = graph.zone_weight.to(device)
        return 0.5 * (node_weights[src] + node_weights[dst])
    if zone_mode == "high" and hasattr(graph, "zone_label"):
        labels = graph.zone_label.to(device)
        return ((labels[src] == 3) | (labels[dst] == 3)).to(torch.float32)
    if zone_mode == "high_interior" and hasattr(graph, "zone_label"):
        labels = graph.zone_label.to(device)
        return ((labels[src] == 3) & (labels[dst] == 3)).to(torch.float32)
    return None


def node_weights_for_zone(graph, zone_mode: str, device) -> torch.Tensor | None:
    if zone_mode == "all":
        return torch.ones(graph.x.shape[0], device=device)
    if zone_mode == "zone_weight" and hasattr(graph, "zone_weight"):
        return graph.zone_weight.to(device)
    if zone_mode == "high" and hasattr(graph, "zone_label"):
        return (graph.zone_label.to(device) == 3).to(torch.float32)
    if (
        zone_mode == "high_interior"
        and hasattr(graph, "zone_label")
        and hasattr(graph, "hecras_boundary_node_mask")
    ):
        labels = graph.zone_label.to(device)
        interior = ~graph.hecras_boundary_node_mask.to(device)
        return ((labels == 3) & interior).to(torch.float32)
    return None


def compute_face_divergence(graph, edge_flux_delta: torch.Tensor) -> torch.Tensor:
    face_index = graph.hecras_face_index.to(edge_flux_delta.device)
    src, dst = face_index
    divergence = torch.zeros(
        graph.x.shape[0], dtype=edge_flux_delta.dtype, device=edge_flux_delta.device
    )
    divergence.index_add_(0, src, -edge_flux_delta.reshape(-1))
    divergence.index_add_(0, dst, edge_flux_delta.reshape(-1))
    return divergence


def transform_face_residual(
    prediction: torch.Tensor,
    target: torch.Tensor,
    graph,
    mode: str,
) -> torch.Tensor:
    if mode == "none":
        return prediction - target
    scale = resolve_face_scale(target, graph, mode)
    prediction = prediction / scale
    target = target / scale
    if mode in (
        "target_rms",
        "per_face_rms",
        "event_rms",
        "transition_rms",
        "face_event_rms",
        "face_transition_rms",
    ):
        return prediction - target
    if mode.startswith("asinh_"):
        return torch.asinh(prediction) - torch.asinh(target)
    if mode.startswith("signed_log1p_"):
        return torch.sign(prediction) * torch.log1p(torch.abs(prediction)) - torch.sign(
            target
        ) * torch.log1p(torch.abs(target))
    return prediction - target


def resolve_face_scale(target: torch.Tensor, graph, mode: str) -> torch.Tensor:
    scale_mode = mode
    for prefix in ("asinh_", "signed_log1p_"):
        if scale_mode.startswith(prefix):
            scale_mode = scale_mode[len(prefix) :]
            break
    target_rms = torch.sqrt(torch.clamp(torch.mean(target**2), min=1e-12))
    if scale_mode == "target_rms":
        return torch.clamp(target_rms, min=1.0)
    if scale_mode in ("per_face_rms", "face_event_rms", "face_transition_rms"):
        if not hasattr(graph, "hecras_internal_face_flow_rms"):
            raise AttributeError(
                f"face_loss_normalization={mode!r} requires "
                "graph.hecras_internal_face_flow_rms."
            )
        face_scale = torch.clamp(
            graph.hecras_internal_face_flow_rms.to(target.device).reshape(-1),
            min=1.0,
        )
    else:
        face_scale = None
    if scale_mode == "per_face_rms":
        return face_scale
    if scale_mode in ("event_rms", "face_event_rms"):
        if not hasattr(graph, "hecras_internal_face_flow_event_rms"):
            raise AttributeError(
                f"face_loss_normalization={mode!r} requires "
                "graph.hecras_internal_face_flow_event_rms."
            )
        event_scale = torch.clamp(
            graph.hecras_internal_face_flow_event_rms.to(target.device).reshape(-1)[0],
            min=1.0,
        )
        if scale_mode == "event_rms":
            return event_scale
        return face_scale * event_scale / global_scale(graph, target)
    if scale_mode in ("transition_rms", "face_transition_rms"):
        if not hasattr(graph, "hecras_internal_face_flow_transition_rms"):
            raise AttributeError(
                f"face_loss_normalization={mode!r} requires "
                "graph.hecras_internal_face_flow_transition_rms."
            )
        transition_scale = torch.clamp(
            graph.hecras_internal_face_flow_transition_rms.to(target.device).reshape(-1)[
                0
            ],
            min=1.0,
        )
        if scale_mode == "transition_rms":
            return transition_scale
        return face_scale * transition_scale / global_scale(graph, target)
    raise ValueError(f"Unknown face loss normalization: {mode!r}")


def global_scale(graph, target: torch.Tensor) -> torch.Tensor:
    if not hasattr(graph, "hecras_internal_face_flow_global_rms"):
        raise AttributeError(
            "face-event/face-transition scaling requires "
            "graph.hecras_internal_face_flow_global_rms."
        )
    return torch.clamp(
        graph.hecras_internal_face_flow_global_rms.to(target.device).reshape(-1)[0],
        min=1.0,
    )


def compute_face_loss(edge_head, graph, args):
    prediction = edge_head(graph).reshape(-1)
    target = graph.hecras_internal_face_flow_delta.to(prediction.device).reshape(-1)
    weights = face_weights_for_zone(graph, args.zone_mode, prediction.device)
    residual = transform_face_residual(
        prediction,
        target,
        graph,
        args.face_loss_normalization,
    )
    loss = masked_mean(residual**2, weights)
    raw_residual = prediction - target
    raw_loss = masked_mean(raw_residual**2, weights)
    div_loss = prediction.new_tensor(0.0)
    raw_div_loss = prediction.new_tensor(0.0)
    divergence = None
    divergence_target = None
    if args.divergence_loss_weight > 0:
        if not hasattr(graph, "hecras_edge_internal_delta"):
            raise AttributeError(
                "--divergence-loss-weight requires graph.hecras_edge_internal_delta."
            )
        divergence = compute_face_divergence(graph, prediction)
        divergence_target = graph.hecras_edge_internal_delta.to(prediction.device).reshape(
            -1
        )
        div_residual = divergence - divergence_target
        div_weights = node_weights_for_zone(
            graph, args.divergence_zone_mode, prediction.device
        )
        raw_div_loss = masked_mean(div_residual**2, div_weights)
        div_scale = torch.clamp(torch.sqrt(masked_mean(divergence_target**2, div_weights)), min=1.0)
        if args.divergence_loss_normalization == "target_rms":
            div_residual = div_residual / div_scale
        elif args.divergence_loss_normalization != "none":
            raise ValueError(
                f"Unknown divergence loss normalization: {args.divergence_loss_normalization!r}"
            )
        div_loss = masked_mean(div_residual**2, div_weights)
        loss = loss + args.divergence_loss_weight * div_loss
    return loss, raw_loss, div_loss, raw_div_loss, prediction, target, divergence, divergence_target


def build_dataset(args, split_file: str, num_samples: int) -> HydroGraphDataset:
    return HydroGraphDataset(
        data_dir=args.data_dir,
        prefix=args.prefix,
        num_samples=num_samples,
        n_time_steps=args.n_time_steps,
        hydrograph_ids_file=split_file,
        split="train",
        return_physics=False,
        use_fidelity_zones=True,
        zone_label_file=args.zone_label_file,
        zone_weight_file=args.zone_weight_file,
        return_hecras_face=True,
        hecras_face_graph_file=args.hecras_face_graph_file,
        return_hecras_edge_flow=True,
        hecras_edge_flow_npz=args.hecras_edge_flow_npz,
        hecras_edge_flow_mode="internal_plus_boundary_source",
        hecras_edge_flow_face_stats_npz=args.hecras_edge_flow_face_stats_npz,
        hecras_edge_flow_scale_stats_npz=args.hecras_edge_flow_scale_stats_npz,
    )


def evaluate(edge_head, dataset, args) -> dict[str, float]:
    sums: dict[str, float] = {}
    count = 0
    edge_head.eval()
    with torch.no_grad():
        for graph in dataset:
            graph = graph.to(args.device_obj)
            (
                loss,
                raw_loss,
                div_loss,
                raw_div_loss,
                prediction,
                target,
                divergence,
                divergence_target,
            ) = compute_face_loss(edge_head, graph, args)
            residual = prediction - target
            sums["selected_face_loss"] = sums.get("selected_face_loss", 0.0) + float(
                loss.detach().cpu()
            )
            sums["raw_face_loss"] = sums.get("raw_face_loss", 0.0) + float(
                raw_loss.detach().cpu()
            )
            sums["selected_divergence_loss"] = sums.get(
                "selected_divergence_loss", 0.0
            ) + float(div_loss.detach().cpu())
            sums["raw_divergence_loss"] = sums.get(
                "raw_divergence_loss", 0.0
            ) + float(raw_div_loss.detach().cpu())
            sums["edge_flux_rms_ft3"] = sums.get("edge_flux_rms_ft3", 0.0) + float(
                rmse(prediction).detach().cpu()
            )
            sums["target_rms_ft3"] = sums.get("target_rms_ft3", 0.0) + float(
                rmse(target).detach().cpu()
            )
            sums["face_rmse_ft3"] = sums.get("face_rmse_ft3", 0.0) + float(
                rmse(residual).detach().cpu()
            )
            if divergence is not None and divergence_target is not None:
                div_residual = divergence - divergence_target
                sums["divergence_rmse_ft3"] = sums.get(
                    "divergence_rmse_ft3", 0.0
                ) + float(rmse(div_residual).detach().cpu())
                sums["divergence_target_rms_ft3"] = sums.get(
                    "divergence_target_rms_ft3", 0.0
                ) + float(rmse(divergence_target).detach().cpu())
            if hasattr(graph, "zone_label"):
                src, dst = graph.hecras_face_index.to(args.device_obj)
                labels = graph.zone_label.to(args.device_obj)
                high_mask = (labels[src] == 3) | (labels[dst] == 3)
                if torch.any(high_mask):
                    high_residual = residual[high_mask]
                    high_target = target[high_mask]
                    high_prediction = prediction[high_mask]
                    sums["zone3_face_rmse_ft3"] = sums.get(
                        "zone3_face_rmse_ft3", 0.0
                    ) + float(rmse(high_residual).detach().cpu())
                    sums["zone3_target_rms_ft3"] = sums.get(
                        "zone3_target_rms_ft3", 0.0
                    ) + float(rmse(high_target).detach().cpu())
                    sums["zone3_edge_flux_rms_ft3"] = sums.get(
                        "zone3_edge_flux_rms_ft3", 0.0
                    ) + float(rmse(high_prediction).detach().cpu())
                if divergence is not None and divergence_target is not None:
                    node_high_mask = labels == 3
                    if torch.any(node_high_mask):
                        high_div_residual = (divergence - divergence_target)[
                            node_high_mask
                        ]
                        high_div_target = divergence_target[node_high_mask]
                        sums["zone3_divergence_rmse_ft3"] = sums.get(
                            "zone3_divergence_rmse_ft3", 0.0
                        ) + float(rmse(high_div_residual).detach().cpu())
                        sums["zone3_divergence_target_rms_ft3"] = sums.get(
                            "zone3_divergence_target_rms_ft3", 0.0
                        ) + float(rmse(high_div_target).detach().cpu())
            count += 1
    rows = {key: value / max(count, 1) for key, value in sums.items()}
    target_rms = max(rows.get("target_rms_ft3", 0.0), 1e-12)
    rows["face_relative_rmse"] = rows.get("face_rmse_ft3", 0.0) / target_rms
    zone3_target_rms = max(rows.get("zone3_target_rms_ft3", 0.0), 1e-12)
    rows["zone3_face_relative_rmse"] = rows.get("zone3_face_rmse_ft3", 0.0) / zone3_target_rms
    div_target_rms = max(rows.get("divergence_target_rms_ft3", 0.0), 1e-12)
    rows["divergence_relative_rmse"] = (
        rows.get("divergence_rmse_ft3", 0.0) / div_target_rms
    )
    zone3_div_target_rms = max(
        rows.get("zone3_divergence_target_rms_ft3", 0.0), 1e-12
    )
    rows["zone3_divergence_relative_rmse"] = (
        rows.get("zone3_divergence_rmse_ft3", 0.0) / zone3_div_target_rms
    )
    return rows


def write_rows(path: Path, rows: list[dict[str, float | int | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    for key in ("split", "epoch"):
        if key in fieldnames:
            fieldnames.remove(key)
    fieldnames = [key for key in ("split", "epoch") if key in rows[0]] + fieldnames
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", required=True)
    parser.add_argument("--train-ids-file", default="train.txt")
    parser.add_argument("--eval-ids-file")
    parser.add_argument("--prefix", default="M80")
    parser.add_argument("--num-train-samples", type=int, default=120)
    parser.add_argument("--num-eval-samples", type=int, default=120)
    parser.add_argument("--n-time-steps", type=int, default=2)
    parser.add_argument("--zone-label-file", default="zone_label.txt")
    parser.add_argument("--zone-weight-file", default="zone_weight.txt")
    parser.add_argument("--hecras-face-graph-file", required=True)
    parser.add_argument("--hecras-edge-flow-npz", required=True)
    parser.add_argument("--eval-hecras-edge-flow-npz")
    parser.add_argument("--hecras-edge-flow-face-stats-npz")
    parser.add_argument("--hecras-edge-flow-scale-stats-npz")
    parser.add_argument("--eval-hecras-edge-flow-scale-stats-npz")
    parser.add_argument("--num-input-features", type=int, default=16)
    parser.add_argument("--edge-head-hidden-dim", type=int, default=128)
    parser.add_argument("--edge-head-num-hidden-layers", type=int, default=2)
    parser.add_argument("--edge-head-scale", type=float, default=1.0)
    parser.add_argument("--edge-head-use-face-normal", action="store_true")
    parser.add_argument("--edge-head-use-surface-features", action="store_true")
    parser.add_argument("--edge-head-use-physical-surface-features", action="store_true")
    parser.add_argument("--edge-head-use-previous-face-flow", action="store_true")
    parser.add_argument("--edge-head-use-edge-physical-features", action="store_true")
    parser.add_argument("--edge-head-output-mode", default="raw")
    parser.add_argument("--zone-mode", default="high")
    parser.add_argument("--face-loss-normalization", default="asinh_per_face_rms")
    parser.add_argument("--divergence-loss-weight", type=float, default=0.0)
    parser.add_argument("--divergence-zone-mode", default="all")
    parser.add_argument("--divergence-loss-normalization", default="target_rms")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--ckpt-path", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    args = parser.parse_args()
    args.device_obj = torch.device(args.device if torch.cuda.is_available() else "cpu")

    train_dataset = build_dataset(args, args.train_ids_file, args.num_train_samples)
    train_loader = PyGDataLoader(train_dataset, batch_size=1, shuffle=True)
    edge_head = HecRasEdgeFluxHead(
        args.num_input_features,
        hidden_dim=args.edge_head_hidden_dim,
        num_hidden_layers=args.edge_head_num_hidden_layers,
        scale=args.edge_head_scale,
        use_face_normal=args.edge_head_use_face_normal,
        use_surface_features=args.edge_head_use_surface_features,
        use_physical_surface_features=args.edge_head_use_physical_surface_features,
        use_previous_face_flow=args.edge_head_use_previous_face_flow,
        use_edge_physical_features=args.edge_head_use_edge_physical_features,
        output_mode=args.edge_head_output_mode,
    ).to(args.device_obj)
    optimizer = torch.optim.AdamW(
        edge_head.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    start_epoch = load_checkpoint(
        args.ckpt_path,
        models=[edge_head],
        optimizer=optimizer,
        device=args.device_obj,
    )
    rows = []
    eval_dataset = None
    original_edge_npz = args.hecras_edge_flow_npz
    original_scale_stats_npz = args.hecras_edge_flow_scale_stats_npz
    if args.eval_ids_file is not None:
        if args.eval_hecras_edge_flow_npz is not None:
            args.hecras_edge_flow_npz = args.eval_hecras_edge_flow_npz
        if args.eval_hecras_edge_flow_scale_stats_npz is not None:
            args.hecras_edge_flow_scale_stats_npz = (
                args.eval_hecras_edge_flow_scale_stats_npz
            )
        eval_dataset = build_dataset(args, args.eval_ids_file, args.num_eval_samples)
        args.hecras_edge_flow_npz = original_edge_npz
        args.hecras_edge_flow_scale_stats_npz = original_scale_stats_npz

    for epoch in range(start_epoch, args.epochs):
        edge_head.train()
        total_loss = 0.0
        total_raw_loss = 0.0
        total_div_loss = 0.0
        total_raw_div_loss = 0.0
        batch_count = 0
        for graph in train_loader:
            if args.max_train_batches is not None and batch_count >= args.max_train_batches:
                break
            graph = graph.to(args.device_obj)
            optimizer.zero_grad(set_to_none=True)
            loss, raw_loss, div_loss, raw_div_loss, *_ = compute_face_loss(edge_head, graph, args)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu())
            total_raw_loss += float(raw_loss.detach().cpu())
            total_div_loss += float(div_loss.detach().cpu())
            total_raw_div_loss += float(raw_div_loss.detach().cpu())
            batch_count += 1
        row = {
            "split": "train_loop",
            "epoch": epoch,
            "selected_face_loss": total_loss / max(batch_count, 1),
            "raw_face_loss": total_raw_loss / max(batch_count, 1),
            "selected_divergence_loss": total_div_loss / max(batch_count, 1),
            "raw_divergence_loss": total_raw_div_loss / max(batch_count, 1),
            "num_batches": batch_count,
        }
        print(
            f"epoch {epoch}: selected={row['selected_face_loss']:.6g} "
            f"div={row['selected_divergence_loss']:.6g} "
            f"raw={row['raw_face_loss']:.6g} batches={batch_count}"
        )
        rows.append(row)
        rows.append({"split": "train_eval", "epoch": epoch, **evaluate(edge_head, train_dataset, args)})
        if eval_dataset is not None:
            rows.append({"split": "eval", "epoch": epoch, **evaluate(edge_head, eval_dataset, args)})
        save_checkpoint(args.ckpt_path, models=[edge_head], optimizer=optimizer, epoch=epoch)

    write_rows(args.output_csv, rows)
    print(f"Wrote {args.output_csv}")


if __name__ == "__main__":
    main()
