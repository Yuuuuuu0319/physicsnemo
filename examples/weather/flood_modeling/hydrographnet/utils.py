# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-FileCopyrightText: All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Utility functions for physics-based loss computation and custom loss definitions.
"""

import torch
import torch.nn.functional as F


def compute_physics_loss(pred, physics_data, graph, delta_t=1200.0):
    """
    Compute a physics-based continuity loss in the denormalized domain.

    For each graph sample, the predicted total volume is computed as:
        predicted_total_volume = past_volume_denorm + volume_std * (sum of predicted volume differences)
    where:
        past_volume_denorm = past_volume_norm * volume_std + (num_nodes * volume_mean)

    Future volume is denormalized similarly:
        future_volume_denorm = future_volume_norm * volume_std + (num_nodes * volume_mean)

    Two continuity terms are computed:
        - term1: Uses average inflow and precipitation (denorm_avg_inflow and denorm_avg_precip)
        - term2: Uses next step's inflow and precipitation (denorm_next_inflow and denorm_next_precip)

    An effective precipitation term is computed as:
        new_precip_term = base_precip * infiltration_area_sum

    Finally, the physics loss is the mean of the sum of term1 and term2 across all graph samples.

    Args:
        pred (torch.Tensor): Model predictions (expected volume difference).
        physics_data (dict): Dictionary containing various denormalized physics parameters.
        graph (PyGData): Batched PyG graph.
        delta_t (float): Time delta over which the continuity is enforced.

    Returns:
        torch.Tensor: Mean physics loss across all graph samples.
    """
    unique_ids = torch.unique(graph.batch)
    predicted_diff = pred[:, 1]  # Predicted volume difference (normalized)
    physics_losses = []

    for uid in unique_ids:
        mask = graph.batch == uid
        pred_diff_sum = predicted_diff[mask].sum()

        idx = (unique_ids == uid).nonzero(as_tuple=False).item()
        past_volume_norm = physics_data["past_volume"][idx]
        future_volume_norm = physics_data["future_volume"][idx]
        # For term1: use average inflow and precipitation
        denorm_avg_inflow = physics_data["avg_inflow"][idx]
        denorm_avg_precip = physics_data["avg_precipitation"][idx]
        # For term2: use next step inflow and precipitation
        denorm_next_inflow = physics_data["next_inflow"][idx]
        denorm_next_precip = physics_data["next_precip"][idx]

        volume_mean = physics_data["volume_mean"][idx]
        volume_std = physics_data["volume_std"][idx]
        num_nodes = physics_data["num_nodes"][idx]
        area_sum = physics_data["area_sum"][idx]
        infiltration_area_sum = physics_data["infiltration_area_sum"][idx]

        # Denormalize past and future volumes.
        past_volume_denorm = past_volume_norm * volume_std + num_nodes * volume_mean
        future_volume_denorm = future_volume_norm * volume_std + num_nodes * volume_mean

        # Compute the predicted total volume.
        pred_total_volume = past_volume_denorm + volume_std * pred_diff_sum

        # Compute effective precipitation terms.
        new_precip_term = denorm_avg_precip * infiltration_area_sum
        new_next_precip_term = denorm_next_precip * infiltration_area_sum

        temp1 = pred_total_volume - (
            past_volume_denorm + delta_t * (denorm_avg_inflow + new_precip_term)
        )

        temp2 = (
            future_volume_denorm
            - pred_total_volume
            - delta_t * (denorm_next_inflow + new_next_precip_term)
        )

        # Compute continuity terms using ReLU to enforce non-negativity.
        term1 = (
            F.relu(
                (
                    pred_total_volume
                    - (
                        past_volume_denorm
                        + delta_t * (denorm_avg_inflow + new_precip_term)
                    )
                )
                / area_sum
            )
            ** 2
        )
        term2 = (
            F.relu(
                (
                    future_volume_denorm
                    - pred_total_volume
                    - delta_t * (denorm_next_inflow + new_next_precip_term)
                )
                / area_sum
            )
            ** 2
        )

        physics_losses.append(term1 + term2)

    if physics_losses:
        return torch.stack(physics_losses).mean()
    else:
        return torch.tensor(0.0, device=pred.device)


def compute_zone_weighted_loss(pred, target, graph):
    """Compute a node-wise prediction loss weighted by fidelity-zone weights."""
    if not hasattr(graph, "zone_weight"):
        return torch.tensor(0.0, device=pred.device)
    weights = graph.zone_weight.to(pred.device).view(-1, 1)
    if torch.sum(weights) <= 0:
        return torch.tensor(0.0, device=pred.device)
    node_loss = (pred - target) ** 2
    return torch.sum(weights * node_loss) / (torch.sum(weights) * pred.shape[1])


def compute_edge_local_proxy_loss(pred, target, graph):
    """Compute a selective velocity-proxy volume-change loss.

    This is a separate bridge toward edge-informed local conservation. It uses
    the current VX/VY field on each directed edge to build a transport proxy,
    calibrates that proxy to the target volume delta for the current sample, and
    applies the residual only through zone weights when available.
    """
    required_attrs = ("edge_unit_vector", "current_vx", "current_vy", "volume_std")
    if not all(hasattr(graph, attr) for attr in required_attrs):
        return torch.tensor(0.0, device=pred.device)

    src, dst = graph.edge_index
    edge_dirs = graph.edge_unit_vector.to(pred.device)
    velocity = torch.stack([graph.current_vx, graph.current_vy], dim=1).to(pred.device)
    edge_velocity = 0.5 * (velocity[src] + velocity[dst])
    edge_flux = torch.sum(edge_velocity * edge_dirs, dim=1)

    divergence = torch.zeros(pred.shape[0], device=pred.device)
    divergence.index_add_(0, src, edge_flux)
    divergence.index_add_(0, dst, -edge_flux)
    proxy = -divergence

    losses = []
    unique_ids = torch.unique(graph.batch) if hasattr(graph, "batch") else [None]
    for local_idx, uid in enumerate(unique_ids):
        if uid is None:
            node_mask = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
            volume_std = graph.volume_std.reshape(-1)[0].to(pred.device)
        else:
            node_mask = graph.batch == uid
            volume_std = graph.volume_std.reshape(-1)[local_idx].to(pred.device)

        pred_delta = pred[node_mask, 1] * volume_std
        target_delta = target[node_mask, 1] * volume_std
        proxy_delta = proxy[node_mask]
        denom = torch.sum(proxy_delta * proxy_delta).clamp_min(1e-12)
        scale = (torch.sum(target_delta * proxy_delta) / denom).detach()
        residual = pred_delta - scale * proxy_delta

        if hasattr(graph, "zone_weight"):
            weights = graph.zone_weight[node_mask].to(pred.device)
            if torch.sum(weights) > 0:
                losses.append(torch.sum(weights * residual**2) / torch.sum(weights))
            else:
                losses.append(torch.mean(residual**2))
        else:
            losses.append(torch.mean(residual**2))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=pred.device)


def compute_hecras_face_local_loss(
    pred,
    target,
    graph,
    delta_t=1200.0,
    zone_mode="zone_weight",
    calibrate_to_target=True,
):
    """Compute selective local loss on HEC-RAS true internal faces.

    This branch is intentionally separate from the kNN/VX/VY proxy loss. It uses
    HEC-RAS internal face connectivity, face length, and optional HDF face
    velocity. The HDF velocity is event-specific, so keep this loss disabled
    unless the event mapping has been reviewed.
    """
    required_attrs = (
        "hecras_face_index",
        "hecras_face_length",
        "hecras_face_velocity",
        "volume_std",
    )
    if not all(hasattr(graph, attr) for attr in required_attrs):
        return torch.tensor(0.0, device=pred.device)

    face_index = graph.hecras_face_index.to(pred.device)
    face_length = graph.hecras_face_length.to(pred.device).reshape(-1)
    face_velocity = graph.hecras_face_velocity.to(pred.device).reshape(-1)
    if face_index.numel() == 0 or face_length.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    src, dst = face_index
    face_flow = face_velocity * face_length
    face_delta = face_flow * delta_t
    proxy = torch.zeros(pred.shape[0], dtype=pred.dtype, device=pred.device)
    proxy.index_add_(0, src, -face_delta.to(pred.dtype))
    proxy.index_add_(0, dst, face_delta.to(pred.dtype))

    losses = []
    batch = getattr(graph, "batch", None)
    unique_ids = torch.unique(batch) if batch is not None else [None]
    for local_idx, uid in enumerate(unique_ids):
        if uid is None:
            node_mask = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
        else:
            node_mask = batch == uid

        volume_std = graph.volume_std.reshape(-1)[local_idx].to(pred.device)
        pred_delta = pred[node_mask, 1] * volume_std
        target_delta = target[node_mask, 1] * volume_std
        proxy_delta = proxy[node_mask]

        if calibrate_to_target:
            denom = torch.sum(proxy_delta * proxy_delta).clamp_min(1e-12)
            scale = (torch.sum(target_delta * proxy_delta) / denom).detach()
            proxy_delta = proxy_delta * scale

        residual = pred_delta - proxy_delta
        weights = None
        if zone_mode == "zone_weight" and hasattr(graph, "zone_weight"):
            weights = graph.zone_weight[node_mask].to(pred.device)
        elif zone_mode == "high" and hasattr(graph, "zone_label"):
            weights = (graph.zone_label[node_mask].to(pred.device) == 3).to(pred.dtype)
        elif zone_mode == "all":
            weights = torch.ones_like(residual)

        if weights is not None and torch.sum(weights) > 0:
            losses.append(torch.sum(weights * residual**2) / torch.sum(weights))
        else:
            losses.append(torch.mean(residual**2))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=pred.device)


def compute_hecras_face_geometry_loss(pred, graph, zone_mode="zone_weight"):
    """Compute geometry-only smoothness on HEC-RAS true internal faces.

    This loss does not use event-specific HDF face velocity. It regularizes the
    predicted volume delta per cell area across true HEC-RAS internal faces,
    weighted by face length and optionally by fidelity-zone weights.
    """
    required_attrs = (
        "hecras_face_index",
        "hecras_face_length",
        "hecras_node_area",
        "volume_std",
    )
    if not all(hasattr(graph, attr) for attr in required_attrs):
        return torch.tensor(0.0, device=pred.device)

    face_index = graph.hecras_face_index.to(pred.device)
    face_length = graph.hecras_face_length.to(pred.device).reshape(-1)
    node_area = graph.hecras_node_area.to(pred.device).reshape(-1).clamp_min(1e-6)
    src, dst = face_index
    if src.numel() == 0:
        return torch.tensor(0.0, device=pred.device)

    losses = []
    batch = getattr(graph, "batch", None)
    unique_ids = torch.unique(batch) if batch is not None else [None]
    for local_idx, uid in enumerate(unique_ids):
        if uid is None:
            node_mask = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
            face_mask = torch.ones(src.shape[0], dtype=torch.bool, device=pred.device)
            node_offset = 0
        else:
            node_mask = batch == uid
            node_ids = torch.nonzero(node_mask, as_tuple=False).reshape(-1)
            node_offset = int(node_ids[0].detach().item())
            face_mask = node_mask[src] & node_mask[dst]

        if not torch.any(face_mask):
            continue

        volume_std = graph.volume_std.reshape(-1)[local_idx].to(pred.device)
        pred_delta_per_area = pred[node_mask, 1] * volume_std / node_area[node_mask]
        local_src = src[face_mask] - node_offset
        local_dst = dst[face_mask] - node_offset
        face_diff = pred_delta_per_area[local_src] - pred_delta_per_area[local_dst]

        weights = face_length[face_mask].to(pred.dtype)
        if zone_mode == "zone_weight" and hasattr(graph, "zone_weight"):
            zone_weight = graph.zone_weight.to(pred.device)
            edge_zone_weight = torch.maximum(zone_weight[src[face_mask]], zone_weight[dst[face_mask]])
            weights = weights * edge_zone_weight.to(pred.dtype)
        elif zone_mode == "high" and hasattr(graph, "zone_label"):
            zone_label = graph.zone_label.to(pred.device)
            high_mask = (zone_label[src[face_mask]] == 3) | (zone_label[dst[face_mask]] == 3)
            weights = weights * high_mask.to(pred.dtype)
        elif zone_mode != "all":
            weights = weights * 0.0 + 1.0

        if torch.sum(weights) > 0:
            losses.append(torch.sum(weights * face_diff**2) / torch.sum(weights))
        else:
            losses.append(torch.mean(face_diff**2))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=pred.device)


def compute_zone_metrics(pred, target, graph, num_zones=4):
    """Return per-zone RMSE metrics for water depth and volume differences."""
    if not hasattr(graph, "zone_label"):
        return {}

    metrics = {}
    labels = graph.zone_label.to(pred.device).view(-1)
    err_sq = (pred - target) ** 2
    for zone in range(num_zones):
        mask = labels == zone
        if torch.any(mask):
            zone_err = err_sq[mask]
            rmse = torch.sqrt(torch.mean(zone_err))
            wd_rmse = torch.sqrt(torch.mean(zone_err[:, 0]))
            volume_rmse = torch.sqrt(torch.mean(zone_err[:, 1]))
            metrics[f"zone_{zone}_rmse"] = rmse
            metrics[f"zone_{zone}_wd_rmse"] = wd_rmse
            metrics[f"zone_{zone}_volume_rmse"] = volume_rmse
    return metrics


def custom_loss(pred, targets):
    """
    Compute a custom loss as the sum of MSE losses on water depth and volume predictions.

    Args:
        pred (torch.Tensor): Model predictions with two columns (depth and volume difference).
        targets (torch.Tensor): Ground truth targets.

    Returns:
        dict: Dictionary containing the total loss and individual losses for depth and volume.
    """
    pred_depth = pred[:, 0]
    pred_volume = pred[:, 1]
    target_depth = targets[:, 0]
    target_volume = targets[:, 1]
    loss_depth = F.mse_loss(pred_depth, target_depth, reduction="mean")
    loss_volume = F.mse_loss(pred_volume, target_volume, reduction="mean")
    total_loss = loss_depth + loss_volume
    return {
        "total_loss": total_loss,
        "loss_depth": loss_depth,
        "loss_volume": loss_volume,
    }
