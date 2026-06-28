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
    """Compute a selective calibrated face-velocity proxy loss.

    This branch is intentionally separate from the kNN/VX/VY proxy loss. It uses
    HEC-RAS internal face connectivity, face length, and optional HDF face
    velocity. Because ``velocity * face_length`` is not a volume transfer,
    this remains a prototype objective rather than formal local conservation.
    Keep it disabled for formal claims until the uncalibrated physical budget
    validator passes on synchronized event HDF and HGN targets.
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


def compute_hecras_cell_balance_loss(pred, graph, zone_mode="zone_weight"):
    """Compute a formal HEC-RAS local storage-budget loss.

    This objective compares the model's denormalized per-cell volume delta to
    the native HEC-RAS ``Cell Flow Balance`` integrated over the matching HGN
    interval plus native precipitation volume. It is separate from face-velocity
    proxy losses and does not apply fitted scale calibration.
    """
    required_attrs = ("hecras_cell_balance_delta", "volume_std")
    if not all(hasattr(graph, attr) for attr in required_attrs):
        return torch.tensor(0.0, device=pred.device)

    budget_delta = graph.hecras_cell_balance_delta.to(pred.device).reshape(-1)
    batch = getattr(graph, "batch", None)
    unique_ids = torch.unique(batch) if batch is not None else [None]
    losses = []
    for local_idx, uid in enumerate(unique_ids):
        if uid is None:
            node_mask = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
        else:
            node_mask = batch == uid

        volume_std = graph.volume_std.reshape(-1)[local_idx].to(pred.device)
        pred_delta = pred[node_mask, 1] * volume_std
        residual = pred_delta - budget_delta[node_mask].to(pred.dtype)

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


def compute_hecras_edge_flow_loss(pred, graph, zone_mode="zone_weight"):
    """Compute an edge-informed HEC-RAS local storage-budget loss.

    This objective compares the model's denormalized per-cell volume delta to a
    precomputed target reconstructed from native HEC-RAS ``Face Flow`` over the
    matching HGN interval.  The target is loaded separately from the
    Cell-Flow-Balance branch so edge-flow ablations stay independent.
    """
    has_combined_target = hasattr(graph, "hecras_edge_flow_delta")
    has_split_target = hasattr(graph, "hecras_edge_internal_delta") and hasattr(
        graph, "hecras_edge_boundary_source_delta"
    )
    if not (has_combined_target or has_split_target) or not hasattr(graph, "volume_std"):
        return torch.tensor(0.0, device=pred.device)

    edge_delta = (
        graph.hecras_edge_flow_delta.to(pred.device).reshape(-1)
        if has_combined_target
        else None
    )
    internal_delta = (
        graph.hecras_edge_internal_delta.to(pred.device).reshape(-1)
        if has_split_target
        else None
    )
    boundary_delta = (
        graph.hecras_edge_boundary_source_delta.to(pred.device).reshape(-1)
        if has_split_target
        else None
    )
    batch = getattr(graph, "batch", None)
    unique_ids = torch.unique(batch) if batch is not None else [None]
    losses = []
    for local_idx, uid in enumerate(unique_ids):
        if uid is None:
            node_mask = torch.ones(pred.shape[0], dtype=torch.bool, device=pred.device)
        else:
            node_mask = batch == uid

        volume_std = graph.volume_std.reshape(-1)[local_idx].to(pred.device)
        pred_delta = pred[node_mask, 1] * volume_std
        if internal_delta is not None and boundary_delta is not None:
            residual = (
                pred_delta
                - boundary_delta[node_mask].to(pred.dtype)
                - internal_delta[node_mask].to(pred.dtype)
            )
        else:
            residual = pred_delta - edge_delta[node_mask].to(pred.dtype)

        weights = None
        if zone_mode == "zone_weight" and hasattr(graph, "zone_weight"):
            weights = graph.zone_weight[node_mask].to(pred.device)
        elif zone_mode == "high" and hasattr(graph, "zone_label"):
            weights = (graph.zone_label[node_mask].to(pred.device) == 3).to(pred.dtype)
        elif (
            zone_mode == "high_interior"
            and hasattr(graph, "zone_label")
            and hasattr(graph, "hecras_boundary_node_mask")
        ):
            high_mask = graph.zone_label[node_mask].to(pred.device) == 3
            interior_mask = ~graph.hecras_boundary_node_mask[node_mask].to(pred.device)
            weights = (high_mask & interior_mask).to(pred.dtype)
        elif zone_mode == "all":
            weights = torch.ones_like(residual)

        if weights is not None and torch.sum(weights) > 0:
            losses.append(torch.sum(weights * residual**2) / torch.sum(weights))
        else:
            losses.append(torch.mean(residual**2))

    return torch.stack(losses).mean() if losses else torch.tensor(0.0, device=pred.device)


def compute_hecras_edge_flux_head_loss(
    pred,
    edge_flux_delta,
    graph,
    zone_mode="zone_weight",
    zone_high_weight=1.0,
    zone_low_weight=1.0,
    closure_target_weight=1.0,
    divergence_target_weight=1.0,
    face_target_weight=0.0,
    face_loss_normalization="none",
):
    """Compute a model-side internal edge-flux local-conservation loss.

    ``edge_flux_delta`` is the model-predicted signed interval volume flux on
    the HEC-RAS internal face graph. Its divergence is compared with the native
    internal Face Flow target, while the node volume prediction is closed with
    the explicit boundary/source residual.
    """
    required_attrs = (
        "hecras_face_index",
        "hecras_edge_internal_delta",
        "hecras_edge_boundary_source_delta",
        "volume_std",
    )
    if not all(hasattr(graph, attr) for attr in required_attrs):
        zero = torch.tensor(0.0, device=pred.device)
        return zero, {"hecras_edge_flux_divergence_loss": zero, "hecras_edge_flux_closure_loss": zero}

    face_index = graph.hecras_face_index.to(pred.device)
    if edge_flux_delta.numel() == 0 or face_index.numel() == 0:
        zero = torch.tensor(0.0, device=pred.device)
        return zero, {"hecras_edge_flux_divergence_loss": zero, "hecras_edge_flux_closure_loss": zero}

    src, dst = face_index
    internal_delta = graph.hecras_edge_internal_delta.to(pred.device).reshape(-1)
    boundary_delta = graph.hecras_edge_boundary_source_delta.to(pred.device).reshape(-1)

    batch = getattr(graph, "batch", None)
    unique_ids = torch.unique(batch) if batch is not None else [None]
    divergence_losses = []
    closure_losses = []
    face_losses = []
    raw_face_losses = []
    edge_flux_delta = edge_flux_delta.reshape(-1).to(pred.device)

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

        local_src = src[face_mask] - node_offset
        local_dst = dst[face_mask] - node_offset
        local_flux = edge_flux_delta[face_mask].to(pred.dtype)
        divergence = torch.zeros(
            int(torch.sum(node_mask).detach().item()), device=pred.device, dtype=pred.dtype
        )
        divergence.index_add_(0, local_src, -local_flux)
        divergence.index_add_(0, local_dst, local_flux)

        volume_std = graph.volume_std.reshape(-1)[local_idx].to(pred.device)
        pred_delta = pred[node_mask, 1] * volume_std
        internal_target = internal_delta[node_mask].to(pred.dtype)
        boundary_target = boundary_delta[node_mask].to(pred.dtype)

        divergence_residual = divergence - internal_target
        closure_residual = pred_delta - boundary_target - divergence

        weights = None
        if zone_mode == "zone_weight" and hasattr(graph, "zone_weight"):
            weights = graph.zone_weight[node_mask].to(pred.device)
        elif zone_mode == "zone_weighted" and hasattr(graph, "zone_label"):
            labels = graph.zone_label[node_mask].to(pred.device)
            weights = torch.full_like(
                divergence_residual,
                float(zone_low_weight),
                dtype=pred.dtype,
                device=pred.device,
            )
            weights = torch.where(
                labels == 3,
                torch.full_like(weights, float(zone_high_weight)),
                weights,
            )
        elif zone_mode == "high" and hasattr(graph, "zone_label"):
            weights = (graph.zone_label[node_mask].to(pred.device) == 3).to(pred.dtype)
        elif (
            zone_mode == "high_interior"
            and hasattr(graph, "zone_label")
            and hasattr(graph, "hecras_boundary_node_mask")
        ):
            high_mask = graph.zone_label[node_mask].to(pred.device) == 3
            interior_mask = ~graph.hecras_boundary_node_mask[node_mask].to(pred.device)
            weights = (high_mask & interior_mask).to(pred.dtype)
        elif zone_mode == "all":
            weights = torch.ones_like(divergence_residual)

        if weights is not None and torch.sum(weights) > 0:
            divergence_losses.append(
                torch.sum(weights * divergence_residual**2) / torch.sum(weights)
            )
            closure_losses.append(
                torch.sum(weights * closure_residual**2) / torch.sum(weights)
            )
        else:
            divergence_losses.append(torch.mean(divergence_residual**2))
            closure_losses.append(torch.mean(closure_residual**2))

        if face_target_weight > 0 and hasattr(graph, "hecras_internal_face_flow_delta"):
            face_target = graph.hecras_internal_face_flow_delta.to(pred.device).reshape(-1)
            face_residual = edge_flux_delta[face_mask].to(pred.dtype) - face_target[
                face_mask
            ].to(pred.dtype)
            face_weights = None
            if zone_mode == "zone_weight" and hasattr(graph, "zone_weight"):
                node_weights = graph.zone_weight.to(pred.device)
                face_weights = 0.5 * (
                    node_weights[src[face_mask]] + node_weights[dst[face_mask]]
                )
            elif zone_mode == "zone_weighted" and hasattr(graph, "zone_label"):
                labels = graph.zone_label.to(pred.device)
                src_high = labels[src[face_mask]] == 3
                dst_high = labels[dst[face_mask]] == 3
                src_weights = torch.where(
                    src_high,
                    torch.full_like(face_residual, float(zone_high_weight)),
                    torch.full_like(face_residual, float(zone_low_weight)),
                )
                dst_weights = torch.where(
                    dst_high,
                    torch.full_like(face_residual, float(zone_high_weight)),
                    torch.full_like(face_residual, float(zone_low_weight)),
                )
                face_weights = 0.5 * (src_weights + dst_weights)
            elif zone_mode == "high" and hasattr(graph, "zone_label"):
                labels = graph.zone_label.to(pred.device)
                face_weights = (
                    (labels[src[face_mask]] == 3) | (labels[dst[face_mask]] == 3)
                ).to(pred.dtype)
            elif zone_mode == "high_interior" and hasattr(graph, "zone_label"):
                labels = graph.zone_label.to(pred.device)
                face_weights = (
                    (labels[src[face_mask]] == 3) & (labels[dst[face_mask]] == 3)
                ).to(pred.dtype)
            elif zone_mode == "all":
                face_weights = torch.ones_like(face_residual)

            local_face_target = face_target[face_mask].to(pred.dtype)
            raw_face_residual = face_residual
            if face_weights is not None and torch.sum(face_weights) > 0:
                target_rms_sq = (
                    torch.sum(face_weights * local_face_target**2)
                    / torch.sum(face_weights)
                )
                raw_face_loss = (
                    torch.sum(face_weights * raw_face_residual**2)
                    / torch.sum(face_weights)
                )
            else:
                target_rms_sq = torch.mean(local_face_target**2)
                raw_face_loss = torch.mean(raw_face_residual**2)
            target_rms = torch.sqrt(torch.clamp(target_rms_sq, min=1e-12))
            scale_mode = face_loss_normalization
            for prefix in ("asinh_", "signed_log1p_"):
                if scale_mode.startswith(prefix):
                    scale_mode = scale_mode[len(prefix) :]
                    break
            scale = torch.clamp(target_rms, min=1.0)
            if scale_mode in ("per_face_rms", "face_event_rms", "face_transition_rms"):
                if not hasattr(graph, "hecras_internal_face_flow_rms"):
                    raise AttributeError(
                        f"face_loss_normalization={face_loss_normalization!r} "
                        "requires graph.hecras_internal_face_flow_rms."
                    )
                face_scale = graph.hecras_internal_face_flow_rms.to(pred.device).reshape(-1)[
                    face_mask
                ].to(pred.dtype)
                face_scale = torch.clamp(face_scale, min=1.0)
                if scale_mode == "per_face_rms":
                    scale = face_scale
                else:
                    if not hasattr(graph, "hecras_internal_face_flow_global_rms"):
                        raise AttributeError(
                            f"face_loss_normalization={face_loss_normalization!r} "
                            "requires graph.hecras_internal_face_flow_global_rms."
                        )
                    global_scale = torch.clamp(
                        graph.hecras_internal_face_flow_global_rms.to(
                            pred.device
                        ).reshape(-1)[0],
                        min=1.0,
                    ).to(pred.dtype)
                    if scale_mode == "face_event_rms":
                        if not hasattr(graph, "hecras_internal_face_flow_event_rms"):
                            raise AttributeError(
                                f"face_loss_normalization={face_loss_normalization!r} "
                                "requires graph.hecras_internal_face_flow_event_rms."
                            )
                        event_scale = torch.clamp(
                            graph.hecras_internal_face_flow_event_rms.to(
                                pred.device
                            ).reshape(-1)[0],
                            min=1.0,
                        ).to(pred.dtype)
                        scale = face_scale * event_scale / global_scale
                    else:
                        if not hasattr(
                            graph, "hecras_internal_face_flow_transition_rms"
                        ):
                            raise AttributeError(
                                f"face_loss_normalization={face_loss_normalization!r} "
                                "requires "
                                "graph.hecras_internal_face_flow_transition_rms."
                            )
                        transition_scale = torch.clamp(
                            graph.hecras_internal_face_flow_transition_rms.to(
                                pred.device
                            ).reshape(-1)[0],
                            min=1.0,
                        ).to(pred.dtype)
                        scale = face_scale * transition_scale / global_scale
            elif scale_mode in ("event_rms", "transition_rms"):
                attr = (
                    "hecras_internal_face_flow_event_rms"
                    if scale_mode == "event_rms"
                    else "hecras_internal_face_flow_transition_rms"
                )
                if not hasattr(graph, attr):
                    raise AttributeError(
                        f"face_loss_normalization={face_loss_normalization!r} "
                        f"requires graph.{attr}."
                    )
                scale = torch.clamp(
                    getattr(graph, attr).to(pred.device).reshape(-1)[0],
                    min=1.0,
                ).to(pred.dtype)
            if face_loss_normalization != "none" and scale_mode in (
                "target_rms",
                "per_face_rms",
                "event_rms",
                "transition_rms",
                "face_event_rms",
                "face_transition_rms",
            ):
                scaled_prediction = edge_flux_delta[face_mask].to(pred.dtype) / scale
                scaled_target = local_face_target / scale
                if face_loss_normalization.startswith("asinh_"):
                    face_residual = torch.asinh(scaled_prediction) - torch.asinh(
                        scaled_target
                    )
                elif face_loss_normalization.startswith("signed_log1p_"):
                    face_residual = torch.sign(scaled_prediction) * torch.log1p(
                        torch.abs(scaled_prediction)
                    ) - torch.sign(scaled_target) * torch.log1p(
                        torch.abs(scaled_target)
                    )
                else:
                    face_residual = scaled_prediction - scaled_target
            if face_weights is not None and torch.sum(face_weights) > 0:
                face_loss = (
                    torch.sum(face_weights * face_residual**2) / torch.sum(face_weights)
                )
            else:
                face_loss = torch.mean(face_residual**2)
            raw_face_losses.append(raw_face_loss)
            if face_loss_normalization == "target_rms":
                face_loss = face_loss / torch.clamp(target_rms_sq, min=1e-12)
            elif face_loss_normalization not in (
                "none",
                "asinh_target_rms",
                "signed_log1p_target_rms",
                "per_face_rms",
                "asinh_per_face_rms",
                "signed_log1p_per_face_rms",
                "event_rms",
                "asinh_event_rms",
                "signed_log1p_event_rms",
                "transition_rms",
                "asinh_transition_rms",
                "signed_log1p_transition_rms",
                "face_event_rms",
                "asinh_face_event_rms",
                "signed_log1p_face_event_rms",
                "face_transition_rms",
                "asinh_face_transition_rms",
                "signed_log1p_face_transition_rms",
            ):
                raise ValueError(
                    "Unknown face_loss_normalization: "
                    f"{face_loss_normalization!r}."
                )
            face_losses.append(face_loss)

    if not divergence_losses:
        zero = torch.tensor(0.0, device=pred.device)
        return zero, {"hecras_edge_flux_divergence_loss": zero, "hecras_edge_flux_closure_loss": zero}

    divergence_loss = torch.stack(divergence_losses).mean()
    closure_loss = torch.stack(closure_losses).mean()
    if face_losses:
        face_loss = torch.stack(face_losses).mean()
        raw_face_loss = torch.stack(raw_face_losses).mean()
    else:
        face_loss = torch.tensor(0.0, device=pred.device)
        raw_face_loss = torch.tensor(0.0, device=pred.device)
    total = (
        closure_target_weight * closure_loss
        + divergence_target_weight * divergence_loss
        + face_target_weight * face_loss
    )
    return total, {
        "hecras_edge_flux_divergence_loss": divergence_loss,
        "hecras_edge_flux_closure_loss": closure_loss,
        "hecras_edge_flux_face_loss": face_loss,
        "hecras_edge_flux_raw_face_loss": raw_face_loss,
    }


def compute_hecras_face_geometry_loss(
    pred,
    graph,
    target=None,
    delta_t=1200.0,
    zone_mode="zone_weight",
    wet_depth_threshold=None,
    reference_mode="smooth",
):
    """Compute geometry-only smoothness on HEC-RAS true internal faces.

    This loss does not use event-specific HDF face velocity. It regularizes the
    predicted volume delta per cell area across true HEC-RAS internal faces,
    weighted by face length and optionally by fidelity-zone weights. The
    default reference mode is smoothing. The target-gradient mode instead
    matches the target cross-face volume-delta gradient, which avoids forcing
    all neighboring cells toward the same update. Source-aware modes subtract
    the node-wise precipitation/IP source term before comparing cross-face
    gradients, which keeps rainfall and infiltration from being regularized as
    if they were transport. When wet_depth_threshold is set, only faces touching
    currently wet cells are regularized.
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
        pred_delta = pred[node_mask, 1] * volume_std
        target_delta = (
            target[node_mask, 1].to(pred.device) * volume_std
            if target is not None
            else None
        )
        if reference_mode.startswith("source_") and hasattr(graph, "local_source_rate"):
            source_delta = (
                graph.local_source_rate[node_mask].to(pred.device) * delta_t
            )
            pred_delta = pred_delta - source_delta
            if target_delta is not None:
                target_delta = target_delta - source_delta

        pred_delta_per_area = pred_delta / node_area[node_mask]
        local_src = src[face_mask] - node_offset
        local_dst = dst[face_mask] - node_offset
        pred_face_diff = (
            pred_delta_per_area[local_src] - pred_delta_per_area[local_dst]
        )
        if (
            reference_mode in ("target_gradient", "source_target_gradient")
            and target_delta is not None
        ):
            target_delta_per_area = target_delta / node_area[node_mask]
            target_face_diff = (
                target_delta_per_area[local_src] - target_delta_per_area[local_dst]
            ).detach()
            face_residual = pred_face_diff - target_face_diff
        else:
            face_residual = pred_face_diff

        weights = face_length[face_mask].to(pred.dtype)
        if zone_mode == "zone_weight" and hasattr(graph, "zone_weight"):
            zone_weight = graph.zone_weight.to(pred.device)
            edge_zone_weight = torch.maximum(
                zone_weight[src[face_mask]], zone_weight[dst[face_mask]]
            )
            weights = weights * edge_zone_weight.to(pred.dtype)
        elif zone_mode in ("high", "high_adjacent") and hasattr(graph, "zone_label"):
            zone_label = graph.zone_label.to(pred.device)
            high_mask = (zone_label[src[face_mask]] == 3) | (
                zone_label[dst[face_mask]] == 3
            )
            weights = weights * high_mask.to(pred.dtype)
        elif zone_mode != "all":
            weights = weights * 0.0 + 1.0

        if wet_depth_threshold is not None and hasattr(graph, "current_water_depth"):
            current_wd = graph.current_water_depth.to(pred.device).reshape(-1)
            if hasattr(graph, "water_depth_mean") and hasattr(graph, "water_depth_std"):
                wd_mean = graph.water_depth_mean.reshape(-1)[local_idx].to(pred.device)
                wd_std = graph.water_depth_std.reshape(-1)[local_idx].to(pred.device)
                current_wd = current_wd * wd_std + wd_mean
            edge_wet = torch.maximum(
                current_wd[src[face_mask]], current_wd[dst[face_mask]]
            ) > wet_depth_threshold
            weights = weights * edge_wet.to(pred.dtype)

        if torch.sum(weights) > 0:
            losses.append(torch.sum(weights * face_residual**2) / torch.sum(weights))
        elif wet_depth_threshold is None and zone_mode not in ("high", "high_adjacent"):
            losses.append(torch.mean(face_residual**2))

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
