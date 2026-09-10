#!/usr/bin/env python3
"""Training objective for selective exact edge-local conservation.

The projection itself never consumes the event Face Flow target. HEC-RAS Face
Flow is used only after projection to supervise the physically feasible flux.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from selective_hard_conservation import (
    SelectiveProjectionResult,
    project_connected_control_volume_flux,
)
from utils import compute_hecras_edge_flux_head_loss


@dataclass(frozen=True)
class SelectiveHardConservationLossResult:
    """Loss terms and projected edge flux used by joint node-edge training."""

    loss: torch.Tensor
    projected_face_loss: torch.Tensor
    raw_control_volume_loss: torch.Tensor
    projected_edge_flux: torch.Tensor
    projection: SelectiveProjectionResult


def face_flux_divergence(
    edge_flux: torch.Tensor,
    face_index: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Return signed per-node divergence for HDF-oriented internal faces."""

    flux = edge_flux.reshape(-1)
    faces = face_index.to(flux.device).long()
    if faces.shape != (2, flux.numel()):
        raise ValueError("face_index must have shape [2, num_faces].")
    src, dst = faces
    divergence = torch.zeros(num_nodes, dtype=flux.dtype, device=flux.device)
    divergence.index_add_(0, src, -flux)
    divergence.index_add_(0, dst, flux)
    return divergence


def projected_volume_feedback_delta(
    pred: torch.Tensor,
    projection: SelectiveProjectionResult,
    graph,
    *,
    delta_t: float,
    alpha: float = 1.0,
) -> torch.Tensor:
    """Build a normalized volume increment for selective rollout feedback.

    Only nodes inside the declared high-fidelity interior control volumes use
    the edge-divergence update. Every other node retains HydroGraphNet's own
    volume increment. This operation is differentiable with respect to both
    the node prediction and the projected edge flux.
    """

    if not 0.0 <= alpha <= 1.0:
        raise ValueError("alpha must be between 0 and 1.")
    if delta_t <= 0.0:
        raise ValueError("delta_t must be positive.")
    if pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError("pred must have shape [num_nodes, 2].")
    required = (
        "hecras_face_index",
        "hecras_high_interior_control_volume_label",
        "volume_std",
    )
    missing = [name for name in required if not hasattr(graph, name)]
    if missing:
        raise AttributeError(
            "Projected rollout feedback requires graph attributes: "
            + ", ".join(missing)
        )

    divergence = face_flux_divergence(
        projection.edge_flux,
        graph.hecras_face_index,
        pred.shape[0],
    )
    source_rate = getattr(graph, "local_source_rate", None)
    source_delta = (
        torch.zeros_like(divergence)
        if source_rate is None
        else source_rate.to(pred.device, pred.dtype).reshape(-1) * float(delta_t)
    )
    physical_delta = divergence + source_delta

    volume_std = graph.volume_std.to(pred.device, pred.dtype).reshape(-1)
    batch = getattr(graph, "batch", None)
    if batch is None:
        if volume_std.numel() != 1:
            raise ValueError("An unbatched graph must contain one volume_std value.")
        normalized_projected_delta = physical_delta / volume_std[0]
    else:
        batch = batch.to(pred.device).long().reshape(-1)
        if batch.shape[0] != pred.shape[0]:
            raise ValueError("graph.batch must align with pred.")
        normalized_projected_delta = physical_delta / volume_std[batch]

    node_delta = pred[:, 1]
    blended = node_delta + float(alpha) * (normalized_projected_delta - node_delta)
    selected = (
        graph.hecras_high_interior_control_volume_label.to(pred.device).reshape(-1)
        >= 0
    )
    return torch.where(selected, blended, node_delta)


def update_hgn_rollout_state(
    x: torch.Tensor,
    pred: torch.Tensor,
    *,
    n_time_steps: int,
    next_inflow: torch.Tensor,
    next_precipitation: torch.Tensor,
    volume_delta_override: torch.Tensor | None = None,
) -> torch.Tensor:
    """Advance the normalized HydroGraphNet state without breaking gradients."""

    expected_features = 12 + 2 * int(n_time_steps)
    if x.ndim != 2 or x.shape[1] != expected_features:
        raise ValueError(
            f"Expected x shape [num_nodes, {expected_features}], got {tuple(x.shape)}."
        )
    if pred.shape != (x.shape[0], 2):
        raise ValueError("pred must align with x and contain depth/volume increments.")
    if n_time_steps < 1:
        raise ValueError("n_time_steps must be positive.")

    static = x[:, :12].clone()
    depth_window = x[:, 12 : 12 + n_time_steps]
    volume_window = x[:, 12 + n_time_steps : expected_features]
    volume_delta = (
        pred[:, 1]
        if volume_delta_override is None
        else volume_delta_override.reshape(-1).to(pred.device, pred.dtype)
    )
    next_depth = depth_window[:, -1] + pred[:, 0]
    next_volume = volume_window[:, -1] + volume_delta
    depth_window = torch.cat((depth_window[:, 1:], next_depth[:, None]), dim=1)
    volume_window = torch.cat((volume_window[:, 1:], next_volume[:, None]), dim=1)
    static[:, 10] = next_inflow.reshape(-1)[0].to(x.device, x.dtype)
    static[:, 11] = next_precipitation.reshape(-1)[0].to(x.device, x.dtype)
    return torch.cat((static, depth_window, volume_window), dim=1)


def refresh_rollout_graph_state(
    graph,
    x: torch.Tensor,
    local_source_rate: torch.Tensor,
    *,
    n_time_steps: int,
) -> None:
    """Refresh edge-head inputs that vary during a differentiable rollout."""

    graph.x = x
    graph.local_source_rate = local_source_rate.reshape(-1).to(x.device, x.dtype)
    if hasattr(graph, "current_surface_elevation") and hasattr(
        graph, "current_water_depth_denorm"
    ):
        terrain = (
            graph.current_surface_elevation.to(x.device)
            - graph.current_water_depth_denorm.to(x.device)
        )
        depth_mean = graph.water_depth_mean.reshape(-1)[0].to(x.device, x.dtype)
        depth_std = graph.water_depth_std.reshape(-1)[0].to(x.device, x.dtype)
        current_depth = x[:, 12 + n_time_steps - 1] * depth_std + depth_mean
        graph.current_water_depth = x[:, 12 + n_time_steps - 1]
        graph.current_water_depth_denorm = current_depth
        graph.current_surface_elevation = terrain + current_depth


def _dataset_storage_delta(pred: torch.Tensor, graph) -> torch.Tensor:
    """Convert normalized HGN volume output to one interval of physical volume."""

    if not hasattr(graph, "volume_std"):
        raise AttributeError("Selective hard conservation requires graph.volume_std.")
    volume_std = graph.volume_std.to(pred.device, pred.dtype).reshape(-1)
    batch = getattr(graph, "batch", None)
    if batch is None:
        if volume_std.numel() != 1:
            raise ValueError("An unbatched graph must contain one volume_std value.")
        scale = volume_std[0]
    else:
        batch = batch.to(pred.device).long().reshape(-1)
        if batch.numel() != pred.shape[0]:
            raise ValueError("graph.batch must align with the node prediction.")
        if batch.numel() and int(batch.max().detach().item()) >= volume_std.numel():
            raise ValueError(
                "graph.volume_std does not cover every graph in the batch."
            )
        scale = volume_std[batch]
    return pred[:, 1] * scale


def project_training_edge_flux(
    pred: torch.Tensor,
    raw_edge_flux: torch.Tensor,
    graph,
    *,
    delta_t: float,
    correction_weight_mode: str = "training_face_rms_squared",
) -> SelectiveProjectionResult:
    """Project raw flux onto exact connected Zone-4 control-volume budgets.

    The target is ``predicted storage change - known local source``. Formal
    control-volume labels exclude every external-domain boundary node, so the
    projection uses only quantities available at operational inference time.
    HEC-RAS boundary and Face Flow targets remain supervision/diagnostics and
    never enter the projection constraint.
    """

    required = (
        "hecras_face_index",
        "hecras_high_interior_control_volume_label",
        "hecras_internal_face_flow_rms",
    )
    missing = [name for name in required if not hasattr(graph, name)]
    if missing:
        raise AttributeError(
            "Selective hard conservation requires graph attributes: "
            + ", ".join(missing)
        )
    if pred.ndim != 2 or pred.shape[1] != 2:
        raise ValueError("pred must have shape [num_nodes, 2].")
    if delta_t <= 0:
        raise ValueError("delta_t must be positive.")

    storage_delta = _dataset_storage_delta(pred, graph)
    source_rate = getattr(graph, "local_source_rate", None)
    if source_rate is None:
        source_delta = torch.zeros_like(storage_delta)
    else:
        source_delta = (
            source_rate.to(pred.device, pred.dtype).reshape(-1) * float(delta_t)
        )
    if source_delta.shape != storage_delta.shape:
        raise ValueError("Node source and storage delta must align.")
    divergence_target = storage_delta - source_delta

    face_index = graph.hecras_face_index.to(pred.device).long()
    labels = graph.hecras_high_interior_control_volume_label.to(pred.device).reshape(
        -1
    )
    src, dst = face_index
    active_face_mask = (labels[src] >= 0) | (labels[dst] >= 0)
    face_rms = graph.hecras_internal_face_flow_rms.to(
        pred.device, pred.dtype
    ).reshape(-1)
    if correction_weight_mode == "training_face_rms_squared":
        correction_weight = face_rms.square()
    elif correction_weight_mode == "equal":
        correction_weight = torch.ones_like(face_rms)
    else:
        raise ValueError(
            "correction_weight_mode must be 'training_face_rms_squared' or "
            f"'equal', got {correction_weight_mode!r}."
        )

    return project_connected_control_volume_flux(
        raw_edge_flux,
        divergence_target,
        face_index,
        labels,
        node_batch=getattr(graph, "batch", None),
        active_face_mask=active_face_mask,
        face_correction_weight=correction_weight,
        ridge=0.0,
    )


def compute_selective_hard_conservation_loss(
    pred: torch.Tensor,
    raw_edge_flux: torch.Tensor,
    graph,
    *,
    delta_t: float,
    raw_control_volume_weight: float,
    correction_weight_mode: str = "training_face_rms_squared",
    face_loss_normalization: str = "asinh_per_face_rms",
    node_loss_normalization: str = "volume_std",
) -> SelectiveHardConservationLossResult:
    """Supervise feasible Face Flow and softly reduce pre-projection residual.

    ``raw_control_volume_weight`` is the only method-specific loss trade-off:
    the hard layer guarantees projected conservation, while this term teaches
    the unprojected node-edge prediction to require progressively less repair.
    """

    if raw_control_volume_weight < 0:
        raise ValueError("raw_control_volume_weight must be nonnegative.")
    projection = project_training_edge_flux(
        pred,
        raw_edge_flux,
        graph,
        delta_t=delta_t,
        correction_weight_mode=correction_weight_mode,
    )
    _, projected_metrics = compute_hecras_edge_flux_head_loss(
        pred,
        projection.edge_flux,
        graph,
        zone_mode="high_interior",
        closure_target_weight=0.0,
        divergence_target_weight=0.0,
        face_target_weight=1.0,
        face_zone_mode="high_interior_touch",
        face_loss_normalization=face_loss_normalization,
        node_loss_normalization=node_loss_normalization,
        storage_delta_mode="dataset_volume",
        closure_granularity="connected_control_volume",
        delta_t=delta_t,
    )
    _, raw_metrics = compute_hecras_edge_flux_head_loss(
        pred,
        raw_edge_flux,
        graph,
        zone_mode="high_interior",
        closure_target_weight=1.0,
        divergence_target_weight=0.0,
        face_target_weight=0.0,
        face_zone_mode="high_interior_touch",
        face_loss_normalization=face_loss_normalization,
        node_loss_normalization=node_loss_normalization,
        storage_delta_mode="dataset_volume",
        closure_granularity="connected_control_volume",
        delta_t=delta_t,
    )
    face_loss = projected_metrics["hecras_edge_flux_face_loss"]
    raw_control_volume_loss = raw_metrics["hecras_edge_flux_closure_loss"]
    total = face_loss + float(raw_control_volume_weight) * raw_control_volume_loss
    return SelectiveHardConservationLossResult(
        loss=total,
        projected_face_loss=face_loss,
        raw_control_volume_loss=raw_control_volume_loss,
        projected_edge_flux=projection.edge_flux,
        projection=projection,
    )
