#!/usr/bin/env python3
"""Differentiable exact conservation on selected connected control volumes."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class SelectiveProjectionResult:
    """Projected flux and diagnostics for a connected-control-volume layer."""

    edge_flux: torch.Tensor
    face_correction: torch.Tensor
    raw_component_residual: torch.Tensor
    projected_component_residual: torch.Tensor
    face_component: torch.Tensor
    face_sign: torch.Tensor
    num_components: int


def _component_topology(
    face_index: torch.Tensor,
    control_volume_label: torch.Tensor,
    node_batch: torch.Tensor | None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Map every control-volume boundary face to one compact component."""

    if face_index.ndim != 2 or face_index.shape[0] != 2:
        raise ValueError("face_index must have shape [2, num_faces].")
    labels = control_volume_label.reshape(-1).long()
    if node_batch is None:
        batch = torch.zeros_like(labels)
    else:
        batch = node_batch.reshape(-1).long().to(labels.device)
        if batch.shape != labels.shape:
            raise ValueError("node_batch and control_volume_label must align.")
        if torch.any(batch < 0):
            raise ValueError("node_batch values must be nonnegative.")
    selected = labels >= 0
    if not torch.any(selected):
        raise ValueError("At least one selected control-volume node is required.")

    stride = int(labels[selected].max().detach().item()) + 1
    component_keys = batch[selected] * stride + labels[selected]
    _, compact = torch.unique(component_keys, sorted=True, return_inverse=True)
    node_component = torch.full_like(labels, -1)
    node_component[selected] = compact
    num_components = int(compact.max().detach().item()) + 1

    src, dst = face_index.long().to(labels.device)
    if src.numel() and (
        int(torch.min(torch.stack((src.min(), dst.min()))).detach().item()) < 0
        or int(torch.max(torch.stack((src.max(), dst.max()))).detach().item())
        >= labels.numel()
    ):
        raise ValueError("face_index contains an out-of-range node index.")
    src_component = node_component[src]
    dst_component = node_component[dst]
    if torch.any(batch[src] != batch[dst]):
        raise ValueError("A physical face cannot connect nodes from different graphs.")
    cross_component = (
        (src_component >= 0)
        & (dst_component >= 0)
        & (src_component != dst_component)
    )
    if torch.any(cross_component):
        raise ValueError(
            "A face connects two distinct selected components; component labels "
            "are inconsistent with the physical face graph."
        )

    src_boundary = (src_component >= 0) & (dst_component < 0)
    dst_boundary = (dst_component >= 0) & (src_component < 0)
    face_component = torch.full_like(src_component, -1)
    face_component[src_boundary] = src_component[src_boundary]
    face_component[dst_boundary] = dst_component[dst_boundary]
    face_sign = torch.zeros(
        src.shape, dtype=torch.get_default_dtype(), device=labels.device
    )
    face_sign[src_boundary] = -1.0
    face_sign[dst_boundary] = 1.0
    return node_component, face_component, face_sign, num_components


def project_connected_control_volume_flux(
    edge_flux: torch.Tensor,
    divergence_target: torch.Tensor,
    face_index: torch.Tensor,
    control_volume_label: torch.Tensor,
    *,
    node_batch: torch.Tensor | None = None,
    active_face_mask: torch.Tensor | None = None,
    face_correction_weight: torch.Tensor | None = None,
    ridge: float = 0.0,
) -> SelectiveProjectionResult:
    """Apply a weighted minimum-change correction that closes each component.

    The constraint is ``C A q = C d``, where ``A`` is the signed face-to-cell
    incidence matrix, ``C`` sums cells in each selected connected control
    volume, and ``d`` is the inference-available divergence target (predicted
    storage change minus known local source). With ``ridge=0`` the constraints
    are exact to floating-point precision. The operation stays in PyTorch and
    preserves gradients to both the raw edge flux and the node-derived target.
    """

    if ridge < 0.0:
        raise ValueError("ridge must be nonnegative.")
    original_shape = edge_flux.shape
    if not torch.is_floating_point(edge_flux):
        raise ValueError("edge_flux must use a floating-point dtype.")
    output_dtype = edge_flux.dtype
    compute_dtype = (
        torch.float64 if output_dtype == torch.float64 else torch.float32
    )
    flux = edge_flux.reshape(-1).to(compute_dtype)
    target = divergence_target.reshape(-1).to(flux.device, flux.dtype)
    labels = control_volume_label.reshape(-1).to(flux.device)
    faces = face_index.to(flux.device)
    if flux.numel() != faces.shape[1]:
        raise ValueError("edge_flux and face_index must contain the same faces.")
    if target.numel() != labels.numel():
        raise ValueError("divergence_target and control_volume_label must align.")

    node_component, face_component, face_sign, num_components = _component_topology(
        faces,
        labels,
        None if node_batch is None else node_batch.to(flux.device),
    )
    face_sign = face_sign.to(flux.dtype)
    adjustable = face_component >= 0
    if active_face_mask is not None:
        active = active_face_mask.reshape(-1).to(flux.device, torch.bool)
        if active.numel() != flux.numel():
            raise ValueError("active_face_mask and edge_flux must align.")
        if torch.any(adjustable & (~active)):
            raise ValueError(
                "active_face_mask excludes a selected control-volume boundary face."
            )
        adjustable = adjustable & active

    if face_correction_weight is None:
        correction_weight = torch.ones_like(flux)
    else:
        correction_weight = face_correction_weight.reshape(-1).to(
            flux.device, flux.dtype
        )
        if correction_weight.numel() != flux.numel():
            raise ValueError("face_correction_weight and edge_flux must align.")
        if torch.any(~torch.isfinite(correction_weight)) or torch.any(
            correction_weight[adjustable] < 0
        ):
            raise ValueError(
                "face_correction_weight must be finite and nonnegative on every "
                "adjustable boundary face. Each component must retain positive "
                "total weight."
            )

    selected_nodes = node_component >= 0
    component_target = torch.zeros(
        num_components, dtype=flux.dtype, device=flux.device
    )
    component_target.index_add_(
        0, node_component[selected_nodes], target[selected_nodes]
    )
    component_divergence = torch.zeros_like(component_target)
    component_divergence.index_add_(
        0,
        face_component[adjustable],
        face_sign[adjustable] * flux[adjustable],
    )
    raw_residual = component_divergence - component_target

    denominator = torch.zeros_like(component_target)
    denominator.index_add_(
        0,
        face_component[adjustable],
        face_sign[adjustable].square() * correction_weight[adjustable],
    )
    if torch.any(denominator <= 0):
        missing = torch.nonzero(denominator <= 0, as_tuple=False).reshape(-1)
        raise ValueError(
            "Selected control volumes without adjustable boundary faces: "
            f"{missing.detach().cpu().tolist()}."
        )

    correction = torch.zeros_like(flux)
    correction[adjustable] = (
        correction_weight[adjustable]
        * face_sign[adjustable]
        * raw_residual[face_component[adjustable]]
        / (denominator[face_component[adjustable]] + float(ridge))
    )
    projected = flux - correction
    projected_divergence = torch.zeros_like(component_target)
    projected_divergence.index_add_(
        0,
        face_component[adjustable],
        face_sign[adjustable] * projected[adjustable],
    )
    projected_residual = projected_divergence - component_target
    return SelectiveProjectionResult(
        edge_flux=projected.to(output_dtype).reshape(original_shape),
        face_correction=correction.to(output_dtype).reshape(original_shape),
        raw_component_residual=raw_residual,
        projected_component_residual=projected_residual,
        face_component=face_component,
        face_sign=face_sign,
        num_components=num_components,
    )
