# SPDX-FileCopyrightText: Copyright (c) 2023 - 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Sparse HEC-RAS edge-flux projection helpers.

These helpers implement a least-change projection from predicted internal
face-flow deltas to node-local divergence targets.  They are intentionally
kept separate from the diagnostic scripts so the same projection can later be
used by rollout evaluation or a differentiable training variant.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import torch
from dataclasses import dataclass


def selected_node_mask(graph, mode: str, device: torch.device) -> torch.Tensor:
    """Return the node mask used by a conservation projection."""

    if mode in ("all", "zone_weighted"):
        return torch.ones(graph.x.shape[0], dtype=torch.bool, device=device)
    if mode == "interior":
        if not hasattr(graph, "hecras_boundary_node_mask"):
            return torch.ones(graph.x.shape[0], dtype=torch.bool, device=device)
        return ~graph.hecras_boundary_node_mask.to(device)

    if not hasattr(graph, "zone_label"):
        raise AttributeError(f"projection_mode={mode!r} requires graph.zone_label")

    zone_label = graph.zone_label.to(device)
    if mode == "high":
        return zone_label == 3
    if mode == "high_interior":
        if not hasattr(graph, "hecras_boundary_node_mask"):
            return zone_label == 3
        return (zone_label == 3) & (~graph.hecras_boundary_node_mask.to(device))
    raise ValueError(f"Unknown projection mode: {mode!r}")


def selected_node_weights(
    graph,
    mode: str,
    mask: torch.Tensor,
    device: torch.device,
    high_weight: float = 1.0,
    low_weight: float = 1.0,
) -> torch.Tensor:
    """Return node residual weights for the selected conservation equations."""

    weights = torch.ones(graph.x.shape[0], dtype=torch.float64, device=device)
    if mode == "zone_weighted":
        if not hasattr(graph, "zone_label"):
            raise AttributeError("projection_mode='zone_weighted' requires graph.zone_label")
        zone_label = graph.zone_label.to(device)
        weights = torch.full_like(weights, float(low_weight))
        weights = torch.where(
            zone_label == 3,
            torch.full_like(weights, float(high_weight)),
            weights,
        )
    return weights[mask]


def build_selected_incidence(graph, mask: torch.Tensor) -> sp.csr_matrix:
    """Build the selected HEC-RAS face-to-cell incidence matrix."""

    face_index = graph.hecras_face_index.detach().cpu().numpy()
    src = face_index[0]
    dst = face_index[1]
    selected = np.flatnonzero(mask.detach().cpu().numpy())
    node_to_row = -np.ones(graph.x.shape[0], dtype=np.int64)
    node_to_row[selected] = np.arange(selected.shape[0], dtype=np.int64)

    src_rows = node_to_row[src]
    dst_rows = node_to_row[dst]
    valid_src = src_rows >= 0
    valid_dst = dst_rows >= 0

    rows = np.concatenate([src_rows[valid_src], dst_rows[valid_dst]])
    cols = np.concatenate(
        [np.flatnonzero(valid_src), np.flatnonzero(valid_dst)]
    ).astype(np.int64)
    data = np.concatenate(
        [
            -np.ones(np.count_nonzero(valid_src), dtype=np.float64),
            np.ones(np.count_nonzero(valid_dst), dtype=np.float64),
        ]
    )
    return sp.csr_matrix(
        (data, (rows, cols)),
        shape=(selected.shape[0], graph.hecras_face_index.shape[1]),
    )


@dataclass
class ProjectionSolver:
    """Reusable sparse projection system for one graph/mode/weight setting."""

    weighted_incidence: sp.csr_matrix
    selected: np.ndarray
    sqrt_weights: np.ndarray
    solve_fn: object
    solver: str

    def solve(self, residual: np.ndarray, cg_rtol: float, cg_maxiter: int):
        if self.solver == "factorized":
            return self.solve_fn(residual), 0
        y, info = spla.cg(
            self.solve_fn,
            residual,
            rtol=cg_rtol,
            maxiter=cg_maxiter,
        )
        if info != 0:
            y = spla.spsolve(self.solve_fn.tocsc(), residual)
        return y, int(info)


def build_projection_solver(
    graph,
    mode: str,
    device: torch.device,
    ridge: float,
    high_weight: float = 1.0,
    low_weight: float = 1.0,
    solver: str = "cg",
) -> ProjectionSolver | None:
    """Build a reusable projection solver for a fixed graph and projection mode."""

    mask = selected_node_mask(graph, mode, device)
    if not torch.any(mask):
        return None

    incidence = build_selected_incidence(graph, mask)
    weights = selected_node_weights(
        graph,
        mode,
        mask,
        device,
        high_weight=high_weight,
        low_weight=low_weight,
    )
    selected = mask.detach().cpu().numpy()
    sqrt_weights = np.sqrt(weights.detach().cpu().double().numpy().reshape(-1))
    weighted_incidence = sp.diags(sqrt_weights, format="csr") @ incidence
    lhs = weighted_incidence @ weighted_incidence.T
    if ridge > 0:
        lhs = lhs + ridge * sp.eye(lhs.shape[0], format="csr")

    if solver == "factorized":
        solve_fn = spla.factorized(lhs.tocsc())
    elif solver == "cg":
        solve_fn = lhs.tocsr()
    else:
        raise ValueError(f"Unknown projection solver: {solver!r}")

    return ProjectionSolver(
        weighted_incidence=weighted_incidence,
        selected=selected,
        sqrt_weights=sqrt_weights,
        solve_fn=solve_fn,
        solver=solver,
    )


def project_edge_flux(
    graph,
    edge_flux: torch.Tensor,
    divergence_target: torch.Tensor,
    mode: str,
    ridge: float,
    cg_rtol: float = 1e-8,
    cg_maxiter: int = 2000,
    high_weight: float = 1.0,
    low_weight: float = 1.0,
    projection_solver: ProjectionSolver | None = None,
) -> tuple[torch.Tensor, int, float]:
    """Project edge flux toward selected node divergence targets.

    The projected flux solves:

    ``q' = q - A^T (A A^T + ridge I)^-1 (A q - d)``.

    Returns the projected tensor, the conjugate-gradient status code, and the
    RMS magnitude of the applied correction in original edge-flow units.
    """

    device = edge_flux.device
    cached_solver = projection_solver
    if cached_solver is None:
        cached_solver = build_projection_solver(
            graph,
            mode,
            device,
            ridge,
            high_weight=high_weight,
            low_weight=low_weight,
            solver="cg",
        )
    if cached_solver is None:
        return edge_flux, 0, 0.0

    q0 = edge_flux.detach().cpu().double().numpy().reshape(-1)
    target = divergence_target.detach().cpu().double().numpy().reshape(-1)
    residual = (
        cached_solver.weighted_incidence @ q0
        - cached_solver.sqrt_weights * target[cached_solver.selected]
    )
    if residual.size == 0:
        return edge_flux, 0, 0.0

    y, info = cached_solver.solve(residual, cg_rtol, cg_maxiter)
    correction = cached_solver.weighted_incidence.T @ y
    projected = q0 - correction
    projected_tensor = torch.as_tensor(projected, dtype=edge_flux.dtype, device=device)
    correction_rms = float(np.sqrt(np.mean(correction**2)))
    return projected_tensor, int(info), correction_rms
