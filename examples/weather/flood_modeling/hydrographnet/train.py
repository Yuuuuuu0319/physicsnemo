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

import time
import random

import hydra
import numpy as np
import torch
import torch.nn as nn
import torch_geometric as pyg
import wandb

from hydra.utils import to_absolute_path
from omegaconf import DictConfig

from torch_geometric.loader import DataLoader as PyGDataLoader

from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data.distributed import DistributedSampler

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset
from physicsnemo.distributed.manager import DistributedManager
from physicsnemo.utils.logging import PythonLogger, RankZeroLoggingWrapper
from physicsnemo.utils.logging.wandb import initialize_wandb
from physicsnemo.utils import load_checkpoint, save_checkpoint
from physicsnemo.models.meshgraphnet.meshgraphkan import MeshGraphKAN
from selective_hard_conservation_training import (
    compute_selective_hard_conservation_loss,
    projected_volume_feedback_delta,
    refresh_rollout_graph_state,
    update_hgn_rollout_state,
)
from utils import (
    compute_edge_local_proxy_loss,
    compute_hecras_cell_balance_loss,
    compute_hecras_edge_flow_loss,
    compute_hecras_edge_flux_head_loss,
    compute_hecras_face_geometry_loss,
    compute_hecras_face_local_loss,
    compute_physics_loss,
    compute_zone_metrics,
    compute_zone_weighted_loss,
)


# Custom collate function that checks if each item is a tuple (graph, physics_data) or a plain graph.
def collate_fn(batch):
    if isinstance(batch[0], tuple):
        graphs, physics_list = zip(*batch)
        batched_graph = pyg.data.from_data_list(graphs)
        physics_data = {}
        # For each key, build a tensor by stacking the scalar values from each sample.
        for key in physics_list[0].keys():
            physics_data[key] = torch.tensor(
                [d[key] for d in physics_list], dtype=torch.float
            )
        return batched_graph, physics_data
    else:
        return pyg.data.from_data_list(batch)


class HecRasEdgeFluxHead(nn.Module):
    """Small experiment-local head for signed internal HEC-RAS face flux."""

    def __init__(
        self,
        num_node_features: int,
        hidden_dim: int = 128,
        num_hidden_layers: int = 2,
        scale: float = 1.0,
        use_face_normal: bool = False,
        use_surface_features: bool = False,
        use_physical_surface_features: bool = False,
        use_previous_face_flow: bool = False,
        use_edge_physical_features: bool = False,
        use_node_prediction_features: bool = False,
        output_mode: str = "raw",
        active_face_mode: str = "all",
    ):
        super().__init__()
        self.scale = scale
        self.num_hidden_layers = num_hidden_layers
        self.use_face_normal = use_face_normal
        self.use_surface_features = use_surface_features
        self.use_physical_surface_features = use_physical_surface_features
        self.use_previous_face_flow = use_previous_face_flow
        self.use_edge_physical_features = use_edge_physical_features
        self.use_node_prediction_features = use_node_prediction_features
        self.output_mode = output_mode
        self.active_face_mode = active_face_mode
        allowed_active_face_modes = {
            "all",
            "zone4_touch",
            "high_interior_touch",
        }
        if active_face_mode not in allowed_active_face_modes:
            raise ValueError(
                "active_face_mode must be one of "
                f"{sorted(allowed_active_face_modes)}, got {active_face_mode!r}."
            )
        allowed_output_modes = {
            "raw",
            "per_face_rms",
            "event_rms",
            "transition_rms",
            "face_event_rms",
            "face_transition_rms",
            "asinh_per_face_rms",
            "asinh_event_rms",
            "asinh_transition_rms",
            "asinh_face_event_rms",
            "asinh_face_transition_rms",
            "signed_log1p_per_face_rms",
            "signed_log1p_event_rms",
            "signed_log1p_transition_rms",
            "signed_log1p_face_event_rms",
            "signed_log1p_face_transition_rms",
        }
        if output_mode not in allowed_output_modes:
            raise ValueError(
                f"output_mode must be one of {sorted(allowed_output_modes)}, "
                f"got {output_mode!r}."
            )
        input_dim = (
            num_node_features * 2
            + 1
            + (2 if use_face_normal else 0)
            + (4 if use_surface_features else 0)
            + (4 if use_physical_surface_features else 0)
            + (2 if use_previous_face_flow else 0)
            + (12 if use_edge_physical_features else 0)
            + (4 if use_node_prediction_features else 0)
        )
        if num_hidden_layers < 1:
            raise ValueError("num_hidden_layers must be at least 1.")
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(num_hidden_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, 1))
        self.net = nn.Sequential(*layers)

    @staticmethod
    def _group_index_and_count(graph, item_index=None):
        """Return graph IDs so feature scaling is invariant to batch makeup."""
        batch = getattr(graph, "batch", None)
        if batch is None:
            return None, 1
        group_index = batch if item_index is None else batch[item_index]
        ptr = getattr(graph, "ptr", None)
        if ptr is not None:
            num_groups = int(ptr.numel() - 1)
        else:
            num_groups = int(batch.max().detach().item()) + 1
        return group_index.to(graph.x.device), num_groups

    @staticmethod
    def _group_rms_scale(values, group_index, num_groups, minimum):
        detached = values.detach()
        if group_index is None:
            return torch.clamp(
                torch.sqrt(torch.mean(detached.square(), dim=0, keepdim=True)),
                min=minimum,
            )
        squared_sum = torch.zeros(
            (num_groups, values.shape[1]),
            dtype=values.dtype,
            device=values.device,
        )
        counts = torch.zeros(
            (num_groups, 1), dtype=values.dtype, device=values.device
        )
        squared_sum.index_add_(0, group_index, detached.square())
        counts.index_add_(
            0,
            group_index,
            torch.ones((values.shape[0], 1), dtype=values.dtype, device=values.device),
        )
        scale = torch.sqrt(squared_sum / torch.clamp(counts, min=1.0))
        return torch.clamp(scale[group_index], min=minimum)

    @staticmethod
    def _group_mean_scale(values, group_index, num_groups, minimum):
        detached = values.detach()
        if group_index is None:
            return torch.clamp(detached.mean(dim=0, keepdim=True), min=minimum)
        value_sum = torch.zeros(
            (num_groups, values.shape[1]),
            dtype=values.dtype,
            device=values.device,
        )
        counts = torch.zeros(
            (num_groups, 1), dtype=values.dtype, device=values.device
        )
        value_sum.index_add_(0, group_index, detached)
        counts.index_add_(
            0,
            group_index,
            torch.ones((values.shape[0], 1), dtype=values.dtype, device=values.device),
        )
        scale = value_sum / torch.clamp(counts, min=1.0)
        return torch.clamp(scale[group_index], min=minimum)

    @staticmethod
    def _group_standardize(values, group_index, num_groups, minimum=1e-6):
        detached = values.detach()
        if group_index is None:
            mean = detached.mean(dim=0, keepdim=True)
            std = detached.std(dim=0, unbiased=False, keepdim=True)
            return (values - mean) / torch.clamp(std, min=minimum)
        value_sum = torch.zeros(
            (num_groups, values.shape[1]),
            dtype=values.dtype,
            device=values.device,
        )
        squared_sum = torch.zeros_like(value_sum)
        counts = torch.zeros(
            (num_groups, 1), dtype=values.dtype, device=values.device
        )
        value_sum.index_add_(0, group_index, detached)
        squared_sum.index_add_(0, group_index, detached.square())
        counts.index_add_(
            0,
            group_index,
            torch.ones((values.shape[0], 1), dtype=values.dtype, device=values.device),
        )
        counts = torch.clamp(counts, min=1.0)
        mean = value_sum / counts
        variance = torch.clamp(squared_sum / counts - mean.square(), min=0.0)
        std = torch.sqrt(variance)
        return (values - mean[group_index]) / torch.clamp(
            std[group_index], min=minimum
        )

    def active_face_mask(self, graph) -> torch.Tensor:
        """Select only faces needed by the declared conservation scope."""
        src, dst = graph.hecras_face_index
        if self.active_face_mode == "all":
            return torch.ones(src.shape[0], dtype=torch.bool, device=src.device)
        if self.active_face_mode == "zone4_touch":
            if not hasattr(graph, "zone_label"):
                raise AttributeError(
                    "active_face_mode='zone4_touch' requires graph.zone_label."
                )
            labels = graph.zone_label.to(src.device)
            return (labels[src] == 3) | (labels[dst] == 3)
        if self.active_face_mode == "high_interior_touch":
            if not hasattr(graph, "hecras_high_interior_control_volume_label"):
                raise AttributeError(
                    "active_face_mode='high_interior_touch' requires "
                    "graph.hecras_high_interior_control_volume_label."
                )
            labels = graph.hecras_high_interior_control_volume_label.to(src.device)
            return (labels[src] >= 0) | (labels[dst] >= 0)
        raise RuntimeError(f"Unhandled active_face_mode: {self.active_face_mode!r}")

    def forward(self, graph, node_prediction=None):
        face_index = graph.hecras_face_index
        full_src, full_dst = face_index
        if full_src.numel() == 0:
            return graph.x.new_zeros((0,))
        active_mask = self.active_face_mask(graph)
        active_indices = torch.nonzero(active_mask, as_tuple=False).reshape(-1)
        if active_indices.numel() == 0:
            return graph.x.new_zeros((full_src.shape[0],))
        src = full_src[active_mask]
        dst = full_dst[active_mask]
        face_length = graph.hecras_face_length.to(graph.x.device).reshape(-1, 1)[
            active_mask
        ]
        face_group, num_groups = self._group_index_and_count(graph, src)
        node_group, _ = self._group_index_and_count(graph)
        face_length_scale = self._group_mean_scale(
            face_length, face_group, num_groups, minimum=1.0
        )
        features = [graph.x[src], graph.x[dst], face_length / face_length_scale]
        if self.use_node_prediction_features:
            if node_prediction is None:
                raise ValueError(
                    "HecRasEdgeFluxHead(use_node_prediction_features=True) "
                    "requires the HydroGraphNet node prediction."
                )
            if node_prediction.shape != (graph.x.shape[0], 2):
                raise ValueError(
                    "node_prediction must have shape (num_nodes, 2), got "
                    f"{tuple(node_prediction.shape)}."
                )
            features.extend([node_prediction[src], node_prediction[dst]])
        if self.use_face_normal:
            if not hasattr(graph, "hecras_face_normal"):
                raise AttributeError(
                    "HecRasEdgeFluxHead(use_face_normal=True) requires "
                    "graph.hecras_face_normal."
                )
            features.append(
                graph.hecras_face_normal.to(graph.x.device)[active_mask]
            )
        if self.use_surface_features:
            latest_wd = graph.x[:, 13:14]
            latest_volume = graph.x[:, 15:16]
            surface_proxy = graph.x[:, 3:4] + latest_wd
            source_rate = getattr(
                graph,
                "local_source_rate",
                torch.zeros(graph.x.shape[0], device=graph.x.device),
            )
            source_rate = source_rate.to(graph.x.device).reshape(-1, 1)
            source_scale = self._group_rms_scale(
                source_rate, node_group, num_groups, minimum=1.0
            )
            source_rate = source_rate / source_scale
            features.extend(
                [
                    latest_wd[dst] - latest_wd[src],
                    latest_volume[dst] - latest_volume[src],
                    surface_proxy[dst] - surface_proxy[src],
                    source_rate[dst] - source_rate[src],
                ]
            )
        if self.use_physical_surface_features:
            required = ("current_surface_elevation", "current_water_depth_denorm")
            if not all(hasattr(graph, attr) for attr in required):
                raise AttributeError(
                    "HecRasEdgeFluxHead(use_physical_surface_features=True) "
                    "requires graph.current_surface_elevation and "
                    "graph.current_water_depth_denorm."
                )
            surface = graph.current_surface_elevation.to(graph.x.device).reshape(-1, 1)
            depth = graph.current_water_depth_denorm.to(graph.x.device).reshape(-1, 1)
            length_scale = torch.clamp(face_length, min=1.0)
            surface_slope = (surface[dst] - surface[src]) / length_scale
            depth_slope = (depth[dst] - depth[src]) / length_scale
            surface_slope = surface_slope / self._group_rms_scale(
                surface_slope, face_group, num_groups, minimum=1e-6
            )
            depth_slope = depth_slope / self._group_rms_scale(
                depth_slope, face_group, num_groups, minimum=1e-6
            )
            features.extend(
                [
                    surface_slope,
                    depth_slope,
                    (depth[src] > 1e-6).to(graph.x.dtype),
                    (depth[dst] > 1e-6).to(graph.x.dtype),
                ]
            )
        if self.use_previous_face_flow:
            if not hasattr(graph, "hecras_previous_internal_face_flow_delta"):
                raise AttributeError(
                    "HecRasEdgeFluxHead(use_previous_face_flow=True) requires "
                    "graph.hecras_previous_internal_face_flow_delta."
                )
            previous_flow = graph.hecras_previous_internal_face_flow_delta.to(
                graph.x.device
            ).reshape(-1, 1)[active_mask]
            if hasattr(graph, "hecras_internal_face_flow_rms"):
                previous_scale = torch.clamp(
                    graph.hecras_internal_face_flow_rms.to(graph.x.device).reshape(
                        -1, 1
                    )[active_mask],
                    min=1.0,
                )
            else:
                previous_scale = torch.clamp(
                    torch.sqrt(torch.mean(previous_flow.detach() ** 2)), min=1.0
                )
            previous_scaled = previous_flow / previous_scale
            features.extend([previous_scaled, torch.asinh(previous_scaled)])
        if self.use_edge_physical_features:
            if not hasattr(graph, "hecras_edge_physical_features"):
                raise AttributeError(
                    "HecRasEdgeFluxHead(use_edge_physical_features=True) "
                    "requires graph.hecras_edge_physical_features."
                )
            physical_features = graph.hecras_edge_physical_features.to(
                graph.x.device
            )[active_mask]
            if physical_features.shape[1] != 12:
                raise ValueError(
                    "graph.hecras_edge_physical_features must have 12 columns, "
                    f"got {physical_features.shape}."
                )
            features.append(
                self._group_standardize(
                    physical_features, face_group, num_groups
                )
            )
        head_input = torch.cat(features, dim=1)
        output = self.net(head_input).reshape(-1) * self.scale
        if self.output_mode == "raw":
            active_output = output
        else:
            face_scale = self._resolve_output_scale(graph, output, active_mask)
            active_output = output * face_scale
        if self.output_mode != "raw" and (
            self.output_mode.startswith("asinh_")
            or self.output_mode.startswith("signed_log1p_")
        ):
            if self.output_mode.startswith("asinh_"):
                active_output = (
                    torch.sinh(torch.clamp(output, min=-20.0, max=20.0))
                    * face_scale
                )
            else:
                active_output = (
                    torch.sign(output) * torch.expm1(torch.abs(output)) * face_scale
                )
        full_output = output.new_zeros((full_src.shape[0],))
        return full_output.index_copy(0, active_indices, active_output)

    def _resolve_output_scale(
        self, graph, output: torch.Tensor, active_mask=None
    ) -> torch.Tensor:
        mode = self.output_mode
        for prefix in ("asinh_", "signed_log1p_"):
            if mode.startswith(prefix):
                mode = mode[len(prefix) :]
                break
        device = graph.x.device
        dtype = output.dtype
        face_scale = None
        if mode in {"per_face_rms", "face_event_rms", "face_transition_rms"}:
            if not hasattr(graph, "hecras_internal_face_flow_rms"):
                raise AttributeError(
                    f"HecRasEdgeFluxHead(output_mode={self.output_mode!r}) requires "
                    "graph.hecras_internal_face_flow_rms."
                )
            face_scale = torch.clamp(
                graph.hecras_internal_face_flow_rms.to(device).reshape(-1),
                min=1.0,
            ).to(dtype)
            if active_mask is not None:
                face_scale = face_scale[active_mask]
        if mode == "per_face_rms":
            return face_scale
        if mode in {"event_rms", "face_event_rms"}:
            if not hasattr(graph, "hecras_internal_face_flow_event_rms"):
                raise AttributeError(
                    f"HecRasEdgeFluxHead(output_mode={self.output_mode!r}) requires "
                    "graph.hecras_internal_face_flow_event_rms."
                )
            event_scale = torch.clamp(
                graph.hecras_internal_face_flow_event_rms.to(device).reshape(-1)[0],
                min=1.0,
            ).to(dtype)
            if mode == "event_rms":
                return torch.ones_like(output) * event_scale
            global_scale = self._global_face_flow_scale(graph, output)
            return face_scale * event_scale / global_scale
        if mode in {"transition_rms", "face_transition_rms"}:
            if not hasattr(graph, "hecras_internal_face_flow_transition_rms"):
                raise AttributeError(
                    f"HecRasEdgeFluxHead(output_mode={self.output_mode!r}) requires "
                    "graph.hecras_internal_face_flow_transition_rms."
                )
            transition_scale = torch.clamp(
                graph.hecras_internal_face_flow_transition_rms.to(device).reshape(-1)[
                    0
                ],
                min=1.0,
            ).to(dtype)
            if mode == "transition_rms":
                return torch.ones_like(output) * transition_scale
            global_scale = self._global_face_flow_scale(graph, output)
            return face_scale * transition_scale / global_scale
        raise RuntimeError(f"Unhandled output scale mode: {self.output_mode!r}")

    def _global_face_flow_scale(self, graph, output: torch.Tensor) -> torch.Tensor:
        if not hasattr(graph, "hecras_internal_face_flow_global_rms"):
            raise AttributeError(
                f"HecRasEdgeFluxHead(output_mode={self.output_mode!r}) requires "
                "graph.hecras_internal_face_flow_global_rms."
            )
        return torch.clamp(
            graph.hecras_internal_face_flow_global_rms.to(graph.x.device).reshape(-1)[0],
            min=1.0,
        ).to(output.dtype)


class MGNTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        # Ensure distributed manager is initialized.
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp = cfg.amp
        self.noise_type = cfg.noise_type
        self.n_time_steps = int(cfg.n_time_steps)

        # Physics loss settings.
        self.use_physics_loss = cfg.get("use_physics_loss", False)
        self.delta_t = cfg.get("delta_t", 1200.0)
        self.physics_loss_weight = cfg.get("physics_loss_weight", 1.0)
        self.use_fidelity_zones = cfg.get("use_fidelity_zones", False)
        self.zone_loss_weight = cfg.get("zone_loss_weight", 0.0)
        self.log_zone_metrics = cfg.get("log_zone_metrics", False)
        self.use_edge_local_proxy = cfg.get("use_edge_local_proxy", False)
        self.edge_local_loss_weight = cfg.get("edge_local_loss_weight", 0.0)
        self.use_hecras_face_loss = cfg.get("use_hecras_face_loss", False)
        self.hecras_face_loss_weight = cfg.get("hecras_face_loss_weight", 0.0)
        self.hecras_face_zone_mode = cfg.get("hecras_face_zone_mode", "zone_weight")
        self.hecras_face_calibrate_to_target = cfg.get(
            "hecras_face_calibrate_to_target", True
        )
        self.use_hecras_face_geometry_loss = cfg.get(
            "use_hecras_face_geometry_loss", False
        )
        self.hecras_face_geometry_loss_weight = cfg.get(
            "hecras_face_geometry_loss_weight", 0.0
        )
        self.hecras_face_geometry_zone_mode = cfg.get(
            "hecras_face_geometry_zone_mode", "zone_weight"
        )
        self.hecras_face_geometry_wet_depth_threshold = cfg.get(
            "hecras_face_geometry_wet_depth_threshold", None
        )
        self.hecras_face_geometry_reference_mode = cfg.get(
            "hecras_face_geometry_reference_mode", "smooth"
        )
        self.use_hecras_cell_balance_loss = cfg.get(
            "use_hecras_cell_balance_loss", False
        )
        self.hecras_cell_balance_loss_weight = cfg.get(
            "hecras_cell_balance_loss_weight", 0.0
        )
        self.hecras_cell_balance_zone_mode = cfg.get(
            "hecras_cell_balance_zone_mode", "zone_weight"
        )
        self.hecras_cell_balance_loss_normalization = cfg.get(
            "hecras_cell_balance_loss_normalization", "none"
        )
        self.use_hecras_edge_flow_loss = cfg.get(
            "use_hecras_edge_flow_loss", False
        )
        self.hecras_edge_flow_loss_weight = cfg.get(
            "hecras_edge_flow_loss_weight", 0.0
        )
        self.hecras_edge_flow_zone_mode = cfg.get(
            "hecras_edge_flow_zone_mode", "zone_weight"
        )
        self.use_hecras_edge_flux_head = cfg.get(
            "use_hecras_edge_flux_head", False
        )
        self.hecras_edge_flux_head_loss_weight = cfg.get(
            "hecras_edge_flux_head_loss_weight", 0.0
        )
        self.hecras_edge_flux_head_zone_mode = cfg.get(
            "hecras_edge_flux_head_zone_mode", "zone_weight"
        )
        self.hecras_edge_flux_head_zone_high_weight = cfg.get(
            "hecras_edge_flux_head_zone_high_weight", 1.0
        )
        self.hecras_edge_flux_head_zone_low_weight = cfg.get(
            "hecras_edge_flux_head_zone_low_weight", 1.0
        )
        self.hecras_edge_flux_head_closure_target_weight = cfg.get(
            "hecras_edge_flux_head_closure_target_weight", 1.0
        )
        self.hecras_edge_flux_head_divergence_target_weight = cfg.get(
            "hecras_edge_flux_head_divergence_target_weight", 1.0
        )
        self.hecras_edge_flux_head_face_target_weight = cfg.get(
            "hecras_edge_flux_head_face_target_weight", 0.0
        )
        self.hecras_edge_flux_head_face_zone_mode = cfg.get(
            "hecras_edge_flux_head_face_zone_mode",
            self.hecras_edge_flux_head_zone_mode,
        )
        self.hecras_edge_flux_head_face_loss_normalization = cfg.get(
            "hecras_edge_flux_head_face_loss_normalization", "none"
        )
        self.hecras_edge_flux_head_node_loss_normalization = cfg.get(
            "hecras_edge_flux_head_node_loss_normalization", "none"
        )
        self.hecras_edge_flux_head_storage_delta_mode = cfg.get(
            "hecras_edge_flux_head_storage_delta_mode", "dataset_volume"
        )
        self.hecras_edge_flux_head_closure_granularity = cfg.get(
            "hecras_edge_flux_head_closure_granularity", "node"
        )
        self.hecras_edge_flux_head_hidden_dim = cfg.get(
            "hecras_edge_flux_head_hidden_dim", 128
        )
        self.hecras_edge_flux_head_scale = cfg.get(
            "hecras_edge_flux_head_scale", 1.0
        )
        self.hecras_edge_flux_head_use_face_normal = cfg.get(
            "hecras_edge_flux_head_use_face_normal", False
        )
        self.hecras_edge_flux_head_use_surface_features = cfg.get(
            "hecras_edge_flux_head_use_surface_features", False
        )
        self.hecras_edge_flux_head_use_physical_surface_features = cfg.get(
            "hecras_edge_flux_head_use_physical_surface_features", False
        )
        self.hecras_edge_flux_head_use_previous_face_flow = cfg.get(
            "hecras_edge_flux_head_use_previous_face_flow", False
        )
        self.hecras_edge_flux_head_use_edge_physical_features = cfg.get(
            "hecras_edge_flux_head_use_edge_physical_features", False
        )
        self.hecras_edge_flux_head_use_node_prediction_features = cfg.get(
            "hecras_edge_flux_head_use_node_prediction_features", False
        )
        self.hecras_edge_flux_head_output_mode = cfg.get(
            "hecras_edge_flux_head_output_mode", "raw"
        )
        self.hecras_edge_flux_head_active_face_mode = cfg.get(
            "hecras_edge_flux_head_active_face_mode", "all"
        )
        self.use_selective_hard_conservation = cfg.get(
            "use_selective_hard_conservation", False
        )
        self.selective_hard_conservation_loss_weight = cfg.get(
            "selective_hard_conservation_loss_weight", 0.0
        )
        self.selective_hard_conservation_raw_control_volume_weight = cfg.get(
            "selective_hard_conservation_raw_control_volume_weight", 0.0
        )
        self.selective_hard_conservation_correction_weight_mode = cfg.get(
            "selective_hard_conservation_correction_weight_mode",
            "training_face_rms_squared",
        )
        self.selective_hard_conservation_face_loss_normalization = cfg.get(
            "selective_hard_conservation_face_loss_normalization",
            "asinh_per_face_rms",
        )
        self.selective_hard_conservation_node_loss_normalization = cfg.get(
            "selective_hard_conservation_node_loss_normalization", "volume_std"
        )
        self.training_rollout_steps = int(cfg.get("training_rollout_steps", 1))
        self.training_rollout_discount = float(
            cfg.get("training_rollout_discount", 1.0)
        )
        self.training_rollout_backprop_through_time = bool(
            cfg.get("training_rollout_backprop_through_time", True)
        )
        self.training_rollout_volume_feedback = cfg.get(
            "training_rollout_volume_feedback", "node"
        )
        self.training_rollout_projected_volume_alpha = float(
            cfg.get("training_rollout_projected_volume_alpha", 1.0)
        )
        if self.training_rollout_steps < 1:
            raise ValueError("training_rollout_steps must be positive.")
        if not 0.0 < self.training_rollout_discount <= 1.0:
            raise ValueError("training_rollout_discount must be in (0, 1].")
        if self.training_rollout_volume_feedback not in {
            "node",
            "selective_projected",
        }:
            raise ValueError(
                "training_rollout_volume_feedback must be 'node' or "
                "'selective_projected'."
            )
        if not 0.0 <= self.training_rollout_projected_volume_alpha <= 1.0:
            raise ValueError(
                "training_rollout_projected_volume_alpha must be in [0, 1]."
            )
        if self.training_rollout_steps > 1:
            if cfg.batch_size != 1:
                raise ValueError(
                    "Differentiable training rollouts currently require batch_size=1."
                )
            if self.noise_type == "pushforward":
                raise ValueError(
                    "training_rollout_steps>1 cannot use legacy pushforward noise."
                )
            unsupported = {
                "edge_local_proxy": self.use_edge_local_proxy,
                "legacy_face": self.use_hecras_face_loss,
                "face_geometry": self.use_hecras_face_geometry_loss,
                "cell_balance": self.use_hecras_cell_balance_loss,
                "legacy_edge_flow": self.use_hecras_edge_flow_loss,
                "soft_edge_flux": self.hecras_edge_flux_head_loss_weight > 0,
            }
            enabled = [name for name, active in unsupported.items() if active]
            if enabled:
                raise ValueError(
                    "Differentiable training rollouts intentionally support only "
                    "the HGN objective and selective hard edge objective; disable: "
                    + ", ".join(enabled)
                )
            if (
                self.training_rollout_volume_feedback == "selective_projected"
                and not self.use_selective_hard_conservation
            ):
                raise ValueError(
                    "selective_projected training feedback requires "
                    "use_selective_hard_conservation=true."
                )

        # Set activation function.
        mlp_act = "relu"
        if cfg.recompute_activation:
            rank_zero_logger.info(
                "Setting MLP activation to SiLU for recompute_activation."
            )
            mlp_act = "silu"

        rank_zero_logger.info("Initializing HydroGraphDataset...")
        # Pass the flag to the dataset so it returns physics data only if needed.
        dataset = HydroGraphDataset(
            name="hydrograph_dataset",
            data_dir=cfg.data_dir,
            prefix="M80",
            num_samples=cfg.num_training_samples,
            n_time_steps=cfg.n_time_steps,
            k=4,
            noise_type=cfg.noise_type,
            noise_std=0.01,
            hydrograph_ids_file=cfg.get("hydrograph_ids_file", "train.txt"),
            split="train",
            return_physics=self.use_physics_loss,
            use_fidelity_zones=self.use_fidelity_zones,
            zone_label_file=cfg.get("zone_label_file", "zone_label.txt"),
            zone_weight_file=cfg.get("zone_weight_file", "zone_weight.txt"),
            return_edge_local=self.use_edge_local_proxy,
            return_hecras_face=(
                self.use_hecras_face_loss or self.use_hecras_face_geometry_loss
                or self.use_hecras_edge_flux_head
            ),
            hecras_face_graph_file=cfg.get("hecras_face_graph_file"),
            hecras_face_velocity_file=cfg.get("hecras_face_velocity_file"),
            hecras_face_velocity_glob=cfg.get("hecras_face_velocity_glob"),
            hecras_face_velocity_path=cfg.get(
                "hecras_face_velocity_path",
                (
                    "Results/Unsteady/Output/Output Blocks/Base Output/"
                    "Unsteady Time Series/2D Flow Areas/per2/Face Velocity"
                ),
            ),
            hecras_face_time_offset=cfg.get("hecras_face_time_offset", 0),
            return_hecras_cell_balance=self.use_hecras_cell_balance_loss,
            hecras_cell_balance_glob=cfg.get("hecras_cell_balance_glob"),
            hecras_cell_balance_npz=cfg.get("hecras_cell_balance_npz"),
            hecras_cell_balance_target_dir=cfg.get(
                "hecras_cell_balance_target_dir"
            ),
            hecras_cell_balance_npz_key_suffix=cfg.get(
                "hecras_cell_balance_npz_key_suffix",
                "_cell_balance_storage_delta",
            ),
            hecras_cell_balance_time_offset=cfg.get(
                "hecras_cell_balance_time_offset"
            ),
            hecras_cell_balance_path=cfg.get(
                "hecras_cell_balance_path",
                (
                    "Results/Unsteady/Output/Output Blocks/Base Output/"
                    "Unsteady Time Series/2D Flow Areas/per2/Cell Flow Balance"
                ),
            ),
            hecras_precipitation_path=cfg.get(
                "hecras_precipitation_path",
                (
                    "Results/Unsteady/Output/Output Blocks/Base Output/"
                    "Unsteady Time Series/2D Flow Areas/per2/"
                    "Cell Cumulative Precipitation Depth"
                ),
            ),
            hecras_result_time_path=cfg.get(
                "hecras_result_time_path",
                (
                    "Results/Unsteady/Output/Output Blocks/Base Output/"
                    "Unsteady Time Series/Time"
                ),
            ),
            hecras_cell_xy_path=cfg.get(
                "hecras_cell_xy_path",
                "Geometry/2D Flow Areas/per2/Cells Center Coordinate",
            ),
            hecras_cell_surface_area_path=cfg.get(
                "hecras_cell_surface_area_path",
                "Geometry/2D Flow Areas/per2/Cells Surface Area",
            ),
            return_hecras_edge_flow=(
                self.use_hecras_edge_flow_loss or self.use_hecras_edge_flux_head
            ),
            hecras_edge_flow_npz=cfg.get("hecras_edge_flow_npz"),
            hecras_conservative_edge_target_dir=cfg.get(
                "hecras_conservative_edge_target_dir"
            ),
            hecras_edge_flow_mode=cfg.get("hecras_edge_flow_mode", "all_touching"),
            hecras_edge_flow_face_stats_npz=cfg.get(
                "hecras_edge_flow_face_stats_npz"
            ),
            hecras_edge_flow_scale_stats_npz=cfg.get(
                "hecras_edge_flow_scale_stats_npz"
            ),
            precipitation_unit_conversion=cfg.get(
                "precipitation_unit_conversion", 2.7778e-7
            ),
            local_source_runoff_mode=cfg.get(
                "local_source_runoff_mode", "ip_fraction"
            ),
            require_node_precipitation=cfg.get(
                "require_node_precipitation", False
            ),
            hecras_edge_flow_time_offset=cfg.get("hecras_edge_flow_time_offset", 0),
            dynamic_skip_steps=cfg.get("dynamic_skip_steps"),
            post_peak_steps=cfg.get("post_peak_steps"),
            training_rollout_steps=self.training_rollout_steps,
        )
        sampler = DistributedSampler(
            dataset,
            shuffle=True,
            drop_last=True,
            num_replicas=self.dist.world_size,
            rank=self.dist.rank,
        )
        self.dataloader = PyGDataLoader(
            dataset,
            batch_size=cfg.batch_size,
            sampler=sampler,
            pin_memory=True,
            num_workers=cfg.num_dataloader_workers,
            collate_fn=collate_fn,
        )
        rank_zero_logger.info("Dataset and dataloader initialization complete.")

        rank_zero_logger.info("Instantiating MeshGraphKAN model...")
        self.model = MeshGraphKAN(
            cfg.num_input_features,
            cfg.num_edge_features,
            cfg.num_output_features,
            mlp_activation_fn=mlp_act,
            do_concat_trick=cfg.do_concat_trick,
            num_processor_checkpoint_segments=cfg.num_processor_checkpoint_segments,
            recompute_activation=cfg.recompute_activation,
        )
        if cfg.jit:
            if not self.model.meta.jit:
                raise ValueError("MeshGraphKAN is not yet JIT-compatible.")
            self.model = torch.compile(self.model).to(self.dist.device)
        else:
            self.model = self.model.to(self.dist.device)
        rank_zero_logger.info("Model instantiated successfully.")

        init_mesh_checkpoint_path = cfg.get("init_mesh_checkpoint_path")
        if init_mesh_checkpoint_path:
            init_mesh_epoch = cfg.get("init_mesh_checkpoint_epoch")
            loaded_init_mesh_epoch = load_checkpoint(
                to_absolute_path(init_mesh_checkpoint_path),
                models=self.model,
                epoch=init_mesh_epoch,
                device=self.dist.device,
            )
            rank_zero_logger.info(
                f"Initialized MeshGraphKAN from {init_mesh_checkpoint_path} "
                f"at epoch {loaded_init_mesh_epoch}."
            )

        self.edge_flux_head = None
        if self.use_hecras_edge_flux_head:
            rank_zero_logger.info("Instantiating HEC-RAS edge-flux head...")
            self.edge_flux_head = HecRasEdgeFluxHead(
                cfg.num_input_features,
                hidden_dim=self.hecras_edge_flux_head_hidden_dim,
                num_hidden_layers=cfg.get("hecras_edge_flux_head_num_hidden_layers", 2),
                scale=self.hecras_edge_flux_head_scale,
                use_face_normal=self.hecras_edge_flux_head_use_face_normal,
                use_surface_features=(
                    self.hecras_edge_flux_head_use_surface_features
                ),
                use_physical_surface_features=(
                    self.hecras_edge_flux_head_use_physical_surface_features
                ),
                use_previous_face_flow=(
                    self.hecras_edge_flux_head_use_previous_face_flow
                ),
                use_edge_physical_features=(
                    self.hecras_edge_flux_head_use_edge_physical_features
                ),
                use_node_prediction_features=(
                    self.hecras_edge_flux_head_use_node_prediction_features
                ),
                output_mode=self.hecras_edge_flux_head_output_mode,
                active_face_mode=self.hecras_edge_flux_head_active_face_mode,
            ).to(self.dist.device)
            rank_zero_logger.info("HEC-RAS edge-flux head instantiated successfully.")

        if cfg.watch_model and not cfg.jit and self.dist.rank == 0:
            wandb.watch(self.model)

        if self.dist.world_size > 1:
            rank_zero_logger.info("Wrapping model in DistributedDataParallel...")
            self.model = DistributedDataParallel(
                self.model,
                device_ids=[self.dist.local_rank],
                output_device=self.dist.device,
                broadcast_buffers=self.dist.broadcast_buffers,
                find_unused_parameters=self.dist.find_unused_parameters,
            )
            if self.edge_flux_head is not None:
                self.edge_flux_head = DistributedDataParallel(
                    self.edge_flux_head,
                    device_ids=[self.dist.local_rank],
                    output_device=self.dist.device,
                    broadcast_buffers=self.dist.broadcast_buffers,
                    find_unused_parameters=self.dist.find_unused_parameters,
                )

        self.model.train()
        if self.edge_flux_head is not None:
            self.edge_flux_head.train()
        self.criterion = nn.MSELoss()
        self.freeze_mesh_model_for_edge_flux_head = bool(
            cfg.get("freeze_mesh_model_for_edge_flux_head", False)
        )
        if self.freeze_mesh_model_for_edge_flux_head and self.edge_flux_head is None:
            raise ValueError(
                "freeze_mesh_model_for_edge_flux_head requires "
                "use_hecras_edge_flux_head=true"
            )
        if self.use_selective_hard_conservation and self.edge_flux_head is None:
            raise ValueError(
                "use_selective_hard_conservation requires "
                "use_hecras_edge_flux_head=true"
            )
        if (
            self.use_selective_hard_conservation
            and self.selective_hard_conservation_loss_weight <= 0
        ):
            raise ValueError(
                "use_selective_hard_conservation requires a positive "
                "selective_hard_conservation_loss_weight"
            )
        if self.freeze_mesh_model_for_edge_flux_head:
            rank_zero_logger.info(
                "Freezing MeshGraphKAN; optimizer will update only "
                "HecRasEdgeFluxHead."
            )
            self.model.eval()
            for parameter in self.model.parameters():
                parameter.requires_grad_(False)
            model_parameters = []
        else:
            model_parameters = list(self.model.parameters())
        if self.edge_flux_head is not None:
            model_parameters += list(self.edge_flux_head.parameters())
        if not model_parameters:
            raise ValueError("No trainable parameters were selected for optimizer.")
        try:
            if cfg.use_apex:
                from apex.optimizers import FusedAdam

                self.optimizer = FusedAdam(model_parameters, lr=cfg.lr)
            else:
                self.optimizer = None
        except ImportError:
            rank_zero_logger.warning(
                "NVIDIA Apex is not installed; FusedAdam optimizer will not be used."
            )
            self.optimizer = None
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(model_parameters, lr=cfg.lr)
        rank_zero_logger.info(f"Using optimizer: {self.optimizer.__class__.__name__}")

        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda=lambda epoch: cfg.lr_decay_rate**epoch
        )
        self.scaler = GradScaler()

        rank_zero_logger.info("Loading checkpoint if available...")
        if self.dist.world_size > 1:
            torch.distributed.barrier()
        self.epoch_init = load_checkpoint(
            to_absolute_path(cfg.ckpt_path),
            models=[self.model, self.edge_flux_head]
            if self.edge_flux_head is not None
            else self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            device=self.dist.device,
        )
        rank_zero_logger.info(
            f"Checkpoint loaded. Starting training from epoch {self.epoch_init}."
        )

    def train(self, batch):
        if self.use_physics_loss:
            graph, physics_data = batch
        else:
            graph = batch
            physics_data = None
        graph = graph.to(self.dist.device)
        if physics_data is not None:
            physics_data = {k: v.to(self.dist.device) for k, v in physics_data.items()}
        self.optimizer.zero_grad()
        loss, loss_dict = self.forward(graph, physics_data)
        self.backward(loss)
        self.scheduler.step()
        return loss, loss_dict

    def add_edge_flux_objectives(self, pred, graph, loss, loss_dict):
        """Add optional soft and exact selective edge-flux objectives."""
        use_soft = (
            self.edge_flux_head is not None
            and self.hecras_edge_flux_head_loss_weight > 0
        )
        use_hard = (
            self.edge_flux_head is not None
            and self.use_selective_hard_conservation
            and self.selective_hard_conservation_loss_weight > 0
        )
        if not (use_soft or use_hard):
            return loss, loss_dict, None

        raw_edge_flux = self.edge_flux_head(graph, pred)
        if use_soft:
            edge_flux_loss, edge_flux_metrics = compute_hecras_edge_flux_head_loss(
                pred,
                raw_edge_flux,
                graph,
                zone_mode=self.hecras_edge_flux_head_zone_mode,
                zone_high_weight=self.hecras_edge_flux_head_zone_high_weight,
                zone_low_weight=self.hecras_edge_flux_head_zone_low_weight,
                closure_target_weight=(
                    self.hecras_edge_flux_head_closure_target_weight
                ),
                divergence_target_weight=(
                    self.hecras_edge_flux_head_divergence_target_weight
                ),
                face_target_weight=self.hecras_edge_flux_head_face_target_weight,
                face_zone_mode=self.hecras_edge_flux_head_face_zone_mode,
                face_loss_normalization=(
                    self.hecras_edge_flux_head_face_loss_normalization
                ),
                node_loss_normalization=(
                    self.hecras_edge_flux_head_node_loss_normalization
                ),
                storage_delta_mode=self.hecras_edge_flux_head_storage_delta_mode,
                closure_granularity=(
                    self.hecras_edge_flux_head_closure_granularity
                ),
                delta_t=self.delta_t,
            )
            loss = loss + self.hecras_edge_flux_head_loss_weight * edge_flux_loss
            loss_dict["hecras_edge_flux_head_loss"] = edge_flux_loss
            loss_dict.update(edge_flux_metrics)

        hard_result = None
        if use_hard:
            hard_result = compute_selective_hard_conservation_loss(
                pred,
                raw_edge_flux,
                graph,
                delta_t=self.delta_t,
                raw_control_volume_weight=(
                    self.selective_hard_conservation_raw_control_volume_weight
                ),
                correction_weight_mode=(
                    self.selective_hard_conservation_correction_weight_mode
                ),
                face_loss_normalization=(
                    self.selective_hard_conservation_face_loss_normalization
                ),
                node_loss_normalization=(
                    self.selective_hard_conservation_node_loss_normalization
                ),
            )
            loss = (
                loss
                + self.selective_hard_conservation_loss_weight * hard_result.loss
            )
            loss_dict["selective_hard_conservation_loss"] = hard_result.loss
            loss_dict["selective_hard_projected_face_loss"] = (
                hard_result.projected_face_loss
            )
            loss_dict["selective_hard_raw_control_volume_loss"] = (
                hard_result.raw_control_volume_loss
            )
            loss_dict["selective_hard_projection_correction_rms"] = torch.sqrt(
                torch.mean(hard_result.projection.face_correction.square())
            )
            loss_dict["selective_hard_projected_closure_rms"] = torch.sqrt(
                torch.mean(
                    hard_result.projection.projected_component_residual.square()
                )
            )
        return loss, loss_dict, hard_result

    def _set_training_rollout_step_attrs(self, graph, step):
        """Select one time column from the dataset's rollout sidecars."""

        local_source = graph.training_rollout_local_source_rate[:, step]
        refresh_rollout_graph_state(
            graph,
            graph.x,
            local_source,
            n_time_steps=self.n_time_steps,
        )
        if self.edge_flux_head is None:
            return

        required = (
            "training_rollout_edge_internal_delta",
            "training_rollout_edge_boundary_source_delta",
            "training_rollout_edge_precipitation_delta",
            "training_rollout_internal_face_flow_delta",
            "training_rollout_previous_internal_face_flow_delta",
        )
        missing = [name for name in required if not hasattr(graph, name)]
        if missing:
            raise AttributeError(
                "Edge rollout training requires graph attributes: "
                + ", ".join(missing)
            )
        graph.hecras_edge_internal_delta = (
            graph.training_rollout_edge_internal_delta[:, step]
        )
        graph.hecras_edge_boundary_source_delta = (
            graph.training_rollout_edge_boundary_source_delta[:, step]
        )
        graph.hecras_local_source_delta = (
            graph.training_rollout_edge_precipitation_delta[:, step]
        )
        graph.hecras_edge_flow_delta = (
            graph.hecras_edge_internal_delta
            + graph.hecras_edge_boundary_source_delta
        )
        graph.hecras_internal_face_flow_delta = (
            graph.training_rollout_internal_face_flow_delta[:, step]
        )
        graph.hecras_previous_internal_face_flow_delta = (
            graph.training_rollout_previous_internal_face_flow_delta[:, step]
        )
        if hasattr(graph, "training_rollout_raw_internal_face_flow_delta"):
            graph.hecras_raw_internal_face_flow_delta = (
                graph.training_rollout_raw_internal_face_flow_delta[:, step]
            )
        if hasattr(
            graph, "training_rollout_internal_face_flow_transition_rms"
        ):
            graph.hecras_internal_face_flow_transition_rms = (
                graph.training_rollout_internal_face_flow_transition_rms[:, step]
            )

    def _forward_differentiable_rollout(self, graph, physics_data):
        """Train on consecutive predicted states with optional edge feedback."""

        required = (
            "training_rollout_target_state",
            "training_rollout_inflow",
            "training_rollout_precipitation",
            "training_rollout_local_source_rate",
        )
        missing = [name for name in required if not hasattr(graph, name)]
        if missing:
            raise AttributeError(
                "Differentiable rollout training requires graph attributes: "
                + ", ".join(missing)
            )

        with autocast(device_type=self.dist.device.type, enabled=self.amp):
            x_iter = graph.x
            target_state = graph.training_rollout_target_state
            if target_state.shape != (
                graph.x.shape[0],
                self.training_rollout_steps,
                2,
            ):
                raise ValueError(
                    "training_rollout_target_state has unexpected shape "
                    f"{tuple(target_state.shape)}."
                )

            weighted_loss = x_iter.new_zeros(())
            weighted_mse = x_iter.new_zeros(())
            weight_sum = 0.0
            physics_loss = None
            loss_dict = {}
            for step in range(self.training_rollout_steps):
                graph.x = x_iter
                self._set_training_rollout_step_attrs(graph, step)
                depth_window = x_iter[
                    :, 12 : 12 + self.n_time_steps
                ]
                volume_window = x_iter[
                    :,
                    12 + self.n_time_steps : 12 + 2 * self.n_time_steps,
                ]
                current_state = torch.stack(
                    (depth_window[:, -1], volume_window[:, -1]), dim=1
                )
                desired_delta = target_state[:, step, :] - current_state
                pred = self.model(x_iter, graph.edge_attr, graph)
                mse_loss = self.criterion(pred, desired_delta)
                step_loss = mse_loss
                step_metrics = {"mse_loss": mse_loss}

                if self.use_fidelity_zones and self.zone_loss_weight > 0:
                    zone_loss = compute_zone_weighted_loss(
                        pred, desired_delta, graph
                    )
                    step_loss = step_loss + self.zone_loss_weight * zone_loss
                    step_metrics["zone_loss"] = zone_loss

                step_loss, step_metrics, hard_result = (
                    self.add_edge_flux_objectives(
                        pred, graph, step_loss, step_metrics
                    )
                )
                weight = self.training_rollout_discount**step
                weighted_loss = weighted_loss + weight * step_loss
                weighted_mse = weighted_mse + weight * mse_loss
                weight_sum += weight

                if step == 0 and self.use_physics_loss and physics_data is not None:
                    physics_loss = compute_physics_loss(
                        pred, physics_data, graph, delta_t=self.delta_t
                    )
                if self.log_zone_metrics:
                    step_metrics.update(
                        compute_zone_metrics(pred, desired_delta, graph)
                    )
                for name, value in step_metrics.items():
                    loss_dict[f"rollout_step_{step + 1}_{name}"] = value

                if step + 1 == self.training_rollout_steps:
                    continue
                volume_delta_override = None
                if self.training_rollout_volume_feedback == "selective_projected":
                    if hard_result is None:
                        raise RuntimeError(
                            "Selective projected feedback requires an active hard "
                            "conservation objective."
                        )
                    volume_delta_override = projected_volume_feedback_delta(
                        pred,
                        hard_result.projection,
                        graph,
                        delta_t=self.delta_t,
                        alpha=self.training_rollout_projected_volume_alpha,
                    )
                x_iter = update_hgn_rollout_state(
                    x_iter,
                    pred,
                    n_time_steps=self.n_time_steps,
                    next_inflow=graph.training_rollout_inflow[:, step],
                    next_precipitation=(
                        graph.training_rollout_precipitation[:, step]
                    ),
                    volume_delta_override=volume_delta_override,
                )
                if not self.training_rollout_backprop_through_time:
                    x_iter = x_iter.detach()

            loss = weighted_loss / weight_sum
            loss_dict["mse_loss"] = weighted_mse / weight_sum
            if physics_loss is not None:
                loss = loss + self.physics_loss_weight * physics_loss
                loss_dict["physics_loss"] = physics_loss
            loss_dict["total_loss"] = loss
            return loss, loss_dict

    def forward(self, graph, physics_data):
        if self.training_rollout_steps > 1:
            return self._forward_differentiable_rollout(graph, physics_data)
        if self.noise_type == "pushforward":
            with autocast(device_type=self.dist.device.type, enabled=self.amp):
                X = graph.x
                n_static = 12  # assumed static features dimension
                n_time = (X.shape[1] - n_static) // 2
                static_part = X[:, :n_static]
                water_depth_full = X[:, n_static : n_static + n_time]
                volume_full = X[:, n_static + n_time : n_static + 2 * n_time]
                # For one-step prediction, use dynamic features from indices 1: (last n_time_steps)
                water_depth_window_one = water_depth_full[:, 1:]
                volume_window_one = volume_full[:, 1:]
                X_one = torch.cat(
                    [static_part, water_depth_window_one, volume_window_one], dim=1
                )
                pred_one = self.model(X_one, graph.edge_attr, graph)
                one_step_loss = self.criterion(pred_one, graph.y)

                # Stability branch (example implementation)
                water_depth_window_stab = water_depth_full[:, : n_time - 1]
                volume_window_stab = volume_full[:, : n_time - 1]
                X_stab = torch.cat(
                    [static_part, water_depth_window_stab, volume_window_stab], dim=1
                )
                pred_stab = self.model(X_stab, graph.edge_attr, graph)
                pred_stab_detached = pred_stab.detach()
                water_depth_updated = torch.cat(
                    [
                        water_depth_full[:, 1:2],
                        water_depth_full[:, 1:2] + pred_stab_detached[:, 0:1],
                    ],
                    dim=1,
                )
                volume_updated = torch.cat(
                    [
                        volume_full[:, 1:2],
                        volume_full[:, 1:2] + pred_stab_detached[:, 1:2],
                    ],
                    dim=1,
                )
                X_stab_updated = torch.cat(
                    [static_part, water_depth_updated, volume_updated], dim=1
                )
                pred_stab2 = self.model(X_stab_updated, graph.edge_attr, graph)
                stability_loss = self.criterion(pred_stab2, graph.y)

                loss = one_step_loss + stability_loss
                loss_dict = {
                    "total_loss": loss,
                    "loss_one": one_step_loss,
                    "loss_stability": stability_loss,
                }
                if self.use_physics_loss and physics_data is not None:
                    phy_loss = compute_physics_loss(
                        pred_one, physics_data, graph, delta_t=self.delta_t
                    )
                    loss = loss + self.physics_loss_weight * phy_loss
                    loss_dict["physics_loss"] = phy_loss
                if self.use_fidelity_zones and self.zone_loss_weight > 0:
                    zone_loss = compute_zone_weighted_loss(pred_one, graph.y, graph)
                    loss = loss + self.zone_loss_weight * zone_loss
                    loss_dict["zone_loss"] = zone_loss
                if self.use_edge_local_proxy and self.edge_local_loss_weight > 0:
                    edge_local_loss = compute_edge_local_proxy_loss(
                        pred_one, graph.y, graph
                    )
                    loss = loss + self.edge_local_loss_weight * edge_local_loss
                    loss_dict["edge_local_proxy_loss"] = edge_local_loss
                if self.use_hecras_face_loss and self.hecras_face_loss_weight > 0:
                    hecras_face_loss = compute_hecras_face_local_loss(
                        pred_one,
                        graph.y,
                        graph,
                        delta_t=self.delta_t,
                        zone_mode=self.hecras_face_zone_mode,
                        calibrate_to_target=self.hecras_face_calibrate_to_target,
                    )
                    loss = loss + self.hecras_face_loss_weight * hecras_face_loss
                    loss_dict["hecras_face_loss"] = hecras_face_loss
                if (
                    self.use_hecras_cell_balance_loss
                    and self.hecras_cell_balance_loss_weight > 0
                ):
                    hecras_cell_balance_loss = compute_hecras_cell_balance_loss(
                        pred_one,
                        graph,
                        zone_mode=self.hecras_cell_balance_zone_mode,
                        normalization=self.hecras_cell_balance_loss_normalization,
                    )
                    loss = (
                        loss
                        + self.hecras_cell_balance_loss_weight
                        * hecras_cell_balance_loss
                    )
                    loss_dict["hecras_cell_balance_loss"] = hecras_cell_balance_loss
                if (
                    self.use_hecras_edge_flow_loss
                    and self.hecras_edge_flow_loss_weight > 0
                ):
                    hecras_edge_flow_loss = compute_hecras_edge_flow_loss(
                        pred_one,
                        graph,
                        zone_mode=self.hecras_edge_flow_zone_mode,
                    )
                    loss = (
                        loss
                        + self.hecras_edge_flow_loss_weight
                        * hecras_edge_flow_loss
                    )
                    loss_dict["hecras_edge_flow_loss"] = hecras_edge_flow_loss
                loss, loss_dict, _ = self.add_edge_flux_objectives(
                    pred_one, graph, loss, loss_dict
                )
                if (
                    self.use_hecras_face_geometry_loss
                    and self.hecras_face_geometry_loss_weight > 0
                ):
                    hecras_face_geometry_loss = compute_hecras_face_geometry_loss(
                        pred_one,
                        graph,
                        target=graph.y,
                        delta_t=self.delta_t,
                        zone_mode=self.hecras_face_geometry_zone_mode,
                        wet_depth_threshold=(
                            self.hecras_face_geometry_wet_depth_threshold
                        ),
                        reference_mode=self.hecras_face_geometry_reference_mode,
                    )
                    loss = (
                        loss
                        + self.hecras_face_geometry_loss_weight
                        * hecras_face_geometry_loss
                    )
                    loss_dict["hecras_face_geometry_loss"] = hecras_face_geometry_loss
                if self.log_zone_metrics:
                    loss_dict.update(compute_zone_metrics(pred_one, graph.y, graph))
                loss_dict["total_loss"] = loss
            return loss, loss_dict
        else:
            with autocast(device_type=self.dist.device.type, enabled=self.amp):
                pred = self.model(graph.x, graph.edge_attr, graph)
                mse_loss = self.criterion(pred, graph.y)
                loss = mse_loss
                loss_dict = {"total_loss": loss, "mse_loss": mse_loss}
                if self.use_physics_loss and physics_data is not None:
                    phy_loss = compute_physics_loss(
                        pred, physics_data, graph, delta_t=self.delta_t
                    )
                    loss = loss + self.physics_loss_weight * phy_loss
                    loss_dict["physics_loss"] = phy_loss
                if self.use_fidelity_zones and self.zone_loss_weight > 0:
                    zone_loss = compute_zone_weighted_loss(pred, graph.y, graph)
                    loss = loss + self.zone_loss_weight * zone_loss
                    loss_dict["zone_loss"] = zone_loss
                if self.use_edge_local_proxy and self.edge_local_loss_weight > 0:
                    edge_local_loss = compute_edge_local_proxy_loss(pred, graph.y, graph)
                    loss = loss + self.edge_local_loss_weight * edge_local_loss
                    loss_dict["edge_local_proxy_loss"] = edge_local_loss
                if self.use_hecras_face_loss and self.hecras_face_loss_weight > 0:
                    hecras_face_loss = compute_hecras_face_local_loss(
                        pred,
                        graph.y,
                        graph,
                        delta_t=self.delta_t,
                        zone_mode=self.hecras_face_zone_mode,
                        calibrate_to_target=self.hecras_face_calibrate_to_target,
                    )
                    loss = loss + self.hecras_face_loss_weight * hecras_face_loss
                    loss_dict["hecras_face_loss"] = hecras_face_loss
                if (
                    self.use_hecras_cell_balance_loss
                    and self.hecras_cell_balance_loss_weight > 0
                ):
                    hecras_cell_balance_loss = compute_hecras_cell_balance_loss(
                        pred,
                        graph,
                        zone_mode=self.hecras_cell_balance_zone_mode,
                        normalization=self.hecras_cell_balance_loss_normalization,
                    )
                    loss = (
                        loss
                        + self.hecras_cell_balance_loss_weight
                        * hecras_cell_balance_loss
                    )
                    loss_dict["hecras_cell_balance_loss"] = hecras_cell_balance_loss
                if (
                    self.use_hecras_edge_flow_loss
                    and self.hecras_edge_flow_loss_weight > 0
                ):
                    hecras_edge_flow_loss = compute_hecras_edge_flow_loss(
                        pred,
                        graph,
                        zone_mode=self.hecras_edge_flow_zone_mode,
                    )
                    loss = (
                        loss
                        + self.hecras_edge_flow_loss_weight
                        * hecras_edge_flow_loss
                    )
                    loss_dict["hecras_edge_flow_loss"] = hecras_edge_flow_loss
                loss, loss_dict, _ = self.add_edge_flux_objectives(
                    pred, graph, loss, loss_dict
                )
                if (
                    self.use_hecras_face_geometry_loss
                    and self.hecras_face_geometry_loss_weight > 0
                ):
                    hecras_face_geometry_loss = compute_hecras_face_geometry_loss(
                        pred,
                        graph,
                        target=graph.y,
                        delta_t=self.delta_t,
                        zone_mode=self.hecras_face_geometry_zone_mode,
                        wet_depth_threshold=(
                            self.hecras_face_geometry_wet_depth_threshold
                        ),
                        reference_mode=self.hecras_face_geometry_reference_mode,
                    )
                    loss = (
                        loss
                        + self.hecras_face_geometry_loss_weight
                        * hecras_face_geometry_loss
                    )
                    loss_dict["hecras_face_geometry_loss"] = hecras_face_geometry_loss
                if self.log_zone_metrics:
                    loss_dict.update(compute_zone_metrics(pred, graph.y, graph))
                loss_dict["total_loss"] = loss
            return loss, loss_dict

    def backward(self, loss):
        if self.amp:
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            loss.backward()
            self.optimizer.step()


@hydra.main(version_base="1.3", config_path="conf", config_name="config")
def main(cfg: DictConfig) -> None:
    if cfg.get("seed") is not None:
        random.seed(cfg.seed)
        np.random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(cfg.seed)
    DistributedManager.initialize()
    dist = DistributedManager()
    initialize_wandb(
        project="Modulus-Launch",
        entity="Modulus",
        name="Vortex_Shedding-Training",
        group="Vortex_Shedding-DDP-Group",
        mode=cfg.wandb_mode,
    )
    logger = PythonLogger("main")
    rank_zero_logger = RankZeroLoggingWrapper(logger, dist)
    rank_zero_logger.file_logging()
    rank_zero_logger.info(f"Starting training process with configuration: {cfg}")
    trainer = MGNTrainer(cfg, rank_zero_logger)
    rank_zero_logger.info("Beginning training loop...")
    start_time = time.time()

    for epoch in range(trainer.epoch_init, cfg.epochs):
        epoch_loss = 0.0
        epoch_metrics = {}
        num_batches = 0
        for batch in trainer.dataloader:
            loss, loss_dict = trainer.train(batch)
            epoch_loss += loss.detach().item()
            for key, value in loss_dict.items():
                if torch.is_tensor(value):
                    epoch_metrics[key] = (
                        epoch_metrics.get(key, 0.0) + value.detach().item()
                    )
            num_batches += 1
            if cfg.get("max_train_batches") and num_batches >= cfg.max_train_batches:
                break

        avg_loss = epoch_loss / num_batches if num_batches > 0 else float("inf")
        avg_metrics = {
            key: value / num_batches for key, value in epoch_metrics.items()
        } if num_batches > 0 else {}
        rank_zero_logger.info(f"Epoch {epoch} completed. Average Loss: {avg_loss:.4e}")
        for key in sorted(avg_metrics):
            rank_zero_logger.info(f"Epoch {epoch} {key}: {avg_metrics[key]:.4e}")

        log_data = {"epoch": epoch, **avg_metrics}
        wandb.log(log_data)

        if dist.world_size > 1:
            torch.distributed.barrier()
        if dist.rank == 0:
            save_checkpoint(
                to_absolute_path(cfg.ckpt_path),
                models=[trainer.model, trainer.edge_flux_head]
                if trainer.edge_flux_head is not None
                else trainer.model,
                optimizer=trainer.optimizer,
                scheduler=trainer.scheduler,
                scaler=trainer.scaler,
                epoch=epoch,
            )
            rank_zero_logger.info(f"Checkpoint saved at epoch {epoch}.")

        elapsed = time.time() - start_time
        rank_zero_logger.info(f"Epoch {epoch} duration: {elapsed:.2f} seconds.")
        start_time = time.time()

    rank_zero_logger.info("Training completed successfully.")


if __name__ == "__main__":
    main()
