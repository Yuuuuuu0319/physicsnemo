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
from utils import (
    compute_edge_local_proxy_loss,
    compute_hecras_cell_balance_loss,
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


class MGNTrainer:
    def __init__(self, cfg: DictConfig, rank_zero_logger: RankZeroLoggingWrapper):
        # Ensure distributed manager is initialized.
        assert DistributedManager.is_initialized()
        self.dist = DistributedManager()
        self.amp = cfg.amp
        self.noise_type = cfg.noise_type

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

        self.model.train()
        self.criterion = nn.MSELoss()
        try:
            if cfg.use_apex:
                from apex.optimizers import FusedAdam

                self.optimizer = FusedAdam(self.model.parameters(), lr=cfg.lr)
            else:
                self.optimizer = None
        except ImportError:
            rank_zero_logger.warning(
                "NVIDIA Apex is not installed; FusedAdam optimizer will not be used."
            )
            self.optimizer = None
        if self.optimizer is None:
            self.optimizer = torch.optim.Adam(self.model.parameters(), lr=cfg.lr)
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
            models=self.model,
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

    def forward(self, graph, physics_data):
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
                    )
                    loss = (
                        loss
                        + self.hecras_cell_balance_loss_weight
                        * hecras_cell_balance_loss
                    )
                    loss_dict["hecras_cell_balance_loss"] = hecras_cell_balance_loss
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
                    )
                    loss = (
                        loss
                        + self.hecras_cell_balance_loss_weight
                        * hecras_cell_balance_loss
                    )
                    loss_dict["hecras_cell_balance_loss"] = hecras_cell_balance_loss
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
                models=trainer.model,
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
