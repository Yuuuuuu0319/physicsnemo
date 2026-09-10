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

# ruff: noqa: S324,F821,S113

"""
HydroGraphDataset module

This module defines a Dataset for hydrograph-based graphs. It includes utility functions
for downloading data, computing normalization statistics, and processing both static and dynamic
data required to build a graph for each hydrograph sample.

The dataset supports two modes:
    - Training: Each sample is a sliding window sample.
    - Testing: Each sample corresponds to an entire hydrograph.

For testing, each sample returns a tuple (graph, rollout_data) containing the initial graph and
a dictionary of future hydrograph data for evaluation.
"""

import hashlib
import json
import logging
import math
import os
import random
import re
import sys
import tarfile
import zipfile
from pathlib import Path
from typing import Any, List, Optional, Union

import numpy as np
import requests
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from physicsnemo.core.version_check import OptionalImport

# Lazy imports for optional dependencies
pyg = OptionalImport("torch_geometric")
scipy_spatial = OptionalImport("scipy.spatial")

# Setup logging
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s] %(message)s")
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


# ---------------------------
# Download Utility Functions
# ---------------------------
def calculate_md5(fpath: Union[str, Path], chunk_size: int = 1024 * 1024) -> str:
    """
    Calculate the MD5 checksum of a file.

    Args:
        fpath (str or Path): Path to the file.
        chunk_size (int): Size of each chunk to read from the file.

    Returns:
        str: MD5 checksum of the file.
    """
    if sys.version_info >= (3, 9):
        md5 = hashlib.md5(usedforsecurity=False)
    else:
        md5 = hashlib.md5()
    with open(fpath, "rb") as f:
        while chunk := f.read(chunk_size):
            md5.update(chunk)
    return md5.hexdigest()


def check_md5(fpath: Union[str, Path], md5: str, **kwargs: Any) -> bool:
    """
    Check if the file at fpath has the expected MD5 checksum.

    Args:
        fpath (str or Path): Path to the file.
        md5 (str): Expected MD5 checksum.
        **kwargs: Additional keyword arguments for calculate_md5.

    Returns:
        bool: True if the file's checksum matches; False otherwise.
    """
    return md5 == calculate_md5(fpath, **kwargs)


def check_integrity(fpath: Union[str, Path], md5: Optional[str] = None) -> bool:
    """
    Verify the integrity of a file by checking its existence and, optionally, its MD5 checksum.

    Args:
        fpath (str or Path): File path to check.
        md5 (Optional[str]): Expected MD5 checksum (if any).

    Returns:
        bool: True if the file exists (and matches the checksum if provided); False otherwise.
    """
    fpath = Path(fpath)
    if not fpath.is_file():
        return False
    if md5 is None:
        return True
    return check_md5(fpath, md5)


def download_from_url(
    url: str,
    root: Union[str, Path],
    filename: Optional[Union[str, Path]] = None,
    md5: Optional[str] = None,
    size: Optional[int] = None,
    chunk_size: int = 256 * 64,
    extract: bool = True,
) -> None:
    """
    Download a file from a URL, verify its integrity, and optionally extract it.

    Args:
        url (str): URL of the file to download.
        root (str or Path): Directory where the file will be saved.
        filename (Optional[str or Path]): Optional file name; if not provided, it is derived from the URL.
        md5 (Optional[str]): Expected MD5 checksum.
        size (Optional[int]): Expected file size.
        chunk_size (int): Chunk size for downloading.
        extract (bool): If True, extract the file if it is a tar or zip archive.
    """
    root = Path(root).expanduser()
    root.mkdir(parents=True, exist_ok=True)
    if not filename:
        filename = url.split("/")[-1]
    fpath = root / filename
    if check_integrity(fpath, md5):
        logger.info(f"Using downloaded and verified file: {fpath}")
    else:
        logger.info(f"Downloading {url} to {fpath} ...")
        with requests.get(url, stream=True, timeout=120) as r:
            r.raise_for_status()
            total_size = int(r.headers.get("content-length", 0))
            with (
                open(fpath, "wb") as f,
                tqdm(
                    desc=str(fpath),
                    total=total_size,
                    unit="iB",
                    unit_scale=True,
                    unit_divisor=1024,
                ) as bar,
            ):
                for chunk in r.iter_content(chunk_size=chunk_size):
                    if chunk:
                        f.write(chunk)
                        f.flush()
                        os.fsync(f.fileno())
                        bar.update(len(chunk))
        if size is not None and fpath.stat().st_size != size:
            raise RuntimeError("Downloaded file has unexpected size.")
        if not check_integrity(fpath, md5):
            raise RuntimeError("File not found or corrupted.")
        logger.info(f"Saved to {fpath} successfully.")
    if extract:
        # Extract tar or zip archives
        if fpath.suffix in [".tar", ".gz", ".tgz"]:
            logger.info(f"Extracting tar archive {fpath}...")
            with tarfile.open(fpath, "r:*") as archive:
                # Safely extract while supporting Python versions < 3.12 that lack the
                # ``filter`` keyword.  Starting with 3.12, ``filter="data"`` is the
                # recommended way to avoid unsafe members;
                extract_kwargs = dict(
                    path=root,
                )
                if "filter" in archive.extractall.__code__.co_varnames:
                    extract_kwargs["filter"] = "data"
                archive.extractall(**extract_kwargs)  # noqa: S202
                names = ", ".join(archive.getnames())
            logger.info(f"Extracted files: {names}")
        elif fpath.suffix == ".zip":
            logger.info(f"Extracting zip archive {fpath}...")
            with zipfile.ZipFile(fpath, "r") as z:
                # Safely extract while supporting Python versions < 3.12 that lack the
                # ``filter`` keyword.  Starting with 3.12, ``filter="data"`` is the
                # recommended way to avoid unsafe members;
                extract_kwargs = dict(
                    path=root,
                )
                if "filter" in z.extractall.__code__.co_varnames:
                    extract_kwargs["filter"] = "data"
                z.extractall(**extract_kwargs)  # noqa: S202
                names = ", ".join(z.namelist())
            logger.info(f"Extracted files: {names}")


def download_from_zenodo_record(
    record_id: str,
    root: Union[str, Path],
    files_to_download: Optional[List[str]] = None,
) -> None:
    """
    Download dataset files from a Zenodo record.

    Args:
        record_id (str): The Zenodo record ID.
        root (str or Path): Directory where files will be saved.
        files_to_download (Optional[List[str]]): Specific files to download; if None, download all.
    """
    zenodo_api_url = "https://zenodo.org/api/records/"
    url = f"{zenodo_api_url}{record_id}"
    logger.info(f"Fetching Zenodo record info for record ID {record_id} ...")
    resp = requests.get(url)
    if resp.status_code != 200:
        raise RuntimeError(f"Error: request failed with status code {resp.status_code}")
    response_json = resp.json()
    for file_record in response_json["files"]:
        fname = file_record["key"]
        if files_to_download is None or fname in files_to_download:
            file_url = file_record["links"]["self"]
            file_md5 = file_record["checksum"][4:]
            file_size = file_record["size"]
            download_from_url(
                url=file_url,
                root=root,
                filename=fname,
                md5=file_md5,
                size=file_size,
                extract=True,
            )


def ensure_data_available(data_dir: Union[str, Path]) -> None:
    """
    Ensure that the dataset is available in the specified directory.
    If not found, download the dataset from Zenodo.

    Args:
        data_dir (str or Path): Path to the data directory.
    """
    data_dir = Path(data_dir)
    if not data_dir.exists():
        logger.info(
            f"Data directory {data_dir} not found. Downloading dataset from Zenodo..."
        )
        download_from_zenodo_record(ZENODO_RECORD_ID, data_dir, FILES_TO_DOWNLOAD)
    else:
        logger.info(f"Data directory {data_dir} already exists. Skipping download.")


# Global constants for Zenodo record and filenames.
ZENODO_RECORD_ID = "14969507"
FILES_TO_DOWNLOAD = None

STATIC_NORM_STATS_FILE = "static_norm_stats.json"
DYNAMIC_NORM_STATS_FILE = "dynamic_norm_stats.json"


# ---------------------------
# HydroGraphDataset Class
# ---------------------------
class HydroGraphDataset(Dataset):
    """
    Dataset for hydrograph-based graphs.

    This dataset processes both static and dynamic data to construct graphs for each hydrograph.
    It supports two modes:
        - Training ("train"): Each sample is a sliding window sample.
        - Testing ("test"): Each sample is a full hydrograph with rollout data.

    Attributes:
        data_dir (str): Directory where the dataset is located.
        prefix (str): Prefix for file names.
        num_samples (int): Maximum number of hydrograph samples.
        n_time_steps (int): Number of time steps used in the sliding window.
        k (int): Number of nearest neighbors for graph connectivity.
        noise_std (float): Standard deviation for added noise.
        noise_type (str): Type of noise to apply.
        hydrograph_ids_file (Optional[str]): File containing hydrograph IDs.
        split (str): Split type ("train" or "test").
        rollout_length (int): Number of rollout time steps (used in test mode).
        return_physics (bool): Flag to include physics data in __getitem__ output.
    """

    def __init__(
        self,
        name: str = "hydrograph_dataset",
        data_dir: Union[str, Path] = "data_directory",
        prefix: str = "M80",
        num_samples: int = 500,
        n_time_steps: int = 10,
        k: int = 4,
        noise_std: float = 0.01,
        noise_type: str = "none",
        hydrograph_ids_file: Optional[str] = None,
        split: str = "train",
        rollout_length: Optional[int] = None,
        return_physics: bool = False,
        use_fidelity_zones: bool = False,
        zone_label_file: str = "zone_label.txt",
        zone_weight_file: str = "zone_weight.txt",
        return_edge_local: bool = False,
        return_hecras_face: bool = False,
        hecras_face_graph_file: Optional[Union[str, Path]] = None,
        hecras_face_velocity_file: Optional[Union[str, Path]] = None,
        hecras_face_velocity_glob: Optional[str] = None,
        hecras_face_velocity_path: str = (
            "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/"
            "2D Flow Areas/per2/Face Velocity"
        ),
        hecras_face_time_offset: int = 0,
        return_hecras_cell_balance: bool = False,
        hecras_cell_balance_glob: Optional[str] = None,
        hecras_cell_balance_npz: Optional[Union[str, Path]] = None,
        hecras_cell_balance_target_dir: Optional[Union[str, Path]] = None,
        hecras_cell_balance_npz_key_suffix: str = (
            "_cell_balance_storage_delta"
        ),
        hecras_cell_balance_time_offset: Optional[int] = None,
        hecras_cell_balance_path: str = (
            "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/"
            "2D Flow Areas/per2/Cell Flow Balance"
        ),
        hecras_precipitation_path: str = (
            "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/"
            "2D Flow Areas/per2/Cell Cumulative Precipitation Depth"
        ),
        hecras_result_time_path: str = (
            "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/Time"
        ),
        hecras_cell_xy_path: str = (
            "Geometry/2D Flow Areas/per2/Cells Center Coordinate"
        ),
        hecras_cell_surface_area_path: str = (
            "Geometry/2D Flow Areas/per2/Cells Surface Area"
        ),
        return_hecras_edge_flow: bool = False,
        hecras_edge_flow_npz: Optional[Union[str, Path]] = None,
        hecras_conservative_edge_target_dir: Optional[Union[str, Path]] = None,
        hecras_edge_flow_mode: str = "all_touching",
        hecras_edge_flow_face_stats_npz: Optional[Union[str, Path]] = None,
        hecras_edge_flow_scale_stats_npz: Optional[Union[str, Path]] = None,
        norm_stats_dir: Optional[Union[str, Path]] = None,
        precipitation_unit_conversion: float = 2.7778e-7,
        local_source_runoff_mode: str = "ip_fraction",
        require_node_precipitation: bool = False,
        hecras_edge_flow_time_offset: Optional[int] = None,
        dynamic_skip_steps: Optional[int] = None,
        post_peak_steps: Optional[int] = None,
        training_rollout_steps: int = 1,
    ):
        if split not in {"train", "test"}:
            raise ValueError(f"Invalid split '{split}'. Expected 'train' or 'test'.")

        # Initialize dataset attributes.
        self.data_dir = str(data_dir)
        ensure_data_available(self.data_dir)
        time_contract = self.load_dataset_time_contract(self.data_dir)
        self.dynamic_skip_steps = int(
            dynamic_skip_steps
            if dynamic_skip_steps is not None
            else time_contract.get("dynamic_skip_steps", 72)
        )
        self.post_peak_steps = int(
            post_peak_steps
            if post_peak_steps is not None
            else time_contract.get("post_peak_steps", 25)
        )
        if self.dynamic_skip_steps < 0:
            raise ValueError("dynamic_skip_steps must be non-negative.")
        if self.post_peak_steps <= 0:
            raise ValueError("post_peak_steps must be positive.")
        self.prefix = prefix
        self.num_samples = num_samples
        self.n_time_steps = n_time_steps
        self.k = k
        self.noise_std = noise_std
        self.noise_type = noise_type
        self.hydrograph_ids_file = hydrograph_ids_file
        self.split = split
        self.training_rollout_steps = int(training_rollout_steps)
        if self.training_rollout_steps < 1:
            raise ValueError("training_rollout_steps must be positive.")
        if self.split != "train" and self.training_rollout_steps != 1:
            raise ValueError(
                "training_rollout_steps is only supported for split='train'."
            )
        if self.training_rollout_steps > 1 and self.noise_type == "pushforward":
            raise ValueError(
                "Explicit training rollouts and legacy pushforward noise cannot "
                "be enabled together."
            )
        # rollout_length is only used when split=="test"
        self.rollout_length = rollout_length if rollout_length is not None else 0
        self.return_physics = return_physics
        self.use_fidelity_zones = use_fidelity_zones
        self.zone_label_file = zone_label_file
        self.zone_weight_file = zone_weight_file
        self.return_edge_local = return_edge_local
        self.return_hecras_face = return_hecras_face
        self.hecras_face_graph_file = (
            str(hecras_face_graph_file) if hecras_face_graph_file is not None else None
        )
        self.hecras_face_velocity_file = (
            str(hecras_face_velocity_file)
            if hecras_face_velocity_file is not None
            else None
        )
        self.hecras_face_velocity_glob = hecras_face_velocity_glob
        self.hecras_face_velocity_path = hecras_face_velocity_path
        self.hecras_face_time_offset = hecras_face_time_offset
        self.return_hecras_cell_balance = return_hecras_cell_balance
        self.hecras_cell_balance_glob = hecras_cell_balance_glob
        self.hecras_cell_balance_npz = (
            str(hecras_cell_balance_npz)
            if hecras_cell_balance_npz is not None
            else None
        )
        self.hecras_cell_balance_target_dir = (
            str(hecras_cell_balance_target_dir)
            if hecras_cell_balance_target_dir is not None
            else None
        )
        self.hecras_cell_balance_npz_key_suffix = (
            hecras_cell_balance_npz_key_suffix
        )
        if hecras_cell_balance_time_offset is None:
            self.hecras_cell_balance_time_offset = (
                self.dynamic_skip_steps
                if self.hecras_cell_balance_target_dir is not None
                else 0
            )
        else:
            self.hecras_cell_balance_time_offset = int(
                hecras_cell_balance_time_offset
            )
        if self.hecras_cell_balance_time_offset < 0:
            raise ValueError("hecras_cell_balance_time_offset must be non-negative.")
        self.hecras_cell_balance_path = hecras_cell_balance_path
        self.hecras_precipitation_path = hecras_precipitation_path
        self.hecras_result_time_path = hecras_result_time_path
        self.hecras_cell_xy_path = hecras_cell_xy_path
        self.hecras_cell_surface_area_path = hecras_cell_surface_area_path
        self.return_hecras_edge_flow = return_hecras_edge_flow
        self.hecras_edge_flow_npz = (
            str(hecras_edge_flow_npz) if hecras_edge_flow_npz is not None else None
        )
        self.hecras_conservative_edge_target_dir = (
            str(hecras_conservative_edge_target_dir)
            if hecras_conservative_edge_target_dir is not None
            else None
        )
        self.hecras_edge_flow_mode = hecras_edge_flow_mode
        self.hecras_edge_flow_face_stats_npz = (
            str(hecras_edge_flow_face_stats_npz)
            if hecras_edge_flow_face_stats_npz is not None
            else None
        )
        self.hecras_edge_flow_scale_stats_npz = (
            str(hecras_edge_flow_scale_stats_npz)
            if hecras_edge_flow_scale_stats_npz is not None
            else None
        )
        self.norm_stats_dir = str(norm_stats_dir) if norm_stats_dir is not None else None
        self.precipitation_unit_conversion = float(precipitation_unit_conversion)
        self.local_source_runoff_mode = local_source_runoff_mode
        self.require_node_precipitation = bool(require_node_precipitation)
        if hecras_edge_flow_time_offset is None:
            self.hecras_edge_flow_time_offset = (
                self.dynamic_skip_steps
                if self.hecras_conservative_edge_target_dir is not None
                else 0
            )
        else:
            self.hecras_edge_flow_time_offset = int(hecras_edge_flow_time_offset)
        if self.hecras_edge_flow_time_offset < 0:
            raise ValueError("hecras_edge_flow_time_offset must be non-negative.")
        if self.local_source_runoff_mode not in {"ip_fraction", "full_area"}:
            raise ValueError(
                "local_source_runoff_mode must be 'ip_fraction' or 'full_area', "
                f"got {self.local_source_runoff_mode!r}."
            )

        # Placeholders for static and dynamic data, indices, and normalization stats.
        self.static_data = {}
        self.static_data_raw_xy = None
        self.dynamic_data = []
        self.sample_index = []
        self.hydrograph_ids = []
        self.static_stats = {}
        self.dynamic_stats = {}
        self.zone_label = None
        self.zone_weight = None
        self.hecras_face_graph = None
        self.hecras_face_velocity = None
        self.hecras_face_velocity_by_hydrograph = {}
        self.hecras_cell_balance_delta_by_hydrograph = {}
        self.hecras_edge_flow_delta_by_hydrograph = {}
        self.hecras_edge_flow_face_stats = None
        self.hecras_edge_flow_scale_stats = None
        self.hecras_boundary_node_mask = None
        self.hecras_high_interior_control_volume_label = None

        self.process()

    def process(self) -> None:
        """
        Process the dataset to load static and dynamic data and compute necessary normalization stats.
        """
        if self.split == "train":
            # For training, load constant data and compute static normalization stats.
            (
                xy_coords,
                area,
                area_denorm,
                elevation,
                slope,
                aspect,
                curvature,
                manning,
                flow_accum,
                infiltration,
                self.static_stats,
            ) = self.load_constant_data(
                self.data_dir, self.prefix, norm_stats_static=None
            )
            self.save_norm_stats(self.static_stats, STATIC_NORM_STATS_FILE)
        else:
            # For test or validation, load precomputed normalization stats.
            self.static_stats = self.load_norm_stats(STATIC_NORM_STATS_FILE)
            (
                xy_coords,
                area,
                area_denorm,
                elevation,
                slope,
                aspect,
                curvature,
                manning,
                flow_accum,
                infiltration,
                _,
            ) = self.load_constant_data(
                self.data_dir, self.prefix, norm_stats_static=self.static_stats
            )

        # Build the graph connectivity using a k-d tree.
        num_nodes = xy_coords.shape[0]
        kdtree = scipy_spatial.KDTree(xy_coords)
        _, neighbors = kdtree.query(xy_coords, k=self.k + 1)
        edge_index = np.vstack(
            [(i, nbr) for i, nbrs in enumerate(neighbors) for nbr in nbrs if nbr != i]
        ).T
        edge_features = self.create_edge_features(xy_coords, edge_index)
        edge_unit_vectors = self.create_edge_unit_vectors(
            self.static_data_raw_xy, edge_index
        )

        # Store static data.
        self.static_data = {
            "xy_coords": xy_coords,
            "area": area,
            "area_denorm": area_denorm,
            "elevation": elevation,
            "slope": slope,
            "aspect": aspect,
            "curvature": curvature,
            "manning": manning,
            "flow_accum": flow_accum,
            "infiltration": infiltration,
            "edge_index": edge_index,
            "edge_features": edge_features,
            "edge_unit_vectors": edge_unit_vectors,
        }
        if self.use_fidelity_zones:
            self.zone_label, self.zone_weight = self.load_fidelity_zones(num_nodes)
        if self.return_hecras_face:
            self.hecras_face_graph = self.load_hecras_face_graph(num_nodes)
            if self.hecras_face_velocity_file is not None:
                self.hecras_face_velocity = self.load_hecras_face_velocity()
            if self.hecras_face_velocity_glob is not None:
                self.hecras_face_velocity_by_hydrograph = (
                    self.load_hecras_face_velocity_by_hydrograph()
                )

        # Read hydrograph IDs either from a file or from the directory.
        if self.hydrograph_ids_file is not None:
            file_path = os.path.join(self.data_dir, self.hydrograph_ids_file)
            if os.path.exists(file_path):
                with open(file_path, "r") as f:
                    lines = f.readlines()
                self.hydrograph_ids = [line.strip() for line in lines if line.strip()]
            else:
                raise FileNotFoundError(f"Hydrograph IDs file not found: {file_path}")
        else:
            all_files = os.listdir(self.data_dir)
            self.hydrograph_ids = []
            for f in all_files:
                if f.startswith(f"{self.prefix}_WD_") and f.endswith(".txt"):
                    parts = f.split("_")
                    if len(parts) >= 3:
                        hid = os.path.splitext(parts[2])[0]
                        self.hydrograph_ids.append(hid)
        if len(self.hydrograph_ids) > self.num_samples:
            self.hydrograph_ids = random.sample(self.hydrograph_ids, self.num_samples)

        # Process dynamic data (water depth, inflow, volume, precipitation) for each hydrograph.
        temp_dynamic_data = []
        water_depth_list = []
        volume_list = []
        precipitation_list = []
        inflow_list = []
        for hid in tqdm(self.hydrograph_ids, desc="Processing Hydrographs"):
            (
                water_depth,
                inflow_hydrograph,
                velocity_x,
                velocity_y,
                volume,
                precipitation,
                local_precipitation,
            ) = self.load_dynamic_data(
                self.data_dir,
                hid,
                self.prefix,
                num_points=num_nodes,
                skip=self.dynamic_skip_steps,
                post_peak_steps=self.post_peak_steps,
            )
            temp_dynamic_data.append(
                {
                    "water_depth": water_depth,
                    "inflow_hydrograph": inflow_hydrograph,
                    "velocity_x": velocity_x,
                    "velocity_y": velocity_y,
                    "volume": volume,
                    "precipitation": precipitation,
                    "local_precipitation": local_precipitation,
                    "hydro_id": hid,
                }
            )
            water_depth_list.append(water_depth.flatten())
            volume_list.append(volume.flatten())
            precipitation_list.append(precipitation.flatten())
            inflow_list.append(inflow_hydrograph.flatten())

        # Compute dynamic normalization statistics for training or load precomputed stats.
        if self.split == "train":
            self.dynamic_stats = {}
            water_depth_all = np.concatenate(water_depth_list)
            self.dynamic_stats["water_depth"] = {
                "mean": float(np.mean(water_depth_all)),
                "std": float(np.std(water_depth_all)),
            }
            volume_all = np.concatenate(volume_list)
            self.dynamic_stats["volume"] = {
                "mean": float(np.mean(volume_all)),
                "std": float(np.std(volume_all)),
            }
            precipitation_all = np.concatenate(precipitation_list)
            self.dynamic_stats["precipitation"] = {
                "mean": float(np.mean(precipitation_all)),
                "std": float(np.std(precipitation_all)),
            }
            inflow_all = np.concatenate(inflow_list)
            self.dynamic_stats["inflow_hydrograph"] = {
                "mean": float(np.mean(inflow_all)),
                "std": float(np.std(inflow_all)),
            }
            self.save_norm_stats(self.dynamic_stats, DYNAMIC_NORM_STATS_FILE)
        else:
            self.dynamic_stats = self.load_norm_stats(DYNAMIC_NORM_STATS_FILE)

        # Normalize the dynamic data.
        self.dynamic_data = []
        for dyn in temp_dynamic_data:
            dyn_std = {
                "water_depth": self.normalize(
                    dyn["water_depth"],
                    self.dynamic_stats["water_depth"]["mean"],
                    self.dynamic_stats["water_depth"]["std"],
                ),
                "volume": self.normalize(
                    dyn["volume"],
                    self.dynamic_stats["volume"]["mean"],
                    self.dynamic_stats["volume"]["std"],
                ),
                "precipitation": self.normalize(
                    dyn["precipitation"],
                    self.dynamic_stats["precipitation"]["mean"],
                    self.dynamic_stats["precipitation"]["std"],
                ),
                "inflow_hydrograph": self.normalize(
                    dyn["inflow_hydrograph"],
                    self.dynamic_stats["inflow_hydrograph"]["mean"],
                    self.dynamic_stats["inflow_hydrograph"]["std"],
                ),
                "velocity_x": dyn["velocity_x"],
                "velocity_y": dyn["velocity_y"],
                "local_precipitation": dyn["local_precipitation"],
                "hydro_id": dyn["hydro_id"],
            }
            self.dynamic_data.append(dyn_std)

        if self.return_hecras_cell_balance:
            self.hecras_cell_balance_delta_by_hydrograph = (
                self.load_hecras_cell_balance_delta_by_hydrograph()
            )
        if self.return_hecras_edge_flow:
            self.hecras_edge_flow_delta_by_hydrograph = (
                self.load_hecras_edge_flow_delta_by_hydrograph()
            )
            self.hecras_edge_flow_face_stats = self.load_hecras_edge_flow_face_stats()
            self.hecras_edge_flow_scale_stats = self.load_hecras_edge_flow_scale_stats()
            self.hecras_boundary_node_mask = self.load_hecras_boundary_node_mask()
            self.hecras_high_interior_control_volume_label = (
                self.build_hecras_high_interior_control_volume_labels()
            )

        # Build sample indices for training (sliding window) or validate test data.
        if self.split == "train":
            for h_idx, dyn in enumerate(self.dynamic_data):
                T = dyn["water_depth"].shape[0]
                if self.noise_type == "pushforward":
                    max_t = T - self.n_time_steps - 1
                else:
                    max_t = (
                        T
                        - self.n_time_steps
                        - self.training_rollout_steps
                        + 1
                    )
                for t in range(max_t):
                    self.sample_index.append((h_idx, t))
            self.length = len(self.sample_index)
        elif self.split == "test":
            for dyn in self.dynamic_data:
                T = dyn["water_depth"].shape[0]
                if T < self.n_time_steps + self.rollout_length:
                    raise ValueError(
                        f"Hydrograph {dyn['hydro_id']} does not have enough time steps for the specified rollout_length."
                    )
            self.length = len(self.dynamic_data)

    def __getitem__(self, idx: int):
        """
        Retrieve a graph sample (and associated physics data if required).

        Args:
            idx (int): Index of the sample.

        Returns:
            Depending on the split:
                - Training: A graph with node features, edge features, and target values, optionally
                  along with a dictionary of physics data.
                - Testing: A tuple (graph, rollout_data) where rollout_data contains future hydrograph data.
        """
        sd = self.static_data
        if self.split != "test":
            # Training mode: use sliding window sample.
            hydro_idx, t_idx = self.sample_index[idx]
            dyn = self.dynamic_data[hydro_idx]

            # Determine the end index for the dynamic window.
            end_index = (
                t_idx + self.n_time_steps + 1
                if self.noise_type == "pushforward"
                else t_idx + self.n_time_steps
            )

            # Compute node features and future flow/precipitation values.
            node_features, future_flow, future_precip = self.create_node_features(
                sd["xy_coords"],
                sd["area"],
                sd["elevation"],
                sd["slope"],
                sd["aspect"],
                sd["curvature"],
                sd["manning"],
                sd["flow_accum"],
                sd["infiltration"],
                dyn["water_depth"][t_idx:end_index, :],
                dyn["volume"][t_idx:end_index, :],
                dyn["precipitation"],
                t_idx,
                self.n_time_steps,
                dyn["inflow_hydrograph"],
            )
            target_time = t_idx + self.n_time_steps
            prev_time = target_time - 1
            # Compute target differences for water depth and volume.
            target_depth = (
                dyn["water_depth"][target_time, :] - dyn["water_depth"][prev_time, :]
            )
            target_volume = dyn["volume"][target_time, :] - dyn["volume"][prev_time, :]
            target = np.stack([target_depth, target_volume], axis=1)

            # Create the graph with PyG.
            src, dst = sd["edge_index"]
            edges = torch.stack([torch.tensor(src), torch.tensor(dst)], dim=0).long()
            g = pyg.data.Data(edge_index=edges)
            g.edge_attr = torch.tensor(sd["edge_features"], dtype=torch.float)
            g.x = torch.tensor(node_features, dtype=torch.float)
            g.y = torch.tensor(target, dtype=torch.float)
            if self.training_rollout_steps > 1:
                rollout_slice = slice(
                    target_time, target_time + self.training_rollout_steps
                )
                rollout_state = np.stack(
                    (
                        dyn["water_depth"][rollout_slice, :].T,
                        dyn["volume"][rollout_slice, :].T,
                    ),
                    axis=2,
                )
                g.training_rollout_target_state = torch.tensor(
                    rollout_state, dtype=torch.float
                )
                g.training_rollout_inflow = torch.tensor(
                    dyn["inflow_hydrograph"][rollout_slice][None, :],
                    dtype=torch.float,
                )
                g.training_rollout_precipitation = torch.tensor(
                    dyn["precipitation"][rollout_slice][None, :],
                    dtype=torch.float,
                )
                local_source_rates = np.stack(
                    [
                        self.compute_local_source_rate(
                            dyn,
                            target_time + step - 1,
                            target_time + step,
                        )
                        for step in range(self.training_rollout_steps)
                    ],
                    axis=1,
                )
                g.training_rollout_local_source_rate = torch.tensor(
                    local_source_rates, dtype=torch.float
                )
            if self.use_fidelity_zones:
                g.zone_label = torch.tensor(self.zone_label, dtype=torch.long)
                g.zone_weight = torch.tensor(self.zone_weight, dtype=torch.float)
            if self.return_edge_local:
                g.edge_unit_vector = torch.tensor(
                    sd["edge_unit_vectors"], dtype=torch.float
                )
                g.current_vx = torch.tensor(
                    dyn["velocity_x"][prev_time, :], dtype=torch.float
                )
                g.current_vy = torch.tensor(
                    dyn["velocity_y"][prev_time, :], dtype=torch.float
                )
                g.volume_std = torch.tensor(
                    [self.dynamic_stats["volume"]["std"]], dtype=torch.float
                )
            if self.return_hecras_face:
                current_wd = dyn["water_depth"][prev_time, :]
                current_wd_denorm = self.denormalize(
                    current_wd,
                    self.dynamic_stats["water_depth"]["mean"],
                    self.dynamic_stats["water_depth"]["std"],
                )
                elevation_denorm = self.denormalize(
                    sd["elevation"],
                    self.static_stats["elevation"]["mean"],
                    self.static_stats["elevation"]["std"],
                ).reshape(-1)
                g.current_water_depth = torch.tensor(
                    current_wd, dtype=torch.float
                )
                g.current_water_depth_denorm = torch.tensor(
                    current_wd_denorm, dtype=torch.float
                )
                g.current_surface_elevation = torch.tensor(
                    elevation_denorm + current_wd_denorm, dtype=torch.float
                )
                g.water_depth_mean = torch.tensor(
                    [self.dynamic_stats["water_depth"]["mean"]], dtype=torch.float
                )
                g.water_depth_std = torch.tensor(
                    [self.dynamic_stats["water_depth"]["std"]], dtype=torch.float
                )
                g.local_source_rate = torch.tensor(
                    self.compute_local_source_rate(dyn, prev_time, target_time),
                    dtype=torch.float,
                )
                self.add_hecras_face_attrs(g, t_idx, self.hydrograph_ids[hydro_idx])
            if self.return_hecras_cell_balance:
                self.add_hecras_cell_balance_attrs(
                    g, prev_time, self.hydrograph_ids[hydro_idx]
                )
            if self.return_hecras_edge_flow:
                self.add_hecras_edge_flow_attrs(
                    g, prev_time, self.hydrograph_ids[hydro_idx]
                )
                if self.training_rollout_steps > 1:
                    self.add_hecras_edge_flow_rollout_attrs(
                        g,
                        prev_time,
                        self.training_rollout_steps,
                        self.hydrograph_ids[hydro_idx],
                    )

            # Determine if physics data should be returned.
            need_physics = self.return_physics or (self.noise_type == "pushforward")
            if need_physics:
                # Compute physics data in the denormalized domain.
                past_volume = float(np.sum(dyn["volume"][prev_time, :]))
                future_volume = (
                    float(np.sum(dyn["volume"][target_time + 1, :]))
                    if (target_time + 1 < dyn["volume"].shape[0])
                    else float(np.sum(dyn["volume"][target_time, :]))
                )
                avg_inflow_norm = float(
                    (
                        dyn["inflow_hydrograph"][prev_time]
                        + dyn["inflow_hydrograph"][target_time]
                    )
                    / 2
                )
                avg_precip_norm = float(
                    (
                        dyn["precipitation"][prev_time]
                        + dyn["precipitation"][target_time]
                    )
                    / 2
                )
                denorm_avg_inflow = (
                    avg_inflow_norm * self.dynamic_stats["inflow_hydrograph"]["std"]
                    + self.dynamic_stats["inflow_hydrograph"]["mean"]
                )
                denorm_avg_precip = (
                    avg_precip_norm * self.dynamic_stats["precipitation"]["std"]
                    + self.dynamic_stats["precipitation"]["mean"]
                )

                # --- New: Compute next-step inflow and precipitation for physics loss term2 ---
                if (target_time + 1) < dyn["inflow_hydrograph"].shape[0]:
                    next_inflow_norm = dyn["inflow_hydrograph"][target_time + 1]
                    next_precip_norm = dyn["precipitation"][target_time + 1]
                else:
                    next_inflow_norm = dyn["inflow_hydrograph"][target_time]
                    next_precip_norm = dyn["precipitation"][target_time]
                denorm_next_inflow = (
                    next_inflow_norm * self.dynamic_stats["inflow_hydrograph"]["std"]
                    + self.dynamic_stats["inflow_hydrograph"]["mean"]
                )
                denorm_next_precip = (
                    next_precip_norm * self.dynamic_stats["precipitation"]["std"]
                    + self.dynamic_stats["precipitation"]["mean"]
                )

                physical_area = sd["area_denorm"].reshape(-1)
                if self.hecras_face_graph is not None and self.hecras_face_graph.get(
                    "node_surface_area"
                ) is not None:
                    physical_area = self.hecras_face_graph[
                        "node_surface_area"
                    ].reshape(-1)
                if self.local_source_runoff_mode == "full_area":
                    effective_precipitation_area_sum = float(np.sum(physical_area))
                else:
                    runoff_percentage = self.denormalize(
                        sd["infiltration"],
                        self.static_stats["infiltration"]["mean"],
                        self.static_stats["infiltration"]["std"],
                    ).reshape(-1)
                    effective_precipitation_area_sum = float(
                        np.sum((runoff_percentage / 100.0) * physical_area)
                    )

                # Build the complete physics data dictionary.
                full_physics_data = {
                    "flow_future": float(
                        future_flow * self.dynamic_stats["inflow_hydrograph"]["std"]
                        + self.dynamic_stats["inflow_hydrograph"]["mean"]
                    ),
                    "precip_future": float(
                        future_precip * self.dynamic_stats["precipitation"]["std"]
                        + self.dynamic_stats["precipitation"]["mean"]
                    ),
                    "past_volume": past_volume,
                    "future_volume": future_volume,
                    "avg_inflow": denorm_avg_inflow,
                    "avg_precipitation": denorm_avg_precip,
                    "next_inflow": denorm_next_inflow,
                    "next_precip": denorm_next_precip,
                    "volume_mean": float(self.dynamic_stats["volume"]["mean"]),
                    "volume_std": float(self.dynamic_stats["volume"]["std"]),
                    "inflow_mean": float(
                        self.dynamic_stats["inflow_hydrograph"]["mean"]
                    ),
                    "inflow_std": float(self.dynamic_stats["inflow_hydrograph"]["std"]),
                    "precip_mean": float(self.dynamic_stats["precipitation"]["mean"]),
                    "precip_std": float(self.dynamic_stats["precipitation"]["std"]),
                    "num_nodes": float(sd["xy_coords"].shape[0]),
                    "area_sum": float(np.sum(physical_area)),
                    "infiltration_area_sum": effective_precipitation_area_sum,
                }
                # For pushforward noise without full physics data requested.
                if not self.return_physics and self.noise_type == "pushforward":
                    physics_data = {
                        "flow_future": full_physics_data["flow_future"],
                        "precip_future": full_physics_data["precip_future"],
                        "next_inflow": full_physics_data["next_inflow"],
                        "next_precip": full_physics_data["next_precip"],
                    }
                else:
                    physics_data = full_physics_data
                return g, physics_data
            else:
                return g
        else:
            # Test mode: Each sample returns a graph and a rollout data dictionary.
            dyn = self.dynamic_data[idx]
            node_features, _, _ = self.create_node_features(
                sd["xy_coords"],
                sd["area"],
                sd["elevation"],
                sd["slope"],
                sd["aspect"],
                sd["curvature"],
                sd["manning"],
                sd["flow_accum"],
                sd["infiltration"],
                dyn["water_depth"][0 : self.n_time_steps, :],
                dyn["volume"][0 : self.n_time_steps, :],
                dyn["precipitation"],
                0,
                self.n_time_steps,
                dyn["inflow_hydrograph"],
            )
            src, dst = sd["edge_index"]
            edges = torch.stack([torch.tensor(src), torch.tensor(dst)], dim=0).long()
            g = pyg.data.Data(edge_index=edges)
            g.edge_attr = torch.tensor(sd["edge_features"], dtype=torch.float)
            g.x = torch.tensor(node_features, dtype=torch.float)
            if self.use_fidelity_zones:
                g.zone_label = torch.tensor(self.zone_label, dtype=torch.long)
                g.zone_weight = torch.tensor(self.zone_weight, dtype=torch.float)
            if self.return_edge_local:
                g.edge_unit_vector = torch.tensor(
                    sd["edge_unit_vectors"], dtype=torch.float
                )
                g.volume_std = torch.tensor(
                    [self.dynamic_stats["volume"]["std"]], dtype=torch.float
                )
            if self.return_hecras_face:
                current_wd = dyn["water_depth"][self.n_time_steps - 1, :]
                current_wd_denorm = self.denormalize(
                    current_wd,
                    self.dynamic_stats["water_depth"]["mean"],
                    self.dynamic_stats["water_depth"]["std"],
                )
                elevation_denorm = self.denormalize(
                    sd["elevation"],
                    self.static_stats["elevation"]["mean"],
                    self.static_stats["elevation"]["std"],
                ).reshape(-1)
                g.current_water_depth = torch.tensor(
                    current_wd, dtype=torch.float
                )
                g.current_water_depth_denorm = torch.tensor(
                    current_wd_denorm, dtype=torch.float
                )
                g.current_surface_elevation = torch.tensor(
                    elevation_denorm + current_wd_denorm, dtype=torch.float
                )
                g.water_depth_mean = torch.tensor(
                    [self.dynamic_stats["water_depth"]["mean"]], dtype=torch.float
                )
                g.water_depth_std = torch.tensor(
                    [self.dynamic_stats["water_depth"]["std"]], dtype=torch.float
                )
                g.local_source_rate = torch.tensor(
                    self.compute_local_source_rate(
                        dyn, self.n_time_steps - 1, self.n_time_steps
                    ),
                    dtype=torch.float,
                )
                self.add_hecras_face_attrs(g, 0, self.hydrograph_ids[idx])
            if self.return_hecras_cell_balance:
                self.add_hecras_cell_balance_attrs(
                    g, self.n_time_steps - 1, self.hydrograph_ids[idx]
                )
            if self.return_hecras_edge_flow:
                self.add_hecras_edge_flow_attrs(
                    g, self.n_time_steps - 1, self.hydrograph_ids[idx]
                )
            rollout_data = {
                "inflow": torch.tensor(
                    dyn["inflow_hydrograph"][
                        self.n_time_steps : self.n_time_steps + self.rollout_length
                    ],
                    dtype=torch.float,
                ),
                "precipitation": torch.tensor(
                    dyn["precipitation"][
                        self.n_time_steps : self.n_time_steps + self.rollout_length
                    ],
                    dtype=torch.float,
                ),
                "water_depth_gt": torch.tensor(
                    dyn["water_depth"][
                        self.n_time_steps : self.n_time_steps + self.rollout_length
                    ],
                    dtype=torch.float,
                ),
                "volume_gt": torch.tensor(
                    dyn["volume"][
                        self.n_time_steps : self.n_time_steps + self.rollout_length
                    ],
                    dtype=torch.float,
                ),
            }
            if dyn["local_precipitation"] is not None:
                rollout_data["local_precipitation"] = torch.tensor(
                    dyn["local_precipitation"][
                        self.n_time_steps : self.n_time_steps + self.rollout_length
                    ],
                    dtype=torch.float,
                )
            return g, rollout_data

    def __len__(self) -> int:
        """Return the number of samples in the dataset."""
        return self.length

    def load_fidelity_zones(self, num_nodes: int) -> tuple[np.ndarray, np.ndarray]:
        """Load node-level fidelity zone labels and local-loss weights."""
        label_path = os.path.join(self.data_dir, self.zone_label_file)
        weight_path = os.path.join(self.data_dir, self.zone_weight_file)
        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Fidelity zone label file not found: {label_path}")
        if not os.path.exists(weight_path):
            raise FileNotFoundError(f"Fidelity zone weight file not found: {weight_path}")

        zone_label = np.loadtxt(label_path, dtype=np.int64).reshape(-1)
        zone_weight = np.loadtxt(weight_path, dtype=np.float32).reshape(-1)
        if zone_label.shape[0] != num_nodes:
            raise ValueError(
                f"Expected {num_nodes} zone labels, found {zone_label.shape[0]}"
            )
        if zone_weight.shape[0] != num_nodes:
            raise ValueError(
                f"Expected {num_nodes} zone weights, found {zone_weight.shape[0]}"
            )
        return zone_label, zone_weight

    def infer_hecras_hydrograph_id(self, hdf_path: Path) -> str:
        """Infer an H1/T1 hydrograph ID from an event-specific HEC-RAS HDF path."""
        for part in reversed(hdf_path.parts):
            match = re.fullmatch(r"plan([HT]\d+)", part, flags=re.IGNORECASE)
            if match is not None:
                return match.group(1).upper()
        for part in reversed(hdf_path.parts):
            match = re.fullmatch(r"([HT]\d+)", part, flags=re.IGNORECASE)
            if match is not None:
                return match.group(1).upper()
        raise ValueError(
            "Could not infer hydrograph ID from HDF path. Expected a path "
            f"containing planH1/H1/T1, got: {hdf_path}"
        )

    def load_hecras_event_hdf_paths(self, hdf_glob: str) -> dict[str, Path]:
        """Load event-specific HEC-RAS HDF paths keyed by hydrograph ID."""
        if hdf_glob.startswith("/"):
            matched_files = sorted(Path("/").glob(hdf_glob[1:]))
        else:
            matched_files = sorted(Path().glob(hdf_glob))
        if not matched_files:
            raise FileNotFoundError(f"No HEC-RAS HDFs matched: {hdf_glob}")

        paths_by_hydrograph = {}
        for hdf_path in matched_files:
            hydrograph_id = self.infer_hecras_hydrograph_id(hdf_path)
            if hydrograph_id in paths_by_hydrograph:
                raise ValueError(
                    f"Multiple HEC-RAS HDF files map to hydrograph {hydrograph_id}."
                )
            paths_by_hydrograph[hydrograph_id] = hdf_path
        return paths_by_hydrograph

    def load_hgn_event_time_days(
        self,
        hydrograph_id: str,
        interval: int = 1,
        skip: Optional[int] = None,
        post_peak_steps: Optional[int] = None,
    ) -> np.ndarray:
        """Load the HGN event time axis after the dataset's skip and peak trim."""
        skip = self.dynamic_skip_steps if skip is None else int(skip)
        post_peak_steps = (
            self.post_peak_steps
            if post_peak_steps is None
            else int(post_peak_steps)
        )
        inflow_path = os.path.join(
            self.data_dir, f"{self.prefix}_US_InF_{hydrograph_id}.txt"
        )
        inflow = np.loadtxt(inflow_path, delimiter="\t")
        time_days = np.asarray(inflow[skip::interval, 0], dtype=np.float64)
        inflow_hydrograph = np.asarray(inflow[skip::interval, 1], dtype=np.float64)
        peak_time_idx = int(np.argmax(inflow_hydrograph))
        return time_days[: peak_time_idx + post_peak_steps]

    @staticmethod
    def load_dataset_time_contract(data_dir: Union[str, Path]) -> dict[str, Any]:
        """Read optional validated timing metadata without changing legacy defaults."""
        contract_path = Path(data_dir) / "hgn_dataset_validation.json"
        if not contract_path.is_file():
            return {}
        try:
            payload = json.loads(contract_path.read_text())
            contract = payload.get("model_time_contract", {})
            if not isinstance(contract, dict):
                raise TypeError("model_time_contract must be a JSON object")
            return contract
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(
                f"Invalid HydroGraphNet dataset timing contract: {contract_path}"
            ) from exc

    @staticmethod
    def match_hgn_times_to_hdf(
        hgn_time_days: np.ndarray, hdf_time_days: np.ndarray
    ) -> np.ndarray:
        """Return HDF output indices corresponding to each HGN target time."""
        matched = np.searchsorted(hdf_time_days, hgn_time_days)
        matched = np.clip(matched, 0, hdf_time_days.shape[0] - 1)
        previous = np.maximum(matched - 1, 0)
        choose_previous = (
            np.abs(hdf_time_days[previous] - hgn_time_days)
            < np.abs(hdf_time_days[matched] - hgn_time_days)
        )
        matched[choose_previous] = previous[choose_previous]
        differences_seconds = np.abs(hdf_time_days[matched] - hgn_time_days) * 86400.0
        if np.max(differences_seconds) > 1e-3:
            raise ValueError(
                "HGN target times are not represented in the HDF output series; "
                f"max mismatch is {np.max(differences_seconds)} seconds."
            )
        return matched

    def map_hgn_nodes_to_hdf_cells(self, hdf_xy: np.ndarray) -> np.ndarray:
        """Map active HGN node order back to original HDF cell indices by XY."""
        if self.static_data_raw_xy is None:
            raise ValueError("HGN raw XY coordinates are not loaded.")
        distances, original_indices = scipy_spatial.KDTree(hdf_xy).query(
            self.static_data_raw_xy, k=1
        )
        if (
            np.max(distances) > 1e-6
            or np.unique(original_indices).size != self.static_data_raw_xy.shape[0]
        ):
            raise ValueError(
                "HGN-to-HDF coordinate mapping is not unique or exceeds tolerance."
            )
        return original_indices.astype(np.int64)

    def load_hecras_cell_balance_delta_by_hydrograph(self) -> dict[str, np.ndarray]:
        """Load cell-budget deltas aligned to HGN prediction intervals.

        Formal SI five-minute targets are derived from HEC-RAS period-average
        face flow plus precipitation, then stored as event-sharded arrays. The
        legacy NPZ and native-HDF paths remain available for compatibility and
        diagnostics; point-sampled five-minute ``Cell Flow Balance`` must not be
        treated as a conservative interval integral.
        """
        target_dir = getattr(self, "hecras_cell_balance_target_dir", None)
        cell_balance_npz = getattr(self, "hecras_cell_balance_npz", None)
        cell_balance_glob = getattr(self, "hecras_cell_balance_glob", None)
        if target_dir is not None:
            if cell_balance_npz is not None or (
                cell_balance_glob is not None
            ):
                raise ValueError(
                    "Specify only one Cell Flow Balance target source."
                )
            directory = Path(target_dir)
            selected_path = directory / "selected_node_index.npy"
            if not selected_path.is_file():
                raise FileNotFoundError(selected_path)
            selected_nodes = np.load(selected_path, mmap_mode="r")
            if selected_nodes.ndim != 1 or (
                selected_nodes.size
                and int(np.max(selected_nodes))
                >= self.static_data["xy_coords"].shape[0]
            ):
                raise ValueError(
                    "Cell-budget selected node indexes do not match the HGN graph."
                )
            loaded = {}
            suffix = "period_average_cell_budget_delta_selected_m3"
            for hydrograph_id in self.hydrograph_ids:
                path = directory / f"{hydrograph_id}_{suffix}.npy"
                if not path.is_file():
                    raise FileNotFoundError(path)
                target = np.load(path, mmap_mode="r")
                if target.ndim != 2 or target.shape[1] != selected_nodes.size:
                    raise ValueError(
                        f"Cell-budget target {path} must have shape "
                        f"(transitions, {selected_nodes.size}), got {target.shape}."
                    )
                loaded[hydrograph_id] = {
                    "selected": target,
                    "selected_nodes": selected_nodes,
                }
            return loaded

        if cell_balance_npz is not None:
            if cell_balance_glob is not None:
                raise ValueError(
                    "Specify only one of hecras_cell_balance_npz and "
                    "hecras_cell_balance_glob."
                )
            path = Path(cell_balance_npz)
            if not path.is_file():
                raise FileNotFoundError(path)
            requested_hydrographs = set(self.hydrograph_ids)
            loaded: dict[str, np.ndarray] = {}
            with np.load(path, allow_pickle=False) as data:
                suffix = self.hecras_cell_balance_npz_key_suffix
                for key in data.files:
                    if not key.endswith(suffix):
                        continue
                    hydrograph_id = key[: -len(suffix)]
                    if hydrograph_id not in requested_hydrographs:
                        continue
                    target = np.asarray(data[key], dtype=np.float32)
                    if target.ndim != 2:
                        raise ValueError(
                            f"{key} in {path} must have shape (transitions, nodes), "
                            f"got {target.shape}."
                        )
                    if target.shape[1] != self.static_data["xy_coords"].shape[0]:
                        raise ValueError(
                            f"{key} has {target.shape[1]} nodes, but the HGN graph "
                            f"has {self.static_data['xy_coords'].shape[0]}."
                        )
                    if not np.all(np.isfinite(target)):
                        raise ValueError(f"{key} in {path} contains non-finite values.")
                    loaded[hydrograph_id] = target
            missing = sorted(requested_hydrographs - set(loaded))
            if missing:
                raise KeyError(
                    f"Missing cell-balance NPZ targets for {missing} in {path}; "
                    f"expected suffix {self.hecras_cell_balance_npz_key_suffix!r}."
                )
            return loaded

        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required when return_hecras_cell_balance=True."
            ) from exc

        if self.hecras_cell_balance_glob is None:
            raise ValueError(
                "return_hecras_cell_balance=True requires either "
                "hecras_cell_balance_npz or hecras_cell_balance_glob."
            )
        hdf_paths = self.load_hecras_event_hdf_paths(self.hecras_cell_balance_glob)
        balance_by_hydrograph = {}
        for dyn in self.dynamic_data:
            hydrograph_id = dyn["hydro_id"]
            hdf_path = hdf_paths.get(hydrograph_id)
            if hdf_path is None:
                available = ", ".join(sorted(hdf_paths))
                raise KeyError(
                    f"No HEC-RAS cell-balance HDF loaded for {hydrograph_id}. "
                    f"Available hydrographs: {available}"
                )

            hgn_time_days = self.load_hgn_event_time_days(hydrograph_id)
            if hgn_time_days.shape[0] != dyn["volume"].shape[0]:
                raise ValueError(
                    f"HGN time count mismatch for {hydrograph_id}: "
                    f"{hgn_time_days.shape[0]} times vs {dyn['volume'].shape[0]} volumes."
                )

            with h5py.File(hdf_path, "r") as hdf:
                for required_path in (
                    self.hecras_cell_balance_path,
                    self.hecras_precipitation_path,
                    self.hecras_result_time_path,
                    self.hecras_cell_xy_path,
                    self.hecras_cell_surface_area_path,
                ):
                    if required_path not in hdf:
                        raise KeyError(
                            f"Required HEC-RAS HDF path not found in {hdf_path}: "
                            f"{required_path}"
                        )
                hdf_time_days = np.asarray(
                    hdf[self.hecras_result_time_path], dtype=np.float64
                )
                target_time_indices = self.match_hgn_times_to_hdf(
                    hgn_time_days, hdf_time_days
                )
                original_indices = self.map_hgn_nodes_to_hdf_cells(
                    np.asarray(hdf[self.hecras_cell_xy_path], dtype=np.float64)
                )
                cell_surface_area = np.asarray(
                    hdf[self.hecras_cell_surface_area_path], dtype=np.float64
                )[original_indices]
                hdf_time_seconds = hdf_time_days * 86400.0
                balance_dataset = hdf[self.hecras_cell_balance_path]
                precip_dataset = hdf[self.hecras_precipitation_path]
                balance_units = balance_dataset.attrs.get("Units", b"")
                precip_units = precip_dataset.attrs.get("Units", b"")
                if isinstance(balance_units, bytes):
                    balance_units = balance_units.decode("utf-8")
                if isinstance(precip_units, bytes):
                    precip_units = precip_units.decode("utf-8")
                balance_units = str(balance_units).replace("^", "").lower()
                precip_units = str(precip_units).lower()
                if balance_units in {"m3/s", "m3/sec"} and precip_units == "mm":
                    precipitation_depth_to_length = 1.0e-3
                elif balance_units in {
                    "ft3/s",
                    "ft3/sec",
                    "cfs",
                } and precip_units in {"in", "inch", "inches"}:
                    precipitation_depth_to_length = 1.0 / 12.0
                else:
                    raise ValueError(
                        "Unsupported HEC-RAS Cell Flow Balance/precipitation "
                        f"unit pair in {hdf_path}: {balance_units!r}, "
                        f"{precip_units!r}."
                    )
                deltas = np.zeros(
                    (target_time_indices.shape[0] - 1, original_indices.size),
                    dtype=np.float64,
                )
                for transition_index, (start, end) in enumerate(
                    zip(target_time_indices[:-1], target_time_indices[1:])
                ):
                    if end <= start:
                        raise ValueError(
                            "Target times must map to increasing HDF output indices."
                        )
                    balance_window = np.nan_to_num(
                        np.asarray(balance_dataset[start : end + 1, :], dtype=np.float64)[
                            :, original_indices
                        ],
                        nan=0.0,
                    )
                    balance_delta = np.trapezoid(
                        balance_window,
                        x=hdf_time_seconds[start : end + 1],
                        axis=0,
                    )
                    precip_delta = (
                        (
                            np.asarray(precip_dataset[end, :], dtype=np.float64)[
                                original_indices
                            ]
                            - np.asarray(precip_dataset[start, :], dtype=np.float64)[
                                original_indices
                            ]
                        )
                        * precipitation_depth_to_length
                        * cell_surface_area
                    )
                    deltas[transition_index] = balance_delta + precip_delta
            balance_by_hydrograph[hydrograph_id] = deltas.astype(np.float32)
        return balance_by_hydrograph

    def load_hecras_face_graph(self, num_nodes: int) -> dict[str, np.ndarray]:
        """Load HEC-RAS internal face connectivity aligned to HGN node order."""
        if self.hecras_face_graph_file is None:
            raise ValueError(
                "return_hecras_face=True requires hecras_face_graph_file."
            )
        face_data = np.load(self.hecras_face_graph_file)
        face_index = face_data["internal_face_index"].astype(np.int64)
        face_length = face_data["internal_face_length"].astype(np.float32)
        face_normal = face_data["internal_normal_unit"].astype(np.float32)
        hdf_face_index = face_data["internal_hdf_face_index"].astype(np.int64)
        if face_index.shape[0] != 2:
            raise ValueError(
                f"Expected HEC-RAS face index shape (2, F), got {face_index.shape}"
            )
        if face_index.size and int(face_index.max()) >= num_nodes:
            raise ValueError(
                "HEC-RAS face graph contains cell indices outside the HGN node count."
            )
        if face_length.shape[0] != face_index.shape[1]:
            raise ValueError("HEC-RAS face length count does not match face count.")
        if hdf_face_index.shape[0] != face_index.shape[1]:
            raise ValueError("HEC-RAS HDF face index count does not match face count.")
        if face_normal.shape[0] != face_index.shape[1] or face_normal.shape[1] != 2:
            raise ValueError("HEC-RAS face normal shape must be (num_faces, 2).")
        node_volume_table_top_elevation = (
            np.asarray(
                face_data["hgn_node_volume_table_top_elevation"], dtype=np.float32
            )
            if "hgn_node_volume_table_top_elevation" in face_data.files
            else None
        )
        if (
            node_volume_table_top_elevation is not None
            and node_volume_table_top_elevation.shape != (num_nodes,)
        ):
            raise ValueError(
                "HEC-RAS node volume-table top elevation must have shape "
                f"({num_nodes},), got {node_volume_table_top_elevation.shape}."
            )
        stored_zone_sha256 = (
            str(np.asarray(face_data["zone_label_sha256"]).reshape(-1)[0])
            if "zone_label_sha256" in face_data.files
            else ""
        )
        if stored_zone_sha256 and self.use_fidelity_zones:
            zone_path = Path(self.data_dir) / self.zone_label_file
            current_zone_sha256 = hashlib.sha256(zone_path.read_bytes()).hexdigest()
            if current_zone_sha256 != stored_zone_sha256:
                raise ValueError(
                    "The HEC-RAS face graph was built for a different fidelity "
                    f"zone mask: {stored_zone_sha256} != {current_zone_sha256}."
                )
        boundary_node_mask = (
            np.asarray(face_data["boundary_node_mask"], dtype=np.bool_)
            if "boundary_node_mask" in face_data.files
            else None
        )
        control_volume_label = (
            np.asarray(
                face_data["high_interior_control_volume_label"], dtype=np.int64
            )
            if "high_interior_control_volume_label" in face_data.files
            else None
        )
        for name, values in (
            ("boundary_node_mask", boundary_node_mask),
            ("high_interior_control_volume_label", control_volume_label),
        ):
            if values is not None and values.shape != (num_nodes,):
                raise ValueError(
                    f"HEC-RAS {name} must have shape ({num_nodes},), got "
                    f"{values.shape}."
                )
        return {
            "face_index": face_index,
            "face_length": face_length,
            "face_normal": face_normal,
            "hdf_face_index": hdf_face_index,
            "node_surface_area": (
                np.asarray(face_data["hgn_node_surface_area"], dtype=np.float32)
                if "hgn_node_surface_area" in face_data.files
                else None
            ),
            "node_volume_table_top_elevation": node_volume_table_top_elevation,
            "boundary_node_mask": boundary_node_mask,
            "high_interior_control_volume_label": control_volume_label,
            "zone_label_sha256": stored_zone_sha256,
        }

    def load_hecras_face_velocity(self) -> np.ndarray:
        """Load HEC-RAS face velocity time series from an HDF result file."""
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required when hecras_face_velocity_file is provided."
            ) from exc

        with h5py.File(self.hecras_face_velocity_file, "r") as hdf:
            if self.hecras_face_velocity_path not in hdf:
                raise KeyError(
                    f"HEC-RAS face velocity path not found: {self.hecras_face_velocity_path}"
                )
            return hdf[self.hecras_face_velocity_path][:].astype(np.float32)

    def load_hecras_face_velocity_by_hydrograph(self) -> dict[str, np.ndarray]:
        """Load event-specific face velocity arrays keyed by hydrograph ID."""
        try:
            import h5py
        except ImportError as exc:
            raise ImportError(
                "h5py is required when hecras_face_velocity_glob is provided."
            ) from exc

        if self.hecras_face_velocity_glob.startswith("/"):
            matched_files = sorted(Path("/").glob(self.hecras_face_velocity_glob[1:]))
        else:
            matched_files = sorted(Path().glob(self.hecras_face_velocity_glob))
        if not matched_files:
            raise FileNotFoundError(
                f"No HEC-RAS face velocity HDFs matched: {self.hecras_face_velocity_glob}"
            )

        velocity_by_hydrograph = {}
        for hdf_path in matched_files:
            hydrograph_id = None
            for part in reversed(hdf_path.parts):
                match = re.fullmatch(r"plan([HT]\d+)", part, flags=re.IGNORECASE)
                if match is not None:
                    hydrograph_id = match.group(1).upper()
                    break
            if hydrograph_id is None:
                for part in reversed(hdf_path.parts):
                    match = re.fullmatch(r"([HT]\d+)", part, flags=re.IGNORECASE)
                    if match is not None:
                        hydrograph_id = match.group(1).upper()
                        break
            if hydrograph_id is None:
                raise ValueError(
                    "Could not infer hydrograph ID from HDF path. Expected a path "
                    f"containing planH1/H1/T1, got: {hdf_path}"
                )
            if hydrograph_id in velocity_by_hydrograph:
                raise ValueError(
                    f"Multiple HEC-RAS HDF files map to hydrograph {hydrograph_id}."
                )
            with h5py.File(hdf_path, "r") as hdf:
                if self.hecras_face_velocity_path not in hdf:
                    raise KeyError(
                        "HEC-RAS face velocity path not found in "
                        f"{hdf_path}: {self.hecras_face_velocity_path}"
                    )
                velocity_by_hydrograph[hydrograph_id] = hdf[
                    self.hecras_face_velocity_path
                ][:].astype(np.float32)
        return velocity_by_hydrograph

    def add_hecras_face_attrs(
        self, graph, time_index: int, hydrograph_id: Optional[str] = None
    ) -> None:
        """Attach HEC-RAS true-face attributes to a PyG graph sample."""
        if self.hecras_face_graph is None:
            return
        graph.hecras_face_index = torch.tensor(
            self.hecras_face_graph["face_index"], dtype=torch.long
        )
        graph.hecras_face_length = torch.tensor(
            self.hecras_face_graph["face_length"], dtype=torch.float
        )
        graph.hecras_face_normal = torch.tensor(
            self.hecras_face_graph["face_normal"], dtype=torch.float
        )
        node_surface_area = self.hecras_face_graph.get("node_surface_area")
        if node_surface_area is None:
            node_surface_area = self.static_data["area_denorm"].reshape(-1)
        graph.hecras_node_area = torch.tensor(node_surface_area, dtype=torch.float)
        node_volume_table_top_elevation = self.hecras_face_graph.get(
            "node_volume_table_top_elevation"
        )
        if node_volume_table_top_elevation is not None:
            graph.hecras_node_volume_table_top_elevation = torch.tensor(
                node_volume_table_top_elevation, dtype=torch.float
            )
        src, dst = self.hecras_face_graph["face_index"]
        xy = self.static_data_raw_xy
        center_dx = xy[dst, 0] - xy[src, 0]
        center_dy = xy[dst, 1] - xy[src, 1]
        center_distance = np.sqrt(center_dx * center_dx + center_dy * center_dy)
        elevation = self.denormalize(
            self.static_data["elevation"],
            self.static_stats["elevation"]["mean"],
            self.static_stats["elevation"]["std"],
        ).reshape(-1)
        manning = self.denormalize(
            self.static_data["manning"],
            self.static_stats["manning"]["mean"],
            self.static_stats["manning"]["std"],
        ).reshape(-1)
        infiltration = self.denormalize(
            self.static_data["infiltration"],
            self.static_stats["infiltration"]["mean"],
            self.static_stats["infiltration"]["std"],
        ).reshape(-1)
        area = self.static_data["area_denorm"].reshape(-1)
        elevation_diff = elevation[dst] - elevation[src]
        bed_slope = elevation_diff / np.maximum(center_distance, 1e-6)
        manning_mean = 0.5 * (manning[src] + manning[dst])
        manning_diff = manning[dst] - manning[src]
        infiltration_mean = 0.5 * (infiltration[src] + infiltration[dst])
        infiltration_diff = infiltration[dst] - infiltration[src]
        area_mean = 0.5 * (area[src] + area[dst])
        area_ratio = np.maximum(area[src], area[dst]) / np.maximum(
            np.minimum(area[src], area[dst]), 1e-6
        )
        if self.use_fidelity_zones:
            zone_src = self.zone_label[src]
            zone_dst = self.zone_label[dst]
            same_zone = (zone_src == zone_dst).astype(np.float32)
            zone3_touching = ((zone_src == 3) | (zone_dst == 3)).astype(np.float32)
            zone3_internal = ((zone_src == 3) & (zone_dst == 3)).astype(np.float32)
        else:
            same_zone = np.ones_like(center_distance, dtype=np.float32)
            zone3_touching = np.zeros_like(center_distance, dtype=np.float32)
            zone3_internal = np.zeros_like(center_distance, dtype=np.float32)
        graph.hecras_edge_physical_features = torch.tensor(
            np.stack(
                [
                    center_distance,
                    elevation_diff,
                    bed_slope,
                    manning_mean,
                    manning_diff,
                    infiltration_mean,
                    infiltration_diff,
                    area_mean,
                    area_ratio,
                    same_zone,
                    zone3_touching,
                    zone3_internal,
                ],
                axis=1,
            ),
            dtype=torch.float,
        )
        graph.volume_std = torch.tensor(
            [self.dynamic_stats["volume"]["std"]], dtype=torch.float
        )

        face_velocity = self.hecras_face_velocity
        if hydrograph_id is not None and self.hecras_face_velocity_by_hydrograph:
            face_velocity = self.hecras_face_velocity_by_hydrograph.get(hydrograph_id)
            if face_velocity is None:
                available = ", ".join(sorted(self.hecras_face_velocity_by_hydrograph))
                raise KeyError(
                    f"No HEC-RAS face velocity HDF loaded for {hydrograph_id}. "
                    f"Available hydrographs: {available}"
                )

        if face_velocity is not None:
            hdf_time_index = time_index + self.hecras_face_time_offset
            if hdf_time_index < 0 or hdf_time_index >= face_velocity.shape[0]:
                raise IndexError(
                    f"HEC-RAS face velocity time index {hdf_time_index} is outside "
                    f"0..{face_velocity.shape[0] - 1}"
                )
            graph.hecras_face_velocity = torch.tensor(
                face_velocity[
                    hdf_time_index, self.hecras_face_graph["hdf_face_index"]
                ],
                dtype=torch.float,
            )

    def add_hecras_cell_balance_attrs(
        self, graph, transition_index: int, hydrograph_id: str
    ) -> None:
        """Attach the formal HEC-RAS cell budget delta for one HGN interval."""
        cell_balance_delta = self.hecras_cell_balance_delta_by_hydrograph.get(
            hydrograph_id
        )
        if cell_balance_delta is None:
            available = ", ".join(sorted(self.hecras_cell_balance_delta_by_hydrograph))
            raise KeyError(
                f"No HEC-RAS cell-balance target loaded for {hydrograph_id}. "
                f"Available hydrographs: {available}"
            )
        transition_index += self.hecras_cell_balance_time_offset
        target_length = (
            cell_balance_delta["selected"].shape[0]
            if isinstance(cell_balance_delta, dict)
            else cell_balance_delta.shape[0]
        )
        if transition_index < 0 or transition_index >= target_length:
            raise IndexError(
                f"HEC-RAS cell-balance transition index {transition_index} is outside "
                f"0..{target_length - 1} for {hydrograph_id}."
            )
        if isinstance(cell_balance_delta, dict):
            values = np.zeros(graph.x.shape[0], dtype=np.float32)
            values[np.asarray(cell_balance_delta["selected_nodes"])] = (
                cell_balance_delta["selected"][transition_index]
            )
        else:
            values = cell_balance_delta[transition_index]
        graph.hecras_cell_balance_delta = torch.tensor(values, dtype=torch.float)
        graph.volume_std = torch.tensor(
            [self.dynamic_stats["volume"]["std"]], dtype=torch.float
        )

    def load_hecras_edge_flow_delta_by_hydrograph(self) -> dict[str, np.ndarray]:
        """Load precomputed HEC-RAS Face Flow deltas aligned to HGN intervals."""
        if self.hecras_conservative_edge_target_dir is not None:
            if self.hecras_edge_flow_mode != "conservative_sharded":
                raise ValueError(
                    "hecras_conservative_edge_target_dir requires "
                    "hecras_edge_flow_mode='conservative_sharded'."
                )
            directory = Path(self.hecras_conservative_edge_target_dir)
            if not directory.is_dir():
                raise FileNotFoundError(directory)
            selected_path = directory / "selected_node_index.npy"
            if not selected_path.is_file():
                raise FileNotFoundError(selected_path)
            selected_nodes = np.load(selected_path, mmap_mode="r")
            targets = {}
            array_names = (
                "raw_internal_face_delta_m3",
                "projected_internal_face_delta_m3",
                "target_internal_divergence_selected_m3",
                "omitted_boundary_source_selected_m3",
                "precipitation_volume_selected_m3",
            )
            for hydrograph_id in self.hydrograph_ids:
                arrays = {}
                for name in array_names:
                    path = directory / f"{hydrograph_id}_{name}.npy"
                    if not path.is_file():
                        raise FileNotFoundError(path)
                    arrays[name] = np.load(path, mmap_mode="r")
                transition_count = arrays[
                    "projected_internal_face_delta_m3"
                ].shape[0]
                if any(
                    values.shape[0] != transition_count
                    for values in arrays.values()
                ):
                    raise ValueError(
                        f"Conservative target row mismatch for {hydrograph_id}."
                    )
                if any(
                    arrays[name].shape[1] != selected_nodes.size
                    for name in array_names[2:]
                ):
                    raise ValueError(
                        f"Selected-node target width mismatch for {hydrograph_id}."
                    )
                if arrays["raw_internal_face_delta_m3"].shape != arrays[
                    "projected_internal_face_delta_m3"
                ].shape:
                    raise ValueError(
                        f"Raw/projected face target shape mismatch for {hydrograph_id}."
                    )
                dynamic_by_id = {
                    item["hydro_id"]: item
                    for item in getattr(self, "dynamic_data", [])
                }
                if hydrograph_id in dynamic_by_id:
                    dynamic = dynamic_by_id[hydrograph_id]
                    required_transition_count = (
                        self.hecras_edge_flow_time_offset
                        + max(int(dynamic["water_depth"].shape[0]) - 1, 0)
                    )
                    if transition_count < required_transition_count:
                        raise ValueError(
                            f"Conservative target for {hydrograph_id} has "
                            f"{transition_count} transitions, but offset "
                            f"{self.hecras_edge_flow_time_offset} and the trimmed "
                            f"dynamic sequence require {required_transition_count}."
                        )
                targets[hydrograph_id] = {
                    "raw_face": arrays["raw_internal_face_delta_m3"],
                    "face": arrays["projected_internal_face_delta_m3"],
                    "internal_selected": arrays[
                        "target_internal_divergence_selected_m3"
                    ],
                    "boundary_selected": arrays[
                        "omitted_boundary_source_selected_m3"
                    ],
                    "precipitation_selected": arrays[
                        "precipitation_volume_selected_m3"
                    ],
                    "selected_nodes": selected_nodes,
                }
            return targets

        if self.hecras_edge_flow_npz is None:
            raise ValueError(
                "return_hecras_edge_flow=True requires hecras_edge_flow_npz."
            )
        path = Path(self.hecras_edge_flow_npz)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        requested_hydrographs = set(self.hydrograph_ids)
        deltas = {}
        if self.hecras_edge_flow_mode == "internal_plus_boundary_source":
            internal_suffix = "_internal_edge_delta"
            boundary_suffix = "_boundary_source_delta"
            face_suffix = "_internal_face_delta"
            internal = {}
            boundary = {}
            face = {}
            for key in data.files:
                if key.endswith(internal_suffix):
                    hydrograph_id = key[: -len(internal_suffix)]
                    if hydrograph_id in requested_hydrographs:
                        internal[hydrograph_id] = np.asarray(
                            data[key], dtype=np.float32
                        )
                elif key.endswith(boundary_suffix):
                    hydrograph_id = key[: -len(boundary_suffix)]
                    if hydrograph_id in requested_hydrographs:
                        boundary[hydrograph_id] = np.asarray(
                            data[key], dtype=np.float32
                        )
                elif key.endswith(face_suffix):
                    hydrograph_id = key[: -len(face_suffix)]
                    if hydrograph_id in requested_hydrographs:
                        face[hydrograph_id] = np.asarray(data[key], dtype=np.float32)
            missing_boundary = sorted(set(internal) - set(boundary))
            missing_internal = sorted(set(boundary) - set(internal))
            missing_requested = sorted(requested_hydrographs - set(internal))
            if missing_boundary or missing_internal or missing_requested:
                raise KeyError(
                    "internal_plus_boundary_source requires paired arrays. "
                    f"Missing boundary for: {missing_boundary}; "
                    f"missing internal for: {missing_internal}; "
                    f"missing requested hydrographs: {missing_requested}."
                )
            if not internal:
                available = ", ".join(data.files)
                raise KeyError(
                    "No paired internal/boundary edge-flow arrays found in "
                    f"{path}. Available arrays: {available}"
                )
            return {
                hydrograph_id: {
                    "internal": internal[hydrograph_id],
                    "boundary": boundary[hydrograph_id],
                    **(
                        {"face": face[hydrograph_id]}
                        if hydrograph_id in face
                        else {}
                    ),
                }
                for hydrograph_id in sorted(internal)
            }

        suffix = f"_{self.hecras_edge_flow_mode}_edge_delta"
        for key in data.files:
            if key.endswith(suffix):
                hydrograph_id = key[: -len(suffix)]
                if hydrograph_id in requested_hydrographs:
                    deltas[hydrograph_id] = np.asarray(data[key], dtype=np.float32)
        if not deltas:
            available = ", ".join(data.files)
            raise KeyError(
                f"No edge-flow arrays ending with {suffix!r} found in {path}. "
                f"Available arrays: {available}"
            )
        return deltas

    def load_hecras_edge_flow_face_stats(self) -> Optional[dict[str, np.ndarray]]:
        """Load per-internal-face target statistics for edge-flux losses."""
        if self.hecras_edge_flow_face_stats_npz is None:
            return None
        path = Path(self.hecras_edge_flow_face_stats_npz)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        required = ("face_mean", "face_std", "face_rms")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(
                f"Missing arrays {missing} in HEC-RAS edge-flow face stats {path}."
            )
        stats = {
            key: np.asarray(data[key], dtype=np.float32).reshape(-1)
            for key in required
        }
        if self.hecras_face_graph is not None:
            expected_faces = self.hecras_face_graph["face_index"].shape[1]
            for key, value in stats.items():
                if value.shape[0] != expected_faces:
                    raise ValueError(
                        f"{key} has {value.shape[0]} entries, but face graph has "
                        f"{expected_faces} internal faces."
                    )
        return stats

    def load_hecras_edge_flow_scale_stats(self) -> Optional[dict[str, np.ndarray]]:
        """Load event and transition scale statistics for Face Flow targets."""
        if self.hecras_edge_flow_scale_stats_npz is None:
            return None
        path = Path(self.hecras_edge_flow_scale_stats_npz)
        if not path.exists():
            raise FileNotFoundError(path)
        data = np.load(path)
        required = ("event_ids", "event_rms", "global_rms")
        missing = [key for key in required if key not in data.files]
        if missing:
            raise KeyError(
                f"Missing arrays {missing} in HEC-RAS edge-flow scale stats {path}."
            )
        event_ids = [str(value) for value in data["event_ids"].tolist()]
        event_rms = np.asarray(data["event_rms"], dtype=np.float32).reshape(-1)
        if len(event_ids) != event_rms.shape[0]:
            raise ValueError(
                f"event_ids has {len(event_ids)} entries, but event_rms has "
                f"{event_rms.shape[0]} entries in {path}."
            )
        transition_rms = {}
        for hydrograph_id in event_ids:
            key = f"{hydrograph_id}_transition_rms"
            if key not in data.files:
                raise KeyError(f"Missing transition scale array {key!r} in {path}.")
            transition_rms[hydrograph_id] = np.asarray(
                data[key], dtype=np.float32
            ).reshape(-1)
        return {
            "event_ids": np.asarray(event_ids),
            "event_rms_by_id": dict(zip(event_ids, event_rms)),
            "transition_rms_by_id": transition_rms,
            "global_rms": np.asarray(data["global_rms"], dtype=np.float32).reshape(-1),
        }

    def add_hecras_edge_flow_attrs(
        self, graph, transition_index: int, hydrograph_id: str
    ) -> None:
        """Attach one precomputed HEC-RAS Face Flow delta target."""
        edge_flow_target = self.hecras_edge_flow_delta_by_hydrograph.get(hydrograph_id)
        if edge_flow_target is None:
            available = ", ".join(sorted(self.hecras_edge_flow_delta_by_hydrograph))
            raise KeyError(
                f"No HEC-RAS edge-flow target loaded for {hydrograph_id}. "
                f"Available hydrographs: {available}"
            )
        transition_index = transition_index + self.hecras_edge_flow_time_offset
        if isinstance(edge_flow_target, dict):
            target_length = edge_flow_target.get(
                "internal", edge_flow_target.get("face")
            ).shape[0]
        else:
            target_length = edge_flow_target.shape[0]
        if transition_index < 0 or transition_index >= target_length:
            raise IndexError(
                f"HEC-RAS edge-flow transition index {transition_index} is outside "
                f"0..{target_length - 1} for {hydrograph_id}."
            )
        if isinstance(edge_flow_target, dict):
            if "internal_selected" in edge_flow_target:
                selected_nodes = np.asarray(
                    edge_flow_target["selected_nodes"], dtype=np.int64
                )
                internal_delta = np.zeros(graph.x.shape[0], dtype=np.float32)
                boundary_delta = np.zeros_like(internal_delta)
                precipitation_delta = np.zeros_like(internal_delta)
                internal_delta[selected_nodes] = edge_flow_target[
                    "internal_selected"
                ][transition_index]
                boundary_delta[selected_nodes] = edge_flow_target[
                    "boundary_selected"
                ][transition_index]
                precipitation_delta[selected_nodes] = edge_flow_target[
                    "precipitation_selected"
                ][transition_index]
                graph.hecras_local_source_delta = torch.tensor(
                    precipitation_delta, dtype=torch.float
                )
                graph.hecras_conservative_selected_node_mask = torch.tensor(
                    np.isin(np.arange(graph.x.shape[0]), selected_nodes),
                    dtype=torch.bool,
                )
            else:
                internal_delta = edge_flow_target["internal"][transition_index]
                boundary_delta = edge_flow_target["boundary"][transition_index]
            graph.hecras_edge_internal_delta = torch.tensor(
                internal_delta, dtype=torch.float
            )
            graph.hecras_edge_boundary_source_delta = torch.tensor(
                boundary_delta, dtype=torch.float
            )
            graph.hecras_edge_flow_delta = torch.tensor(
                internal_delta + boundary_delta, dtype=torch.float
            )
            if "face" in edge_flow_target:
                previous_face_delta = (
                    np.zeros_like(edge_flow_target["face"][transition_index])
                    if transition_index == 0
                    else edge_flow_target["face"][transition_index - 1]
                )
                graph.hecras_internal_face_flow_delta = torch.tensor(
                    edge_flow_target["face"][transition_index], dtype=torch.float
                )
                if "raw_face" in edge_flow_target:
                    graph.hecras_raw_internal_face_flow_delta = torch.tensor(
                        edge_flow_target["raw_face"][transition_index],
                        dtype=torch.float,
                    )
                graph.hecras_previous_internal_face_flow_delta = torch.tensor(
                    previous_face_delta, dtype=torch.float
                )
                if self.hecras_edge_flow_face_stats is not None:
                    graph.hecras_internal_face_flow_mean = torch.tensor(
                        self.hecras_edge_flow_face_stats["face_mean"],
                        dtype=torch.float,
                    )
                    graph.hecras_internal_face_flow_std = torch.tensor(
                        self.hecras_edge_flow_face_stats["face_std"],
                        dtype=torch.float,
                    )
                    graph.hecras_internal_face_flow_rms = torch.tensor(
                        self.hecras_edge_flow_face_stats["face_rms"],
                        dtype=torch.float,
                    )
                if self.hecras_edge_flow_scale_stats is not None:
                    event_rms = self.hecras_edge_flow_scale_stats[
                        "event_rms_by_id"
                    ].get(hydrograph_id)
                    transition_rms_array = self.hecras_edge_flow_scale_stats[
                        "transition_rms_by_id"
                    ].get(hydrograph_id)
                    if event_rms is None or transition_rms_array is None:
                        available = ", ".join(
                            sorted(
                                self.hecras_edge_flow_scale_stats[
                                    "transition_rms_by_id"
                                ]
                            )
                        )
                        raise KeyError(
                            f"No Face Flow scale stats for {hydrograph_id}. "
                            f"Available hydrographs: {available}"
                        )
                    if transition_index >= transition_rms_array.shape[0]:
                        raise IndexError(
                            f"Transition {transition_index} is outside scale stats "
                            f"0..{transition_rms_array.shape[0] - 1} for "
                            f"{hydrograph_id}."
                        )
                    graph.hecras_internal_face_flow_global_rms = torch.tensor(
                        self.hecras_edge_flow_scale_stats["global_rms"],
                        dtype=torch.float,
                    )
                    graph.hecras_internal_face_flow_event_rms = torch.tensor(
                        [event_rms], dtype=torch.float
                    )
                    graph.hecras_internal_face_flow_transition_rms = torch.tensor(
                        [transition_rms_array[transition_index]], dtype=torch.float
                    )
        else:
            graph.hecras_edge_flow_delta = torch.tensor(
                edge_flow_target[transition_index], dtype=torch.float
            )
        if self.hecras_boundary_node_mask is not None:
            graph.hecras_boundary_node_mask = torch.tensor(
                self.hecras_boundary_node_mask, dtype=torch.bool
            )
        if self.hecras_high_interior_control_volume_label is not None:
            graph.hecras_high_interior_control_volume_label = torch.tensor(
                self.hecras_high_interior_control_volume_label, dtype=torch.long
            )
        graph.volume_std = torch.tensor(
            [self.dynamic_stats["volume"]["std"]], dtype=torch.float
        )

    def add_hecras_edge_flow_rollout_attrs(
        self,
        graph,
        first_transition_index: int,
        rollout_steps: int,
        hydrograph_id: str,
    ) -> None:
        """Attach consecutive edge targets for differentiable training rollout."""

        if rollout_steps < 2:
            raise ValueError("Edge rollout attributes require at least two steps.")
        target = self.hecras_edge_flow_delta_by_hydrograph.get(hydrograph_id)
        if not isinstance(target, dict) or "face" not in target:
            raise ValueError(
                "Differentiable edge rollout requires paired sharded Face Flow "
                f"targets for {hydrograph_id}."
            )
        first = first_transition_index + self.hecras_edge_flow_time_offset
        last = first + rollout_steps
        if first < 0 or last > target["face"].shape[0]:
            raise IndexError(
                f"Edge rollout transitions {first}..{last - 1} are outside "
                f"0..{target['face'].shape[0] - 1} for {hydrograph_id}."
            )
        rows = slice(first, last)

        if "internal_selected" in target:
            selected_nodes = np.asarray(target["selected_nodes"], dtype=np.int64)
            shape = (graph.x.shape[0], rollout_steps)
            internal = np.zeros(shape, dtype=np.float32)
            boundary = np.zeros(shape, dtype=np.float32)
            precipitation = np.zeros(shape, dtype=np.float32)
            internal[selected_nodes] = np.asarray(
                target["internal_selected"][rows], dtype=np.float32
            ).T
            boundary[selected_nodes] = np.asarray(
                target["boundary_selected"][rows], dtype=np.float32
            ).T
            precipitation[selected_nodes] = np.asarray(
                target["precipitation_selected"][rows], dtype=np.float32
            ).T
        else:
            internal = np.asarray(target["internal"][rows], dtype=np.float32).T
            boundary = np.asarray(target["boundary"][rows], dtype=np.float32).T
            precipitation = np.zeros_like(internal)

        face = np.asarray(target["face"][rows], dtype=np.float32).T
        previous_rows = np.arange(first, last, dtype=np.int64) - 1
        previous = np.zeros_like(face)
        valid_previous = previous_rows >= 0
        if np.any(valid_previous):
            previous[:, valid_previous] = np.asarray(
                target["face"][previous_rows[valid_previous]], dtype=np.float32
            ).T

        graph.training_rollout_edge_internal_delta = torch.tensor(
            internal, dtype=torch.float
        )
        graph.training_rollout_edge_boundary_source_delta = torch.tensor(
            boundary, dtype=torch.float
        )
        graph.training_rollout_edge_precipitation_delta = torch.tensor(
            precipitation, dtype=torch.float
        )
        graph.training_rollout_internal_face_flow_delta = torch.tensor(
            face, dtype=torch.float
        )
        graph.training_rollout_previous_internal_face_flow_delta = torch.tensor(
            previous, dtype=torch.float
        )
        if "raw_face" in target:
            graph.training_rollout_raw_internal_face_flow_delta = torch.tensor(
                np.asarray(target["raw_face"][rows], dtype=np.float32).T,
                dtype=torch.float,
            )
        if self.hecras_edge_flow_scale_stats is not None:
            transition_rms = self.hecras_edge_flow_scale_stats[
                "transition_rms_by_id"
            ].get(hydrograph_id)
            if transition_rms is None or last > transition_rms.shape[0]:
                raise IndexError(
                    f"Face Flow scale stats do not cover rollout for {hydrograph_id}."
                )
            graph.training_rollout_internal_face_flow_transition_rms = torch.tensor(
                np.asarray(transition_rms[rows], dtype=np.float32)[None, :],
                dtype=torch.float,
            )

    def load_hecras_boundary_node_mask(self) -> Optional[np.ndarray]:
        """Load nodes touched by HEC-RAS boundary/ghost faces, if available."""
        if self.hecras_face_graph is not None:
            stored = self.hecras_face_graph.get("boundary_node_mask")
            if stored is not None:
                return np.asarray(stored, dtype=np.bool_).copy()
        if self.hecras_face_graph_file is None:
            return None

        path = Path(self.hecras_face_graph_file)
        if not path.exists():
            raise FileNotFoundError(path)

        data = np.load(path)
        if "boundary_face_cell_index" not in data.files:
            return None

        num_nodes = self.static_data["xy_coords"].shape[0]
        boundary_cells = np.asarray(data["boundary_face_cell_index"], dtype=np.int64)
        if "hdf_to_hgn_node_index" not in data.files:
            return None
        hdf_to_hgn = np.asarray(data["hdf_to_hgn_node_index"], dtype=np.int64)
        if hdf_to_hgn.size == 0:
            return None

        boundary_cells = boundary_cells.reshape(-1)
        boundary_cells = boundary_cells[
            (boundary_cells >= 0) & (boundary_cells < hdf_to_hgn.size)
        ]
        mapped = hdf_to_hgn[boundary_cells]
        mapped = mapped[(mapped >= 0) & (mapped < num_nodes)]
        mask = np.zeros(num_nodes, dtype=np.bool_)
        mask[mapped] = True
        return mask

    def build_hecras_high_interior_control_volume_labels(
        self,
    ) -> Optional[np.ndarray]:
        """Label connected control volumes in the fixed high-zone interior mask."""

        if (
            self.hecras_face_graph is None
            or self.zone_label is None
            or self.hecras_boundary_node_mask is None
        ):
            return None
        stored = self.hecras_face_graph.get("high_interior_control_volume_label")
        if stored is not None:
            stored = np.asarray(stored, dtype=np.int64)
            expected_mask = (self.zone_label == 3) & (
                ~self.hecras_boundary_node_mask
            )
            if not np.array_equal(stored >= 0, expected_mask):
                raise ValueError(
                    "Stored high-interior control-volume scope does not match "
                    "the current zone labels and boundary mask."
                )
            return stored.copy()
        node_mask = (self.zone_label == 3) & (~self.hecras_boundary_node_mask)
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

        src, dst = self.hecras_face_graph["face_index"]
        selected_faces = node_mask[src] & node_mask[dst]
        for first, second in zip(src[selected_faces], dst[selected_faces]):
            union(int(first), int(second))

        labels = np.full(node_mask.size, -1, dtype=np.int64)
        selected_nodes = np.flatnonzero(node_mask)
        roots = np.asarray(
            [find(int(node)) for node in selected_nodes], dtype=np.int64
        )
        _, compact_labels = np.unique(roots, return_inverse=True)
        labels[selected_nodes] = compact_labels
        return labels

    def compute_local_source_rate(
        self, dyn: dict[str, np.ndarray], prev_time: int, target_time: int
    ) -> np.ndarray:
        """Estimate node-wise source volume rate from precipitation and IP.

        The HydroGraphNet precipitation values are normalized after converting to
        m/s. `M80_IP` is treated consistently with the existing global physics
        loss, where infiltration is used as a percentage multiplier.
        """
        if dyn.get("local_precipitation") is not None:
            # M80_PrNode stores the rate applied over the interval ending at
            # target_time. It is a known model input in physical m/s.
            precipitation_rate = dyn["local_precipitation"][target_time]
        else:
            prev_precip = dyn["precipitation"][prev_time]
            target_precip = dyn["precipitation"][target_time]
            avg_precip_norm = 0.5 * (prev_precip + target_precip)
            precipitation_rate = self.denormalize(
                avg_precip_norm,
                self.dynamic_stats["precipitation"]["mean"],
                self.dynamic_stats["precipitation"]["std"],
            )
        if self.hecras_face_graph is not None and self.hecras_face_graph.get(
            "node_surface_area"
        ) is not None:
            area = self.hecras_face_graph["node_surface_area"].reshape(-1)
        else:
            area = self.static_data["area_denorm"].reshape(-1)
        if self.local_source_runoff_mode == "full_area":
            runoff_fraction = np.ones_like(area)
        else:
            infiltration = self.denormalize(
                self.static_data["infiltration"],
                self.static_stats["infiltration"]["mean"],
                self.static_stats["infiltration"]["std"],
            ).reshape(-1)
            runoff_fraction = infiltration / 100.0
        return precipitation_rate * area * runoff_fraction

    @staticmethod
    def normalize(
        data: np.ndarray,
        mean: Union[float, list, np.ndarray],
        std: Union[float, list, np.ndarray],
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """
        Normalize the data using the provided mean and standard deviation.

        Args:
            data (np.ndarray): Data to normalize.
            mean (float, list, or np.ndarray): Mean value(s).
            std (float, list, or np.ndarray): Standard deviation value(s).
            epsilon (float): Small constant to avoid division by zero.

        Returns:
            np.ndarray: Normalized data.
        """
        mean = np.array(mean) if isinstance(mean, list) else mean
        std = np.array(std) if isinstance(std, list) else std
        return (data - mean) / (std + epsilon)

    @staticmethod
    def denormalize(
        data: np.ndarray,
        mean: Union[float, list, np.ndarray],
        std: Union[float, list, np.ndarray],
        epsilon: float = 1e-8,
    ) -> np.ndarray:
        """
        Denormalize the data using the provided mean and standard deviation.

        Args:
            data (np.ndarray): Normalized data.
            mean (float, list, or np.ndarray): Mean value(s) used for normalization.
            std (float, list, or np.ndarray): Standard deviation used for normalization.
            epsilon (float): Small constant to avoid division by zero.

        Returns:
            np.ndarray: Denormalized data.
        """
        mean = np.array(mean) if isinstance(mean, list) else mean
        std = np.array(std) if isinstance(std, list) else std
        return data * (std + epsilon) + mean

    def apply_noise_to_feature(
        self, data: np.ndarray, noise_type: str, noise_std: float
    ) -> np.ndarray:
        """
        Apply specified noise to a feature matrix.

        Args:
            data (np.ndarray): Input data of shape (T, num_nodes).
            noise_type (str): Type of noise ("none", "only_last", "correlated", "uncorrelated", "random_walk").
            noise_std (float): Standard deviation of the noise.

        Returns:
            np.ndarray: Data with noise applied.
        """
        if noise_type in ["none", "pushforward"]:
            return data
        T, num_nodes = data.shape
        if noise_type == "only_last":
            noise = np.random.normal(0, noise_std, size=(1, num_nodes))
            data_modified = data.copy()
            data_modified[-1] += noise[0]
            return data_modified
        elif noise_type == "correlated":
            noise = np.random.normal(0, noise_std, size=(1, num_nodes))
            return data + noise
        elif noise_type == "uncorrelated":
            noise = np.random.normal(0, noise_std, size=(T, num_nodes))
            return data + noise
        elif noise_type == "random_walk":
            noise_increments = np.random.normal(
                0, noise_std / math.sqrt(T), size=(T, num_nodes)
            )
            noise_cumulative = np.cumsum(noise_increments, axis=0)
            return data + noise_cumulative
        else:
            logger.warning(f"Unknown noise_type={noise_type}, skipping noise.")
            return data

    def save_norm_stats(self, stats: dict, filename: str) -> None:
        """
        Save normalization statistics to a JSON file.

        Args:
            stats (dict): Dictionary of normalization statistics.
            filename (str): Filename to save the stats.
        """
        filepath = os.path.join(self.data_dir, filename)
        with open(filepath, "w") as f:
            json.dump(stats, f)

    def load_norm_stats(self, filename: str) -> dict:
        """
        Load normalization statistics from a JSON file.

        Args:
            filename (str): Filename from which to load the stats.

        Returns:
            dict: Normalization statistics.
        """
        stats_dir = self.norm_stats_dir or self.data_dir
        filepath = os.path.join(stats_dir, filename)
        with open(filepath, "r") as f:
            stats = json.load(f)
        return stats

    def load_constant_data(
        self, folder: str, prefix: str, norm_stats_static: Optional[dict] = None
    ):
        """
        Load and standardize static (constant) data such as coordinates, elevation, and flow accumulation.

        Args:
            folder (str): Directory where the static data files are located.
            prefix (str): Prefix for file names.
            norm_stats_static (Optional[dict]): Precomputed static normalization statistics.

        Returns:
            Tuple containing standardized static data and the updated normalization stats.
        """
        epsilon = 1e-8
        stats = norm_stats_static if norm_stats_static is not None else {}

        def standardize(data: np.ndarray, key: str) -> np.ndarray:
            """
            Standardize data by subtracting the mean and dividing by the standard deviation.
            """
            if key in stats:
                mean_val = np.array(stats[key]["mean"])
                std_val = np.array(stats[key]["std"])
            else:
                mean_val = np.mean(data, axis=0)
                std_val = np.std(data, axis=0)
                stats[key] = {"mean": mean_val.tolist(), "std": std_val.tolist()}
            return (data - mean_val) / (std_val + epsilon)

        # Load each file using the given prefix.
        xy_path = os.path.join(folder, f"{prefix}_XY.txt")
        ca_path = os.path.join(folder, f"{prefix}_CA.txt")
        ce_path = os.path.join(folder, f"{prefix}_CE.txt")
        cs_path = os.path.join(folder, f"{prefix}_CS.txt")
        aspect_path = os.path.join(folder, f"{prefix}_A.txt")
        curvature_path = os.path.join(folder, f"{prefix}_CU.txt")
        manning_path = os.path.join(folder, f"{prefix}_N.txt")
        flow_accum_path = os.path.join(folder, f"{prefix}_FA.txt")
        infiltration_path = os.path.join(folder, f"{prefix}_IP.txt")

        raw_xy_coords = np.loadtxt(xy_path, delimiter="\t")
        self.static_data_raw_xy = raw_xy_coords
        xy_coords = standardize(raw_xy_coords, "xy_coords")
        area_denorm = np.loadtxt(ca_path, delimiter="\t")[: xy_coords.shape[0]].reshape(
            -1, 1
        )
        area = standardize(area_denorm, "area")
        elevation = np.loadtxt(ce_path, delimiter="\t")[: xy_coords.shape[0]].reshape(
            -1, 1
        )
        elevation = standardize(elevation, "elevation")
        slope = np.loadtxt(cs_path, delimiter="\t")[: xy_coords.shape[0]].reshape(-1, 1)
        slope = standardize(slope, "slope")
        aspect = np.loadtxt(aspect_path, delimiter="\t")[: xy_coords.shape[0]].reshape(
            -1, 1
        )
        aspect = standardize(aspect, "aspect")
        curvature = np.loadtxt(curvature_path, delimiter="\t")[
            : xy_coords.shape[0]
        ].reshape(-1, 1)
        curvature = standardize(curvature, "curvature")
        manning = np.loadtxt(manning_path, delimiter="\t")[
            : xy_coords.shape[0]
        ].reshape(-1, 1)
        manning = standardize(manning, "manning")
        flow_accum = np.loadtxt(flow_accum_path, delimiter="\t")[
            : xy_coords.shape[0]
        ].reshape(-1, 1)
        flow_accum = standardize(flow_accum, "flow_accum")
        infiltration = np.loadtxt(infiltration_path, delimiter="\t")[
            : xy_coords.shape[0]
        ].reshape(-1, 1)
        infiltration = standardize(infiltration, "infiltration")
        return (
            xy_coords,
            area,
            area_denorm,
            elevation,
            slope,
            aspect,
            curvature,
            manning,
            flow_accum,
            infiltration,
            stats,
        )

    def load_dynamic_data(
        self,
        folder: str,
        hydrograph_id: str,
        prefix: str,
        num_points: int,
        interval: int = 1,
        skip: Optional[int] = None,
        post_peak_steps: Optional[int] = None,
    ):
        """
        Load dynamic data (water depth, inflow, volume, and precipitation) for a given hydrograph.

        Args:
            folder (str): Directory where the dynamic data files are located.
            hydrograph_id (str): Identifier for the hydrograph.
            prefix (str): Prefix for file names.
            num_points (int): Number of spatial points (nodes).
            interval (int): Sampling interval.
            skip (int): Number of initial time steps to skip.
            post_peak_steps (int): Number of samples retained after peak inflow.

        Returns:
            Tuple of np.ndarray: (water_depth, inflow_hydrograph, volume, precipitation)
        """
        skip = self.dynamic_skip_steps if skip is None else int(skip)
        post_peak_steps = (
            self.post_peak_steps
            if post_peak_steps is None
            else int(post_peak_steps)
        )
        wd_path = os.path.join(folder, f"{prefix}_WD_{hydrograph_id}.txt")
        inflow_path = os.path.join(folder, f"{prefix}_US_InF_{hydrograph_id}.txt")
        volume_path = os.path.join(folder, f"{prefix}_V_{hydrograph_id}.txt")
        vx_path = os.path.join(folder, f"{prefix}_VX_{hydrograph_id}.txt")
        vy_path = os.path.join(folder, f"{prefix}_VY_{hydrograph_id}.txt")
        precipitation_path = os.path.join(folder, f"{prefix}_Pr_{hydrograph_id}.txt")
        node_precipitation_path = os.path.join(
            folder, f"{prefix}_PrNode_{hydrograph_id}.npz"
        )
        water_depth = np.loadtxt(wd_path, delimiter="\t")[skip::interval, :num_points]
        inflow_table = np.loadtxt(inflow_path, delimiter="\t")
        time_days = inflow_table[:, 0]
        inflow_hydrograph = inflow_table[skip::interval, 1]
        volume = np.loadtxt(volume_path, delimiter="\t")[skip::interval, :num_points]
        velocity_x = np.loadtxt(vx_path, delimiter="\t")[skip::interval, :num_points]
        velocity_y = np.loadtxt(vy_path, delimiter="\t")[skip::interval, :num_points]
        precipitation = np.loadtxt(precipitation_path, delimiter="\t")[skip::interval]
        local_precipitation = None
        if os.path.exists(node_precipitation_path):
            with np.load(node_precipitation_path, allow_pickle=False) as node_precip:
                required_keys = {
                    "input_rate_mm_per_hour",
                    "hgn_time_origin_days",
                    "input_interval_seconds",
                    "units",
                    "source",
                }
                missing_keys = required_keys.difference(node_precip.files)
                if missing_keys:
                    raise ValueError(
                        f"Node precipitation file {node_precipitation_path} is "
                        f"missing keys: {sorted(missing_keys)}"
                    )
                node_rate = np.asarray(
                    node_precip["input_rate_mm_per_hour"], dtype=np.float32
                )
                origin_days = float(node_precip["hgn_time_origin_days"])
                input_interval_seconds = float(
                    node_precip["input_interval_seconds"]
                )
                units = str(node_precip["units"])
                source = str(node_precip["source"])
            if node_rate.ndim != 2 or node_rate.shape[1] != num_points:
                raise ValueError(
                    f"Node precipitation shape {node_rate.shape} does not match "
                    f"{num_points} HGN nodes."
                )
            if units != "mm/h" or source != (
                "pre_simulation_raster_values_and_cell_weights"
            ):
                raise ValueError(
                    "Node precipitation must be inference-side HEC-RAS raster "
                    f"input in mm/h; got units={units!r}, source={source!r}."
                )
            if input_interval_seconds <= 0.0:
                raise ValueError("Node precipitation input interval must be positive.")
            elapsed_seconds = (time_days - origin_days) * 86400.0
            if np.min(elapsed_seconds) < -1.0e-3:
                raise ValueError(
                    "HGN time axis begins before node precipitation origin."
                )
            input_rows = np.ceil(
                elapsed_seconds / input_interval_seconds - 1.0e-6
            ).astype(np.int64)
            if np.min(input_rows) < 0 or np.max(input_rows) >= node_rate.shape[0]:
                raise ValueError(
                    "HGN time axis exceeds the node precipitation input table."
                )
            local_precipitation = (
                node_rate[input_rows][skip::interval]
                * self.precipitation_unit_conversion
            )
        elif self.require_node_precipitation:
            raise FileNotFoundError(
                "Formal local conservation requires inference-side node "
                f"precipitation: {node_precipitation_path}"
            )
        # Keep a configurable physical window after peak inflow.
        peak_time_idx = np.argmax(inflow_hydrograph)
        trim_end = peak_time_idx + post_peak_steps
        water_depth = water_depth[:trim_end]
        volume = volume[:trim_end]
        velocity_x = velocity_x[:trim_end]
        velocity_y = velocity_y[:trim_end]
        precipitation = (
            precipitation[:trim_end] * self.precipitation_unit_conversion
        )
        if local_precipitation is not None:
            local_precipitation = local_precipitation[:trim_end]
        inflow_hydrograph = inflow_hydrograph[:trim_end]
        return (
            water_depth,
            inflow_hydrograph,
            velocity_x,
            velocity_y,
            volume,
            precipitation,
            local_precipitation,
        )

    def create_node_features(
        self,
        xy_coords: np.ndarray,
        area: np.ndarray,
        elevation: np.ndarray,
        slope: np.ndarray,
        aspect: np.ndarray,
        curvature: np.ndarray,
        manning: np.ndarray,
        flow_accum: np.ndarray,
        infiltration: np.ndarray,
        water_depth: np.ndarray,
        volume: np.ndarray,
        precipitation_data: np.ndarray,
        time_step: int,
        n_time_steps: int,
        inflow_hydrograph: np.ndarray,
    ) -> (np.ndarray, float, float):
        """
        Create node features by combining static and dynamic data.

        Args:
            xy_coords (np.ndarray): Spatial coordinates.
            area (np.ndarray): Normalized area.
            elevation (np.ndarray): Normalized elevation.
            slope (np.ndarray): Normalized slope.
            aspect (np.ndarray): Normalized aspect.
            curvature (np.ndarray): Normalized curvature.
            manning (np.ndarray): Normalized Manning coefficient.
            flow_accum (np.ndarray): Normalized flow accumulation.
            infiltration (np.ndarray): Normalized infiltration.
            water_depth (np.ndarray): Dynamic water depth data (time x nodes).
            volume (np.ndarray): Dynamic volume data (time x nodes).
            precipitation_data (np.ndarray): Dynamic precipitation data.
            time_step (int): Starting time step.
            n_time_steps (int): Number of time steps in the window.
            inflow_hydrograph (np.ndarray): Dynamic inflow data.

        Returns:
            Tuple:
                - features (np.ndarray): Node feature matrix.
                - future_inflow (float): Future inflow at time_step+n_time_steps.
                - future_precip (float): Future precipitation at time_step+n_time_steps.
        """
        # Apply noise if required (excluding "none" and "pushforward").
        if self.noise_type not in ["none", "pushforward"]:
            window_slice = slice(time_step, time_step + n_time_steps)
            water_depth[window_slice, :] = self.apply_noise_to_feature(
                water_depth[window_slice, :], self.noise_type, self.noise_std
            )
            volume[window_slice, :] = self.apply_noise_to_feature(
                volume[window_slice, :], self.noise_type, self.noise_std
            )
        num_nodes = xy_coords.shape[0]
        # Create static copies of inflow and precipitation for each node.
        flow_hydrograph_current_step = np.full(
            (num_nodes, 1), inflow_hydrograph[time_step]
        )
        precip_current_step = np.full((num_nodes, 1), precipitation_data[time_step])
        # Concatenate all features horizontally.
        features = np.hstack(
            [
                xy_coords,
                area,
                elevation,
                slope,
                aspect,
                curvature,
                manning,
                flow_accum,
                infiltration,
                flow_hydrograph_current_step,
                precip_current_step,
                water_depth.T,
                volume.T,
            ]
        )
        future_inflow = inflow_hydrograph[time_step + n_time_steps]
        future_precip = precipitation_data[time_step + n_time_steps]
        return features, future_inflow, future_precip

    def create_edge_features(
        self, xy_coords: np.ndarray, edge_index: np.ndarray
    ) -> np.ndarray:
        """
        Create edge features based on the relative positions of connected nodes.

        Args:
            xy_coords (np.ndarray): Node spatial coordinates.
            edge_index (np.ndarray): Array containing source and destination indices for each edge.

        Returns:
            np.ndarray: Concatenated edge features (relative coordinates and normalized distance).
        """
        row, col = edge_index
        relative_coords = xy_coords[row] - xy_coords[col]
        distance = np.linalg.norm(relative_coords, axis=1)
        epsilon = 1e-8
        # Normalize relative coordinates and distance.
        relative_coords = (relative_coords - np.mean(relative_coords, axis=0)) / (
            np.std(relative_coords, axis=0) + epsilon
        )
        distance = (distance - np.mean(distance)) / (np.std(distance) + epsilon)
        return np.hstack([relative_coords, distance[:, None]])

    def create_edge_unit_vectors(
        self, xy_coords: np.ndarray, edge_index: np.ndarray
    ) -> np.ndarray:
        """Create raw-coordinate unit vectors for directed local-flow proxies."""
        row, col = edge_index
        relative_coords = xy_coords[col] - xy_coords[row]
        distance = np.linalg.norm(relative_coords, axis=1, keepdims=True)
        return relative_coords / (distance + 1e-8)
