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
Unit tests for the HydroGraphDataset datapipe.
"""

import json
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from test.conftest import requires_module

from . import common

Tensor = torch.Tensor


def test_hydrograph_time_contract(tmp_path):
    """Validated datasets expose sample-based windows for their output interval."""
    from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset

    contract = {
        "model_time_contract": {
            "delta_t_seconds": 300.0,
            "dynamic_skip_steps": 288,
            "post_peak_steps": 100,
        }
    }
    (tmp_path / "hgn_dataset_validation.json").write_text(
        json.dumps(contract)
    )

    loaded = HydroGraphDataset.load_dataset_time_contract(tmp_path)
    assert loaded["delta_t_seconds"] == 300.0
    assert loaded["dynamic_skip_steps"] == 288
    assert loaded["post_peak_steps"] == 100


@requires_module(["torch_geometric", "torch_scatter", "scipy", "tqdm"])
def test_node_precipitation_uses_interval_ending_spatial_rate(tmp_path):
    """Known raster rain remains node-wise and follows HEC interval indexing."""
    from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset

    prefix = "M80"
    event_id = "H1"
    time_days = np.arange(5, dtype=np.float64) * 300.0 / 86400.0
    inflow = np.column_stack([time_days, [5.0, 4.0, 3.0, 2.0, 1.0]])
    np.savetxt(tmp_path / f"{prefix}_US_InF_{event_id}.txt", inflow, delimiter="\t")
    for field in ("WD", "V", "VX", "VY"):
        np.savetxt(
            tmp_path / f"{prefix}_{field}_{event_id}.txt",
            np.zeros((5, 2)),
            delimiter="\t",
        )
    np.savetxt(
        tmp_path / f"{prefix}_Pr_{event_id}.txt",
        np.zeros((5, 1)),
        delimiter="\t",
    )
    np.savez_compressed(
        tmp_path / f"{prefix}_PrNode_{event_id}.npz",
        input_rate_mm_per_hour=np.asarray([[1.0, 2.0], [10.0, 20.0]]),
        hgn_time_origin_days=np.asarray(0.0),
        input_interval_seconds=np.asarray(3600.0),
        units=np.asarray("mm/h"),
        source=np.asarray("pre_simulation_raster_values_and_cell_weights"),
    )

    dataset = object.__new__(HydroGraphDataset)
    dataset.dynamic_skip_steps = 0
    dataset.post_peak_steps = 5
    dataset.precipitation_unit_conversion = 1.0e-3 / 3600.0
    dataset.require_node_precipitation = True
    loaded = dataset.load_dynamic_data(
        str(tmp_path), event_id, prefix, num_points=2
    )
    local_precipitation = loaded[-1]
    np.testing.assert_allclose(
        local_precipitation[0],
        np.asarray([1.0, 2.0]) * dataset.precipitation_unit_conversion,
    )
    np.testing.assert_allclose(
        local_precipitation[1:],
        np.tile(
            np.asarray([10.0, 20.0]) * dataset.precipitation_unit_conversion,
            (4, 1),
        ),
    )

    dataset.local_source_runoff_mode = "full_area"
    dataset.hecras_face_graph = {
        "node_surface_area": np.asarray([2.0, 3.0])
    }
    source_rate = dataset.compute_local_source_rate(
        {"local_precipitation": local_precipitation}, 0, 1
    )
    np.testing.assert_allclose(
        source_rate,
        local_precipitation[1] * np.asarray([2.0, 3.0]),
    )


@requires_module(["torch_geometric", "torch_scatter", "scipy", "tqdm"])
def test_sharded_period_average_cell_budget_is_expanded_at_offset(tmp_path):
    """Cell-only controls load the selected-node five-minute budget lazily."""
    from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset

    selected_nodes = np.asarray([1, 3], dtype=np.int64)
    target = np.arange(10, dtype=np.float32).reshape(5, 2)
    np.save(tmp_path / "selected_node_index.npy", selected_nodes)
    np.save(
        tmp_path / "H1_period_average_cell_budget_delta_selected_m3.npy",
        target,
    )

    dataset = object.__new__(HydroGraphDataset)
    dataset.hecras_cell_balance_target_dir = str(tmp_path)
    dataset.hecras_cell_balance_npz = None
    dataset.hecras_cell_balance_glob = None
    dataset.hydrograph_ids = ["H1"]
    dataset.static_data = {"xy_coords": np.zeros((4, 2), dtype=np.float32)}
    loaded = dataset.load_hecras_cell_balance_delta_by_hydrograph()
    dataset.hecras_cell_balance_delta_by_hydrograph = loaded
    dataset.hecras_cell_balance_time_offset = 2
    dataset.dynamic_stats = {"volume": {"std": 7.0}}

    graph = SimpleNamespace(x=torch.zeros((4, 1), dtype=torch.float32))
    dataset.add_hecras_cell_balance_attrs(graph, 1, "H1")
    expected = np.zeros(4, dtype=np.float32)
    expected[selected_nodes] = target[3]
    np.testing.assert_array_equal(graph.hecras_cell_balance_delta.numpy(), expected)
    assert graph.volume_std.item() == 7.0


@pytest.fixture(scope="session")
def hydrograph_data_dir(nfs_data_dir, tmp_path_factory):
    """
    Make a **writable copy** of the tiny HydroGraph dataset so tests can
    freely create cache files without touching the pristine NFS copy.
    """
    src = nfs_data_dir.joinpath("datasets/hydrographnet_tiny")
    dst = tmp_path_factory.mktemp("hydrograph_unit_test")
    shutil.copytree(src, dst, dirs_exist_ok=True)
    return Path(dst)


@requires_module(["torch_geometric", "torch_scatter", "scipy", "tqdm"])
@pytest.mark.parametrize("device", ["cuda:0", "cpu"])
def test_hydrograph_constructor(hydrograph_data_dir, device, pytestconfig):
    """Constructor & basic iteration checks."""

    from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset

    # -- build a tiny train‑split dataset ------------------------------------
    dataset = HydroGraphDataset(
        data_dir=hydrograph_data_dir,
        split="train",
        num_samples=2,
        n_time_steps=2,
        k=2,
        noise_type="none",
    )

    common.check_datapipe_iterable(dataset)
    assert len(dataset) > 0

    sample = dataset[0]
    if isinstance(sample, tuple):  # physics / push‑forward mode
        g, physics = sample
        assert isinstance(physics, dict)
    else:
        g = sample
    assert g.x.shape[0] == g.num_nodes
    assert g.edge_attr.shape[0] == g.num_edges

    # -- invalid split --------------------------------------------------------
    with pytest.raises(ValueError):
        _ = HydroGraphDataset(
            data_dir=hydrograph_data_dir,
            split="validation",
            num_samples=1,
        )

    # -- test‑split rollout length -------------------------------------------
    rollout_len = 5
    test_ds = HydroGraphDataset(
        data_dir=hydrograph_data_dir,
        split="test",
        num_samples=1,
        n_time_steps=2,
        rollout_length=rollout_len,
    )
    g_test, rollout = test_ds[0]
    for key in ["inflow", "precipitation", "water_depth_gt", "volume_gt"]:
        assert rollout[key].shape[0] == rollout_len
    assert g_test.num_nodes > 0
