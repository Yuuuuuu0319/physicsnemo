# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0

"""Focused tests for the HydroGraphNet physical-edge conservation extension."""

import sys
from pathlib import Path
from types import SimpleNamespace

import h5py
import pytest
import numpy as np
import torch
from torch_geometric.data import Batch, Data


EXAMPLE_DIR = (
    Path(__file__).resolve().parents[2]
    / "examples"
    / "weather"
    / "flood_modeling"
    / "hydrographnet"
)
sys.path.insert(0, str(EXAMPLE_DIR))

from hecras_projection import project_edge_flux  # noqa: E402
from evaluate_hecras_physical_budget import infer_native_volume_contract  # noqa: E402
from validate_hecras_edge_flow_delta import infer_integrated_flow_unit  # noqa: E402
from physicsnemo.datapipes.gnn.hydrographnet_dataset import (  # noqa: E402
    HydroGraphDataset,
)
from train import HecRasEdgeFluxHead  # noqa: E402
from utils import (  # noqa: E402
    compute_area_extrapolated_storage_delta,
    compute_hecras_cell_balance_loss,
    compute_hecras_edge_flux_head_loss,
)


def test_si_face_flow_and_precipitation_units_are_m3(tmp_path):
    hdf_path = tmp_path / "si.hdf"
    precipitation_path = (
        "Results/Unsteady/Output/Output Blocks/Base Output/Unsteady Time Series/"
        "2D Flow Areas/per2/Cell Cumulative Precipitation Depth"
    )
    with h5py.File(hdf_path, "w") as hdf:
        hdf.attrs["Units System"] = "SI Units"
        dataset = hdf.create_dataset(precipitation_path, data=np.zeros((2, 2)))
        dataset.attrs["Units"] = "mm"
    with h5py.File(hdf_path, "r") as hdf:
        assert infer_integrated_flow_unit(hdf) == "m3"
        volume_unit, precipitation_factor = infer_native_volume_contract(hdf)
    assert volume_unit == "m3"
    assert precipitation_factor == pytest.approx(1.0e-3)


def test_cell_balance_npz_loader_attaches_requested_transition(tmp_path):
    target = np.arange(18, dtype=np.float32).reshape(3, 6)
    path = tmp_path / "cell_balance_targets.npz"
    np.savez_compressed(path, H1_cell_balance_storage_delta=target)

    dataset = HydroGraphDataset.__new__(HydroGraphDataset)
    dataset.hecras_cell_balance_npz = str(path)
    dataset.hecras_cell_balance_glob = None
    dataset.hecras_cell_balance_npz_key_suffix = "_cell_balance_storage_delta"
    dataset.hecras_cell_balance_time_offset = 1
    dataset.hydrograph_ids = ["H1"]
    dataset.static_data = {"xy_coords": np.zeros((6, 2), dtype=np.float32)}
    dataset.dynamic_stats = {"volume": {"std": 7.0}}

    loaded = dataset.load_hecras_cell_balance_delta_by_hydrograph()
    dataset.hecras_cell_balance_delta_by_hydrograph = loaded
    graph = SimpleNamespace()
    dataset.add_hecras_cell_balance_attrs(graph, 0, "H1")

    np.testing.assert_array_equal(
        graph.hecras_cell_balance_delta.numpy(), target[1]
    )
    assert graph.volume_std.item() == pytest.approx(7.0)


def test_cell_balance_npz_loader_fails_closed_on_missing_event(tmp_path):
    path = tmp_path / "cell_balance_targets.npz"
    np.savez_compressed(
        path,
        H1_cell_balance_storage_delta=np.zeros((3, 2), dtype=np.float32),
    )
    dataset = HydroGraphDataset.__new__(HydroGraphDataset)
    dataset.hecras_cell_balance_npz = str(path)
    dataset.hecras_cell_balance_glob = None
    dataset.hecras_cell_balance_npz_key_suffix = "_cell_balance_storage_delta"
    dataset.hydrograph_ids = ["H2"]
    dataset.static_data = {"xy_coords": np.zeros((2, 2), dtype=np.float32)}

    with pytest.raises(KeyError, match="Missing cell-balance NPZ targets"):
        dataset.load_hecras_cell_balance_delta_by_hydrograph()


def test_cell_balance_loss_volume_std_normalization_is_dimensionless():
    graph = SimpleNamespace(
        hecras_cell_balance_delta=torch.tensor([2.0, -2.0]),
        volume_std=torch.tensor([4.0]),
        zone_label=torch.tensor([3, 3]),
    )
    prediction = torch.tensor([[0.0, 1.0], [0.0, -1.0]])

    raw = compute_hecras_cell_balance_loss(
        prediction, graph, zone_mode="all", normalization="none"
    )
    normalized = compute_hecras_cell_balance_loss(
        prediction, graph, zone_mode="all", normalization="volume_std"
    )

    assert raw.item() == pytest.approx(4.0)
    assert normalized.item() == pytest.approx(0.25)


def test_edge_head_can_require_hydrographnet_node_prediction():
    graph = SimpleNamespace(
        x=torch.zeros((3, 16)),
        hecras_face_index=torch.tensor([[0, 1], [1, 2]]),
        hecras_face_length=torch.ones(2),
    )
    head = HecRasEdgeFluxHead(
        16,
        hidden_dim=8,
        use_node_prediction_features=True,
    )

    with pytest.raises(ValueError, match="node prediction"):
        head(graph)

    prediction = torch.zeros((3, 2), requires_grad=True)
    flux = head(graph, prediction)
    assert flux.shape == (2,)
    flux.sum().backward()
    assert prediction.grad is not None


def test_edge_head_dynamic_features_are_invariant_to_batch_companions():
    """One event must predict the same flux alone or beside another event."""

    def make_graph(surface_scale: float, source_scale: float) -> Data:
        graph = Data(x=torch.randn((3, 16)))
        graph.hecras_face_index = torch.tensor([[0, 1], [1, 2]])
        graph.hecras_face_length = torch.tensor([10.0, 20.0])
        graph.hecras_face_normal = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        graph.local_source_rate = source_scale * torch.tensor([1.0, 2.0, 4.0])
        graph.current_surface_elevation = surface_scale * torch.tensor(
            [0.0, 1.0, 3.0]
        )
        graph.current_water_depth_denorm = surface_scale * torch.tensor(
            [0.1, 0.3, 0.9]
        )
        graph.hecras_edge_physical_features = surface_scale * torch.arange(
            24, dtype=torch.float
        ).reshape(2, 12)
        return graph

    torch.manual_seed(19)
    first = make_graph(surface_scale=1.0, source_scale=1.0)
    second = make_graph(surface_scale=1000.0, source_scale=1000.0)
    head = HecRasEdgeFluxHead(
        16,
        hidden_dim=16,
        use_face_normal=True,
        use_surface_features=True,
        use_physical_surface_features=True,
        use_edge_physical_features=True,
    )
    head.eval()

    alone = head(first).detach()
    batched = Batch.from_data_list([first, second])
    beside_other_event = head(batched).detach()[: first.hecras_face_index.shape[1]]

    torch.testing.assert_close(alone, beside_other_event, rtol=1e-6, atol=1e-6)


def test_edge_head_selective_modes_compute_only_declared_faces():
    graph = SimpleNamespace(
        x=torch.zeros((5, 16)),
        hecras_face_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        hecras_face_length=torch.ones(4),
        zone_label=torch.tensor([3, 3, 0, 0, 3]),
        hecras_high_interior_control_volume_label=torch.tensor([0, 0, -1, -1, -1]),
    )

    expected_masks = {
        "all": [True, True, True, True],
        "zone4_touch": [True, True, False, True],
        "high_interior_touch": [True, True, False, False],
    }
    for mode, expected in expected_masks.items():
        head = HecRasEdgeFluxHead(16, hidden_dim=8, active_face_mode=mode)
        with torch.no_grad():
            for parameter in head.parameters():
                parameter.zero_()
            head.net[-1].bias.fill_(2.0)
        assert head.active_face_mask(graph).tolist() == expected
        output = head(graph)
        expected_output = torch.tensor(
            [2.0 if selected else 0.0 for selected in expected]
        )
        torch.testing.assert_close(output, expected_output)


def test_selective_edge_head_blocks_inactive_node_prediction_gradients():
    graph = SimpleNamespace(
        x=torch.randn((5, 16)),
        hecras_face_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        hecras_face_length=torch.ones(4),
        hecras_high_interior_control_volume_label=torch.tensor([0, 0, -1, -1, -1]),
    )
    prediction = torch.randn((5, 2), requires_grad=True)
    head = HecRasEdgeFluxHead(
        16,
        hidden_dim=8,
        use_node_prediction_features=True,
        active_face_mode="high_interior_touch",
    )

    output = head(graph, prediction)
    assert torch.equal(output[2:], torch.zeros_like(output[2:]))
    output.sum().backward()
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad[:3]) > 0
    assert torch.equal(prediction.grad[3:], torch.zeros_like(prediction.grad[3:]))


def test_joint_control_volume_loss_updates_node_backbone_and_edge_head():
    """The selective local loss must train both halves of the proposed model."""
    torch.manual_seed(7)
    graph = SimpleNamespace(
        x=torch.randn((3, 16)),
        hecras_face_index=torch.tensor([[0, 1], [1, 2]]),
        hecras_face_length=torch.ones(2),
        hecras_edge_internal_delta=torch.zeros(3),
        hecras_edge_boundary_source_delta=torch.zeros(3),
        local_source_rate=torch.zeros(3),
        volume_std=torch.ones(1),
        zone_label=torch.tensor([3, 3, 0]),
        hecras_boundary_node_mask=torch.zeros(3, dtype=torch.bool),
        hecras_high_interior_control_volume_label=torch.tensor([0, 0, -1]),
    )
    node_backbone = torch.nn.Linear(16, 2, bias=False)
    edge_head = HecRasEdgeFluxHead(
        16,
        hidden_dim=8,
        use_node_prediction_features=True,
    )

    prediction = node_backbone(graph.x)
    edge_flux = edge_head(graph, prediction)
    loss, _ = compute_hecras_edge_flux_head_loss(
        prediction,
        edge_flux,
        graph,
        zone_mode="high_interior",
        closure_target_weight=1.0,
        divergence_target_weight=0.0,
        face_target_weight=0.0,
        closure_granularity="connected_control_volume",
    )
    loss.backward()

    assert node_backbone.weight.grad is not None
    assert torch.count_nonzero(node_backbone.weight.grad) > 0
    edge_gradients = [
        parameter.grad for parameter in edge_head.parameters() if parameter.grad is not None
    ]
    assert edge_gradients
    assert any(torch.count_nonzero(gradient) > 0 for gradient in edge_gradients)


def test_edge_closure_includes_known_precipitation_source():
    graph = SimpleNamespace(
        x=torch.zeros((3, 16)),
        hecras_face_index=torch.tensor([[0], [1]]),
        hecras_edge_internal_delta=torch.tensor([-2.0, 2.0, 0.0]),
        hecras_edge_boundary_source_delta=torch.zeros(3),
        local_source_rate=torch.tensor([1.0, 0.0, 0.0]),
        volume_std=torch.tensor([1.0]),
        zone_label=torch.full((3,), 3),
    )
    prediction = torch.tensor([[0.0, 0.0], [0.0, 2.0], [0.0, 0.0]])
    edge_flux = torch.tensor([2.0])

    loss, metrics = compute_hecras_edge_flux_head_loss(
        prediction,
        edge_flux,
        graph,
        zone_mode="high",
        closure_target_weight=1.0,
        divergence_target_weight=1.0,
        delta_t=2.0,
    )

    assert loss.item() == pytest.approx(0.0, abs=1e-8)
    assert metrics["hecras_edge_flux_closure_loss"].item() == pytest.approx(
        0.0, abs=1e-8
    )


def test_area_extrapolated_storage_delta_restores_above_table_volume():
    graph = SimpleNamespace(
        volume_std=torch.tensor([10.0]),
        water_depth_std=torch.tensor([2.0]),
        current_surface_elevation=torch.tensor([9.0, 11.0]),
        hecras_node_area=torch.tensor([100.0, 200.0]),
        hecras_node_volume_table_top_elevation=torch.tensor([10.0, 10.0]),
    )
    prediction = torch.tensor(
        [
            [1.0, 3.0],
            [-1.0, 4.0],
        ],
        requires_grad=True,
    )

    storage_delta = compute_area_extrapolated_storage_delta(prediction, graph)

    assert storage_delta.tolist() == pytest.approx([130.0, -160.0])
    storage_delta.sum().backward()
    assert prediction.grad is not None
    assert prediction.grad[0, 0].item() == pytest.approx(200.0)


def test_connected_control_volume_closure_aggregates_internal_node_residuals():
    graph = SimpleNamespace(
        hecras_face_index=torch.tensor([[0], [1]]),
        hecras_edge_internal_delta=torch.zeros(2),
        hecras_edge_boundary_source_delta=torch.zeros(2),
        local_source_rate=torch.zeros(2),
        volume_std=torch.tensor([1.0]),
        zone_label=torch.full((2,), 3),
        hecras_boundary_node_mask=torch.zeros(2, dtype=torch.bool),
        hecras_high_interior_control_volume_label=torch.zeros(2, dtype=torch.long),
    )
    prediction = torch.tensor([[0.0, 1.0], [0.0, -1.0]])
    edge_flux = torch.tensor([0.0])

    node_loss, _ = compute_hecras_edge_flux_head_loss(
        prediction,
        edge_flux,
        graph,
        zone_mode="high_interior",
        closure_target_weight=1.0,
        divergence_target_weight=0.0,
        closure_granularity="node",
    )
    control_volume_loss, metrics = compute_hecras_edge_flux_head_loss(
        prediction,
        edge_flux,
        graph,
        zone_mode="high_interior",
        closure_target_weight=1.0,
        divergence_target_weight=0.0,
        closure_granularity="connected_control_volume",
    )

    assert node_loss.item() == pytest.approx(1.0)
    assert control_volume_loss.item() == pytest.approx(0.0, abs=1e-8)
    assert metrics["hecras_edge_flux_closure_loss"].item() == pytest.approx(
        0.0, abs=1e-8
    )


def test_connected_control_volume_closure_weights_components_by_node_count():
    graph = SimpleNamespace(
        hecras_face_index=torch.tensor([[0, 1], [1, 2]]),
        hecras_edge_internal_delta=torch.zeros(3),
        hecras_edge_boundary_source_delta=torch.zeros(3),
        local_source_rate=torch.zeros(3),
        volume_std=torch.tensor([1.0]),
        zone_label=torch.full((3,), 3),
        hecras_boundary_node_mask=torch.zeros(3, dtype=torch.bool),
        hecras_high_interior_control_volume_label=torch.tensor([0, 0, 1]),
    )
    prediction = torch.tensor(
        [[0.0, 1.0], [0.0, 3.0], [0.0, 5.0]], requires_grad=True
    )
    edge_flux = torch.zeros(2)

    loss, metrics = compute_hecras_edge_flux_head_loss(
        prediction,
        edge_flux,
        graph,
        zone_mode="high_interior",
        closure_target_weight=1.0,
        divergence_target_weight=0.0,
        closure_granularity="connected_control_volume",
    )

    # Component means are 2 and 5. Weighting by component node counts gives
    # (2 * 2^2 + 1 * 5^2) / 3 = 11.
    assert loss.item() == pytest.approx(11.0)
    assert metrics["hecras_edge_flux_closure_loss"].item() == pytest.approx(11.0)
    loss.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()
    assert torch.count_nonzero(prediction.grad[:, 1]) == 3


def test_face_supervision_scope_is_independent_from_local_closure_scope():
    graph = SimpleNamespace(
        hecras_face_index=torch.tensor([[0, 1], [1, 2]]),
        hecras_edge_internal_delta=torch.zeros(3),
        hecras_edge_boundary_source_delta=torch.zeros(3),
        hecras_internal_face_flow_delta=torch.tensor([1.0, 100.0]),
        volume_std=torch.tensor([1.0]),
        zone_label=torch.tensor([3, 3, 0]),
        hecras_boundary_node_mask=torch.zeros(3, dtype=torch.bool),
    )

    loss, metrics = compute_hecras_edge_flux_head_loss(
        torch.zeros((3, 2)),
        torch.zeros(2),
        graph,
        zone_mode="high_interior",
        closure_target_weight=0.0,
        divergence_target_weight=0.0,
        face_target_weight=1.0,
        face_zone_mode="all",
    )

    # Both faces are supervised even though local closure selects only
    # high-interior nodes: mean([1^2, 100^2]) = 5000.5.
    assert loss.item() == pytest.approx(5000.5)
    assert metrics["hecras_edge_flux_face_loss"].item() == pytest.approx(5000.5)


def test_high_interior_touch_face_supervision_ignores_inactive_targets():
    graph = SimpleNamespace(
        hecras_face_index=torch.tensor([[0, 1, 2, 3], [1, 2, 3, 4]]),
        hecras_edge_internal_delta=torch.zeros(5),
        hecras_edge_boundary_source_delta=torch.zeros(5),
        hecras_internal_face_flow_delta=torch.tensor([1.0, 2.0, 100.0, 100.0]),
        volume_std=torch.tensor([1.0]),
        zone_label=torch.tensor([3, 3, 0, 0, 3]),
        hecras_boundary_node_mask=torch.zeros(5, dtype=torch.bool),
        hecras_high_interior_control_volume_label=torch.tensor([0, 0, -1, -1, -1]),
    )
    edge_prediction = torch.tensor([0.0, 0.0, 999.0, 999.0])

    loss, metrics = compute_hecras_edge_flux_head_loss(
        torch.zeros((5, 2)),
        edge_prediction,
        graph,
        zone_mode="high_interior",
        closure_target_weight=0.0,
        divergence_target_weight=0.0,
        face_target_weight=1.0,
        face_zone_mode="high_interior_touch",
    )

    assert loss.item() == pytest.approx(2.5)
    assert metrics["hecras_edge_flux_face_loss"].item() == pytest.approx(2.5)


def test_high_zone_projection_never_changes_low_low_edges():
    graph = SimpleNamespace(
        x=torch.zeros((4, 16)),
        zone_label=torch.tensor([3, 0, 0, 0]),
        hecras_face_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
    )
    edge_flux = torch.tensor([0.0, 5.0, -7.0])
    divergence_target = torch.tensor([-2.0, 0.0, 0.0, 0.0])

    projected, info, _ = project_edge_flux(
        graph,
        edge_flux,
        divergence_target,
        mode="high",
        ridge=0.0,
    )

    assert info == 0
    assert projected[0].item() == pytest.approx(2.0, abs=1e-6)
    assert torch.equal(projected[1:], edge_flux[1:])


def test_control_volume_projection_changes_only_component_boundary_flux():
    graph = SimpleNamespace(
        x=torch.zeros((3, 16)),
        zone_label=torch.tensor([3, 3, 0]),
        hecras_boundary_node_mask=torch.zeros(3, dtype=torch.bool),
        hecras_high_interior_control_volume_label=torch.tensor([0, 0, -1]),
        hecras_face_index=torch.tensor([[0, 1], [1, 2]]),
    )
    edge_flux = torch.tensor([5.0, 3.0])
    divergence_target = torch.tensor([1.0, -1.0, 0.0])

    projected, info, _ = project_edge_flux(
        graph,
        edge_flux,
        divergence_target,
        mode="high_interior_control_volume",
        ridge=0.0,
    )

    assert info == 0
    assert projected[0].item() == pytest.approx(5.0, abs=1e-6)
    assert projected[1].item() == pytest.approx(0.0, abs=1e-6)
