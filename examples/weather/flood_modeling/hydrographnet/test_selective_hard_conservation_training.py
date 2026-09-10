from types import SimpleNamespace

import torch

from evaluate_rollout_selective_hard_conservation import (
    expanded_node_target,
    manning_face_volume_heuristic,
    rollout_local_source_delta,
)
from selective_hard_conservation_training import (
    compute_selective_hard_conservation_loss,
    projected_volume_feedback_delta,
    project_training_edge_flux,
    update_hgn_rollout_state,
)


def _graph(face_target=None, boundary_target=None):
    if face_target is None:
        face_target = torch.tensor([0.5, -3.0, 4.0])
    if boundary_target is None:
        boundary_target = torch.zeros(4)
    return SimpleNamespace(
        hecras_face_index=torch.tensor([[0, 1, 2], [1, 2, 3]]),
        hecras_high_interior_control_volume_label=torch.tensor([0, 0, -1, -1]),
        hecras_edge_boundary_source_delta=boundary_target,
        hecras_edge_internal_delta=torch.zeros(4),
        hecras_internal_face_flow_rms=torch.tensor([2.0, 3.0, 5.0]),
        hecras_internal_face_flow_delta=face_target,
        local_source_rate=torch.zeros(4),
        volume_std=torch.tensor([1.0]),
        zone_label=torch.tensor([3, 3, 0, 0]),
        hecras_boundary_node_mask=torch.zeros(4, dtype=torch.bool),
    )


def test_projection_uses_node_budget_and_changes_only_component_boundary_face():
    graph = _graph()
    pred = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
    raw = torch.tensor([0.5, 8.0, 4.0])
    result = project_training_edge_flux(pred, raw, graph, delta_t=1800.0)

    assert torch.allclose(result.edge_flux, torch.tensor([0.5, -3.0, 4.0]))
    assert torch.allclose(result.face_correction, torch.tensor([0.0, 11.0, 0.0]))
    assert torch.max(torch.abs(result.projected_component_residual)) < 1e-6


def test_projection_does_not_consume_event_face_target():
    pred = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
    raw = torch.tensor([0.5, 8.0, 4.0])
    first = project_training_edge_flux(pred, raw, _graph(), delta_t=1800.0)
    second = project_training_edge_flux(
        pred,
        raw,
        _graph(face_target=torch.tensor([999.0, -777.0, 555.0])),
        delta_t=1800.0,
    )
    assert torch.equal(first.edge_flux, second.edge_flux)


def test_projection_does_not_consume_hecras_boundary_target():
    pred = torch.tensor([[0.0, 1.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]])
    raw = torch.tensor([0.5, 8.0, 4.0])
    first = project_training_edge_flux(pred, raw, _graph(), delta_t=1800.0)
    second = project_training_edge_flux(
        pred,
        raw,
        _graph(boundary_target=torch.tensor([999.0, -777.0, 555.0, 333.0])),
        delta_t=1800.0,
    )
    assert torch.equal(first.edge_flux, second.edge_flux)


def test_joint_loss_backpropagates_to_node_and_edge_predictions():
    graph = _graph(face_target=torch.tensor([1.0, -2.0, 4.0]))
    pred = torch.tensor(
        [[0.0, 1.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
        requires_grad=True,
    )
    raw = torch.tensor([0.5, 8.0, 4.0], requires_grad=True)
    result = compute_selective_hard_conservation_loss(
        pred,
        raw,
        graph,
        delta_t=1800.0,
        raw_control_volume_weight=0.3,
    )
    result.loss.backward()

    assert torch.isfinite(result.loss)
    assert result.projected_face_loss > 0
    assert result.raw_control_volume_loss > 0
    assert pred.grad is not None and torch.any(torch.abs(pred.grad) > 0)
    assert raw.grad is not None and torch.any(torch.abs(raw.grad) > 0)


def test_selected_node_evaluation_target_expands_to_full_graph():
    target = expanded_node_target(
        {
            "selected_nodes": torch.tensor([1, 3]),
            "internal_selected": torch.tensor([[2.0, -5.0]]),
        },
        "internal",
        0,
        4,
        dtype=torch.float32,
        device=torch.device("cpu"),
    )
    assert torch.equal(target, torch.tensor([0.0, 2.0, 0.0, -5.0]))


def test_rollout_uses_inference_side_node_precipitation_directly():
    dataset = SimpleNamespace(
        dynamic_stats={"precipitation": {"mean": 100.0, "std": 50.0}},
        hecras_face_graph={"node_surface_area": torch.tensor([2.0, 3.0])},
        local_source_runoff_mode="full_area",
    )
    x_iter = torch.zeros((2, 16))
    node_rate = torch.tensor([1.0e-6, 4.0e-6])
    source_delta = rollout_local_source_delta(
        dataset,
        x_iter,
        target_precipitation=torch.tensor(999.0),
        delta_t=300.0,
        device=torch.device("cpu"),
        target_node_precipitation=node_rate,
    )
    assert torch.allclose(
        source_delta,
        node_rate * torch.tensor([2.0, 3.0]) * 300.0,
    )


def test_manning_face_heuristic_follows_water_surface_direction():
    graph = SimpleNamespace(
        hecras_face_index=torch.tensor([[0], [1]]),
        hecras_face_length=torch.tensor([2.0]),
        hecras_edge_physical_features=torch.tensor(
            [[10.0, 0.0, 0.0, 0.04]]
        ),
        current_water_depth_denorm=torch.tensor([1.0, 1.0]),
        current_surface_elevation=torch.tensor([2.0, 1.0]),
    )
    volume = manning_face_volume_heuristic(graph, 300.0)
    assert volume.item() > 0.0
    graph.current_surface_elevation = torch.tensor([1.0, 2.0])
    reverse = manning_face_volume_heuristic(graph, 300.0)
    assert torch.allclose(reverse, -volume)


def test_projected_rollout_feedback_preserves_cross_step_gradients():
    graph = _graph()
    pred = torch.tensor(
        [[0.0, 1.0], [0.0, 2.0], [0.0, 0.5], [0.0, -0.5]],
        requires_grad=True,
    )
    raw = torch.tensor([0.5, 8.0, 4.0], requires_grad=True)
    projection = project_training_edge_flux(pred, raw, graph, delta_t=300.0)
    volume_delta = projected_volume_feedback_delta(
        pred, projection, graph, delta_t=300.0, alpha=1.0
    )
    x = torch.zeros((4, 16))
    updated = update_hgn_rollout_state(
        x,
        pred,
        n_time_steps=2,
        next_inflow=torch.tensor([0.25]),
        next_precipitation=torch.tensor([-0.5]),
        volume_delta_override=volume_delta,
    )
    second_step_loss = updated[:, 15].square().sum()
    second_step_loss.backward()

    assert pred.grad is not None and torch.any(torch.abs(pred.grad) > 0)
    assert raw.grad is not None and torch.any(torch.abs(raw.grad) > 0)
    assert updated[0, 10].item() == 0.25
    assert updated[0, 11].item() == -0.5
    # Nodes outside the selected control volume retain the HGN node increment.
    assert torch.allclose(volume_delta[2:], pred.detach()[2:, 1])


def test_dataset_attaches_consecutive_sharded_edge_rollout_targets():
    from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset

    dataset = object.__new__(HydroGraphDataset)
    dataset.hecras_edge_flow_time_offset = 2
    dataset.hecras_edge_flow_scale_stats = None
    selected_nodes = torch.tensor([1, 3]).numpy()
    face = torch.arange(30, dtype=torch.float).reshape(10, 3).numpy()
    selected = torch.arange(20, dtype=torch.float).reshape(10, 2).numpy()
    dataset.hecras_edge_flow_delta_by_hydrograph = {
        "H1": {
            "selected_nodes": selected_nodes,
            "internal_selected": selected,
            "boundary_selected": selected + 100.0,
            "precipitation_selected": selected + 200.0,
            "face": face,
            "raw_face": face + 300.0,
        }
    }
    graph = SimpleNamespace(x=torch.zeros((4, 16)))
    dataset.add_hecras_edge_flow_rollout_attrs(graph, 1, 3, "H1")

    assert graph.training_rollout_internal_face_flow_delta.shape == (3, 3)
    assert torch.equal(
        graph.training_rollout_internal_face_flow_delta,
        torch.as_tensor(face[3:6].T),
    )
    assert torch.equal(
        graph.training_rollout_edge_internal_delta[selected_nodes],
        torch.as_tensor(selected[3:6].T),
    )
    assert torch.count_nonzero(
        graph.training_rollout_edge_internal_delta[[0, 2]]
    ) == 0
