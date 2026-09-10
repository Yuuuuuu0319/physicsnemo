from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from physicsnemo.datapipes.gnn.hydrographnet_dataset import HydroGraphDataset


def test_sharded_conservative_targets_are_memory_mapped_and_attached(tmp_path: Path):
    selected = np.asarray([1, 3], dtype=np.int64)
    face = np.asarray([[10.0, 20.0, 30.0], [11.0, 21.0, 31.0]], dtype=np.float32)
    internal = np.asarray([[2.0, -3.0], [4.0, -5.0]], dtype=np.float32)
    boundary = np.asarray([[0.5, 0.25], [0.75, 1.0]], dtype=np.float32)
    precipitation = np.asarray([[1.5, 2.5], [3.5, 4.5]], dtype=np.float32)
    np.save(tmp_path / "selected_node_index.npy", selected)
    arrays = {
        "raw_internal_face_delta_m3": face + 100.0,
        "projected_internal_face_delta_m3": face,
        "target_internal_divergence_selected_m3": internal,
        "omitted_boundary_source_selected_m3": boundary,
        "precipitation_volume_selected_m3": precipitation,
    }
    for name, values in arrays.items():
        np.save(tmp_path / f"H1_{name}.npy", values)

    dataset = object.__new__(HydroGraphDataset)
    dataset.hecras_conservative_edge_target_dir = str(tmp_path)
    dataset.hecras_edge_flow_mode = "conservative_sharded"
    dataset.hecras_edge_flow_npz = None
    dataset.hydrograph_ids = ["H1"]
    dataset.hecras_face_graph = {
        "face_index": np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    }
    loaded = dataset.load_hecras_edge_flow_delta_by_hydrograph()
    assert isinstance(loaded["H1"]["face"], np.memmap)
    assert isinstance(loaded["H1"]["raw_face"], np.memmap)

    dataset.hecras_edge_flow_delta_by_hydrograph = loaded
    dataset.hecras_edge_flow_time_offset = 0
    dataset.hecras_edge_flow_face_stats = None
    dataset.hecras_edge_flow_scale_stats = None
    dataset.hecras_boundary_node_mask = None
    dataset.hecras_high_interior_control_volume_label = None
    dataset.dynamic_stats = {"volume": {"std": 7.0}}
    graph = SimpleNamespace(x=torch.zeros((4, 2)))
    dataset.add_hecras_edge_flow_attrs(graph, transition_index=1, hydrograph_id="H1")

    torch.testing.assert_close(
        graph.hecras_internal_face_flow_delta, torch.tensor(face[1])
    )
    torch.testing.assert_close(
        graph.hecras_raw_internal_face_flow_delta,
        torch.tensor(face[1] + 100.0),
    )
    torch.testing.assert_close(
        graph.hecras_edge_internal_delta,
        torch.tensor([0.0, 4.0, 0.0, -5.0]),
    )
    torch.testing.assert_close(
        graph.hecras_edge_boundary_source_delta,
        torch.tensor([0.0, 0.75, 0.0, 1.0]),
    )
    torch.testing.assert_close(
        graph.hecras_local_source_delta,
        torch.tensor([0.0, 3.5, 0.0, 4.5]),
    )
    torch.testing.assert_close(graph.volume_std, torch.tensor([7.0]))
