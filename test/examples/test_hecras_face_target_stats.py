import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[2]
    / "examples/weather/flood_modeling/hydrographnet/compute_hecras_face_target_stats.py"
)
SPEC = importlib.util.spec_from_file_location(
    "compute_hecras_face_target_stats", MODULE_PATH
)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_usable_transition_slice_matches_dataset_sampling(tmp_path: Path):
    validation = {
        "model_time_contract": {
            "dynamic_skip_steps": 3,
            "post_peak_steps": 4,
        }
    }
    (tmp_path / "hgn_dataset_validation.json").write_text(json.dumps(validation))
    # Peak is at trimmed index 4, so retained length is 4 + 4 = 8 frames.
    inflow = np.asarray([0, 0, 0, 1, 2, 3, 4, 9, 5, 3, 1, 0], dtype=float)
    np.savetxt(
        tmp_path / "M80_US_InF_H1.txt",
        np.column_stack((np.arange(inflow.size), inflow)),
        delimiter="\t",
    )

    selected = MODULE.usable_transition_slice(
        tmp_path, "H1", target_transition_count=11, n_time_steps=2
    )

    assert selected == slice(4, 10)


def test_sharded_targets_follow_explicit_event_split(tmp_path: Path):
    target_dir = tmp_path / "targets"
    target_dir.mkdir()
    np.save(
        target_dir / "H2_projected_internal_face_delta_m3.npy",
        np.ones((3, 2), dtype=np.float32),
    )
    np.save(
        target_dir / "H1_projected_internal_face_delta_m3.npy",
        np.zeros((3, 2), dtype=np.float32),
    )
    ids_path = tmp_path / "train_ids.txt"
    ids_path.write_text("H2\nH1\n")
    args = SimpleNamespace(
        target_dir=target_dir,
        edge_flow_npz=None,
        event_ids_file=ids_path,
    )

    loaded = MODULE.load_target_arrays(args)

    assert [event_id for event_id, _ in loaded] == ["H2", "H1"]
    assert all(isinstance(values, np.memmap) for _, values in loaded)
