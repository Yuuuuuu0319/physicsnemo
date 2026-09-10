import importlib.util
import json
from pathlib import Path

import numpy as np
import scipy.sparse.linalg as spla


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/weather/flood_modeling/hydrographnet/build_conservative_edge_targets.py"
)
SPEC = importlib.util.spec_from_file_location("build_conservative_edge_targets", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)

VALIDATOR_PATH = (
    Path(__file__).resolve().parents[2]
    / "examples/weather/flood_modeling/hydrographnet/validate_conservative_edge_targets.py"
)
VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_conservative_edge_targets", VALIDATOR_PATH
)
VALIDATOR = importlib.util.module_from_spec(VALIDATOR_SPEC)
assert VALIDATOR_SPEC.loader is not None
VALIDATOR_SPEC.loader.exec_module(VALIDATOR)


def test_minimum_change_projection_closes_selected_nodes():
    face_index = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    selected_nodes = np.asarray([1, 2], dtype=np.int64)
    incidence = MODULE.build_selected_incidence(face_index, selected_nodes, 4)
    solve_fn = spla.factorized((incidence @ incidence.T).tocsc())
    raw = np.asarray([[1.0, 4.0, -2.0], [3.0, -1.0, 2.0]])
    target = np.asarray([[2.0, -3.0], [-4.0, 1.0]])

    projected, correction = MODULE.project_minimum_change(
        incidence, raw, target, solve_fn
    )

    np.testing.assert_allclose((incidence @ projected.T).T, target, atol=1e-12)
    np.testing.assert_allclose(projected, raw - correction, atol=1e-12)


def test_independent_validator_uses_the_same_edge_orientation():
    face_index = np.asarray([[0, 1, 2], [1, 2, 3]], dtype=np.int64)
    selected_nodes = np.asarray([1, 2], dtype=np.int64)
    builder_incidence = MODULE.build_selected_incidence(
        face_index, selected_nodes, 4
    )
    validator_incidence = VALIDATOR.selected_incidence(
        face_index, selected_nodes, 4
    )
    np.testing.assert_array_equal(
        builder_incidence.toarray(), validator_incidence.toarray()
    )


def test_resume_rejects_shard_when_any_source_contract_changes(tmp_path):
    event_id = "H1"
    hdf_path = tmp_path / "event.hdf"
    face_graph_path = tmp_path / "face_graph.npz"
    zone_path = tmp_path / "zone_label.txt"
    volume_path = tmp_path / "M80_V_H1.txt"
    inflow_path = tmp_path / "M80_US_InF_H1.txt"
    training_array_path = tmp_path / "H1_projected.npy"
    output_path = tmp_path / "H1_conservative_edge_target.npz"
    hdf_path.write_bytes(b"hdf-v1")
    face_graph_path.write_bytes(b"face-v1")
    zone_path.write_bytes(b"zone-v1")
    volume_path.write_bytes(b"volume-v1")
    inflow_path.write_bytes(b"time-v1")
    np.save(training_array_path, np.asarray([1.0], dtype=np.float32))
    face_hash = MODULE.file_sha256(face_graph_path)
    zone_hash = MODULE.file_sha256(zone_path)
    summary = {
        "target_builder_schema_version": MODULE.TARGET_BUILDER_SCHEMA_VERSION,
        "event_id": event_id,
        "hdf_sha256": MODULE.file_sha256(hdf_path),
        "hgn_volume_sha256": MODULE.file_sha256(volume_path),
        "hgn_inflow_time_sha256": MODULE.file_sha256(inflow_path),
        "face_graph_sha256": face_hash,
        "zone_label_sha256": zone_hash,
        "training_arrays": {
            "projected_internal_face_delta_m3": {
                "path": str(training_array_path),
                "sha256": MODULE.file_sha256(training_array_path),
            }
        },
    }
    np.savez_compressed(
        output_path,
        metadata_json=np.asarray(
            json.dumps({"schema_version": MODULE.TARGET_BUILDER_SCHEMA_VERSION})
        ),
        summary_json=np.asarray(json.dumps(summary)),
    )

    assert MODULE.load_current_resumable_summary(
        output_path,
        event_id,
        hdf_path,
        tmp_path,
        "M80",
        face_hash,
        zone_hash,
    ) is not None

    hdf_path.write_bytes(b"hdf-v2")
    assert MODULE.load_current_resumable_summary(
        output_path,
        event_id,
        hdf_path,
        tmp_path,
        "M80",
        face_hash,
        zone_hash,
    ) is None
