import importlib.util
from pathlib import Path

import numpy as np


MODULE_PATH = (
    Path(__file__).parents[2]
    / "examples/weather/flood_modeling/hydrographnet/extract_hecras_face_graph.py"
)
SPEC = importlib.util.spec_from_file_location("extract_hecras_face_graph", MODULE_PATH)
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)


def test_selective_scope_excludes_boundary_and_partitions_components():
    face_graph = {
        "internal_face_index": np.asarray(
            [[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=np.int64
        ),
        # HDF cell 2 maps to node 2 and is therefore excluded from the scope.
        "boundary_face_cell_index": np.asarray([[2, 9]], dtype=np.int64),
    }
    hdf_to_hgn = np.asarray([0, 1, 2, 3, 4, 5, -1, -1, -1, -1])
    zones = np.asarray([3, 3, 3, 0, 3, 3], dtype=np.int64)

    arrays, summary = MODULE.build_selective_scope(
        face_graph, hdf_to_hgn, zones
    )

    assert np.array_equal(
        arrays["boundary_node_mask"],
        np.asarray([False, False, True, False, False, False]),
    )
    labels = arrays["high_interior_control_volume_label"]
    assert labels[0] == labels[1] >= 0
    assert labels[4] == labels[5] >= 0
    assert labels[0] != labels[4]
    assert labels[2] == labels[3] == -1
    assert summary["num_high_interior_control_volumes"] == 2
    assert summary["num_active_high_interior_touch_faces"] == 4
