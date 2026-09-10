#!/usr/bin/env bash
set -euo pipefail

ROOT=/mnt/8tb_hdd2/joyce
PYTHON="$ROOT/conda-envs/yutest/bin/python"
HGN="$ROOT/hecras-dataset/Linux_RAS_v66/Minxiong/outputs_si5m_v2_h1h100/HGN_dataset"
HDF_ROOT="$ROOT/hecras-dataset/Linux_RAS_v66/Minxiong/outputs_si5m_v2_h1h100"
HERE="$ROOT/physicsnemo/examples/weather/flood_modeling/hydrographnet"
FACE_GRAPH="$HGN/hecras_face_graph_si5m_formal_h1h100.npz"
FACE_GRAPH_SUMMARY="$HGN/hecras_face_graph_si5m_formal_h1h100.json"
TARGET_DIR="$HGN/conservative_edge_targets"
FACE_STATS="$TARGET_DIR/train_only_face_stats.npz"

"$PYTHON" -c \
  'import json,sys; d=json.load(open(sys.argv[1])); assert d["ready_for_training"] is True; assert d["event_count"] == 100; assert d["unit_contract"]["project_unit_system"] == "SI Units"; assert d["model_time_contract"]["delta_t_seconds"] == 300.0' \
  "$HGN/hgn_dataset_validation.json"

"$PYTHON" "$HERE/extract_hecras_face_graph.py" \
  --hdf-file "$HDF_ROOT/planH1/Minxiong.p01.hdf" \
  --hgn-data-dir "$HGN" \
  --output-npz "$FACE_GRAPH" \
  --summary-file "$FACE_GRAPH_SUMMARY"

"$PYTHON" "$HERE/build_conservative_edge_targets.py" \
  --data-dir "$HGN" \
  --ids-file train.txt \
  --hdf-template "$HDF_ROOT/plan{event_id}/Minxiong.p01.hdf" \
  --face-graph-file "$FACE_GRAPH" \
  --zone-label-file "$HGN/zone_label.txt" \
  --selected-zone 3 \
  --output-dir "$TARGET_DIR" \
  --resume

"$PYTHON" "$HERE/validate_conservative_edge_targets.py" \
  --target-dir "$TARGET_DIR" \
  --output-json "$TARGET_DIR/conservative_edge_target_validation.json"

"$PYTHON" "$HERE/compute_hecras_face_target_stats.py" \
  --target-dir "$TARGET_DIR" \
  --event-ids-file "$HGN/train_ids.txt" \
  --hgn-data-dir "$HGN" \
  --n-time-steps 2 \
  --output-npz "$FACE_STATS"

"$PYTHON" "$HERE/audit_conservative_face_target_stats.py" \
  --target-dir "$TARGET_DIR" \
  --face-stats-npz "$FACE_STATS" \
  --event-ids-file "$HGN/train_ids.txt" \
  --hgn-data-dir "$HGN" \
  --n-time-steps 2 \
  --output-json "$TARGET_DIR/train_only_face_stats_audit.json"

date --iso-8601=seconds > "$TARGET_DIR/FORMAL_EDGE_DATA_READY.marker"
printf 'Formal SI 5-minute conservative edge data are ready.\n'
