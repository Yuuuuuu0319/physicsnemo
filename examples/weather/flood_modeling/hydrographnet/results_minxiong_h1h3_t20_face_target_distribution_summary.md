# HEC-RAS Internal Face Target Distribution

- edge flow npz: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_edge_flow_delta_h1h24_internal_plus_boundary_source_with_face.npz`
- face graph: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_face_graph_h1h30.npz`
- zone labels: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/zone_label.txt`
- events: `3`
- transitions: `60`
- global face target RMS: `88795.339608` ft^3
- global face target |max|: `3783204.750000` ft^3
- largest transition RMS: `H3` transition `19` RMS `141245.442431` ft^3
- Zone 3 touching face RMS: `23433.564381` ft^3
- Zone 3 internal face RMS: `23063.975061` ft^3

## Outputs

- event CSV: `examples/weather/flood_modeling/hydrographnet/results_minxiong_h1h3_t20_face_target_distribution_by_event.csv`
- transition CSV: `examples/weather/flood_modeling/hydrographnet/results_minxiong_h1h3_t20_face_target_distribution_by_transition.csv`
