# HEC-RAS Internal Face Target Distribution

- edge flow npz: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_edge_flow_delta_h1h24_internal_plus_boundary_source_with_face.npz`
- face graph: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_face_graph_h1h30.npz`
- zone labels: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/zone_label.txt`
- events: `24`
- transitions: `2880`
- global face target RMS: `307006.148697` ft^3
- global face target |p50|: `1254.350952` ft^3
- global face target |p90|: `98798.211719` ft^3
- global face target |p99|: `1096146.302500` ft^3
- global face target |max|: `12032229.000000` ft^3
- largest transition RMS: `H18` transition `101` RMS `503410.456638` ft^3
- Zone 3 touching face RMS: `275590.155578` ft^3
- Zone 3 internal face RMS: `250869.882959` ft^3

## Outputs

- event CSV: `examples/weather/flood_modeling/hydrographnet/results_minxiong_h1h24_face_target_distribution_by_event.csv`
- transition CSV: `examples/weather/flood_modeling/hydrographnet/results_minxiong_h1h24_face_target_distribution_by_transition.csv`
