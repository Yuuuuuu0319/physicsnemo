# Minxiong HEC-RAS Edge Feature Audit

## Inputs

- data dir: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset`
- face graph: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_face_graph_h1h30.npz`
- face stats: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_internal_face_target_stats_h1h24.npz`
- ids file: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/train_h1h24.txt`
- landcover HDF: `/mnt/8tb_hdd2/joyce/hecras-dataset/winres/LandCover.hdf`
- output CSV: `examples/weather/flood_modeling/hydrographnet/results_minxiong_h1h24_edge_feature_audit.csv`

## Feature Availability

| Feature | Status | Source / note |
|---|---|---|
| HEC-RAS internal face connectivity | available | `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_face_graph_h1h30.npz` |
| Face length and normal | available | `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_face_graph_h1h30.npz` |
| Cell elevation/area/Manning/IP | available | `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset` |
| LandCover Manning-to-IP lookup | available | `/mnt/8tb_hdd2/joyce/hecras-dataset/winres/LandCover.hdf` |
| Dynamic water-surface gradient proxy | available | `24 events / 2880 transitions` |
| Native internal Face Flow target stats | available | `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_internal_face_target_stats_h1h24.npz` |
| Boundary face flow supervision | not in this audit | `edge NPZ stores node boundary/source delta` |
| HEC-RAS face hydraulic conveyance/open area | missing | `not exported to current NPZ` |
| Autoregressive predicted edge state | missing | `not a deployed model feature yet` |

## LandCover / IP Check

- LandCover classes read: `6`
- Cells matched by Manning's n lookup: `12715` / `12715`
- HGN `M80_IP` unique values: `[0.0, 2.0, 10.0, 85.0]`
- LandCover Manning to IP mapping: `{-9999.0: -9999.0, 0.12: 85.0, 0.045: 10.0, 0.03: 0.0, 0.1: 2.0, 0.05: 5.0}`

## Face Target RMS by Zone Group

| Group | Faces | Median ft^3 | Mean ft^3 | P90 ft^3 | P99 ft^3 | Max ft^3 |
|---|---:|---:|---:|---:|---:|---:|
| all | 28633 | 7958.847 | 74729.545 | 122988.628 | 1297282.285 | 6700509.000 |
| zone3_touching | 3564 | 154.072 | 49941.136 | 28053.593 | 1216095.442 | 5288590.500 |
| zone3_internal | 2015 | 126.622 | 30446.842 | 2697.517 | 679036.086 | 5288590.500 |
| cross_zone | 6170 | 1188.891 | 114549.084 | 255962.623 | 2030621.744 | 6700509.000 |

## Feature Correlations

Pearson correlation uses `log1p(face_target_rms)` as the target.

| Feature comparison | Pearson r |
|---|---:|
| log1p(face_target_rms) vs log1p(face_length) | 0.429555 |
| log1p(face_target_rms) vs abs(elevation_diff) | -0.277877 |
| log1p(face_target_rms) vs abs(bed_slope) | -0.297451 |
| log1p(face_target_rms) vs manning_mean | -0.226458 |
| log1p(face_target_rms) vs ip_mean | 0.096717 |
| log1p(face_target_rms) vs log1p(area_mean) | 0.488796 |
| log1p(face_target_rms) vs log1p(surface_gradient_rms) | -0.385408 |
| log1p(face_target_rms) vs zone3_touching | -0.325258 |
| log1p(face_target_rms) vs zone3_internal | -0.269145 |

## Top 10 High-RMS Faces

| Face | Src | Dst | Zone pair | Target RMS ft^3 | Surface grad RMS | Face length | Manning mean | IP mean |
|---:|---:|---:|---|---:|---:|---:|---:|---:|
| 23632 | 9326 | 9327 | 0-1 | 6700509.000 | 0.0114834 | 111.278 | 0.120000 | 85.000 |
| 26987 | 9338 | 9337 | 0-0 | 6636475.500 | 0.000492487 | 100.044 | 0.120000 | 85.000 |
| 17649 | 8736 | 8737 | 0-0 | 6249445.000 | 0.00134762 | 124.532 | 0.120000 | 85.000 |
| 17643 | 8735 | 8736 | 0-0 | 6090936.500 | 0.000299035 | 107.416 | 0.120000 | 85.000 |
| 18556 | 9348 | 9347 | 1-1 | 5977050.000 | 0.0420654 | 85.607 | 0.075000 | 42.500 |
| 26450 | 8899 | 8898 | 1-1 | 5973260.500 | 5.75027e-05 | 115.522 | 0.030000 | 0.000 |
| 18291 | 9331 | 11915 | 1-2 | 5951015.500 | 4.65898e-05 | 115.463 | 0.075000 | 42.500 |
| 18038 | 8738 | 8737 | 0-0 | 5824680.500 | 0.000694918 | 112.681 | 0.120000 | 85.000 |
| 23628 | 9342 | 10750 | 1-2 | 5822670.500 | 0.000353025 | 84.348 | 0.120000 | 85.000 |
| 18420 | 9328 | 10748 | 1-1 | 5628407.000 | 7.19263e-05 | 64.461 | 0.120000 | 85.000 |

## Interpretation

- The current data now has enough topology and static hydraulic context to build a better edge feature matrix: face length, normal, cell area, bed elevation difference, Manning's n, IP, zone pair, and a dynamic water-surface gradient proxy.
- The current edge head did not consume this full audited feature set; it only used a subset through node features and optional face normal / previous face flow. This audit identifies the next concrete feature block for a publishable edge-local conservation branch.
- Missing pieces for a DUALFloodGNN-level edge method remain: autoregressive edge state, stronger edge message passing, and direct boundary-face flow representation rather than only node-level boundary/source residuals.
