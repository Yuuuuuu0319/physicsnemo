# HEC-RAS Edge Flow Delta Validation

This diagnostic compares interval-integrated native HEC-RAS `Face Flow` against the existing `Cell Flow Balance` transport delta over HGN time intervals. It is a gate before training an edge-informed loss.

## Inputs

- HGN data dir: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset`
- HDF glob: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/planH*/Minxiong.p01.hdf`
- Face graph: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_face_graph_h1h30.npz`
- Event ids file: `test_h25h30.txt`

## Summary

| Event | Mode | Relative RMSE | RMSE (ft^3) | Correlation | Zone 3 RMSE (ft^3) | Zone 3 Relative RMSE |
|---|---|---:|---:|---:|---:|---:|
| H25 | internal | 4.677908e+01 | 7.129300e+04 | 0.016901 | 1.479977e+05 | 2.287506e+02 |
| H25 | all_touching | 1.444774e-06 | 2.201887e-03 | 1.000000 | 1.548349e-03 | 2.393184e-06 |
| H26 | internal | 4.363487e+01 | 6.318608e+04 | 0.018583 | 1.283475e+05 | 1.971350e+02 |
| H26 | all_touching | 1.438873e-06 | 2.083579e-03 | 1.000000 | 1.399729e-03 | 2.149911e-06 |
| H27 | internal | 4.853909e+01 | 7.068217e+04 | 0.017252 | 1.532013e+05 | 2.360083e+02 |
| H27 | all_touching | 1.535009e-06 | 2.235265e-03 | 1.000000 | 1.538822e-03 | 2.370571e-06 |
| H28 | internal | 4.941003e+01 | 7.683868e+04 | 0.016427 | 1.682431e+05 | 2.544731e+02 |
| H28 | all_touching | 1.565293e-06 | 2.434224e-03 | 1.000000 | 1.811785e-03 | 2.740384e-06 |
| H29 | internal | 4.723118e+01 | 7.006462e+04 | 0.017108 | 1.455018e+05 | 2.258395e+02 |
| H29 | all_touching | 1.464105e-06 | 2.171913e-03 | 1.000000 | 1.521239e-03 | 2.361180e-06 |
| H30 | internal | 4.429707e+01 | 6.356442e+04 | 0.017959 | 1.258004e+05 | 1.946880e+02 |
| H30 | all_touching | 1.478526e-06 | 2.121622e-03 | 1.000000 | 1.333980e-03 | 2.064460e-06 |

## Interpretation Guide

- `all_touching` includes any HEC-RAS face that touches an HGN node, including boundary/ghost faces.
- `internal` includes only faces where both adjacent cells are HGN nodes.
- If `all_touching` matches Cell Flow Balance but `internal` does not, the boundary-face contribution is required for exact local closure.
- A future high-zone edge loss can still use an internal physical edge graph, but boundary nodes need explicit treatment or masking.
