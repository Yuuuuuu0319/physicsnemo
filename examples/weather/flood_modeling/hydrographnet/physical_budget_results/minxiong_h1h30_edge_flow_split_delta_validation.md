# HEC-RAS Edge Flow Delta Validation

This diagnostic compares interval-integrated native HEC-RAS `Face Flow` against the existing `Cell Flow Balance` transport delta over HGN time intervals. It is a gate before training an edge-informed loss.

## Inputs

- HGN data dir: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset`
- HDF glob: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/planH*/Minxiong.p01.hdf`
- Face graph: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong_hgn_h1h30_20260611/outputs/HGN_dataset/hecras_face_graph_h1h30.npz`
- Event ids file: `auto-discovered`

## Summary

| Event | Mode | Relative RMSE | RMSE (ft^3) | Correlation | Zone 3 RMSE (ft^3) | Zone 3 Relative RMSE |
|---|---|---:|---:|---:|---:|---:|
| H1 | internal | 4.929663e+01 | 7.330092e+04 | 0.016408 | 1.527969e+05 | 2.368639e+02 |
| H1 | all_touching | 1.548073e-06 | 2.301885e-03 | 1.000000 | 1.578016e-03 | 2.446220e-06 |
| H2 | internal | 4.507364e+01 | 6.925610e+04 | 0.017047 | 1.382528e+05 | 2.122457e+02 |
| H2 | all_touching | 1.442871e-06 | 2.216986e-03 | 1.000000 | 1.587413e-03 | 2.436997e-06 |
| H3 | internal | 4.887619e+01 | 7.490928e+04 | 0.016282 | 1.585882e+05 | 2.371095e+02 |
| H3 | all_touching | 1.542384e-06 | 2.363908e-03 | 1.000000 | 1.653822e-03 | 2.472675e-06 |
| H4 | internal | 4.710884e+01 | 6.988356e+04 | 0.016968 | 1.419726e+05 | 2.185036e+02 |
| H4 | all_touching | 1.496912e-06 | 2.220592e-03 | 1.000000 | 1.499425e-03 | 2.307696e-06 |
| H5 | internal | 4.709248e+01 | 7.261611e+04 | 0.017134 | 1.567187e+05 | 2.401631e+02 |
| H5 | all_touching | 1.480625e-06 | 2.283108e-03 | 1.000000 | 1.629786e-03 | 2.497561e-06 |
| H6 | internal | 4.767168e+01 | 6.912833e+04 | 0.017351 | 1.474003e+05 | 2.285266e+02 |
| H6 | all_touching | 1.494946e-06 | 2.167809e-03 | 1.000000 | 1.562742e-03 | 2.422845e-06 |
| H7 | internal | 4.733558e+01 | 7.217050e+04 | 0.016475 | 1.461736e+05 | 2.268202e+02 |
| H7 | all_touching | 1.488824e-06 | 2.269945e-03 | 1.000000 | 1.566587e-03 | 2.430902e-06 |
| H8 | internal | 4.814230e+01 | 7.173846e+04 | 0.017115 | 1.563404e+05 | 2.390177e+02 |
| H8 | all_touching | 1.511695e-06 | 2.252627e-03 | 1.000000 | 1.654255e-03 | 2.529072e-06 |
| H9 | internal | 4.744706e+01 | 7.000586e+04 | 0.017402 | 1.520514e+05 | 2.332704e+02 |
| H9 | all_touching | 1.506741e-06 | 2.223124e-03 | 1.000000 | 1.695922e-03 | 2.601807e-06 |
| H10 | internal | 4.580248e+01 | 6.746547e+04 | 0.018005 | 1.461277e+05 | 2.259983e+02 |
| H10 | all_touching | 1.462931e-06 | 2.154848e-03 | 1.000000 | 1.466378e-03 | 2.267872e-06 |
| H11 | internal | 4.707985e+01 | 6.810488e+04 | 0.018057 | 1.498552e+05 | 2.291184e+02 |
| H11 | all_touching | 1.502451e-06 | 2.173419e-03 | 1.000000 | 1.613292e-03 | 2.466614e-06 |
| H12 | internal | 4.790933e+01 | 7.075117e+04 | 0.017624 | 1.544820e+05 | 2.369222e+02 |
| H12 | all_touching | 1.533097e-06 | 2.264035e-03 | 1.000000 | 1.558476e-03 | 2.390165e-06 |
| H13 | internal | 4.732912e+01 | 7.261998e+04 | 0.017079 | 1.578476e+05 | 2.411029e+02 |
| H13 | all_touching | 1.498528e-06 | 2.299283e-03 | 1.000000 | 1.720104e-03 | 2.627359e-06 |
| H14 | internal | 4.406158e+01 | 6.761980e+04 | 0.017507 | 1.354477e+05 | 2.085890e+02 |
| H14 | all_touching | 1.390939e-06 | 2.134626e-03 | 1.000000 | 1.480889e-03 | 2.280565e-06 |
| H15 | internal | 4.444196e+01 | 6.553438e+04 | 0.017746 | 1.298141e+05 | 2.013684e+02 |
| H15 | all_touching | 1.440441e-06 | 2.124083e-03 | 1.000000 | 1.407202e-03 | 2.182861e-06 |
| H16 | internal | 4.521645e+01 | 6.884192e+04 | 0.017617 | 1.443304e+05 | 2.237242e+02 |
| H16 | all_touching | 1.433747e-06 | 2.182875e-03 | 1.000000 | 1.518490e-03 | 2.353788e-06 |
| H17 | internal | 4.820874e+01 | 7.142743e+04 | 0.017392 | 1.578156e+05 | 2.391736e+02 |
| H17 | all_touching | 1.525363e-06 | 2.260020e-03 | 1.000000 | 1.576728e-03 | 2.389571e-06 |
| H18 | internal | 5.047696e+01 | 7.526432e+04 | 0.016463 | 1.644341e+05 | 2.510321e+02 |
| H18 | all_touching | 1.599590e-06 | 2.385089e-03 | 1.000000 | 1.641014e-03 | 2.505242e-06 |
| H19 | internal | 4.534558e+01 | 6.883077e+04 | 0.017814 | 1.495491e+05 | 2.296298e+02 |
| H19 | all_touching | 1.424203e-06 | 2.161820e-03 | 1.000000 | 1.502053e-03 | 2.306373e-06 |
| H20 | internal | 4.824673e+01 | 7.314032e+04 | 0.017017 | 1.591923e+05 | 2.432230e+02 |
| H20 | all_touching | 1.537016e-06 | 2.330061e-03 | 1.000000 | 1.781459e-03 | 2.721814e-06 |
| H21 | internal | 4.866295e+01 | 7.158794e+04 | 0.017397 | 1.577185e+05 | 2.396061e+02 |
| H21 | all_touching | 1.544829e-06 | 2.272594e-03 | 1.000000 | 1.720598e-03 | 2.613934e-06 |
| H22 | internal | 4.545658e+01 | 6.432418e+04 | 0.017942 | 1.293359e+05 | 2.013812e+02 |
| H22 | all_touching | 1.488235e-06 | 2.105955e-03 | 1.000000 | 1.355378e-03 | 2.110379e-06 |
| H23 | internal | 4.564449e+01 | 6.681172e+04 | 0.017460 | 1.357331e+05 | 2.114043e+02 |
| H23 | all_touching | 1.477311e-06 | 2.162401e-03 | 1.000000 | 1.429525e-03 | 2.226484e-06 |
| H24 | internal | 4.651095e+01 | 6.931124e+04 | 0.017305 | 1.453689e+05 | 2.213350e+02 |
| H24 | all_touching | 1.469645e-06 | 2.190084e-03 | 1.000000 | 1.535725e-03 | 2.338255e-06 |
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
