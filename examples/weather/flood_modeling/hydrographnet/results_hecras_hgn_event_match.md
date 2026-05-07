# HEC-RAS / HGN Event Match Report

HDF file: `/mnt/8tb_hdd2/joyce/Minxiong/Minxiong.p03.hdf`

## Best Matches

| HGN ID | Split | HDF series | Corr | z-NRMSE | Max abs diff | Peak index diff | Exact/close |
|---|---|---|---:|---:|---:|---:|---:|
| H1 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H2 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H3 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H4 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H5 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H6 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H7 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H8 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H9 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H10 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H11 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H12 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H13 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H14 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H15 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H16 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H17 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H18 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H19 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H20 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H21 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H22 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H23 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H24 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |
| H25 | HGN_train | result_upstreamBC1 | 1.000000 | 3.931133e-13 | 5.002221e-10 | 0 | 1 |

## Interpretation

- `50` HGN hydrographs exactly/closely match `result_upstreamBC1` within numerical tolerance.
- If all H1-H50 match the same HDF upstream hydrograph, then this HDF appears to use a shared inflow template rather than uniquely identifying one HGN event by upstream flow alone.
- Event matching for face velocity is stronger when inflow, precipitation, and output state/volume timing all match; this report focuses on boundary hydrographs and should be treated as a necessary check, not the only check.

## Precipitation Check

- HDF event precipitation max: `1.0`
- HGN precipitation files checked: `50`
- HGN precipitation nonzero files: `0`
- HGN precipitation max abs: `0.0`

If HDF precipitation is nonzero while all HGN precipitation files are zero, the upstream boundary can still match exactly, but the HDF face velocity should not yet be treated as a fully matched event-specific target.
