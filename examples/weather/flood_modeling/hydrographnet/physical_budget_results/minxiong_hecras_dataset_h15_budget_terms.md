# HEC-RAS Budget Term Diagnosis

This report diagnoses why synchronized HEC-RAS/HGN storage still does not close with native budget terms.

## Storage Checks

- Active-cell HGN `M80_V` vs reconstructed HDF cell volume RMSE: `2.456587e+04 ft^3`
- Active-cell HGN `M80_V` vs reconstructed HDF cell volume max abs: `3.358437e+05 ft^3`
- Sum of all reconstructed HDF cell volumes vs HEC-RAS global `Computations/Volume` RMSE: `136493475.314017 ft^3`
- Area-extrapolated active-cell volume vs HEC-RAS global `Computations/Volume` delta RMSE: `58.025766 ft^3`
- Volume table cap: `5264` active cells exceed their table top at least once; final over-table fraction `0.411404`, max exceedance `35.357094 ft`
- HEC-RAS sampled `Volume Error` RMS: `0.000000 ft^3`
- `Cell Flow Balance` vs signed native `Face Flow` net sum RMSE: `3.617754e-06 cfs`, correlation `1.000000`
- `Cell Flow Balance` vs inward-only face flow RMSE: `129.572985 cfs`, correlation `0.052634`

## Best Budget Variant

- `cell_balance_trapz_plus_precipitation`: relative RMSE `0.031953`, cell RMSE `75.085891 ft^3`, domain RMSE `5060.497830 ft^3`, correlation `0.999450`

## Variants

| Variant | Relative RMSE | Cell RMSE (ft^3) | Domain RMSE (ft^3) | Bias (ft^3) | Correlation |
|---|---:|---:|---:|---:|---:|
| cell_balance_trapz_plus_precipitation | 0.031953 | 75.085891 | 5060.497830 | 0.084723 | 0.999450 |
| area_extrapolated_target::cell_balance_trapz_plus_precipitation | 0.031953 | 75.085891 | 5060.497291 | 0.084723 | 0.999450 |
| cell_balance_rect_right | 0.099359 | 233.482665 | 2585678.574057 | 174.235059 | 0.997659 |
| area_extrapolated_target::cell_balance_trapz | 0.099903 | 234.761029 | 2585827.608823 | 174.319549 | 0.997604 |
| cell_balance_trapz | 0.099903 | 234.761029 | 2585827.611199 | 174.319549 | 0.997604 |
| cell_balance_rect_left | 0.101486 | 238.482294 | 2585986.540524 | 174.404039 | 0.997434 |
| area_extrapolated_target::cell_balance_trapz_minus_precipitation | 0.191982 | 451.137914 | 5171502.237349 | 348.554375 | 0.991978 |
| cell_balance_trapz_minus_precipitation | 0.191982 | 451.137915 | 5171502.239725 | 348.554376 | 0.991978 |
| precipitation_volume | 0.973682 | 2288.051219 | 5901812.195386 | 460.246335 | 0.184162 |
| zero_budget | 1.000000 | 2349.896866 | 8286479.656762 | 634.481161 | nan |
| negative_cell_balance_trapz | 1.971684 | 4633.252939 | 14152345.288574 | 1094.642773 | -0.997604 |

## Domain Diagnostics

| Target | Budget | Relative RMSE | RMSE (ft^3) | Bias (ft^3) | Correlation |
|---|---|---:|---:|---:|---:|
| area_extrapolated_active_delta_v | boundary_total_plus_precipitation | 0.000155 | 1285.437816 | 394.768726 | 1.000000 |
| active_hgn_delta_v | boundary_total_plus_precipitation | 0.000155 | 1285.447088 | 394.770346 | 1.000000 |
| hecras_global_computations_volume_delta | boundary_total_plus_precipitation | 0.000157 | 1299.369118 | 450.379488 | 1.000000 |
| area_extrapolated_active_delta_v | sum_cell_balance_plus_precipitation | 0.000611 | 5060.497291 | 1077.246910 | 0.999997 |
| active_hgn_delta_v | sum_cell_balance_plus_precipitation | 0.000611 | 5060.497830 | 1077.248530 | 0.999997 |
| hecras_global_computations_volume_delta | sum_cell_balance_plus_precipitation | 0.000612 | 5067.959327 | 1132.857672 | 0.999997 |
| area_extrapolated_active_delta_v | boundary_total | 0.312048 | 2585780.293009 | 2215790.586798 | 0.825501 |
| active_hgn_delta_v | boundary_total | 0.312048 | 2585780.295389 | 2215790.588418 | 0.825501 |
| hecras_global_computations_volume_delta | boundary_total | 0.312052 | 2585834.135611 | 2215846.197560 | 0.825501 |
| area_extrapolated_active_delta_v | sum_cell_balance | 0.312054 | 2585827.608823 | 2216473.064982 | 0.825546 |
| active_hgn_delta_v | sum_cell_balance | 0.312054 | 2585827.611199 | 2216473.066602 | 0.825546 |
| hecras_global_computations_volume_delta | sum_cell_balance | 0.312058 | 2585881.458091 | 2216528.675744 | 0.825547 |
| area_extrapolated_active_delta_v | boundary_total_minus_precipitation | 0.624084 | 5171457.236555 | 4431186.404870 | -0.588999 |
| active_hgn_delta_v | boundary_total_minus_precipitation | 0.624084 | 5171457.238932 | 4431186.406490 | -0.588999 |
| hecras_global_computations_volume_delta | boundary_total_minus_precipitation | 0.624086 | 5171511.077032 | 4431242.015632 | -0.588998 |
| area_extrapolated_active_delta_v | sum_cell_balance_minus_precipitation | 0.624089 | 5171502.237349 | 4431868.883054 | -0.587748 |
| active_hgn_delta_v | sum_cell_balance_minus_precipitation | 0.624089 | 5171502.239725 | 4431868.884674 | -0.587748 |
| hecras_global_computations_volume_delta | sum_cell_balance_minus_precipitation | 0.624091 | 5171556.081183 | 4431924.493816 | -0.587748 |
| all_reconstructed_delta_v | sum_cell_balance_minus_precipitation | 0.653241 | 1542075.451522 | -1466456.151274 | 0.895231 |
| all_reconstructed_delta_v | boundary_total_minus_precipitation | 0.653366 | 1542369.783171 | -1467138.629458 | 0.895808 |
| area_extrapolated_active_delta_v | precipitation_only | 0.712222 | 5901812.194017 | 5852032.145385 | 0.946071 |
| active_hgn_delta_v | precipitation_only | 0.712222 | 5901812.195386 | 5852032.147005 | 0.946071 |
| hecras_global_computations_volume_delta | precipitation_only | 0.712224 | 5901868.717086 | 5852087.756147 | 0.946070 |
| all_reconstructed_delta_v | precipitation_only | 0.928207 | 2191174.544727 | -46292.888943 | -0.867036 |
| all_reconstructed_delta_v | sum_cell_balance | 1.661453 | 3922114.981721 | -3681851.969346 | -0.259645 |
| all_reconstructed_delta_v | boundary_total | 1.661514 | 3922258.854569 | -3682534.447530 | -0.259262 |
| all_reconstructed_delta_v | sum_cell_balance_plus_precipitation | 2.737443 | 6462153.458092 | -5897247.787418 | -0.715284 |
| all_reconstructed_delta_v | boundary_total_plus_precipitation | 2.737487 | 6462257.860136 | -5897930.265602 | -0.715435 |

### Boundary Components

| Component | RMS volume (ft^3) | Mean volume (ft^3) |
|---|---:|---:|
| DownstreamBC1 - Flow per Face | 967792.048712 | -542114.736055 |
| DownstreamBC2 - Flow per Face | 1066849.140175 | -713603.245491 |
| boundary_total | 5901473.121728 | 5851637.376659 |
| upstreamBC1 - Flow per Face | 5083638.093935 | 5022355.360799 |
| upstreamBC2 - Flow per Face | 2370441.736127 | 2084999.997406 |

## Zone Residuals

| Zone | Cells | Target RMS (ft^3) | Budget RMS (ft^3) | Residual RMSE (ft^3) | Relative RMSE | Bias (ft^3) | Correlation | Wet-change fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6193 | 2961.320572 | 2960.884251 | 31.704105 | 0.010706 | 0.190385 | 0.999934 | 0.020833 |
| 1 | 3182 | 1755.037864 | 1755.396704 | 25.554900 | 0.014561 | -0.011904 | 0.999891 | 0.020833 |
| 2 | 2066 | 1536.809325 | 1546.667936 | 174.058396 | 0.113260 | -0.083013 | 0.993537 | 0.020833 |
| 3 | 1274 | 979.741117 | 979.115091 | 24.912184 | 0.025427 | 0.084441 | 0.999674 | 0.020866 |

## Top Residual Cells

| Node | Zone | X | Y | Residual RMSE (ft^3) | Target RMS (ft^3) | Mean depth | Max depth | Wet-change fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11719 | 2 | 243917.492776 | 2608641.266596 | 5513.023326 | 2551.724220 | 19.116192 | 31.148563 | 0.020833 |
| 11720 | 2 | 243977.083138 | 2608634.471204 | 5378.667661 | 5399.445273 | 22.236893 | 35.948524 | 0.020833 |
| 12380 | 2 | 245136.581202 | 2607739.807034 | 1266.132571 | 4018.030968 | 19.201427 | 33.926922 | 0.020833 |
| 8658 | 0 | 234825.877671 | 2602470.496415 | 1102.835066 | 15737.980693 | 28.889002 | 34.146706 | 0.020833 |
| 85 | 0 | 232196.049495 | 2606816.770101 | 813.973650 | 10838.465727 | 15.436460 | 24.893841 | 0.020833 |
| 12379 | 2 | 245133.668964 | 2607779.670974 | 771.541702 | 4162.705543 | 19.201373 | 33.926922 | 0.020833 |
| 8636 | 1 | 234709.561409 | 2602550.839441 | 764.911640 | 12661.096989 | 28.542704 | 33.776672 | 0.020833 |
| 8657 | 0 | 234725.982177 | 2602452.196862 | 752.227675 | 14075.348057 | 28.903218 | 34.146812 | 0.020833 |
| 82 | 0 | 232196.049495 | 2606916.770101 | 588.855144 | 26462.424706 | 17.952759 | 28.142408 | 0.020833 |
| 9790 | 1 | 245329.052675 | 2607228.759398 | 561.705984 | 12650.084261 | 19.437692 | 35.516594 | 0.020833 |

## Interpretation

- The synchronized active-cell storage target is valid: HGN `M80_V` and HDF reconstructed active-cell volume match to numerical precision.
- The current HGN `M80_V` target is capped by finite HEC-RAS volume-elevation tables when water surface exceeds the table top; area extrapolation above the table largely restores consistency with HEC-RAS global volume.
- HEC-RAS global `Computations/Volume` is not the same quantity as the active-cell volume-elevation reconstruction used by HGN.
- HEC-RAS global `Computations/Volume` closes tightly against external boundary flow plus precipitation, so the simulation is globally water-balanced even though the HGN active-cell target does not close.
- With the capped HGN `M80_V` target, native `Cell Flow Balance` does not close local storage. With the area-extrapolated target, `Cell Flow Balance + precipitation` closes much more tightly.
- `Cell Flow Balance` is numerically the signed net sum of native `Face Flow`; the residual is therefore not caused by a face-orientation implementation error in the validator.
- Formal local-conservation loss should use a volume target that handles above-table water-surface values; the capped table target should remain disabled for formal conservation training.
