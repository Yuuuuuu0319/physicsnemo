# HEC-RAS Budget Term Diagnosis

This report diagnoses why synchronized HEC-RAS/HGN storage still does not close with native budget terms.

## Storage Checks

- Active-cell HGN `M80_V` vs reconstructed HDF cell volume RMSE: `2.832007e-10 ft^3`
- Active-cell HGN `M80_V` vs reconstructed HDF cell volume max abs: `5.020411e-10 ft^3`
- Sum of all reconstructed HDF cell volumes vs HEC-RAS global `Computations/Volume` RMSE: `71638907.760645 ft^3`
- Area-extrapolated active-cell volume vs HEC-RAS global `Computations/Volume` delta RMSE: `39.235807 ft^3`
- Volume table cap: `4079` active cells exceed their table top at least once; final over-table fraction `0.320409`, max exceedance `35.306687 ft`
- HEC-RAS sampled `Volume Error` RMS: `0.000000 ft^3`
- `Cell Flow Balance` vs signed native `Face Flow` net sum RMSE: `2.789812e-06 cfs`, correlation `1.000000`
- `Cell Flow Balance` vs inward-only face flow RMSE: `101.170189 cfs`, correlation `0.028261`

## Best Budget Variant

- `area_extrapolated_target::cell_balance_trapz_plus_precipitation`: relative RMSE `0.061336`, cell RMSE `119.169194 ft^3`, domain RMSE `2173.014667 ft^3`, correlation `0.998037`

## Variants

| Variant | Relative RMSE | Cell RMSE (ft^3) | Domain RMSE (ft^3) | Bias (ft^3) | Correlation |
|---|---:|---:|---:|---:|---:|
| area_extrapolated_target::cell_balance_trapz_plus_precipitation | 0.061336 | 119.169194 | 2173.014667 | 0.090706 | 0.998037 |
| area_extrapolated_target::cell_balance_trapz | 0.072809 | 141.460404 | 885695.937868 | 59.726161 | 0.997718 |
| area_extrapolated_target::cell_balance_trapz_minus_precipitation | 0.099554 | 193.423036 | 1770741.503907 | 119.361616 | 0.996787 |
| cell_balance_trapz_minus_precipitation | 0.824699 | 1166.760020 | 2187272.689208 | -143.798527 | 0.791457 |
| cell_balance_rect_right | 0.835521 | 1182.070275 | 3066086.003943 | -203.523689 | 0.790526 |
| cell_balance_trapz | 0.837387 | 1184.710374 | 3065463.776181 | -203.433982 | 0.789302 |
| cell_balance_rect_left | 0.839441 | 1187.615636 | 3064842.950342 | -203.344276 | 0.788007 |
| cell_balance_trapz_plus_precipitation | 0.853286 | 1207.203410 | 3946718.905135 | -263.069437 | 0.786683 |
| precipitation_volume | 0.996384 | 1409.654435 | 1310627.821787 | 86.448881 | 0.021188 |
| zero_budget | 1.000000 | 1414.770405 | 1910959.605190 | 146.084335 | nan |
| negative_cell_balance_trapz | 2.238356 | 3166.759497 | 6504761.363024 | 495.602653 | -0.789302 |

## Domain Diagnostics

| Target | Budget | Relative RMSE | RMSE (ft^3) | Bias (ft^3) | Correlation |
|---|---|---:|---:|---:|---:|
| area_extrapolated_active_delta_v | boundary_total_plus_precipitation | 0.000243 | 1355.167342 | 934.588765 | 1.000000 |
| hecras_global_computations_volume_delta | boundary_total_plus_precipitation | 0.000247 | 1377.068508 | 970.780105 | 1.000000 |
| area_extrapolated_active_delta_v | sum_cell_balance_plus_precipitation | 0.000390 | 2173.014667 | 1153.332824 | 1.000000 |
| hecras_global_computations_volume_delta | sum_cell_balance_plus_precipitation | 0.000392 | 2187.707722 | 1189.524165 | 1.000000 |
| area_extrapolated_active_delta_v | boundary_total | 0.158795 | 885680.645109 | 759199.397558 | 0.998026 |
| area_extrapolated_active_delta_v | sum_cell_balance | 0.158798 | 885695.937868 | 759418.141618 | 0.998014 |
| hecras_global_computations_volume_delta | boundary_total | 0.158801 | 885718.837755 | 759235.588899 | 0.998026 |
| hecras_global_computations_volume_delta | sum_cell_balance | 0.158804 | 885734.132243 | 759454.332958 | 0.998014 |
| area_extrapolated_active_delta_v | boundary_total_minus_precipitation | 0.317476 | 1770727.020380 | 1517464.206352 | 0.984850 |
| area_extrapolated_active_delta_v | sum_cell_balance_minus_precipitation | 0.317479 | 1770741.503907 | 1517682.950411 | 0.984812 |
| hecras_global_computations_volume_delta | boundary_total_minus_precipitation | 0.317481 | 1770765.210222 | 1517500.397692 | 0.984850 |
| hecras_global_computations_volume_delta | sum_cell_balance_minus_precipitation | 0.317484 | 1770779.694631 | 1517719.141752 | 0.984812 |
| all_reconstructed_delta_v | precipitation_only | 0.685848 | 1310627.821787 | 1099197.516743 | -0.242914 |
| active_hgn_delta_v | precipitation_only | 0.685848 | 1310627.821787 | 1099197.516743 | -0.242914 |
| area_extrapolated_active_delta_v | precipitation_only | 0.844967 | 4712809.761472 | 4445278.742298 | 0.976526 |
| hecras_global_computations_volume_delta | precipitation_only | 0.844968 | 4712848.606144 | 4445314.933638 | 0.976526 |
| active_hgn_delta_v | sum_cell_balance_minus_precipitation | 1.144594 | 2187272.689208 | -1828398.275144 | 0.046532 |
| all_reconstructed_delta_v | sum_cell_balance_minus_precipitation | 1.144594 | 2187272.689208 | -1828398.275144 | 0.046532 |
| active_hgn_delta_v | boundary_total_minus_precipitation | 1.144604 | 2187291.144802 | -1828617.019203 | 0.046223 |
| all_reconstructed_delta_v | boundary_total_minus_precipitation | 1.144604 | 2187291.144802 | -1828617.019203 | 0.046223 |
| active_hgn_delta_v | sum_cell_balance | 1.604149 | 3065463.776181 | -2586663.083938 | -0.037094 |
| all_reconstructed_delta_v | sum_cell_balance | 1.604149 | 3065463.776181 | -2586663.083938 | -0.037094 |
| active_hgn_delta_v | boundary_total | 1.604158 | 3065480.892450 | -2586881.827997 | -0.037341 |
| all_reconstructed_delta_v | boundary_total | 1.604158 | 3065480.892450 | -2586881.827997 | -0.037341 |
| active_hgn_delta_v | sum_cell_balance_plus_precipitation | 2.065307 | 3946718.905135 | -3344927.892731 | -0.084119 |
| all_reconstructed_delta_v | sum_cell_balance_plus_precipitation | 2.065307 | 3946718.905135 | -3344927.892731 | -0.084119 |
| active_hgn_delta_v | boundary_total_plus_precipitation | 2.065316 | 3946735.265857 | -3345146.636790 | -0.084322 |
| all_reconstructed_delta_v | boundary_total_plus_precipitation | 2.065316 | 3946735.265857 | -3345146.636790 | -0.084322 |

### Boundary Components

| Component | RMS volume (ft^3) | Mean volume (ft^3) |
|---|---:|---:|
| DownstreamBC1 - Flow per Face | 128895.218105 | -30441.500430 |
| DownstreamBC2 - Flow per Face | 1022746.346203 | -672714.360975 |
| boundary_total | 4712039.013894 | 4444344.153533 |
| upstreamBC1 - Flow per Face | 3272143.487970 | 3062500.017532 |
| upstreamBC2 - Flow per Face | 2370441.736127 | 2084999.997406 |

## Zone Residuals

| Zone | Cells | Target RMS (ft^3) | Budget RMS (ft^3) | Residual RMSE (ft^3) | Relative RMSE | Bias (ft^3) | Correlation | Wet-change fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6193 | 2373.812765 | 2373.518867 | 44.055032 | 0.018559 | 0.198378 | 0.999813 | 0.020833 |
| 1 | 3182 | 1500.480504 | 1504.933771 | 119.228303 | 0.079460 | 0.087042 | 0.996800 | 0.020833 |
| 2 | 2066 | 1482.241584 | 1502.122345 | 233.681537 | 0.157654 | -0.340350 | 0.987679 | 0.020833 |
| 3 | 1274 | 1047.064824 | 1053.004880 | 90.776688 | 0.086696 | 0.275490 | 0.996260 | 0.020866 |

## Top Residual Cells

| Node | Zone | X | Y | Residual RMSE (ft^3) | Target RMS (ft^3) | Mean depth | Max depth | Wet-change fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11719 | 2 | 243917.492776 | 2608641.266596 | 6922.135019 | 2615.466979 | 18.901337 | 31.098446 | 0.020833 |
| 11720 | 2 | 243977.083138 | 2608634.471204 | 6740.115296 | 5602.097088 | 21.953368 | 35.898327 | 0.020833 |
| 9790 | 1 | 245329.052675 | 2607228.759398 | 4970.423884 | 12339.640979 | 18.803434 | 35.462051 | 0.020833 |
| 9686 | 1 | 245386.419441 | 2607310.668214 | 4429.543935 | 13893.013793 | 18.802056 | 35.462044 | 0.020833 |
| 9791 | 2 | 245403.680721 | 2607171.074857 | 2791.287174 | 8437.675923 | 18.833507 | 35.461983 | 0.020833 |
| 12143 | 3 | 245438.684859 | 2607372.307080 | 2466.901573 | 5564.758670 | 17.388620 | 33.461700 | 0.020833 |
| 10380 | 0 | 233225.301202 | 2602460.938521 | 2212.380776 | 15710.639897 | 22.752320 | 26.912487 | 0.020833 |
| 12384 | 2 | 245318.127267 | 2607334.301803 | 2095.326863 | 6476.002721 | 18.592753 | 35.224586 | 0.020833 |
| 12544 | 2 | 245425.230703 | 2607238.211662 | 1906.379301 | 7739.939412 | 18.453720 | 35.045185 | 0.020833 |
| 8612 | 0 | 233207.061265 | 2602414.360165 | 1840.759231 | 16865.446311 | 22.821516 | 26.983126 | 0.020833 |

## Interpretation

- The synchronized active-cell storage target is valid: HGN `M80_V` and HDF reconstructed active-cell volume match to numerical precision.
- The current HGN `M80_V` target is capped by finite HEC-RAS volume-elevation tables when water surface exceeds the table top; area extrapolation above the table largely restores consistency with HEC-RAS global volume.
- HEC-RAS global `Computations/Volume` is not the same quantity as the active-cell volume-elevation reconstruction used by HGN.
- HEC-RAS global `Computations/Volume` closes tightly against external boundary flow plus precipitation, so the simulation is globally water-balanced even though the HGN active-cell target does not close.
- With the capped HGN `M80_V` target, native `Cell Flow Balance` does not close local storage. With the area-extrapolated target, `Cell Flow Balance + precipitation` closes much more tightly.
- `Cell Flow Balance` is numerically the signed net sum of native `Face Flow`; the residual is therefore not caused by a face-orientation implementation error in the validator.
- Formal local-conservation loss should use a volume target that handles above-table water-surface values; the capped table target should remain disabled for formal conservation training.
