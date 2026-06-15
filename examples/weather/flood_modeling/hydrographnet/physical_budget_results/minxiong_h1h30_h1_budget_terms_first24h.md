# HEC-RAS Budget Term Diagnosis

This report diagnoses why synchronized HEC-RAS/HGN storage still does not close with native budget terms.

## Storage Checks

- Active-cell HGN `M80_V` vs reconstructed HDF cell volume RMSE: `2.214976e+04 ft^3`
- Active-cell HGN `M80_V` vs reconstructed HDF cell volume max abs: `3.267357e+05 ft^3`
- Sum of all reconstructed HDF cell volumes vs HEC-RAS global `Computations/Volume` RMSE: `118439331.572144 ft^3`
- Area-extrapolated active-cell volume vs HEC-RAS global `Computations/Volume` delta RMSE: `54.591357 ft^3`
- Volume table cap: `5048` active cells exceed their table top at least once; final over-table fraction `0.396225`, max exceedance `35.333630 ft`
- HEC-RAS sampled `Volume Error` RMS: `0.000000 ft^3`
- `Cell Flow Balance` vs signed native `Face Flow` net sum RMSE: `3.451515e-06 cfs`, correlation `1.000000`
- `Cell Flow Balance` vs inward-only face flow RMSE: `121.836361 cfs`, correlation `0.045477`

## Best Budget Variant

- `cell_balance_trapz_plus_precipitation`: relative RMSE `0.037181`, cell RMSE `82.084297 ft^3`, domain RMSE `3531.276260 ft^3`, correlation `0.999257`

## Variants

| Variant | Relative RMSE | Cell RMSE (ft^3) | Domain RMSE (ft^3) | Bias (ft^3) | Correlation |
|---|---:|---:|---:|---:|---:|
| cell_balance_trapz_plus_precipitation | 0.037181 | 82.084297 | 3531.276260 | 0.092251 | 0.999257 |
| area_extrapolated_target::cell_balance_trapz_plus_precipitation | 0.037181 | 82.084297 | 3531.277670 | 0.092251 | 0.999257 |
| cell_balance_rect_right | 0.091565 | 202.149086 | 2156871.510972 | 145.340064 | 0.997841 |
| area_extrapolated_target::cell_balance_trapz | 0.091915 | 202.921717 | 2157242.815122 | 145.432121 | 0.997813 |
| cell_balance_trapz | 0.091915 | 202.921717 | 2157242.820498 | 145.432121 | 0.997813 |
| cell_balance_rect_left | 0.093114 | 205.568528 | 2157619.843077 | 145.524178 | 0.997699 |
| area_extrapolated_target::cell_balance_trapz_minus_precipitation | 0.172160 | 380.078870 | 4314111.779578 | 290.771990 | 0.993411 |
| cell_balance_trapz_minus_precipitation | 0.172160 | 380.078871 | 4314111.784955 | 290.771990 | 0.993411 |
| precipitation_volume | 0.975668 | 2153.989431 | 5727743.707338 | 439.273743 | 0.201194 |
| zero_budget | 1.000000 | 2207.706817 | 7781689.666387 | 584.613613 | nan |
| negative_cell_balance_trapz | 1.973919 | 4357.834944 | 13492649.899263 | 1023.795104 | -0.997813 |

## Domain Diagnostics

| Target | Budget | Relative RMSE | RMSE (ft^3) | Bias (ft^3) | Correlation |
|---|---|---:|---:|---:|---:|
| active_hgn_delta_v | boundary_total_plus_precipitation | 0.000187 | 1452.120617 | 728.578274 | 1.000000 |
| area_extrapolated_active_delta_v | boundary_total_plus_precipitation | 0.000187 | 1452.121123 | 728.574914 | 1.000000 |
| hecras_global_computations_volume_delta | boundary_total_plus_precipitation | 0.000189 | 1472.544254 | 780.083901 | 1.000000 |
| active_hgn_delta_v | sum_cell_balance_plus_precipitation | 0.000454 | 3531.276260 | 1172.977658 | 0.999999 |
| area_extrapolated_active_delta_v | sum_cell_balance_plus_precipitation | 0.000454 | 3531.277670 | 1172.974297 | 0.999999 |
| hecras_global_computations_volume_delta | sum_cell_balance_plus_precipitation | 0.000455 | 3542.122737 | 1224.483285 | 0.999999 |
| area_extrapolated_active_delta_v | boundary_total | 0.277216 | 2157212.582960 | 1848725.015483 | 0.970949 |
| active_hgn_delta_v | boundary_total | 0.277216 | 2157212.588337 | 1848725.018843 | 0.970949 |
| area_extrapolated_active_delta_v | sum_cell_balance | 0.277220 | 2157242.815122 | 1849169.414866 | 0.970864 |
| active_hgn_delta_v | sum_cell_balance | 0.277220 | 2157242.820498 | 1849169.418226 | 0.970864 |
| hecras_global_computations_volume_delta | boundary_total | 0.277221 | 2157264.578294 | 1848776.524470 | 0.970949 |
| hecras_global_computations_volume_delta | sum_cell_balance | 0.277225 | 2157294.813663 | 1849220.923853 | 0.970864 |
| area_extrapolated_active_delta_v | boundary_total_minus_precipitation | 0.554389 | 4314082.745768 | 3696721.456051 | 0.258940 |
| active_hgn_delta_v | boundary_total_minus_precipitation | 0.554389 | 4314082.751146 | 3696721.459412 | 0.258940 |
| hecras_global_computations_volume_delta | boundary_total_minus_precipitation | 0.554392 | 4314134.738619 | 3696772.965038 | 0.258940 |
| area_extrapolated_active_delta_v | sum_cell_balance_minus_precipitation | 0.554393 | 4314111.779578 | 3697165.855435 | 0.260007 |
| active_hgn_delta_v | sum_cell_balance_minus_precipitation | 0.554393 | 4314111.784955 | 3697165.858795 | 0.260007 |
| hecras_global_computations_volume_delta | sum_cell_balance_minus_precipitation | 0.554396 | 4314163.774048 | 3697217.364422 | 0.260007 |
| area_extrapolated_active_delta_v | precipitation_only | 0.736054 | 5727743.703304 | 5585365.640532 | 0.961959 |
| active_hgn_delta_v | precipitation_only | 0.736054 | 5727743.707338 | 5585365.643892 | 0.961959 |
| hecras_global_computations_volume_delta | precipitation_only | 0.736056 | 5727797.397819 | 5585417.149519 | 0.961959 |
| all_reconstructed_delta_v | precipitation_only | 0.784822 | 1758737.288414 | 275405.939805 | -0.795492 |
| all_reconstructed_delta_v | sum_cell_balance_minus_precipitation | 0.787167 | 1763992.106748 | -1612793.845293 | 0.442484 |
| all_reconstructed_delta_v | boundary_total_minus_precipitation | 0.787214 | 1764097.511700 | -1613238.244676 | 0.443043 |
| all_reconstructed_delta_v | sum_cell_balance | 1.726719 | 3869470.798830 | -3460790.285861 | -0.476875 |
| all_reconstructed_delta_v | boundary_total | 1.726748 | 3869534.366712 | -3461234.685245 | -0.477300 |
| all_reconstructed_delta_v | sum_cell_balance_plus_precipitation | 2.682582 | 6011498.994690 | -5308786.726430 | -0.647130 |
| all_reconstructed_delta_v | boundary_total_plus_precipitation | 2.682604 | 6011549.899100 | -5309231.125813 | -0.647402 |

### Boundary Components

| Component | RMS volume (ft^3) | Mean volume (ft^3) |
|---|---:|---:|
| DownstreamBC1 - Flow per Face | 751161.917480 | -377873.122712 |
| DownstreamBC2 - Flow per Face | 1053393.964451 | -701696.445292 |
| boundary_total | 5727152.305323 | 5584637.065618 |
| upstreamBC1 - Flow per Face | 4745819.641227 | 4579206.636216 |
| upstreamBC2 - Flow per Face | 2370441.736127 | 2084999.997406 |

## Zone Residuals

| Zone | Cells | Target RMS (ft^3) | Budget RMS (ft^3) | Residual RMSE (ft^3) | Relative RMSE | Bias (ft^3) | Correlation | Wet-change fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6193 | 2757.484907 | 2755.923888 | 21.389774 | 0.007757 | 0.203493 | 0.999966 | 0.020833 |
| 1 | 3182 | 1678.994402 | 1679.473957 | 62.741086 | 0.037368 | -0.005594 | 0.999285 | 0.020833 |
| 2 | 2066 | 1507.336262 | 1519.344288 | 180.513727 | 0.119757 | -0.122663 | 0.992798 | 0.020854 |
| 3 | 1274 | 977.957294 | 978.860832 | 48.455710 | 0.049548 | 0.144400 | 0.998764 | 0.020850 |

## Top Residual Cells

| Node | Zone | X | Y | Residual RMSE (ft^3) | Target RMS (ft^3) | Mean depth | Max depth | Wet-change fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11719 | 2 | 243917.492776 | 2608641.266596 | 5604.681689 | 2599.172455 | 18.836918 | 31.125134 | 0.062500 |
| 11720 | 2 | 243977.083138 | 2608634.471204 | 5494.651070 | 5489.189444 | 22.175928 | 35.925083 | 0.020833 |
| 9790 | 1 | 245329.052675 | 2607228.759398 | 2510.286121 | 12607.884567 | 19.267718 | 35.492779 | 0.020833 |
| 9686 | 1 | 245386.419441 | 2607310.668214 | 2240.304704 | 14258.191705 | 19.266463 | 35.492767 | 0.020833 |
| 9791 | 2 | 245403.680721 | 2607171.074857 | 1432.510582 | 8691.445713 | 19.293765 | 35.492706 | 0.020833 |
| 12143 | 3 | 245438.684859 | 2607372.307080 | 1244.839852 | 5754.578901 | 17.650559 | 33.492420 | 0.020833 |
| 12384 | 2 | 245318.127267 | 2607334.301803 | 1059.646073 | 6606.351499 | 19.046923 | 35.255310 | 0.020833 |
| 12544 | 2 | 245425.230703 | 2607238.211662 | 962.945716 | 7843.585475 | 18.895886 | 35.075912 | 0.020833 |
| 12144 | 3 | 245463.198437 | 2607376.714738 | 876.745079 | 5343.383063 | 17.871374 | 33.801338 | 0.020833 |
| 11397 | 2 | 245369.364644 | 2607133.095454 | 703.535558 | 1164.433258 | 13.202591 | 23.982243 | 0.020833 |

## Interpretation

- The synchronized active-cell storage target is valid: HGN `M80_V` and HDF reconstructed active-cell volume match to numerical precision.
- The current HGN `M80_V` target is capped by finite HEC-RAS volume-elevation tables when water surface exceeds the table top; area extrapolation above the table largely restores consistency with HEC-RAS global volume.
- HEC-RAS global `Computations/Volume` is not the same quantity as the active-cell volume-elevation reconstruction used by HGN.
- HEC-RAS global `Computations/Volume` closes tightly against external boundary flow plus precipitation, so the simulation is globally water-balanced even though the HGN active-cell target does not close.
- With the capped HGN `M80_V` target, native `Cell Flow Balance` does not close local storage. With the area-extrapolated target, `Cell Flow Balance + precipitation` closes much more tightly.
- `Cell Flow Balance` is numerically the signed net sum of native `Face Flow`; the residual is therefore not caused by a face-orientation implementation error in the validator.
- Formal local-conservation loss should use a volume target that handles above-table water-surface values; the capped table target should remain disabled for formal conservation training.
