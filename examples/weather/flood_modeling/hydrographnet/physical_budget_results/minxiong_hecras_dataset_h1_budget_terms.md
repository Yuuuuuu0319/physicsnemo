# HEC-RAS Budget Term Diagnosis

This report diagnoses why synchronized HEC-RAS/HGN storage still does not close with native budget terms.

## Storage Checks

- Active-cell HGN `M80_V` vs reconstructed HDF cell volume RMSE: `2.429147e+04 ft^3`
- Active-cell HGN `M80_V` vs reconstructed HDF cell volume max abs: `3.347594e+05 ft^3`
- Sum of all reconstructed HDF cell volumes vs HEC-RAS global `Computations/Volume` RMSE: `134247344.191687 ft^3`
- Area-extrapolated active-cell volume vs HEC-RAS global `Computations/Volume` delta RMSE: `57.860559 ft^3`
- Volume table cap: `5191` active cells exceed their table top at least once; final over-table fraction `0.406056`, max exceedance `35.342698 ft`
- HEC-RAS sampled `Volume Error` RMS: `0.000000 ft^3`
- `Cell Flow Balance` vs signed native `Face Flow` net sum RMSE: `3.666586e-06 cfs`, correlation `1.000000`
- `Cell Flow Balance` vs inward-only face flow RMSE: `129.479690 cfs`, correlation `0.112915`

## Best Budget Variant

- `area_extrapolated_target::cell_balance_trapz_plus_precipitation`: relative RMSE `0.044605`, cell RMSE `103.279931 ft^3`, domain RMSE `4788.236933 ft^3`, correlation `0.998928`

## Variants

| Variant | Relative RMSE | Cell RMSE (ft^3) | Domain RMSE (ft^3) | Bias (ft^3) | Correlation |
|---|---:|---:|---:|---:|---:|
| area_extrapolated_target::cell_balance_trapz_plus_precipitation | 0.044605 | 103.279931 | 4788.236933 | 0.092426 | 0.998928 |
| cell_balance_trapz_plus_precipitation | 0.044605 | 103.279931 | 4788.234223 | 0.092426 | 0.998928 |
| cell_balance_rect_right | 0.088896 | 205.830147 | 2084366.055055 | 140.454339 | 0.997730 |
| area_extrapolated_target::cell_balance_trapz | 0.089380 | 206.951940 | 2084577.444724 | 140.546550 | 0.997687 |
| cell_balance_trapz | 0.089380 | 206.951940 | 2084577.444366 | 140.546550 | 0.997687 |
| cell_balance_rect_left | 0.090836 | 210.322345 | 2084799.806249 | 140.638762 | 0.997547 |
| area_extrapolated_target::cell_balance_trapz_minus_precipitation | 0.161184 | 373.207886 | 4168939.830697 | 281.000675 | 0.993934 |
| cell_balance_trapz_minus_precipitation | 0.161184 | 373.207886 | 4168939.830342 | 281.000675 | 0.993934 |
| precipitation_volume | 0.977566 | 2263.471438 | 6278773.948816 | 487.019890 | 0.188695 |
| zero_budget | 1.000000 | 2315.416520 | 8222534.585717 | 627.474015 | nan |
| negative_cell_balance_trapz | 1.976574 | 4576.592996 | 14481028.165221 | 1114.401479 | -0.997687 |

## Domain Diagnostics

| Target | Budget | Relative RMSE | RMSE (ft^3) | Bias (ft^3) | Correlation |
|---|---|---:|---:|---:|---:|
| active_hgn_delta_v | boundary_total_plus_precipitation | 0.000164 | 1349.142522 | 567.018503 | 1.000000 |
| area_extrapolated_active_delta_v | boundary_total_plus_precipitation | 0.000164 | 1349.151661 | 567.018592 | 1.000000 |
| hecras_global_computations_volume_delta | boundary_total_plus_precipitation | 0.000167 | 1369.327821 | 622.199443 | 1.000000 |
| active_hgn_delta_v | sum_cell_balance_plus_precipitation | 0.000582 | 4788.234223 | 1175.190894 | 0.999998 |
| area_extrapolated_active_delta_v | sum_cell_balance_plus_precipitation | 0.000582 | 4788.236933 | 1175.190982 | 0.999998 |
| hecras_global_computations_volume_delta | sum_cell_balance_plus_precipitation | 0.000583 | 4797.307320 | 1230.371833 | 0.999998 |
| active_hgn_delta_v | boundary_total | 0.253515 | 2084534.358818 | 1786441.215543 | 0.939550 |
| area_extrapolated_active_delta_v | boundary_total | 0.253515 | 2084534.359176 | 1786441.215631 | 0.939550 |
| hecras_global_computations_volume_delta | boundary_total | 0.253520 | 2084588.321383 | 1786496.396482 | 0.939550 |
| active_hgn_delta_v | sum_cell_balance | 0.253520 | 2084577.444366 | 1787049.387933 | 0.939435 |
| area_extrapolated_active_delta_v | sum_cell_balance | 0.253520 | 2084577.444724 | 1787049.388022 | 0.939435 |
| hecras_global_computations_volume_delta | sum_cell_balance | 0.253525 | 2084631.413514 | 1787104.568873 | 0.939435 |
| active_hgn_delta_v | boundary_total_minus_precipitation | 0.507009 | 4168899.274279 | 3572315.412582 | -0.054721 |
| area_extrapolated_active_delta_v | boundary_total_minus_precipitation | 0.507009 | 4168899.274634 | 3572315.412670 | -0.054721 |
| hecras_global_computations_volume_delta | boundary_total_minus_precipitation | 0.507012 | 4168953.232805 | 3572370.593521 | -0.054721 |
| active_hgn_delta_v | sum_cell_balance_minus_precipitation | 0.507014 | 4168939.830342 | 3572923.584972 | -0.052871 |
| area_extrapolated_active_delta_v | sum_cell_balance_minus_precipitation | 0.507014 | 4168939.830697 | 3572923.585061 | -0.052871 |
| hecras_global_computations_volume_delta | sum_cell_balance_minus_precipitation | 0.507017 | 4168993.792193 | 3572978.765912 | -0.052871 |
| area_extrapolated_active_delta_v | precipitation_only | 0.763606 | 6278773.948384 | 6192457.902383 | 0.943715 |
| active_hgn_delta_v | precipitation_only | 0.763606 | 6278773.948816 | 6192457.902295 | 0.943715 |
| hecras_global_computations_volume_delta | precipitation_only | 0.763607 | 6278830.477218 | 6192513.083234 | 0.943715 |
| all_reconstructed_delta_v | precipitation_only | 0.825693 | 1923308.276520 | 365481.765459 | -0.840223 |
| all_reconstructed_delta_v | sum_cell_balance_minus_precipitation | 1.010271 | 2353250.470863 | -2254052.551863 | 0.666490 |
| all_reconstructed_delta_v | boundary_total_minus_precipitation | 1.010346 | 2353424.716580 | -2254660.724254 | 0.667863 |
| all_reconstructed_delta_v | sum_cell_balance | 1.869522 | 4354724.354297 | -4039926.748903 | -0.411717 |
| all_reconstructed_delta_v | boundary_total | 1.869570 | 4354836.718105 | -4040534.921293 | -0.411848 |
| all_reconstructed_delta_v | sum_cell_balance_plus_precipitation | 2.751553 | 6409260.223947 | -5825800.945942 | -0.668575 |
| all_reconstructed_delta_v | boundary_total_plus_precipitation | 2.751591 | 6409348.935671 | -5826409.118332 | -0.668766 |

### Boundary Components

| Component | RMS volume (ft^3) | Mean volume (ft^3) |
|---|---:|---:|
| DownstreamBC1 - Flow per Face | 955050.317803 | -528692.622604 |
| DownstreamBC2 - Flow per Face | 1051809.820192 | -700130.760958 |
| boundary_total | 6278311.017670 | 6191890.883792 |
| upstreamBC1 - Flow per Face | 5450361.177982 | 5335714.269948 |
| upstreamBC2 - Flow per Face | 2370441.736127 | 2084999.997406 |

## Zone Residuals

| Zone | Cells | Target RMS (ft^3) | Budget RMS (ft^3) | Residual RMSE (ft^3) | Relative RMSE | Bias (ft^3) | Correlation | Wet-change fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0 | 6193 | 2903.759611 | 2903.789077 | 34.711101 | 0.011954 | 0.194504 | 0.999917 | 0.020833 |
| 1 | 3182 | 1752.463377 | 1755.014168 | 110.175804 | 0.062869 | 0.076725 | 0.997981 | 0.020833 |
| 2 | 2066 | 1546.156645 | 1559.109241 | 197.533780 | 0.127758 | -0.297676 | 0.991805 | 0.020833 |
| 3 | 1274 | 985.570567 | 989.037447 | 83.703997 | 0.084929 | 0.268043 | 0.996384 | 0.020833 |

## Top Residual Cells

| Node | Zone | X | Y | Residual RMSE (ft^3) | Target RMS (ft^3) | Mean depth | Max depth | Wet-change fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 11719 | 2 | 243917.492776 | 2608641.266596 | 5699.197405 | 2602.309083 | 19.053335 | 31.134254 | 0.020833 |
| 11720 | 2 | 243977.083138 | 2608634.471204 | 5558.918805 | 5508.528379 | 22.150901 | 35.934189 | 0.020833 |
| 9790 | 1 | 245329.052675 | 2607228.759398 | 4578.195495 | 12591.871642 | 19.243429 | 35.508495 | 0.020833 |
| 9686 | 1 | 245386.419441 | 2607310.668214 | 4083.682892 | 14269.531135 | 19.242133 | 35.508484 | 0.020833 |
| 9791 | 2 | 245403.680721 | 2607171.074857 | 2608.115859 | 8678.348815 | 19.270452 | 35.508423 | 0.020833 |
| 12143 | 3 | 245438.684859 | 2607372.307080 | 2268.217581 | 5741.492362 | 17.634897 | 33.508156 | 0.020833 |
| 12384 | 2 | 245318.127267 | 2607334.301803 | 1931.859684 | 6631.032595 | 19.022910 | 35.271027 | 0.020833 |
| 12544 | 2 | 245425.230703 | 2607238.211662 | 1754.206236 | 7832.210853 | 18.872464 | 35.091629 | 0.020833 |
| 12144 | 3 | 245463.198437 | 2607376.714738 | 1597.058420 | 5330.378709 | 17.854434 | 33.817085 | 0.020833 |
| 11397 | 2 | 245369.364644 | 2607133.095454 | 1280.297092 | 1162.485768 | 13.185602 | 23.997959 | 0.020833 |

## Interpretation

- The synchronized active-cell storage target is valid: HGN `M80_V` and HDF reconstructed active-cell volume match to numerical precision.
- The current HGN `M80_V` target is capped by finite HEC-RAS volume-elevation tables when water surface exceeds the table top; area extrapolation above the table largely restores consistency with HEC-RAS global volume.
- HEC-RAS global `Computations/Volume` is not the same quantity as the active-cell volume-elevation reconstruction used by HGN.
- HEC-RAS global `Computations/Volume` closes tightly against external boundary flow plus precipitation, so the simulation is globally water-balanced even though the HGN active-cell target does not close.
- With the capped HGN `M80_V` target, native `Cell Flow Balance` does not close local storage. With the area-extrapolated target, `Cell Flow Balance + precipitation` closes much more tightly.
- `Cell Flow Balance` is numerically the signed net sum of native `Face Flow`; the residual is therefore not caused by a face-orientation implementation error in the validator.
- Formal local-conservation loss should use a volume target that handles above-table water-surface values; the capped table target should remain disabled for formal conservation training.
