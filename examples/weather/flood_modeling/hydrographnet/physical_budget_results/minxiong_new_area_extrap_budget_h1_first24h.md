# HEC-RAS / HydroGraphNet Physical Budget Validation

Formal local conservation status: **CANDIDATE**

## Inputs

- HGN data: `/mnt/8tb_hdd2/joyce/Minxiong/Minxiong_new/hgn_extract_workspace/outputs_native_30min_area_extrap/HGN_dataset` (`train.txt`)
- Event-specific HDF glob: `/mnt/8tb_hdd2/joyce/Minxiong/Minxiong_new/hgn_extract_workspace/outputs_native/planH*/Minxiong.p01.hdf`
- Evaluated event transitions: `48`
- HGN budget interval: `1799.999971 s`
- HEC-RAS computation interval: `30.000000 s` (`60.0` solver steps per HGN budget interval)
- HDF stored face-output interval: `30.000000 s`

## Formal Gates

| Gate | Result | Evidence |
|---|---|---|
| HDF reconstructed cell storage matches HGN `M80_V` target | PASS | RMSE `0.000000 ft^3`, max abs `0.000000 ft^3` |
| HDF stores face output at solver-computation frequency | PASS | HDF `30.000000 s`, computation `30.000000 s` |
| Native HEC-RAS internal `Face Flow` is available | PASS | Required for formal internal-face flux evidence |
| Evaluation uses no fitted scale factor | PASS | Budget is evaluated directly in native volume units |
| Uncalibrated residual is acceptable for a training target | PASS | Best HGN-target relative RMSE `0.061336` (threshold `0.100000`) |
| HDF self-storage can be closed by reconstructed face transport | PASS | Best HDF-self relative RMSE `0.061336` (threshold `0.100000`) |

## Best Uncalibrated Variants

| Target | Flux source | Face stage rule | Sign mode | Boundary mode | Source mode | Residual RMSE (ft^3) | Relative RMSE | Correlation |
|---|---|---|---|---|---|---:|---:|---:|
| HGN `M80_V` | native_cell_flow_balance | native | cell_order | reconstructed_all_touching_faces | hdf_precipitation | 119.169194 | 0.061336 | 0.998037 |
| HDF reconstructed storage | native_cell_flow_balance | native | cell_order | reconstructed_all_touching_faces | hdf_precipitation | 119.169194 | 0.061336 | 0.998037 |

## Interpretation

- The result is uncalibrated: no regression scale factor is fitted to target volume changes.
- HDF reconstructed dynamic storage reproduces the HGN `M80_V` target, so the event/HGN target pair is synchronized at the storage level.
- Native HEC-RAS `Face Flow` and `Cell Flow Balance` are present in this HDF.
- Temporal sampling check: the HDF stores face samples every `30.000000 s`, while the solver computation interval is `30.000000 s`. The high-frequency preflight output therefore addresses the prior sampling blocker.
- This corrected area-extrapolated target is a candidate for formal HEC-RAS local-conservation training; keep it separate from the capped `M80_V` target for ablation and backward comparison.
