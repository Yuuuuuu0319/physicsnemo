# HEC-RAS / HydroGraphNet Physical Budget Validation

Formal local conservation status: **REQUIRES RESIDUAL REVIEW**

## Inputs

- HGN data: `/mnt/8tb_hdd2/joyce/Minxiong/Minxiong_new/hgn_extract_workspace/outputs_native/HGN_dataset` (`train.txt`)
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
| Uncalibrated residual is acceptable for a training target | FAIL | Best HGN-target relative RMSE `0.837387` |
| HDF self-storage can be closed by reconstructed face transport | FAIL | Best HDF-self relative RMSE `0.837387` |

## Best Uncalibrated Variants

| Target | Flux source | Face stage rule | Sign mode | Boundary mode | Source mode | Residual RMSE (ft^3) | Relative RMSE | Correlation |
|---|---|---|---|---|---|---:|---:|---:|
| HGN `M80_V` | native_face_flow | native | cell_order | reconstructed_all_touching_faces | none | 1184.710373 | 0.837387 | 0.789302 |
| HDF reconstructed storage | native_face_flow | native | cell_order | reconstructed_all_touching_faces | none | 1184.710373 | 0.837387 | 0.789302 |

## Interpretation

- The result is uncalibrated: no regression scale factor is fitted to target volume changes.
- HDF reconstructed dynamic storage reproduces the HGN `M80_V` target, so the event/HGN target pair is synchronized at the storage level.
- Native HEC-RAS `Face Flow` and `Cell Flow Balance` are present in this HDF.
- Temporal sampling check: the HDF stores face samples every `30.000000 s`, while the solver computation interval is `30.000000 s`. The high-frequency preflight output therefore addresses the prior sampling blocker.
- Do not activate a formal HEC-RAS local-conservation training loss until a synchronized dataset/HDF pair and an interval-integrated or sufficiently sampled internal-face flux budget pass these gates.
