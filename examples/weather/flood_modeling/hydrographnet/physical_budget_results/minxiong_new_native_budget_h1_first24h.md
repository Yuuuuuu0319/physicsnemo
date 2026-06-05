# HEC-RAS / HydroGraphNet Physical Budget Validation

Formal local conservation status: **BLOCKED**

## Inputs

- HGN data: `/mnt/8tb_hdd2/joyce/train` (`train_h1.txt`)
- Event-specific HDF glob: `/mnt/8tb_hdd2/joyce/Minxiong/Minxiong_new/planH*/Minxiong.p03.hdf`
- Evaluated event transitions: `48`
- HGN budget interval: `1799.999971 s`
- HEC-RAS computation interval: `30.000000 s` (`60.0` solver steps per HGN budget interval)
- HDF stored face-output interval: `30.000000 s`

## Formal Gates

| Gate | Result | Evidence |
|---|---|---|
| HDF reconstructed cell storage matches HGN `M80_V` target | FAIL | RMSE `3863.649841 ft^3`, max abs `177406.298159 ft^3` |
| HDF stores face output at solver-computation frequency | PASS | HDF `30.000000 s`, computation `30.000000 s` |
| Native HEC-RAS internal `Face Flow` is available | PASS | Required for formal internal-face flux evidence |
| Evaluation uses no fitted scale factor | PASS | Budget is evaluated directly in native volume units |
| Uncalibrated residual is acceptable for a training target | FAIL | Best HGN-target relative RMSE `1.563688` |
| HDF self-storage can be closed by reconstructed face transport | FAIL | Best HDF-self relative RMSE `0.837387` |

## Best Uncalibrated Variants

| Target | Flux source | Face stage rule | Sign mode | Boundary mode | Source mode | Residual RMSE (ft^3) | Relative RMSE | Correlation |
|---|---|---|---|---|---|---:|---:|---:|
| HGN `M80_V` | native_face_flow | native | cell_order | reconstructed_all_touching_faces | none | 1797.863984 | 1.563688 | 0.394723 |
| HDF reconstructed storage | native_face_flow | native | cell_order | reconstructed_all_touching_faces | none | 1184.710373 | 0.837387 | 0.789302 |

## Interpretation

- The result is uncalibrated: no regression scale factor is fitted to target volume changes.
- Current HDF forcing matching is insufficient because its reconstructed dynamic storage does not reproduce the HGN target storage.
- Native HEC-RAS `Face Flow` and `Cell Flow Balance` are present in this HDF.
- Temporal sampling check: the HDF stores face samples every `30.000000 s`, while the solver computation interval is `30.000000 s`. The high-frequency preflight output therefore addresses the prior sampling blocker.
- Do not activate a formal HEC-RAS local-conservation training loss until a synchronized dataset/HDF pair and an interval-integrated or sufficiently sampled internal-face flux budget pass these gates.
