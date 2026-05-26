# HEC-RAS / HydroGraphNet Physical Budget Validation

Formal local conservation status: **BLOCKED**

## Inputs

- HGN data: `/mnt/8tb_hdd2/joyce/test` (`test.txt`)
- Event-specific HDF glob: `/mnt/8tb_hdd2/joyce/hecras-dataset/origin/Linux_RAS_v66/Minxiong_hgn_test_h*/outputs_seed/planH*/Minxiong.p01.hdf`
- Evaluated event transitions: `835`
- HGN budget interval: `1799.999971 s`
- HEC-RAS computation interval: `30.000000 s` (`60.0` solver steps per HGN budget interval)
- HDF stored face-output interval: `1800.000000 s`

## Formal Gates

| Gate | Result | Evidence |
|---|---|---|
| HDF reconstructed cell storage matches HGN `M80_V` target | FAIL | RMSE `15777.974689 ft^3`, max abs `435862.596611 ft^3` |
| HDF stores face output at solver-computation frequency | FAIL | HDF `1800.000000 s`, computation `30.000000 s` |
| Native HEC-RAS internal `Face Flow` is available | FAIL | Required for formal internal-face flux evidence |
| Evaluation uses no fitted scale factor | PASS | Budget is evaluated directly in native volume units |
| Uncalibrated residual is acceptable for a training target | FAIL | Best HGN-target relative RMSE `12789.891145` |
| HDF self-storage can be closed by reconstructed face transport | FAIL | Best HDF-self relative RMSE `1296.609494` |

## Best Uncalibrated Variants

| Target | Flux source | Face stage rule | Sign mode | Boundary mode | Source mode | Residual RMSE (ft^3) | Relative RMSE | Correlation |
|---|---|---|---|---|---|---:|---:|---:|
| HGN `M80_V` | reconstructed_velocity_area | minimum | negated | reconstructed_all_touching_faces | none | 845431.069823 | 12789.891145 | 0.002973 |
| HDF reconstructed storage | reconstructed_velocity_area | minimum | negated | reconstructed_all_touching_faces | none | 845426.598808 | 1296.609494 | 0.007477 |

## Interpretation

- The result is uncalibrated: no regression scale factor is fitted to target volume changes.
- Current HDF forcing matching is insufficient because its reconstructed dynamic storage does not reproduce the HGN target storage.
- HEC-RAS documentation identifies optional `Face flow` and `Cell flow balance` HDF variables; they are not present in the evaluated HDF and should be enabled for formal budget evidence.
- Temporal sampling check: the HDF stores face samples every `1800.000000 s`, while the solver computation interval is `30.000000 s`. The stored output is still too sparse to integrate every solver-step face transfer.
- Do not activate a formal HEC-RAS local-conservation training loss until a synchronized dataset/HDF pair and an interval-integrated or sufficiently sampled internal-face flux budget pass these gates.
