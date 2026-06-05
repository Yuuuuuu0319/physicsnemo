# Minxiong HEC-RAS Matched H1-H15 Cell-Balance Experiment

Date: 2026-06-04

## Data

- HEC-RAS event HDFs: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong/outputs/planH*/Minxiong.p01.hdf`
- HGN dataset: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong/outputs/HGN_dataset`
- Train split: `train_h1h12.txt`
- Held-out split: `test_h13h15.txt`
- HEC-RAS HDF output interval: 30 seconds
- HGN output interval: 30 minutes
- Active cells: 12715

## Data Generation Notes

- H1-H15 HEC-RAS batch simulation completed successfully.
- HDFs contain native `Face Flow`, `Cell Flow Balance`, and `Cell Cumulative Precipitation Depth`.
- `batch_HGN_dataset.py` was corrected to sort `planH*` folders numerically, avoiding `planH10` being assigned to H2.
- `LandCover.hdf` was not available, so `M80_IP.txt` uses the existing fallback value `100.0`.
- Fidelity zones were regenerated from `M80_CA.txt` with nominal resolutions 100, 80, 60, and 40 ft.

## Fidelity Zones

| Zone label | Study meaning | Cells | Fraction |
|---:|---|---:|---:|
| 0 | Zone 1, low-fidelity 100 ft | 6193 | 0.4871 |
| 1 | Zone 2, 80 ft | 3182 | 0.2503 |
| 2 | Zone 3, 60 ft | 2066 | 0.1625 |
| 3 | Zone 4, high-fidelity 40 ft | 1274 | 0.1002 |

## Physical Budget Evidence

Native HEC-RAS `Cell Flow Balance + precipitation` closes the synchronized local storage target without fitted scale calibration:

| Event | Best local budget variant | Relative RMSE | Cell RMSE (ft^3) | Correlation |
|---|---|---:|---:|---:|
| H1 | `cell_balance_trapz_plus_precipitation` | 0.044605 | 103.279931 | 0.998928 |
| H15 | `cell_balance_trapz_plus_precipitation` | 0.031953 | 75.085891 | 0.999450 |

Additional checks:

- H1 `Cell Flow Balance` vs signed native `Face Flow` net sum RMSE: `3.666586e-06 cfs`, correlation `1.000000`.
- H15 `Cell Flow Balance` vs signed native `Face Flow` net sum RMSE: `3.617754e-06 cfs`, correlation `1.000000`.
- This supports using native `Cell Flow Balance` as the formal local-conservation target.

## Training

Both runs used H1-H12, 20 epochs, 240 training samples, batch size 1, and zone metrics enabled.

| Run | Checkpoint | Cell-balance loss | Final total loss | Final MSE | Final Zone 4 volume RMSE | Final Zone 4 WD RMSE |
|---|---|---:|---:|---:|---:|---:|
| Baseline | `checkpoints_minxiong_hecras_h1h12_noface_e20` | disabled | 1.2370e-03 | 1.5184e-04 | 8.7054e-03 | 1.6158e-02 |
| Cell-balance | `checkpoints_minxiong_hecras_h1h12_cellbalance_w1e9_e20` | enabled, weight 1e-9 | 3.6221e-03 | 1.6204e-04 | 8.3526e-03 | 1.9143e-02 |

The raw training `hecras_cell_balance_loss` decreased from `1.9338e+07` at epoch 0 to a low of about `3.5765e+05` around epoch 17.

## Held-Out H13-H15 Evaluation

One-step zone metrics:

| Run | MSE | Zone 4 volume RMSE | Zone 4 WD RMSE |
|---|---:|---:|---:|
| Baseline | 1.861854e-04 | 9.307462e-03 | 1.590883e-02 |
| Cell-balance | 1.092053e-04 | 6.657264e-03 | 1.698810e-02 |

Formal cell-balance residual metrics:

| Run | MSE | Cell-balance RMSE (ft^3) | Relative cell-balance RMSE | Zone 4 cell-balance RMSE (ft^3) |
|---|---:|---:|---:|---:|
| Baseline | 1.735292e-04 | 965.380239 | 1.433235 | 681.205240 |
| Cell-balance | 1.102220e-04 | 578.911615 | 0.859471 | 542.316826 |

## Interpretation

The matched H1-H15 dataset now provides formal local-conservation evidence on event-specific HDFs. The cell-balance run improves held-out one-step MSE, high-fidelity Zone 4 volume RMSE, global cell-balance residual, and Zone 4 cell-balance residual. The held-out Zone 4 WD RMSE is slightly worse, so the next tuning target is balancing water-depth accuracy against conservation strength rather than assuming a single weight is optimal.
