# Minxiong LandCover IP + HEC-RAS Cell-Balance Weight Sweep

Date: 2026-06-05

## Purpose

This run continues the formal local-conservation path after adding:

`/mnt/8tb_hdd2/joyce/hecras-dataset/winres/LandCover.hdf`

The goal was to remove the old fallback impervious percentage (`IP=100` everywhere), re-export the HGN dataset with spatially varying IP, then test whether HEC-RAS native `Cell Flow Balance` supervision improves held-out H13-H15 behavior.

## Dataset Update

Re-exported the Minxiong HGN dataset from:

`/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong`

Output:

`/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong/outputs/HGN_dataset`

The exporter successfully loaded LandCover lookup from:

`/mnt/8tb_hdd2/joyce/hecras-dataset/winres/LandCover.hdf`

LandCover `Variables` mapping:

| Manning n | Percent Impervious |
|---:|---:|
| 0.030 | 0 |
| 0.045 | 10 |
| 0.100 | 2 |
| 0.120 | 85 |

Exported `M80_IP.txt` now has spatial variability:

| IP | Cell Count |
|---:|---:|
| 0 | 369 |
| 2 | 2382 |
| 10 | 4915 |
| 85 | 5049 |

This replaces the previous fallback state where all 12715 cells had `IP=100`.

Fidelity zones were regenerated after export:

| Zone | Cells | Fraction |
|---:|---:|---:|
| 0 | 6193 | 0.4871 |
| 1 | 3182 | 0.2503 |
| 2 | 2066 | 0.1625 |
| 3 | 1274 | 0.1002 |

## Training Setup

Training IDs:

`train_h1h12.txt`

Held-out evaluation IDs:

`test_h13h15.txt`

Common training settings:

| Setting | Value |
|---|---|
| Epochs | 20 |
| Training samples | 240 |
| Batch size | 1 |
| Zone metrics | enabled |
| Zone loss weight | 0.0 |
| Cell-balance zone mode | all |
| HEC-RAS source | `outputs/planH*/Minxiong.p01.hdf` |

Checkpoint groups:

| Name | Checkpoint Directory |
|---|---|
| noface | `checkpoints_minxiong_landcover_h1h12_noface_e20` |
| cellbalance_w1e10 | `checkpoints_minxiong_landcover_h1h12_cellbalance_w1e10_e20` |
| cellbalance_w3e10 | `checkpoints_minxiong_landcover_h1h12_cellbalance_w3e10_e20` |
| cellbalance_w1e9 | `checkpoints_minxiong_landcover_h1h12_cellbalance_w1e9_e20` |
| cellbalance_w3e9 | `checkpoints_minxiong_landcover_h1h12_cellbalance_w3e9_e20` |

## Held-Out Zone Evaluation

Source CSV:

`examples/weather/flood_modeling/hydrographnet/results_minxiong_landcover_h13h15_zone_eval_e20.csv`

| Checkpoint | MSE | Zone Loss | Zone 3 Volume RMSE | Zone 3 WD RMSE |
|---|---:|---:|---:|---:|
| noface | 1.285661e-04 | 1.409804e-04 | 7.705450e-03 | 1.550552e-02 |
| cellbalance_w1e10 | 1.521938e-04 | 1.455697e-04 | 8.117892e-03 | 1.647418e-02 |
| cellbalance_w3e10 | 1.062054e-04 | 1.201730e-04 | 7.185424e-03 | 1.548878e-02 |
| cellbalance_w1e9 | 2.242838e-04 | 2.328601e-04 | 6.741908e-03 | 2.283940e-02 |
| cellbalance_w3e9 | 9.667044e-05 | 1.039937e-04 | 4.884281e-03 | 1.517889e-02 |

Best current held-out zone result: `cellbalance_w3e9`.

Compared with `noface`, `cellbalance_w3e9` improved:

| Metric | Improvement |
|---|---:|
| MSE | 24.8% lower |
| Zone loss | 26.2% lower |
| Zone 3 volume RMSE | 36.6% lower |
| Zone 3 WD RMSE | 2.1% lower |

## Held-Out Formal Cell-Balance Evaluation

Source CSV:

`examples/weather/flood_modeling/hydrographnet/results_minxiong_landcover_h13h15_cellbalance_eval_e20.csv`

Evaluation used 77 available H13-H15 samples with matched HEC-RAS native cell-balance terms.

| Checkpoint | MSE | Cell-Balance RMSE (ft^3) | Relative Cell-Balance RMSE | Zone 3 Cell-Balance RMSE (ft^3) |
|---|---:|---:|---:|---:|
| noface | 1.359704e-04 | 769.831 | 1.1429 | 600.861 |
| cellbalance_w1e10 | 1.267822e-04 | 749.304 | 1.1124 | 642.919 |
| cellbalance_w3e10 | 9.746143e-05 | 579.033 | 0.8597 | 590.070 |
| cellbalance_w1e9 | 1.931572e-04 | 582.833 | 0.8653 | 593.207 |
| cellbalance_w3e9 | 8.721895e-05 | 443.137 | 0.6579 | 431.563 |

Best current formal local-conservation result: `cellbalance_w3e9`.

Compared with `noface`, `cellbalance_w3e9` improved:

| Metric | Improvement |
|---|---:|
| MSE | 35.9% lower |
| Cell-balance RMSE | 42.4% lower |
| Relative cell-balance RMSE | 42.4% lower |
| Zone 3 cell-balance RMSE | 28.2% lower |

## Interpretation

After the LandCover/IP correction, the stronger `3e-9` HEC-RAS cell-balance weight is no longer merely over-constraining the model. It improves the formal conservation residual and also improves held-out prediction metrics.

The current best candidate for the formal local-conservation branch is:

`cellbalance_w3e9`

This is stronger evidence than the previous pre-LandCover runs because:

1. The static IP input is now spatially meaningful instead of constant fallback `100`.
2. The loss uses HEC-RAS native `Cell Flow Balance` rather than inferred/scaled face-flow budget terms.
3. The held-out events H13-H15 show improvement in both prediction error and physical residual.

## Caveats

Current evaluation loads the latest checkpoint from each checkpoint directory. Some earlier runs showed epoch-level instability, especially around larger physical weights. A future improvement should add best-checkpoint selection or validation-based early stopping.

The reported results are still 20-epoch experiments. They are enough to show a working trend, but a final report-quality experiment should run longer and repeat the best weights with fixed seeds.
