# Minxiong Selective High-Zone Cell-Balance Ablation

Date: 2026-06-05

## Purpose

This experiment directly tests the current research hypothesis:

HydroGraphNet's graph-level/global conservation can be retained, while native
HEC-RAS local conservation is applied only to the high-fidelity zone selected
by Automatic Fidelity Zoning.

The comparison separates:

- `noface`: HydroGraphNet-style baseline without HEC-RAS local cell-balance loss.
- `cellbalance_all_w3e9`: native cell-balance loss on all nodes.
- `cellbalance_high_w3e9`: native cell-balance loss only on Zone 3 / high-fidelity cells.

## Setup

| Setting | Value |
|---|---|
| Dataset | `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong/outputs/HGN_dataset` |
| Train split | `train_h1h12.txt` |
| Test split | `test_h13h15.txt` |
| Epochs | 20 |
| Training samples | 240 |
| Cell-balance weight | `3e-9` |
| Selective mode | `hecras_cell_balance_zone_mode=high` |
| HEC-RAS source | `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong/outputs/planH*/Minxiong.p01.hdf` |

New checkpoint:

`checkpoints_minxiong_landcover_h1h12_cellbalance_high_w3e9_e20`

Final high-zone training epoch:

| Metric | Value |
|---|---:|
| total loss | `1.8668e-03` |
| hecras cell-balance loss | `8.4767e+04` |
| mse loss | `1.1698e-04` |
| physics loss | `1.4955e-03` |
| Zone 3 volume RMSE | `3.3237e-03` |
| Zone 3 WD RMSE | `1.5738e-02` |

## Held-Out One-Step Metrics

Source:

`results_minxiong_landcover_h13h15_zone_eval_selective_high_e20.csv`

| Checkpoint | MSE | Zone Loss | Zone 3 Volume RMSE | Zone 3 WD RMSE |
|---|---:|---:|---:|---:|
| noface | `1.285661e-04` | `1.409804e-04` | `7.705449e-03` | `1.550552e-02` |
| cellbalance_all_w3e9 | `9.667042e-05` | `1.039937e-04` | `4.884281e-03` | `1.517889e-02` |
| cellbalance_high_w3e9 | `1.113887e-04` | `1.087874e-04` | `3.427553e-03` | `1.600297e-02` |

Interpretation:

- `cellbalance_all_w3e9` has the best overall one-step MSE.
- `cellbalance_high_w3e9` has the best Zone 3 volume RMSE.
- Relative to `noface`, `cellbalance_high_w3e9` lowers Zone 3 volume RMSE by about 55.5%.
- Relative to `cellbalance_all_w3e9`, `cellbalance_high_w3e9` lowers Zone 3 volume RMSE by about 29.8%.

## HEC-RAS Native Cell-Balance Residual

Source:

`results_minxiong_landcover_h13h15_cellbalance_eval_selective_high_e20.csv`

| Checkpoint | MSE | Global Cell-Balance RMSE ft3 | Relative RMSE | Zone 3 Cell-Balance RMSE ft3 |
|---|---:|---:|---:|---:|
| noface | `1.359704e-04` | `769.831` | `1.1429` | `600.861` |
| cellbalance_all_w3e9 | `8.721894e-05` | `443.137` | `0.6579` | `431.563` |
| cellbalance_high_w3e9 | `1.102048e-04` | `643.716` | `0.9557` | `299.977` |

Interpretation:

- `cellbalance_all_w3e9` has the best global residual.
- `cellbalance_high_w3e9` has the best high-fidelity-zone residual.
- Relative to `noface`, `cellbalance_high_w3e9` lowers Zone 3 cell-balance RMSE by about 50.1%.
- Relative to `cellbalance_all_w3e9`, `cellbalance_high_w3e9` lowers Zone 3 cell-balance RMSE by about 30.5%.

## 25-Step Rollout

Source:

`results_minxiong_landcover_h13h15_zone_rollout_selective_high_e20_len25.csv`

| Checkpoint | Overall RMSE | Volume RMSE | WD RMSE | Zone 3 Volume RMSE | Zone 3 WD RMSE |
|---|---:|---:|---:|---:|---:|
| noface | `0.127483` | `0.117569` | `0.136157` | `0.084840` | `0.170271` |
| cellbalance_all_w3e9 | `0.103100` | `0.066232` | `0.129830` | `0.061251` | `0.170093` |
| cellbalance_high_w3e9 | `0.109055` | `0.088820` | `0.126067` | `0.039164` | `0.171009` |

Interpretation:

- `cellbalance_all_w3e9` has the best overall rollout RMSE and volume RMSE.
- `cellbalance_high_w3e9` has the best rollout WD RMSE and the best Zone 3 rollout volume RMSE.
- Relative to `noface`, `cellbalance_high_w3e9` lowers Zone 3 rollout volume RMSE by about 53.8%.
- Relative to `cellbalance_all_w3e9`, `cellbalance_high_w3e9` lowers Zone 3 rollout volume RMSE by about 36.1%.

## Research Takeaway

This experiment supports a narrower but stronger claim:

Selective high-fidelity-zone local conservation is not currently the best global
error minimizer, but it is the strongest option for Zone 3 volume and physical
budget consistency among the tested 20-epoch checkpoints.

This aligns with the intended middle path between HydroGraphNet and DUALFloodGNN:

- keep HydroGraphNet's global/physics-informed backbone,
- avoid strict local conservation everywhere,
- focus HEC-RAS native local conservation on zones identified as high fidelity.

The method should not yet be described as outperforming DUALFloodGNN. It should
be described as a lower-intrusion, zone-aware selective local conservation
strategy whose strongest current evidence is high-fidelity-zone volume and
cell-budget improvement.
