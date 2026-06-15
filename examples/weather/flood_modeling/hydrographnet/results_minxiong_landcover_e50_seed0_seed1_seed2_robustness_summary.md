# Minxiong LandCover E50 Robustness Summary - Seeds 0, 1, and 2

## Purpose

This note compares the 50-epoch checkpoint sweeps for seeds 0, 1, and 2 on
the same held-out H13-H15 split. It is intended to check whether the current
Automatic Fidelity Zoning / selective local-conservation conclusion is stable
across random initialization.

## Inputs

- Seed 0 CSV:
  `results_minxiong_landcover_h13h15_epoch_sweep_e50.csv`
- Seed 1 CSV:
  `results_minxiong_landcover_h13h15_epoch_sweep_e50_seed1.csv`
- Seed 2 CSV:
  `results_minxiong_landcover_h13h15_epoch_sweep_e50_seed2.csv`
- Held-out split: `test_h13h15.txt`
- Rollout length: `25`
- Branches:
  - `noface`: HydroGraphNet physics loss only
  - `all`: native HEC-RAS `Cell Flow Balance` on all zones
  - `high`: native HEC-RAS `Cell Flow Balance` only on high-fidelity Zone 3

## Best Rollout Checkpoint Per Branch

| Seed | Branch | Best epoch | Rollout RMSE | One-step MSE | Global cell-balance RMSE | Zone 3 cell-balance RMSE | Zone 3 rollout volume RMSE | Zone 3 rollout WD RMSE |
|---:|---|---:|---:|---:|---:|---:|---:|---:|
| 0 | `noface` | 43 | `0.085997` | `7.105205e-05` | `623.027 ft^3` | `378.459 ft^3` | `0.055886` | `0.178864` |
| 0 | `all` | 49 | `0.054639` | `3.066061e-05` | `268.863 ft^3` | `275.491 ft^3` | `0.026194` | `0.149414` |
| 0 | `high` | 49 | `0.069137` | `4.907576e-05` | `500.586 ft^3` | `189.805 ft^3` | `0.049511` | `0.115084` |
| 1 | `noface` | 49 | `0.102112` | `8.811070e-05` | `709.605 ft^3` | `838.479 ft^3` | `0.146871` | `0.165964` |
| 1 | `all` | 48 | `0.060910` | `3.427456e-05` | `340.462 ft^3` | `367.413 ft^3` | `0.046748` | `0.132438` |
| 1 | `high` | 48 | `0.080271` | `6.480511e-05` | `591.463 ft^3` | `172.433 ft^3` | `0.045106` | `0.113761` |
| 2 | `noface` | 47 | `0.081478` | `7.177839e-05` | `587.129 ft^3` | `528.465 ft^3` | `0.078567` | `0.140076` |
| 2 | `all` | 45 | `0.052871` | `2.943424e-05` | `272.728 ft^3` | `286.423 ft^3` | `0.029452` | `0.141417` |
| 2 | `high` | 44 | `0.076220` | `5.936272e-05` | `510.923 ft^3` | `167.203 ft^3` | `0.030570` | `0.145533` |

## Mean / Standard Deviation Across Seeds

Values are computed from each branch's best-rollout checkpoint in each seed.

| Branch | Metric | Mean | Std |
|---|---|---:|---:|
| `noface` | One-step MSE | `7.698040e-05` | `7.875910e-06` |
| `noface` | Global cell-balance RMSE | `639.920 ft^3` | `51.408 ft^3` |
| `noface` | Zone 3 cell-balance RMSE | `581.801 ft^3` | `191.552 ft^3` |
| `noface` | Rollout RMSE | `0.089862` | `0.008856` |
| `noface` | Zone 3 rollout volume RMSE | `0.093775` | `0.038670` |
| `noface` | Zone 3 rollout WD RMSE | `0.161635` | `0.016128` |
| `all` | One-step MSE | `3.145650e-05` | `2.054630e-06` |
| `all` | Global cell-balance RMSE | `294.018 ft^3` | `32.879 ft^3` |
| `all` | Zone 3 cell-balance RMSE | `309.776 ft^3` | `40.999 ft^3` |
| `all` | Rollout RMSE | `0.056140` | `0.003449` |
| `all` | Zone 3 rollout volume RMSE | `0.034131` | `0.009020` |
| `all` | Zone 3 rollout WD RMSE | `0.141090` | `0.006934` |
| `high` | One-step MSE | `5.774790e-05` | `6.522210e-06` |
| `high` | Global cell-balance RMSE | `534.324 ft^3` | `40.623 ft^3` |
| `high` | Zone 3 cell-balance RMSE | `176.480 ft^3` | `9.661 ft^3` |
| `high` | Rollout RMSE | `0.075209` | `0.004601` |
| `high` | Zone 3 rollout volume RMSE | `0.041729` | `0.008093` |
| `high` | Zone 3 rollout WD RMSE | `0.124793` | `0.014675` |

## Best Branch Per Claim

| Claim | Seed 0 best | Seed 1 best | Seed 2 best | Robust interpretation |
|---|---|---|---|---|
| Overall 25-step rollout RMSE | `all`, epoch 49, `0.054639` | `all`, epoch 48, `0.060910` | `all`, epoch 45, `0.052871` | Stable: all-zone branch is best for overall rollout in all three seeds. |
| Global formal cell-balance RMSE | `all`, epoch 49, `268.863 ft^3` | `all`, epoch 42, `300.928 ft^3` | `all`, epoch 49, `266.674 ft^3` | Stable: all-zone branch is best for global physical residual in all three seeds. |
| Zone 3 formal cell-balance RMSE | `high`, epoch 33, `143.425 ft^3` | `high`, epoch 47, `136.551 ft^3` | `high`, epoch 44, `167.203 ft^3` | Stable: selective high-zone branch is best for high-fidelity-zone local budget in all three seeds. |
| Zone 3 rollout volume RMSE | `high`, epoch 42, `0.018663` | `all`, epoch 45, `0.032253` | `high`, epoch 40, `0.022568` | Mostly supportive but not fully stable: high-zone wins in 2/3 seeds. |
| Zone 3 rollout WD RMSE | `high`, epoch 49, `0.115084` | `high`, epoch 48, `0.113761` | `noface`, epoch 27, `0.123689` | Mixed: high-zone wins in 2/3 seeds, but this should not be claimed as seed-robust yet. |

## Updated Interpretation

- The third seed sharpens the main conservative claim:
  - all-zone native cell-balance training is the strongest overall/global
    baseline across all three seeds.
  - high-zone selective native cell-balance training is consistently best for
    Zone 3 formal local-budget residual across all three seeds.
- The claim needs to remain precise:
  - Do not say the high-zone branch is always best for every Zone 3 metric.
  - Zone 3 rollout volume is promising but mixed: high-zone wins in seeds 0
    and 2, while all-zone wins in seed 1.
  - Zone 3 rollout water depth is also mixed after seed 2: high-zone wins in
    seeds 0 and 1, while noface wins in seed 2.
- The best current wording is:

```text
All-zone cell-balance training provides the most stable overall and global
conservation improvement, while selective high-zone cell-balance training
consistently improves formal local-budget closure in the high-fidelity zone.
Zone 3 rollout-volume benefits are promising but not fully seed-robust, and
Zone 3 rollout water-depth benefits remain mixed under the current H1-H15
event set.
```

## Remaining Work

- Generate more HEC-RAS / HGN events beyond H1-H15 before making stronger
  statistical claims.
- Export longer time series if 45-step or 48-step rollout evidence is needed;
  current H13-H15 loaded time steps only support a fair rollout length of 25.
- For paper figures, report all-zone as the strongest overall/global baseline
  and high-zone as the strongest selective local-budget method, instead of
  claiming high-zone universally improves every accuracy metric.
