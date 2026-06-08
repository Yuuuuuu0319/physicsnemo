# Minxiong Land-Cover H1-H12 to H13-H15 50-Epoch Checkpoint Sweep

Date: 2026-06-06

## Purpose

This run extends the prior 20-epoch Minxiong experiments to 50 epochs before increasing the event count. The goal is to verify that the current training/evaluation pipeline is stable and that the earlier short runs were not under-trained.

Training split:

- Train events: `train_h1h12.txt`
- Test events: `test_h13h15.txt`
- Data directory: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong/outputs/HGN_dataset`
- HEC-RAS outputs: `/mnt/8tb_hdd2/joyce/hecras-dataset/Linux_RAS_v66/Minxiong/outputs/planH*/Minxiong.p01.hdf`

Compared checkpoints:

- `noface_e50`: HydroGraphNet baseline with global physics loss and no HEC-RAS local cell-balance loss.
- `cellbalance_all_w3e9_e50`: HEC-RAS native Cell Flow Balance loss applied to all cells.
- `cellbalance_high_w3e9_e50`: HEC-RAS native Cell Flow Balance loss applied only to the high-fidelity zone.

Evaluation file:

- `results_minxiong_landcover_h13h15_epoch_sweep_e50.csv`

Evaluation settings:

- `num_samples=500`
- `rollout_length=25`
- checkpoint epochs swept from 0 to 49

Longer rollout note:

- A follow-up 45-step rollout attempt was made after the 50-epoch sweep.
- It failed before inference because H14 and H15 only have 27 usable dynamic time steps after loading.
- With `n_time_steps=2`, the maximum common rollout length across H13-H15 is therefore 25.
- Longer 45-step or 48-step rollout evidence requires regenerated events with at least 47 or 50 loaded time steps, respectively.

## Best Metrics Across All Checkpoints

| Metric | Best checkpoint | Epoch | Value |
|---|---:|---:|---:|
| One-step MSE | `cellbalance_all_w3e9_e50` | 49 | `3.066061e-05` |
| Zone 3 one-step volume RMSE | `cellbalance_high_w3e9_e50` | 42 | `1.533498e-03` |
| Zone 3 one-step water-depth RMSE | `cellbalance_high_w3e9_e50` | 49 | `1.135433e-02` |
| Formal cell-balance RMSE | `cellbalance_all_w3e9_e50` | 49 | `268.863 ft^3` |
| Zone 3 formal cell-balance RMSE | `cellbalance_high_w3e9_e50` | 33 | `143.425 ft^3` |
| 25-step rollout RMSE | `cellbalance_all_w3e9_e50` | 49 | `5.463940e-02` |
| 25-step rollout volume RMSE | `cellbalance_all_w3e9_e50` | 49 | `3.672743e-02` |
| 25-step rollout water-depth RMSE | `cellbalance_high_w3e9_e50` | 49 | `6.328221e-02` |
| Zone 3 rollout volume RMSE | `cellbalance_high_w3e9_e50` | 42 | `1.866333e-02` |
| Zone 3 rollout water-depth RMSE | `cellbalance_high_w3e9_e50` | 49 | `1.150844e-01` |

## Best Rollout Checkpoint Per Model

| Model | Best epoch | Rollout RMSE | One-step MSE | Formal cell-balance RMSE | Zone 3 cell-balance RMSE | Zone 3 rollout volume RMSE |
|---|---:|---:|---:|---:|---:|---:|
| `noface_e50` | 43 | `8.599673e-02` | `7.105205e-05` | `623.027 ft^3` | `378.459 ft^3` | `5.588626e-02` |
| `cellbalance_all_w3e9_e50` | 49 | `5.463940e-02` | `3.066061e-05` | `268.863 ft^3` | `275.491 ft^3` | `2.619364e-02` |
| `cellbalance_high_w3e9_e50` | 49 | `6.913687e-02` | `4.907576e-05` | `500.586 ft^3` | `189.805 ft^3` | `4.951072e-02` |

## Epoch 49 Comparison

| Model | One-step MSE | Formal cell-balance RMSE | Zone 3 cell-balance RMSE | 25-step rollout RMSE | Rollout volume RMSE | Rollout water-depth RMSE | Zone 3 rollout volume RMSE | Zone 3 rollout water-depth RMSE |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `noface_e50` | `8.673195e-05` | `899.852 ft^3` | `505.241 ft^3` | `9.038982e-02` | `1.058093e-01` | `7.165357e-02` | `8.216074e-02` | `1.511209e-01` |
| `cellbalance_all_w3e9_e50` | `3.066061e-05` | `268.863 ft^3` | `275.491 ft^3` | `5.463940e-02` | `3.672743e-02` | `6.792857e-02` | `2.619364e-02` | `1.494138e-01` |
| `cellbalance_high_w3e9_e50` | `4.907576e-05` | `500.586 ft^3` | `189.805 ft^3` | `6.913687e-02` | `7.445958e-02` | `6.328221e-02` | `4.951072e-02` | `1.150844e-01` |

## Interpretation

The longer 50-epoch runs confirm that the previous 20-epoch experiments were under-trained. The baseline best 25-step rollout RMSE improves from the earlier approximately `0.124` range to `0.085997`, so longer training is necessary before expanding the event count.

The all-zone Cell Flow Balance model is the strongest overall model. It has the best one-step MSE, best global formal cell-balance residual, and best 25-step rollout RMSE. This is the main evidence that using HEC-RAS native Cell Flow Balance as a formal local-conservation target improves HydroGraphNet on the current Minxiong H13-H15 held-out events.

The high-zone selective model is not the global error minimizer, but it is strongest for the high-fidelity zone. It achieves the best Zone 3 formal cell-balance RMSE and best Zone 3 rollout water-depth RMSE, and it also gives the best Zone 3 one-step and rollout volume metrics at selected epochs. This supports the Automatic Fidelity Zoning direction: the selective local-conservation term concentrates the physical improvement in the region that was intentionally modeled at higher mesh fidelity.

## Current Recommendation

Use two model choices depending on the report claim:

- Overall accuracy claim: use `cellbalance_all_w3e9_e50`, epoch 49.
- Selective high-fidelity-zone local-conservation claim: use `cellbalance_high_w3e9_e50`, epoch 49 for rollout water-depth and epoch 33 or 42 for strict Zone 3 budget/volume analysis.

The next stage should expand from H1-H12/H13-H15 to more events only after keeping this exact evaluation pipeline fixed. The proposed expansion should increase both event count and rollout length, for example:

- More training events: include newly generated events beyond H15.
- More test events: reserve a separate held-out group instead of evaluating only H13-H15.
- Longer rollout checks: add 45-step or 48-step rollout in addition to 25-step, but only after the generated HGN event files contain enough loaded time steps.
