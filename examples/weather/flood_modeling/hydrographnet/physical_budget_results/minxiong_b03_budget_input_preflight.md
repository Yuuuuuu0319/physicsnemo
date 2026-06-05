# HEC-RAS Budget Input Preflight

This report checks whether candidate HEC-RAS plan/result inputs are ready for formal HydroGraphNet physical-budget validation.

| Type | Path | Status | Output interval | Native Face Flow | Cell Flow Balance | Note |
|---|---|---|---:|---|---|---|
| plan | `/mnt/8tb_hdd2/joyce/Minxiong/Minxiong.b03` | BLOCKED | 1800.000 s | None | None | Plan text does not prove native Face Flow / Cell Flow Balance output; inspect the result HDF after simulation. |
| plan | `/mnt/8tb_hdd2/joyce/hecras-dataset/origin/Linux_RAS_v66/Minxiong_hgn_physical_budget_h1/outputs_budget_seed/Minxiong.b01` | PASS | 30.000 s | None | None | Plan text does not prove native Face Flow / Cell Flow Balance output; inspect the result HDF after simulation. |
| hdf | `/mnt/8tb_hdd2/joyce/Minxiong/Minxiong.p03.hdf` | BLOCKED | 1800.000 s | False | False | Native budget-ready HDF requires Face Flow, Cell Flow Balance, and stored output matching the computation interval. |

## Interpretation

- `BLOCKED` means the file is not sufficient for formal local-conservation training yet.
- A plan file can show the intended output interval, but native optional variables must be verified in the completed HDF.
- The next heavy check remains `evaluate_hecras_physical_budget.py` after a candidate HDF passes this preflight.
