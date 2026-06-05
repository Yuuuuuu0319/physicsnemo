# HEC-RAS Budget Input Preflight

This report checks whether candidate HEC-RAS plan/result inputs are ready for formal HydroGraphNet physical-budget validation.

| Type | Path | Status | Output interval | Native Face Flow | Cell Flow Balance | Note |
|---|---|---|---:|---|---|---|
| plan | `/mnt/8tb_hdd2/joyce/Minxiong/Minxiong_new/Minxiong.b03` | PASS | 30.000 s | None | None | Plan text does not prove native Face Flow / Cell Flow Balance output; inspect the result HDF after simulation. |
| hdf | `/mnt/8tb_hdd2/joyce/Minxiong/Minxiong_new/Minxiong.p03.hdf` | PASS | 30.000 s | True | True | Native budget-ready HDF requires Face Flow, Cell Flow Balance, and stored output matching the computation interval. |

## Interpretation

- `BLOCKED` means the file is not sufficient for formal local-conservation training yet.
- A plan file can show the intended output interval, but native optional variables must be verified in the completed HDF.
- The next heavy check remains `evaluate_hecras_physical_budget.py` after a candidate HDF passes this preflight.
