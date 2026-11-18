# Optimizer coverage benchmark

Release builds, 1 pinned thread, 12 order-flipped rounds, no warm-up and no rescue reruns. Convergence is 0.01 eV/A; cell rows also require 0.001 eV/A^3. Energy agreement tolerance is 0.001 eV.

| System | Method | Code | Success | Wall ms, median [IQR] | Outside potential ms | Force calls to converge | Final energy eV | Wall p | Outside p | Calls p | Success p | Energy agreement |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cu108-cell | cell-lbfgs | rgmin | 0/12 | 3.6e+04 [3.54e+04, 3.62e+04] | 1.19e+04 [1.17e+04, 1.2e+04] | — | — | — | — | — | — | — |
| Cu108-cell | cell-lbfgs | ase | 12/12 | 619 [612, 636] | 133 [132, 138] | 20 [20, 20] | -292 [-292, -292] | 3.66e-05 | 3.66e-05 | — | 7.4e-07 | 0/0; max Δ=— |
| Cu108-cell | cell-lbfgs | eon | gap | — | — | — | — | — | — | — | — | The eOn client optimizer does not expose variable-cell relaxation. |
| Cu255-vacancy | fire | rgmin | 12/12 | 40.5 [40.2, 43.1] | 26.9 [26.6, 28.5] | 45 [45, 45] | -684 [-684, -684] | — | — | — | — | — |
| Cu255-vacancy | fire | ase | 12/12 | 34.5 [33.7, 35.2] | 20.9 [20.4, 21.4] | 44 [44, 44] | -684 [-684, -684] | 3.66e-05 | 3.66e-05 | 1.91e-06 | 1 | 12/12; max Δ=0.000224 |
| Cu255-vacancy | fire | eon | 12/12 | 19.7 [19.2, 20.5] | 6.42 [6.22, 6.66] | 43 [43, 43] | -684 [-684, -684] | 3.66e-05 | 3.66e-05 | 1.91e-06 | 1 | 12/12; max Δ=0.000396 |
| Cu255-vacancy | lbfgs | rgmin | 12/12 | 12.6 [12.3, 13.2] | 7.97 [7.77, 8.43] | 12 [12, 12] | -684 [-684, -684] | — | — | — | — | — |
| Cu255-vacancy | lbfgs | ase | 12/12 | 16.5 [16, 17.2] | 10.2 [9.97, 10.7] | 18 [18, 18] | -684 [-684, -684] | 3.66e-05 | 3.66e-05 | 1.91e-06 | 1 | 12/12; max Δ=0.000423 |
| Cu255-vacancy | lbfgs | eon | 12/12 | 6.91 [6.69, 7.1] | 2.22 [2.1, 2.3] | 12 [12, 12] | -684 [-684, -684] | 3.66e-05 | 3.66e-05 | 1 | 1 | 12/12; max Δ=6.82e-13 |
| Cu255-vacancy | sd | rgmin | 12/12 | 15.9 [15.7, 16.4] | 9.69 [9.54, 10] | 18 [18, 18] | -684 [-684, -684] | — | — | — | — | — |
| Cu255-vacancy | sd | ase | gap | — | — | — | — | — | — | — | — | No steepest-descent comparator was requested for ASE. |
| Cu255-vacancy | sd | eon | 0/12 | 411 [403, 418] | 135 [132, 139] | — | — | 3.66e-05 | 3.66e-05 | — | 7.4e-07 | 0/0; max Δ=— |
| LJ38 | fire | rgmin | 120/120 | 40.6 [37.6, 43.4] | 39 [36.1, 41.7] | 85 [78, 92] | -174 [-174, -174] | — | — | — | — | — |
| LJ38 | fire | ase | 120/120 | 27.1 [25.7, 28.6] | 25.8 [24.5, 27.2] | 69.5 [67, 73] | -174 [-174, -174] | 1.8e-40 | 1.47e-40 | 1.38e-36 | 1 | 120/120; max Δ=3.95e-05 |
| LJ38 | fire | eon | 120/120 | 7.57 [7.27, 8.14] | 6.13 [5.88, 6.6] | 86.5 [83, 91] | -174 [-174, -174] | 7.14e-41 | 7.14e-41 | 0.283 | 1 | 120/120; max Δ=2.42e-05 |
| LJ38 | lbfgs | rgmin | 120/120 | 11.3 [10.8, 12.1] | 10.8 [10.3, 11.6] | 21.5 [21, 24] | -174 [-174, -174] | — | — | — | — | — |
| LJ38 | lbfgs | ase | 120/120 | 49.6 [45.2, 55.2] | 47.8 [43.6, 53.3] | 89.5 [82, 95] | -174 [-174, -174] | 1.21e-39 | 1.15e-39 | 2.18e-41 | 1 | 120/120; max Δ=1.61e-05 |
| LJ38 | lbfgs | eon | 120/120 | 2.32 [2.18, 2.47] | 1.89 [1.78, 2.01] | 21.5 [21, 24] | -174 [-174, -174] | 1.37e-39 | 1.37e-39 | 0.487 | 1 | 120/120; max Δ=2.83e-06 |
| LJ38 | sd | rgmin | 120/120 | 42.7 [31.9, 60.1] | 40.3 [30.1, 56.7] | 114 [90, 185] | -174 [-174, -174] | — | — | — | — | — |
| LJ38 | sd | ase | gap | — | — | — | — | — | — | — | — | No steepest-descent comparator was requested for ASE. |
| LJ38 | sd | eon | 0/120 | 85.9 [83.7, 90] | 69.9 [67.8, 73.2] | — | — | 1.74e-38 | 1.04e-37 | — | 2.2e-71 | 0/0; max Δ=— |
| LJ75 | fire | rgmin | 120/120 | 42.6 [41, 44.1] | 38.1 [36.7, 39.5] | 80 [79, 84] | -397 [-397, -397] | — | — | — | — | — |
| LJ75 | fire | ase | 120/120 | 31.3 [29.6, 33.1] | 27.3 [25.9, 28.9] | 72.5 [69, 75] | -397 [-397, -397] | 1.67e-40 | 9.64e-41 | 6.47e-29 | 1 | 120/120; max Δ=2.65e-05 |
| LJ75 | fire | eon | 120/120 | 11.4 [10.6, 13.1] | 6.85 [6.39, 7.85] | 86 [80, 99] | -397 [-397, -397] | 7.14e-41 | 7.14e-41 | 1.77e-17 | 1 | 120/120; max Δ=1.98e-05 |
| LJ75 | lbfgs | rgmin | 120/120 | 13.4 [12.4, 14] | 11.9 [11.1, 12.5] | 23.5 [21, 24] | -397 [-397, -397] | — | — | — | — | — |
| LJ75 | lbfgs | ase | 0/120 | 762 [743, 781] | 707 [691, 725] | — | — | 7.14e-41 | 7.14e-41 | — | 2.2e-71 | 0/0; max Δ=— |
| LJ75 | lbfgs | eon | 120/120 | 3.37 [3.07, 3.6] | 2.08 [1.91, 2.24] | 23 [20, 24] | -397 [-397, -397] | 7.14e-41 | 7.14e-41 | 0.566 | 1 | 120/120; max Δ=3.5e-06 |
| LJ75 | sd | rgmin | 120/120 | 71.1 [67.3, 78.8] | 61 [57.8, 67.7] | 187 [177, 213] | -397 [-397, -397] | — | — | — | — | — |
| LJ75 | sd | ase | gap | — | — | — | — | — | — | — | — | No steepest-descent comparator was requested for ASE. |
| LJ75 | sd | eon | 0/120 | 126 [123, 130] | 74.9 [72.9, 77.6] | — | — | 7.14e-41 | 3.59e-31 | — | 2.2e-71 | 0/0; max Δ=— |
| Si64 | fire | rgmin | 12/12 | 19.7 [18.8, 20] | 18 [17.2, 18.3] | 39 [39, 39] | -277 [-277, -277] | — | — | — | — | — |
| Si64 | fire | ase | 12/12 | 18.8 [18.5, 19.4] | 16.9 [16.6, 17.3] | 49 [49, 49] | -277 [-277, -277] | 0.0885 | 0.0141 | 1.91e-06 | 1 | 12/12; max Δ=7.84e-05 |
| Si64 | fire | eon | 12/12 | 4.49 [4.36, 4.6] | 2.53 [2.43, 2.59] | 45 [45, 45] | -277 [-277, -277] | 3.66e-05 | 3.66e-05 | 1.91e-06 | 1 | 12/12; max Δ=1.5e-06 |
| Si64 | lbfgs | rgmin | 12/12 | 5.7 [5.63, 6.34] | 5.12 [5.05, 5.7] | 11 [11, 11] | -277 [-277, -277] | — | — | — | — | — |
| Si64 | lbfgs | ase | 12/12 | 8.51 [8.09, 8.8] | 7.66 [7.26, 7.91] | 18 [18, 18] | -277 [-277, -277] | 3.66e-05 | 3.66e-05 | 1.91e-06 | 1 | 12/12; max Δ=1.32e-05 |
| Si64 | lbfgs | eon | 12/12 | 1.55 [1.53, 1.63] | 0.948 [0.924, 1] | 11 [11, 11] | -277 [-277, -277] | 3.66e-05 | 3.66e-05 | 1 | 1 | 12/12; max Δ=0 |
| Si64 | sd | rgmin | 12/12 | 9.27 [8.99, 9.64] | 8.3 [8.05, 8.62] | 21 [21, 21] | -277 [-277, -277] | — | — | — | — | — |
| Si64 | sd | ase | gap | — | — | — | — | — | — | — | — | No steepest-descent comparator was requested for ASE. |
| Si64 | sd | eon | 0/12 | 92.5 [90.4, 95.3] | 51 [50.1, 52.9] | — | — | 3.66e-05 | 3.66e-05 | — | 7.4e-07 | 0/0; max Δ=— |
| Si64-cell | cell-lbfgs | rgmin | 0/12 | 1.33e+04 [1.29e+04, 1.36e+04] | 1.2e+04 [1.16e+04, 1.23e+04] | — | — | — | — | — | — | — |
| Si64-cell | cell-lbfgs | ase | 12/12 | 66.1 [64.2, 67] | 55.1 [53.6, 56.2] | 9 [9, 9] | -277 [-277, -277] | 3.66e-05 | 3.66e-05 | — | 7.4e-07 | 0/0; max Δ=— |
| Si64-cell | cell-lbfgs | eon | gap | — | — | — | — | — | — | — | — | The eOn client optimizer does not expose variable-cell relaxation. |

The result JSON contains the odd/even A/A wall-time noise band and its Mann-Whitney p-value for every system/method/code cell. The widest relative band is 6.93% for Cu255-vacancy lbfgs rgmin.

## Failures and gaps

Nonconverged or failed runs: 408.
- Cu108-cell cell-lbfgs rgmin: 12 nonconverged; force calls/attempt 1002 [1002, 1002].
- Cu255-vacancy sd eon: 12 nonconverged; force calls/attempt 1001 [1001, 1001].
- LJ38 sd eon: 120 nonconverged; force calls/attempt 1001 [1001, 1001].
- LJ75 lbfgs ase: 120 nonconverged; force calls/attempt 1001 [1001, 1001].
- LJ75 sd eon: 120 nonconverged; force calls/attempt 1001 [1001, 1001].
- Si64 sd eon: 12 nonconverged; force calls/attempt 1001 [1001, 1001].
- Si64-cell cell-lbfgs rgmin: 12 nonconverged; force calls/attempt 1002 [1002, 1002].
The eOn client has no variable-cell optimizer. A generic rgpot EMT Python binding was absent and is delivered as a separate source revision; the selected Cu systems use the predeclared Morse potential.

## Bugs

- Cu108-cell: rgmin capped L-BFGS converged 0/12 times while ASE converged 12/12; rgmin median final stress was 0.016 eV/A^3.
- Si64-cell: rgmin capped L-BFGS converged 0/12 times while ASE converged 12/12; rgmin median final stress was 0.0746 eV/A^3.
- Minimal reproduction: `bench/optimizers/repro_cell.py` runs the same capped L-BFGS and filter without the benchmark scheduler.
