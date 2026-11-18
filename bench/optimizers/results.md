# Optimizer coverage benchmark

Release builds, 1 pinned thread, 12 order-flipped rounds, no warm-up and no rescue reruns. Convergence is 0.01 eV/A; cell rows also require 0.001 eV/A^3. Energy agreement tolerance is 0.001 eV.

| System | Method | Code | Success | Wall ms, median [IQR] | Outside potential ms | Force calls to converge | Final energy eV | Wall p | Outside p | Calls p | Success p | Energy agreement |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Cu108-cell | cell-lbfgs | rgmin | 0/12 | 3.4e+04 [3.37e+04, 3.43e+04] | 1.1e+04 [1.09e+04, 1.12e+04] | — | — | — | — | — | — | — |
| Cu108-cell | cell-lbfgs | ase | 12/12 | 592 [589, 598] | 124 [122, 127] | 20 [20, 20] | -292 [-292, -292] | 3.66e-05 | 3.66e-05 | — | 7.4e-07 | 0/0; max Δ=— |
| Cu108-cell | cell-lbfgs | eon | gap | — | — | — | — | — | — | — | — | The eOn client optimizer does not expose variable-cell relaxation. |
| Cu255-vacancy | fire | rgmin | 12/12 | 38.9 [38.4, 39.3] | 25.8 [25.4, 26] | 45 [45, 45] | -684 [-684, -684] | — | — | — | — | — |
| Cu255-vacancy | fire | ase | 12/12 | 31.8 [31.7, 32.5] | 19.2 [19.1, 19.8] | 44 [44, 44] | -684 [-684, -684] | 3.66e-05 | 3.66e-05 | 1.91e-06 | 1 | 12/12; max Δ=0.000224 |
| Cu255-vacancy | fire | eon | 12/12 | 18.6 [18.5, 18.8] | 5.96 [5.88, 6.06] | 43 [43, 43] | -684 [-684, -684] | 3.66e-05 | 3.66e-05 | 1.91e-06 | 1 | 12/12; max Δ=0.000396 |
| Cu255-vacancy | lbfgs | rgmin | 12/12 | 11.9 [11.8, 12.1] | 7.48 [7.41, 7.68] | 12 [12, 12] | -684 [-684, -684] | — | — | — | — | — |
| Cu255-vacancy | lbfgs | ase | 12/12 | 15.8 [15.5, 15.9] | 9.76 [9.54, 9.85] | 18 [18, 18] | -684 [-684, -684] | 3.66e-05 | 3.66e-05 | 1.91e-06 | 1 | 12/12; max Δ=0.000423 |
| Cu255-vacancy | lbfgs | eon | 12/12 | 6.47 [6.4, 6.51] | 2.03 [1.99, 2.06] | 12 [12, 12] | -684 [-684, -684] | 3.66e-05 | 3.66e-05 | 1 | 1 | 12/12; max Δ=6.82e-13 |
| Cu255-vacancy | sd | rgmin | 12/12 | 15.1 [14.9, 15.2] | 9.11 [8.97, 9.25] | 18 [18, 18] | -684 [-684, -684] | — | — | — | — | — |
| Cu255-vacancy | sd | ase | gap | — | — | — | — | — | — | — | — | No steepest-descent comparator was requested for ASE. |
| Cu255-vacancy | sd | eon | 0/12 | 389 [386, 391] | 126 [125, 128] | — | — | 3.66e-05 | 3.66e-05 | — | 7.4e-07 | 0/0; max Δ=— |
| LJ38 | fire | rgmin | 120/120 | 37.7 [34.3, 40.6] | 36.2 [33, 39] | 85 [78, 92] | -174 [-174, -174] | — | — | — | — | — |
| LJ38 | fire | ase | 120/120 | 25.2 [24.1, 26] | 24.1 [22.9, 24.8] | 69.5 [67, 73] | -174 [-174, -174] | 7.14e-41 | 7.14e-41 | 1.38e-36 | 1 | 120/120; max Δ=3.95e-05 |
| LJ38 | fire | eon | 120/120 | 7.06 [6.78, 7.34] | 5.73 [5.5, 5.96] | 86.5 [83, 91] | -174 [-174, -174] | 7.14e-41 | 7.14e-41 | 0.283 | 1 | 120/120; max Δ=2.42e-05 |
| LJ38 | lbfgs | rgmin | 120/120 | 10.4 [9.95, 11.2] | 9.91 [9.51, 10.8] | 21.5 [21, 24] | -174 [-174, -174] | — | — | — | — | — |
| LJ38 | lbfgs | ase | 120/120 | 46.4 [40.9, 49.1] | 44.8 [39.5, 47.4] | 89.5 [82, 95] | -174 [-174, -174] | 9.89e-41 | 1.51e-40 | 2.18e-41 | 1 | 120/120; max Δ=1.61e-05 |
| LJ38 | lbfgs | eon | 120/120 | 2.09 [1.97, 2.19] | 1.69 [1.6, 1.78] | 21.5 [21, 24] | -174 [-174, -174] | 9.17e-41 | 9.41e-41 | 0.487 | 1 | 120/120; max Δ=2.83e-06 |
| LJ38 | sd | rgmin | 120/120 | 39 [29, 57.6] | 37 [27.5, 54.6] | 114 [90, 185] | -174 [-174, -174] | — | — | — | — | — |
| LJ38 | sd | ase | gap | — | — | — | — | — | — | — | — | No steepest-descent comparator was requested for ASE. |
| LJ38 | sd | eon | 0/120 | 79.3 [78.9, 80.2] | 64.6 [64.2, 65.2] | — | — | 7.14e-41 | 1.55e-39 | — | 2.2e-71 | 0/0; max Δ=— |
| LJ75 | fire | rgmin | 120/120 | 39.2 [38.7, 41.1] | 35.1 [34.6, 36.8] | 80 [79, 84] | -397 [-397, -397] | — | — | — | — | — |
| LJ75 | fire | ase | 120/120 | 29.2 [27.7, 30.5] | 25.5 [24.3, 26.7] | 72.5 [69, 75] | -397 [-397, -397] | 3.99e-40 | 2.43e-40 | 6.47e-29 | 1 | 120/120; max Δ=2.65e-05 |
| LJ75 | fire | eon | 120/120 | 10.5 [9.74, 11.9] | 6.3 [5.85, 7.14] | 86 [80, 99] | -397 [-397, -397] | 7.14e-41 | 7.14e-41 | 1.77e-17 | 1 | 120/120; max Δ=1.98e-05 |
| LJ75 | lbfgs | rgmin | 120/120 | 12.6 [11.3, 13] | 11.3 [10.1, 11.7] | 23.5 [21, 24] | -397 [-397, -397] | — | — | — | — | — |
| LJ75 | lbfgs | ase | 0/120 | 711 [705, 720] | 661 [655, 669] | — | — | 7.14e-41 | 7.14e-41 | — | 2.2e-71 | 0/0; max Δ=— |
| LJ75 | lbfgs | eon | 120/120 | 3.17 [2.88, 3.3] | 1.96 [1.78, 2.04] | 23 [20, 24] | -397 [-397, -397] | 7.14e-41 | 7.14e-41 | 0.566 | 1 | 120/120; max Δ=3.5e-06 |
| LJ75 | sd | rgmin | 120/120 | 66.1 [63.4, 75] | 56.8 [54.4, 64.5] | 187 [177, 213] | -397 [-397, -397] | — | — | — | — | — |
| LJ75 | sd | ase | gap | — | — | — | — | — | — | — | — | No steepest-descent comparator was requested for ASE. |
| LJ75 | sd | eon | 0/120 | 117 [117, 118] | 69.7 [69.3, 70.2] | — | — | 7.14e-41 | 2.45e-38 | — | 2.2e-71 | 0/0; max Δ=— |
| Si64 | fire | rgmin | 12/12 | 18.3 [18.2, 18.8] | 16.7 [16.6, 17] | 39 [39, 39] | -277 [-277, -277] | — | — | — | — | — |
| Si64 | fire | ase | 12/12 | 17.8 [17.7, 18.1] | 16 [15.9, 16.2] | 49 [49, 49] | -277 [-277, -277] | 0.000592 | 6.01e-05 | 1.91e-06 | 1 | 12/12; max Δ=7.84e-05 |
| Si64 | fire | eon | 12/12 | 4.25 [4.16, 4.37] | 2.33 [2.28, 2.42] | 45 [45, 45] | -277 [-277, -277] | 3.66e-05 | 3.66e-05 | 1.91e-06 | 1 | 12/12; max Δ=1.5e-06 |
| Si64 | lbfgs | rgmin | 12/12 | 5.62 [5.51, 5.88] | 5.05 [4.95, 5.27] | 11 [11, 11] | -277 [-277, -277] | — | — | — | — | — |
| Si64 | lbfgs | ase | 12/12 | 8.04 [7.98, 8.24] | 7.23 [7.17, 7.38] | 18 [18, 18] | -277 [-277, -277] | 3.66e-05 | 3.66e-05 | 1.91e-06 | 1 | 12/12; max Δ=1.32e-05 |
| Si64 | lbfgs | eon | 12/12 | 1.45 [1.42, 1.52] | 0.868 [0.847, 0.921] | 11 [11, 11] | -277 [-277, -277] | 3.66e-05 | 3.66e-05 | 1 | 1 | 12/12; max Δ=0 |
| Si64 | sd | rgmin | 12/12 | 9.03 [8.8, 9.08] | 8.07 [7.87, 8.17] | 21 [21, 21] | -277 [-277, -277] | — | — | — | — | — |
| Si64 | sd | ase | gap | — | — | — | — | — | — | — | — | No steepest-descent comparator was requested for ASE. |
| Si64 | sd | eon | 0/12 | 88.8 [88.4, 91.2] | 49.2 [48.8, 50.3] | — | — | 3.66e-05 | 3.66e-05 | — | 7.4e-07 | 0/0; max Δ=— |
| Si64-cell | cell-lbfgs | rgmin | 0/12 | 1.23e+04 [1.22e+04, 1.25e+04] | 1.11e+04 [1.1e+04, 1.12e+04] | — | — | — | — | — | — | — |
| Si64-cell | cell-lbfgs | ase | 12/12 | 61 [60.6, 61.7] | 50.6 [50.3, 51.3] | 9 [9, 9] | -277 [-277, -277] | 3.66e-05 | 3.66e-05 | — | 7.4e-07 | 0/0; max Δ=— |
| Si64-cell | cell-lbfgs | eon | gap | — | — | — | — | — | — | — | — | The eOn client optimizer does not expose variable-cell relaxation. |

The result JSON contains the odd/even A/A wall-time noise band and its Mann-Whitney p-value for every system/method/code cell. The widest relative band is 7.4% for LJ38 sd rgmin.

## Failures and gaps

Nonconverged or failed runs: 408.
- Cu108-cell cell-lbfgs rgmin: 12 nonconverged.
- Cu255-vacancy sd eon: 12 nonconverged.
- LJ38 sd eon: 120 nonconverged.
- LJ75 lbfgs ase: 120 nonconverged.
- LJ75 sd eon: 120 nonconverged.
- Si64 sd eon: 12 nonconverged.
- Si64-cell cell-lbfgs rgmin: 12 nonconverged.
The eOn client has no variable-cell optimizer. A generic rgpot EMT Python binding was absent and is delivered as a separate source revision; the selected Cu systems use the predeclared Morse potential.

## Bugs

- Cu108-cell: rgmin capped L-BFGS converged 0/12 times while ASE converged 12/12; rgmin median final stress was 0.016 eV/A^3.
- Si64-cell: rgmin capped L-BFGS converged 0/12 times while ASE converged 12/12; rgmin median final stress was 0.0746 eV/A^3.
- Minimal reproduction: `bench/optimizers/repro_cell.py` runs the same capped L-BFGS and filter without the benchmark scheduler.
