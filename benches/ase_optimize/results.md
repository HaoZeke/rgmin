# rgmin.optimize against ase.optimize

- rgmin 0.2.0, git 1f6d19076ef6d64c1e5a5253e0f06e27d0afb9be
- ASE 3.29.0, NumPy 2.5.4, Python 3.12.3
- Linux-6.12.113+-x86_64-with-glibc2.39, Intel(R) Xeon(R) Processor, 8 logical CPUs
- fmax 0.01, step budget 2000, 5 serial repeats per row,
  default constructor arguments on both sides.

Steps and force calls are identical on every repeat. Wall time is
the whole `run`, Python calculator included, in milliseconds.
ODE12r has no rgmin counterpart; its rows are ASE only.

| case | optimizer | library | steps | force calls | energy | fmax | median ms | min ms | converged |
|---|---|---|---:|---:|---:|---:|---:|---:|---|
| Cu55 | BFGS | ase | 28 | 29 | 24.648760 | 8.057e-03 | 130.9 | 130.4 | yes |
| Cu55 | BFGS | rgmin | 28 | 29 | 24.648760 | 8.057e-03 | 96.8 | 96.1 | yes |
| Cu55 | LBFGS | ase | 28 | 29 | 24.648760 | 8.057e-03 | 93.0 | 92.6 | yes |
| Cu55 | LBFGS | rgmin | 28 | 29 | 24.648760 | 8.057e-03 | 94.9 | 93.9 | yes |
| Cu55 | LBFGSLineSearch | ase | 18 | 34 | 24.648764 | 9.120e-03 | 108.5 | 107.8 | yes |
| Cu55 | LBFGSLineSearch | rgmin | 13 | 14 | 24.648750 | 7.781e-03 | 48.0 | 47.7 | yes |
| Cu55 | FIRE | ase | 51 | 52 | 24.648883 | 9.270e-03 | 161.6 | 158.1 | yes |
| Cu55 | FIRE | rgmin | 51 | 52 | 24.648883 | 9.270e-03 | 163.5 | 163.0 | yes |
| Cu55 | FIRE2 | ase | 30 | 32 | 24.648879 | 8.889e-03 | 100.7 | 100.5 | yes |
| Cu55 | FIRE2 | rgmin | 27 | 30 | 24.648921 | 9.503e-03 | 96.1 | 95.8 | yes |
| Cu55 | ODE12r | ase | 18 | 19 | 24.648824 | 6.161e-03 | 60.7 | 60.2 | yes |
| Pt38 | BFGS | ase | 26 | 27 | 18.135801 | 9.990e-03 | 81.1 | 80.3 | yes |
| Pt38 | BFGS | rgmin | 26 | 27 | 18.135801 | 9.990e-03 | 66.5 | 65.9 | yes |
| Pt38 | LBFGS | ase | 26 | 27 | 18.135801 | 9.990e-03 | 66.5 | 64.5 | yes |
| Pt38 | LBFGS | rgmin | 26 | 27 | 18.135801 | 9.990e-03 | 68.9 | 63.4 | yes |
| Pt38 | LBFGSLineSearch | ase | 16 | 30 | 18.135855 | 7.644e-03 | 72.7 | 70.9 | yes |
| Pt38 | LBFGSLineSearch | rgmin | 11 | 12 | 18.135867 | 9.789e-03 | 31.6 | 30.4 | yes |
| Pt38 | FIRE | ase | 63 | 64 | 18.135820 | 7.729e-03 | 143.8 | 142.2 | yes |
| Pt38 | FIRE | rgmin | 63 | 64 | 18.135820 | 7.729e-03 | 152.0 | 148.2 | yes |
| Pt38 | FIRE2 | ase | 47 | 50 | 18.136016 | 9.418e-03 | 112.8 | 112.3 | yes |
| Pt38 | FIRE2 | rgmin | 29 | 33 | 18.135953 | 9.575e-03 | 78.0 | 76.0 | yes |
| Pt38 | ODE12r | ase | 20 | 21 | 18.135845 | 5.753e-03 | 46.8 | 46.4 | yes |
| Cu(111)+O | BFGS | ase | 36 | 37 | 6.855077 | 9.960e-03 | 117.5 | 113.9 | yes |
| Cu(111)+O | BFGS | rgmin | 36 | 37 | 6.855077 | 9.960e-03 | 99.1 | 98.4 | yes |
| Cu(111)+O | LBFGS | ase | 36 | 37 | 6.855077 | 9.960e-03 | 95.6 | 95.3 | yes |
| Cu(111)+O | LBFGS | rgmin | 36 | 37 | 6.855077 | 9.960e-03 | 97.7 | 97.5 | yes |
| Cu(111)+O | LBFGSLineSearch | ase | 22 | 41 | 6.855031 | 8.078e-03 | 106.6 | 106.4 | yes |
| Cu(111)+O | LBFGSLineSearch | rgmin | 13 | 14 | 6.855010 | 9.346e-03 | 40.6 | 40.3 | yes |
| Cu(111)+O | FIRE | ase | 63 | 64 | 6.856021 | 9.259e-03 | 154.8 | 154.3 | yes |
| Cu(111)+O | FIRE | rgmin | 63 | 64 | 6.856021 | 9.259e-03 | 162.4 | 162.4 | yes |
| Cu(111)+O | FIRE2 | ase | 34 | 36 | 6.855607 | 9.333e-03 | 91.8 | 91.4 | yes |
| Cu(111)+O | FIRE2 | rgmin | 30 | 33 | 6.855606 | 9.230e-03 | 86.6 | 86.2 | yes |
| Cu(111)+O | ODE12r | ase | 21 | 22 | 6.856898 | 9.711e-03 | 56.7 | 56.5 | yes |
| Pt(111)+O | BFGS | ase | 34 | 35 | 6.038290 | 7.701e-03 | 106.6 | 106.2 | yes |
| Pt(111)+O | BFGS | rgmin | 34 | 35 | 6.038290 | 7.701e-03 | 91.9 | 91.1 | yes |
| Pt(111)+O | LBFGS | ase | 34 | 35 | 6.038290 | 7.701e-03 | 88.3 | 87.9 | yes |
| Pt(111)+O | LBFGS | rgmin | 34 | 35 | 6.038290 | 7.701e-03 | 90.1 | 89.4 | yes |
| Pt(111)+O | LBFGSLineSearch | ase | 20 | 37 | 6.038307 | 7.780e-03 | 94.1 | 93.9 | yes |
| Pt(111)+O | LBFGSLineSearch | rgmin | 13 | 14 | 6.038249 | 5.156e-03 | 39.3 | 39.2 | yes |
| Pt(111)+O | FIRE | ase | 62 | 63 | 6.038885 | 9.886e-03 | 149.6 | 148.5 | yes |
| Pt(111)+O | FIRE | rgmin | 62 | 63 | 6.038885 | 9.886e-03 | 158.0 | 156.6 | yes |
| Pt(111)+O | FIRE2 | ase | 50 | 53 | 6.039083 | 9.515e-03 | 129.0 | 128.5 | yes |
| Pt(111)+O | FIRE2 | rgmin | 25 | 28 | 6.038831 | 9.961e-03 | 72.5 | 72.2 | yes |
| Pt(111)+O | ODE12r | ase | 24 | 25 | 6.039042 | 8.034e-03 | 62.1 | 61.7 | yes |
| LJ38 | BFGS | ase | 93 | 94 | -173.925614 | 6.177e-03 | 164.7 | 164.4 | yes |
| LJ38 | BFGS | rgmin | 93 | 94 | -173.925614 | 6.177e-03 | 111.0 | 110.1 | yes |
| LJ38 | LBFGS | ase | 93 | 94 | -173.925614 | 6.177e-03 | 113.9 | 112.0 | yes |
| LJ38 | LBFGS | rgmin | 93 | 94 | -173.925614 | 6.177e-03 | 109.3 | 108.9 | yes |
| LJ38 | LBFGSLineSearch | ase | 29 | 59 | -173.925613 | 7.029e-03 | 64.2 | 64.0 | yes |
| LJ38 | LBFGSLineSearch | rgmin | 24 | 26 | -173.925608 | 8.537e-03 | 29.3 | 29.1 | yes |
| LJ38 | FIRE | ase | 85 | 86 | -173.925607 | 9.517e-03 | 85.3 | 84.3 | yes |
| LJ38 | FIRE | rgmin | 85 | 86 | -173.925607 | 9.517e-03 | 93.2 | 93.1 | yes |
| LJ38 | FIRE2 | ase | 2000 | 2107 | -20.855947 | 4.844e-02 | 2023.8 | 2020.8 | no |
| LJ38 | FIRE2 | rgmin | 51 | 71 | -173.925605 | 8.766e-03 | 73.1 | 72.9 | yes |
| LJ38 | ODE12r | ase | 34 | 39 | -173.925580 | 9.617e-03 | 37.7 | 37.3 | yes |
| Cu107-vac-cell | BFGS | ase | 41 | 42 | 0.470939 | 9.437e-03 | 2787.2 | 2139.7 | yes |
| Cu107-vac-cell | BFGS | rgmin | 41 | 42 | 0.470939 | 9.437e-03 | 952.1 | 949.6 | yes |
| Cu107-vac-cell | LBFGS | ase | 41 | 42 | 0.470939 | 9.437e-03 | 863.7 | 860.7 | yes |
| Cu107-vac-cell | LBFGS | rgmin | 41 | 42 | 0.470939 | 9.437e-03 | 937.0 | 929.8 | yes |
| Cu107-vac-cell | LBFGSLineSearch | ase | 26 | 77 | 0.470807 | 7.416e-03 | 1398.4 | 1395.4 | yes |
| Cu107-vac-cell | LBFGSLineSearch | rgmin | 38 | 40 | 0.470715 | 8.689e-03 | 883.8 | 879.8 | yes |
| Cu107-vac-cell | FIRE | ase | 61 | 62 | 0.471151 | 9.141e-03 | 1263.8 | 1258.9 | yes |
| Cu107-vac-cell | FIRE | rgmin | 61 | 62 | 0.471151 | 9.141e-03 | 1374.4 | 1370.5 | yes |
| Cu107-vac-cell | FIRE2 | ase | 48 | 50 | 0.471083 | 8.700e-03 | 1087.9 | 1082.6 | yes |
| Cu107-vac-cell | FIRE2 | rgmin | 54 | 57 | 0.471206 | 9.890e-03 | 1254.3 | 1252.8 | yes |
| Cu107-vac-cell | ODE12r | ase | 71 | 73 | 0.471193 | 9.765e-03 | 1336.6 | 1336.2 | yes |
