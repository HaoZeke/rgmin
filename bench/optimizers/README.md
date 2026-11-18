# Optimizer coverage harness

This harness compares capped L-BFGS, FIRE 2.0, and steepest descent from
rgmin with ASE L-BFGS and FIRE and with eOn L-BFGS, FIRE, and steepest
descent. Variable-cell rows compare rgmin and ASE L-BFGS through the same
`FrechetCellFilter`; the eOn client does not expose variable-cell
relaxation.

The fixed protocol uses release builds, one pinned core and one thread, 12
interleaved order-flipped rounds, no warm-up, and no rescue reruns. Every LJ
seed is run 12 times. Potential dispatch is pinned to scalar outside the harness.
Optimizers keep their documented defaults. The only matched
optimizer setting is the stopping rule: maximum atomic force
0.01 eV/A, with maximum stress 0.001 eV/A^3 for cell relaxation. The
predeclared iteration budget is 1000 and final-energy agreement is tested at
0.001 eV.

LJ38 and LJ75 use ten deterministic 0.02 A perturbations of the committed
minima and `LJPot` in the calculator's open-cell geometry. The Cu255 vacancy and Cu108 cell use `MorsePot`
with `De=0.3429`, `a=1.3588`, `re=2.866`, a 5.0 A cutoff, and a 1.0 A
switch width. Si64 uses the published `StillingerWeberSi` defaults. The
cell systems start under 2% isotropic strain. Every implementation reaches
the potential through `RgpotCalculator`; eOn uses its generic ASE-potential
client binding.

The benchmark command requires every input:

```console
python bench/optimizers/run.py \
  --fixtures bench/optimizers/data \
  --output bench/optimizers/results.json \
  --rounds 12 \
  --cores 0 \
  --thread-count 1 \
  --fmax 0.01 \
  --stress-max 0.001 \
  --max-steps 1000 \
  --energy-tolerance 0.001 \
  --rgmin-sha <source-sha> \
  --rgpot-tag <source-tag> \
  --rgpot-sha <source-sha> \
  --rgpot-patch-sha <source-sha> \
  --eon-tag <source-tag> \
  --eon-sha <source-sha> \
  --eon-rgpot-sha <source-sha> \
  --rgmin-wheel <wheel-file> \
  --rgpot-wheel <wheel-file> \
  --eon-wheel <wheel-file> \
  --component linkcell=<version> \
  --component minimage=<version> \
  --component vesin=<version> \
  --component readcon-core=<version> \
  --component rgsaddle=not-used \
  --rust-version <compiler-version> \
  --cxx-version <compiler-version>
```

`run.py` creates `results.json`, the raw benchmark artifact. Its top-level
keys are `schema`, `protocol`, `sources`, `artifacts`, `build`, `runtime`,
`gaps`, and `records`; it never writes analysis keys. The artifact section
contains hashes computed from the three supplied wheel archives after their
distribution names and versions are checked against the installed packages.
The runtime section records CPU model, architecture, logical CPU counts,
memory, package versions, and operating-system version, but no hostname.

The recorded wall times are 12 measured repetitions per cell on the machine
described by that runtime header. They support comparisons within this run;
they are not single-run speedup claims. Nonconverged eOn endpoints have null
energy and force values because only their convergence verdict and force-call
budget are reproducible and compared.

Analysis is a separate deterministic pass over the untouched raw records:

```console
python bench/optimizers/analyze.py \
  --input bench/optimizers/results.json \
  --output bench/optimizers/results-analyzed.json \
  --markdown bench/optimizers/results.md \
  --energy-tolerance 0.001
```

`analyze.py` refuses in-place output and raw input that already contains
analysis keys. `results-analyzed.json` copies all raw provenance and adds
`analysis`, `summaries`, `comparisons`, and `failures`; `results.md` is the
short rendered table. The analyzed JSON includes medians, interquartile
ranges, odd/even A/A noise bands per cell, Mann-Whitney p-values for
continuous metrics, Fisher exact p-values for success rates, and paired
final-energy agreement. Repeating the two commands with the same inputs
reproduces both committed JSON documents byte for byte except for raw timing
fields and statistics derived from them.

The variable-cell failure can be reproduced independently:

```console
python bench/optimizers/repro_cell.py \
  --system Cu108-cell \
  --fixtures bench/optimizers/data \
  --steps 200 \
  --fmax 0.01 \
  --stress-max 0.001
```
