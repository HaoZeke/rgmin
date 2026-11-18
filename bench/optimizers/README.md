# Optimizer coverage harness

This harness compares capped L-BFGS, FIRE 2.0, and steepest descent from
rgmin with ASE L-BFGS and FIRE and with eOn L-BFGS, FIRE, and steepest
descent. Variable-cell rows compare rgmin and ASE L-BFGS through the same
`FrechetCellFilter`; the eOn client does not expose variable-cell
relaxation.

The fixed protocol uses release builds, one pinned core and one thread, 12
order-flipped rounds, no warm-up, and no rescue reruns. Every LJ seed is run
12 times. Potential dispatch is pinned to scalar outside the harness.
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
  --rgpot-sha <source-sha> \
  --rgpot-patch-sha <source-sha> \
  --eon-tag <source-tag> \
  --eon-sha <source-sha> \
  --eon-rgpot-sha <source-sha> \
  --rgmin-wheel-sha256 <wheel-sha256> \
  --rgpot-wheel-sha256 <wheel-sha256> \
  --eon-wheel-sha256 <wheel-sha256> \
  --component linkcell=<version> \
  --component minimage=<version> \
  --component vesin=<version> \
  --component readcon-core=<version> \
  --component rgsaddle=not-used \
  --rust-version <compiler-version> \
  --cxx-version <compiler-version>
```

Analysis is a separate deterministic pass over the completed records:

```console
python bench/optimizers/analyze.py \
  --input bench/optimizers/results.json \
  --output bench/optimizers/results-analyzed.json \
  --markdown bench/optimizers/results.md \
  --energy-tolerance 0.001
```

The analyzed JSON includes medians, interquartile ranges, odd/even A/A
noise bands per cell, Mann-Whitney p-values for continuous metrics, Fisher
exact p-values for success rates, and paired final-energy agreement.

The variable-cell failure can be reproduced independently:

```console
python bench/optimizers/repro_cell.py \
  --system Cu108-cell \
  --fixtures bench/optimizers/data \
  --steps 200 \
  --fmax 0.01 \
  --stress-max 0.001
```
