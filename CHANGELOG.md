# Changelog

## 2.0.0 — 2026-09

**Results produced with 1.0.0 are not reproducible under this release, and
should not be used.** Three defects in 1.0.0 changed numerical output. They
are corrections, not choices, which is why this is a major version.

### Corrected

- `dlpmi/params.py` — free-parameter names were matched by substring, so
  `"mean age"` matched inside `"2nd Mean Age"` and the modern-component
  transit time was freed where TracerLPM Mode C holds it fixed. This changed
  the free-parameter count, the degrees of freedom, and therefore every
  chi-square probability for the 8 affected samples in the Edwards
  configuration. Now an exact token match on the comma-split list.
- `dlpmi/uncertainty.py` — the covariance was formed as `H⁻¹·(χ²/dof)`. For
  χ² = Σr² the Gauss–Newton Hessian is H ≈ 2JᵀJ, so a factor of 2 is
  required. **Every Hessian standard deviation from 1.0.0 is low by
  √2 ≈ 1.414.** Verified against a closed-form OLS problem to eight
  significant figures.
- `dlpmi/uncertainty.py` — the lower-side interval search called
  `_bisect(lo68, lo68 + step)` with the endpoints reversed. Reversed it does
  not fail; it marches one endpoint onto the other and returns it, so every
  lower bound sat at an exact integer multiple of the coarse step. The
  bisection now takes named `inside`/`outside` arguments and asserts their
  roles.
- `dlpmi/metrics.py` — percentiles were computed without filtering
  non-finite draws, so a single NaN among the Monte Carlo draws returned NaN
  for the whole sample.
- The dissolved-gas metadata wrapper signature never accepted `dgmeta` or
  `uz_tt`, so binary-mixture fits silently used defaulted metadata. SF₆-only
  effect.

### Added

- `dlpmi/metrics.py`, absent from 1.0.0.
- `analysis/` — the full chain: identifiability and leave-one-tracer-out
  sweeps, true profile likelihood, the parameterization exchanges, Monte
  Carlo, mean-age uncertainty, the scale-marginalised Bayesian posterior and
  its grid-convergence study.
- `scripts/` — the figure and table generators, and
  `verify_manuscript_numbers.py`, which restates every quantitative claim of
  the associated manuscript as an assertion against the file it comes from.
- `reference/00_summary_table.csv` — the baseline fit every script reads.
- `results/` — the sweep outputs and Monte Carlo draws.
- `verify_setup.py`, `requirements.txt`, `REPRODUCE.md`, `LICENSE`,
  `CITATION.cff`.

### Moved

- `examples/edwards_aquifer/edwards_pinn_estimate.py` → `deprecated/`. It does
  not run; see the header in that file.

## 1.0.0

Initial public release.
