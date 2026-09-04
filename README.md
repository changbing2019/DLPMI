# DLPMI — Differentiable Lumped-Parameter Model Inversion

Lumped-parameter model (LPM) inversion for environmental tracer data, using
automatic differentiation rather than finite-difference Jacobians, with
identifiability, tracer information-content, model-structure and Bayesian
diagnostics.

This release accompanies a manuscript on the identifiability and information
content of groundwater ages estimated from environmental tracers in a karst
aquifer, applied to 60 samples from the San Antonio segment of the Edwards
Aquifer, Texas.

---

## Version 2.0.0 — results from 1.0.0 should not be used

Three defects in 1.0.0 changed numerical output. They are corrections, not
choices, which is why this is a major version. See `CHANGELOG.md` for the
detail and the verification of each.

| | 1.0.0 | 2.0.0 |
|---|---|---|
| free-parameter matching | substring — freed a parameter that Mode C fixes, for 8 samples | exact token match |
| Hessian covariance | `H⁻¹·(χ²/dof)` — every σ low by √2 | the factor of 2 applied |
| lower interval bound | reversed bisection — quantised to the search step | named `inside`/`outside`, roles asserted |

---

## Start here

```
pip install -r requirements.txt
python verify_setup.py
```

`verify_setup.py` must print **τ₁ = 29.57 yr, χ² = 0.045** for EDTRPAS1-36. If
it does not, the package tree does not match the one the published results came
from and nothing downstream is comparable.

Then follow `REPRODUCE.md`, which gives the ordered pipeline and the exact
command for each stage.

## Layout

| path | contents |
|---|---|
| `dlpmi/` | the package: forward models, kernels, inverter, parameters, uncertainty, metrics |
| `examples/edwards_aquifer/` | `edwards_run.py` and the input configuration — **`edwards_input_config.json` is the authoritative input**; every published number derives from it |
| `analysis/` | the analysis chain: identifiability and leave-one-tracer-out sweeps, true profile likelihood, the parameterization exchanges, Monte Carlo, mean-age uncertainty, the scale-marginalised Bayesian posterior and its grid-convergence study |
| `scripts/` | figure and table generators, and `verify_manuscript_numbers.py` |
| `reference/00_summary_table.csv` | the baseline fit every script reads |
| `results/` | sweep outputs (`sweeps_trueprofile/`) and Monte Carlo draws (`mc/`, `mc_tau2free/`) |
| `figures/` | the published figures and table data |
| `deprecated/` | `edwards_pinn_estimate.py`, which does not run; see its header |

## Verification

`scripts/verify_manuscript_numbers.py` restates every quantitative claim of the
manuscript as an assertion against the file it should come from, and fails if a
retired value reappears. Exit 0 means every claim reproduces.

It reads the manuscript and Supporting Information `.docx` files, which are
**not** in this release — they are added after acceptance. Until then the
data-derived checks run and the document checks are skipped.

## Two things to know about the data

**The input tracer values are not the analytical measurements.** All 218 active
tracer–sample observations and all 218 standard errors in
`edwards_input_config.json` reproduce `LPM_Meas_Tracer_NN` and
`LPM_Meas_Tracer_NN_Err` in `Table_4_LPM.txt` of the source data release
exactly. Those are the inputs to the original published inversion, not the
analytical measurements tabulated separately in Tables 3 and 5 of the same
release. A reproducer diffing against Tables 3 and 5 will find discrepancies
that are not errors. Note the file is UTF-16.

**`profile_true_loto.csv` carries retired columns.** Its `*_cond` columns
belong to a superseded conditional-χ² run. Use the `*_true` columns. Do not mix
them.

## Source data

Musgrove, M., Jurgens, B. C., and Opsahl, S. P. (2023), *Data Release for
Recharge and Groundwater Vulnerability Analysis in the Edwards Aquifer, Texas*,
U.S. Geological Survey. https://doi.org/10.5066/P9CWM574

Not redistributed here. `REPRODUCE.md` section 0 explains which table to read
and what happens if you read the wrong one.

## Citation

See `CITATION.cff`. Please cite the archived release DOI rather than this
repository URL, so that the version you used is unambiguous.

## Licence

MIT — see `LICENSE`.
