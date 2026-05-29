# DLPMI — Differentiable LPM Inversion  `v1.1.0`

**A Differentiable Framework for Lumped Parameter Model Inversion and
Uncertainty Quantification in Groundwater Age Dating.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

---

## Overview

DLPMI implements Lumped Parameter Model (LPM) inversion for groundwater
age dating using **automatic differentiation** (PyTorch) rather than
finite-difference Jacobians.  Unlike Physics-Informed Neural Networks
(Raissi et al., 2019), DLPMI has **no hidden layers** — the learnable
variables are the physical model parameters directly (τ, P_D, f₁) and the
governing equations are embedded entirely in the forward model.

### Validated results (Edwards Aquifer, 60 samples)

| Metric | DLPMI | TracerLPM |
|---|---|---|
| χ² probability p > 0.05 | **60/60** | 48/60 |
| Lower χ² | **49/60** | 11/60 |
| ³H mean error (Modern) | 20% | 5% |
| f₁ R² (BMM) | **0.76** | — |

---

## Supported models

| Model | Name | Free parameters |
|---|---|---|
| PFM | Piston Flow | τ |
| EMM | Exponential Mixing | τ |
| EPM | Exponential-Piston Flow | τ, η |
| DM  | Dispersion Model | τ, P_D |
| PEM | Partial Exponential (= EPM) | τ, η |
| **BMM** | **Any two models mixed** | f₁ + component params |

---

## Supported tracers

| Tracer | Unit | Input function | Pre-correction by user? |
|---|---|---|---|
| ³H | TU | GNIP N-hemisphere (calibrated) | None |
| ³He(trit) | TU | In-situ ³H decay | **Yes — run DGMETA first** |
| SF₆ | pptv | NOAA/ESRL + Bullister (2002) + DGMETA | Workflow A or B (see docs) |
| ¹⁴C | pmC | IntCal09 + NH Zone 2 bomb | None |
| ⁴He | cc STP/g | Linear accumulation | None |

---

## ³H latitude calibration — important

The built-in GNIP ³H record is split at 1981 and scaled by latitude
correction factors.  **The default is calibrated for subtropical / semi-arid
sites (≈ 30°N)**:

```python
H3_SCALE_PRE1981  = 0.668   # pre-1981  (all latitudes similar)
H3_SCALE_POST1980 = 0.668   # post-1980 subtropical default
```

For other latitudes, change these before running:

```python
import dlpmi.tracers as tr
# High-latitude site (Ottawa, Vienna; > 45°N):
tr.H3_SCALE_POST1980 = 1.336
# Mid-latitude (≈ 35–45°N):
tr.H3_SCALE_POST1980 = 1.000
# Re-apply calibration
tr._3H_VALS = tr._3H_VALS_RAW.clone()
tr._3H_VALS[:28] *= tr.H3_SCALE_PRE1981
tr._3H_VALS[28:] *= tr.H3_SCALE_POST1980
```

> **Tip:** For a new site, determine the correct scale by back-calculating
> from DM Modern samples whose age is well-constrained by SF₆ and ³He(trit).
> See Section 8.3 of the companion paper.

---

## Installation

```bash
git clone https://github.com/changbing2019/DLPMI.git
cd DLPMI
pip install -e .
pip install -e ".[plot]"   # with matplotlib
```

---

## Quick start

```python
from dlpmi import Inverter, Sample

# ── Single-component DM ─────────────────────────────────────────────────────
s = Sample(
    sample_id   = "My_Well",
    sample_date = 2020.0,
    tracers = [
        {"name": "3H",  "obs": 5.2,  "scale": 1.0},
        {"name": "SF6", "obs": 3.1,  "scale": 1.0},
        {"name": "14C", "obs": 72.0, "scale": 0.85},
    ],
    model  = "DM",
    bounds = {"tau_lo": 1, "tau_hi": 200,
              "pd_lo": 0.001, "pd_hi": 2.0},
)
result = Inverter(n_adam=5000, n_starts=3).fit(s)
print(f"τ = {result['params']['tau']:.1f} yr,  P_D = {result['params']['PD']:.4f}")
print(f"χ² = {result['chi2']:.4f}")
```

```python
# ── BMM-DM-DM ───────────────────────────────────────────────────────────────
s2 = Sample(
    sample_id   = "Mixed_Well",
    sample_date = 2018.5,
    tracers = [
        {"name": "3H",  "obs": 1.2,  "scale": 1.0},
        {"name": "14C", "obs": 35.0, "scale": 0.75},
    ],
    model       = "BMM",
    model1      = "DM",
    model2      = "DM",
    free_params = "Mean Age, Fraction",
    bounds      = {
        "tau1_0": 20., "tau1_lo": 5., "tau1_hi": 100.,
        "pd1":    0.05,
        "tau2":   10000.,
        "pd2":    0.1,
        "f1_0":   0.3, "f1_lo": 0.01, "f1_hi": 0.95,
    },
    dic_c1 = 80., dic_c2 = 80.,
)
result2 = Inverter().fit(s2)
print(f"τ₁ = {result2['params']['tau']:.1f} yr,  f₁ = {result2['f1']*100:.1f}%")
```

```python
# ── Custom ³H record (e.g., Albuquerque NM from TracerLPM file) ─────────────
import torch
from dlpmi import Sample, Inverter

alb_yrs  = torch.tensor([1953., 1960, 1963, 1970, 1975, 1980, 1985,
                          1990, 1995, 2000, 2005, 2010, 2015, 2018])
alb_vals = torch.tensor([  35.,   74, 1265,  120,   80,   31,  16.3,
                             11,   13,  8.71,  8.0, 10.0,  9.37, 9.15])
s3 = Sample(
    sample_id   = "Well_Albuquerque",
    sample_date = 2007.5,
    tracers     = [{"name": "3H", "obs": 2.35, "scale": 1.0}],
    model       = "DM",
    bounds      = {"tau_lo": 10., "tau_hi": 50.},
    h3_record   = (alb_yrs, alb_vals),   # custom record
)
result3 = Inverter().fit(s3)
```

---

## Package structure

```
DLPMI/
├── dlpmi/
│   ├── __init__.py       ← public API
│   ├── tracers.py        ← ³H (GNIP + custom), SF₆, ¹⁴C, ⁴He
│   ├── kernels.py        ← g_PFM, g_EMM, g_EPM, g_DM, g_PEM
│   ├── forward.py        ← forward_single, forward_BMM
│   ├── params.py         ← DLPMIModel, BMMModel
│   ├── inverter.py       ← Sample, Inverter
│   └── uncertainty.py    ← hessian, profile, Monte Carlo
├── examples/
│   ├── quickstart.py              ← TracerLPM Cases 1–3
│   └── edwards_aquifer/
│       ├── edwards_run.py         ← 60-sample Edwards application
│       └── edwards_input_config.json
├── setup.py
└── README.md
```

---

## Uncertainty quantification

```python
from dlpmi.uncertainty import (chi2_probability, hessian_uncertainty,
                                profile_uncertainty, mc_uncertainty)

# Chi-square probability (TracerLPM LPM_Prob equivalent)
prob, dof = chi2_probability(result["chi2"], n_active=3, n_free=2)

# Exact Hessian-based 1-sigma errors
sigmas, cov, dof = hessian_uncertainty(loss_fn, params_opt,
                                        result["chi2"], 3, 2)

# Profile 68% CI (asymmetric)
lo68, hi68 = profile_uncertainty(loss_fn_1d, tau_opt,
                                   result["chi2"], 1., 200.)
```

---

## Changelog

### v1.1.0
- **Critical fix:** ³H post-1980 latitude calibration corrected from ×1.336
  to ×0.668 for subtropical sites (Edwards Aquifer, Texas).  This reduced
  Modern ³H mean error from 90% → 20% and improved BMM f₁ R² from 0.25 → 0.76.
- Pre-1953 ³H background lowered from 5.0 → 2.0 TU for subtropical sites.
- Added `input_3H_custom()` for user-supplied regional ³H records.
- Added `h3_record` field to `Sample` dataclass.
- Module-level constants `H3_SCALE_PRE1981` / `H3_SCALE_POST1980` exposed
  for easy latitude override.
- Improved docstrings documenting tracer input requirements for new users.

### v1.0.0
- Initial release: PFM, EMM, EPM, DM, PEM kernels; BMM; SF₆ DGMETA;
  Hessian, profile, and Monte Carlo uncertainty.

---

## References

- Jurgens et al. (2012) TracerLPM, USGS TM 4-F3
- Jurgens et al. (2020) DGMETA, USGS TM 4-F5
- Bullister et al. (2002) Deep-Sea Research I, 49, 175–187
- Musgrove et al. (2023) GRL, doi:10.1029/2023GL102853
- Maloszewski & Zuber (1982) J. Hydrology, 57, 207–231
- Reimer et al. (2009) Radiocarbon, 51(4), 1111–1150
- Raissi et al. (2019) J. Computational Physics, 378, 686–707
