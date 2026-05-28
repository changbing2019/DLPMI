# DLPMI — Differentiable LPM Inversion

**Open-source Python framework for groundwater age dating using Lumped Parameter Models and automatic differentiation.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![PyTorch 2.0+](https://img.shields.io/badge/PyTorch-2.0+-orange.svg)](https://pytorch.org/)

---

## What is DLPMI?

DLPMI implements Lumped Parameter Model (LPM) inversion for interpreting
environmental tracer data in groundwater using **automatic differentiation**
(autodiff) rather than finite-difference Jacobians.

Unlike Physics-Informed Neural Networks (Raissi et al. 2019), DLPMI has
**no hidden layers** — the learnable variables are the physical model
parameters themselves (τ, P_D, f₁). The governing equations are embedded
entirely in the forward model, not as a loss-function residual.
The method is therefore described as **differentiable inverse modelling**
rather than a PINN in the strict sense.

---

## Supported models

| Model | Name | Parameters |
|-------|------|-----------|
| PFM | Piston Flow Model | τ |
| EMM | Exponential Mixing Model | τ |
| EPM | Exponential-Piston Flow Model | τ, η |
| DM  | Dispersion Model | τ, P_D |
| PEM | Partial Exponential Model (≡ EPM) | τ, η |
| **BMM** | **Any two models above mixed** | f₁ + params of each component |

**Any binary combination is supported**: BMM-DM-DM, BMM-EMM-PFM,
BMM-DM-EMM, BMM-EPM-DM, etc.

---

## Supported tracers

| Tracer | Unit | Input function |
|--------|------|----------------|
| ³H | TU | GNIP N-hemisphere precipitation |
| ³He(trit) | TU | In-situ accumulation from ³H decay |
| SF₆ | pptv | NOAA/ESRL atmospheric + Bullister (2002) solubility + DGMETA |
| ¹⁴C | pmC | IntCal09 + NH Zone 2 bomb curve |
| ⁴He | cc STP/g | Linear accumulation at user-supplied rate |

---

## Installation

```bash
# From source
git clone https://github.com/your-org/DLPMI.git
cd DLPMI
pip install -e .

# With plotting support
pip install -e ".[plot]"
```

---

## Quick start

```python
from dlpmi import Inverter, Sample

# ── Single-component DM ──────────────────────────────────────
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
print(f"tau = {result['params']['tau']:.2f} yr")
print(f"PD  = {result['params']['PD']:.4f}")
print(f"chi2 = {result['chi2']:.5f}")
```

```python
# ── BMM-DM-DM ────────────────────────────────────────────────
s2 = Sample(
    sample_id   = "Mixed_Well",
    sample_date = 2018.5,
    tracers = [
        {"name": "3H",  "obs": 1.2,  "scale": 1.0},
        {"name": "14C", "obs": 35.0, "scale": 0.75},
    ],
    model       = "BMM",
    model1      = "DM",    # young component
    model2      = "DM",    # old component
    free_params = "Mean Age, Fraction",
    bounds      = {
        "tau1_0": 20., "tau1_lo": 5., "tau1_hi": 100.,
        "pd1":    0.05,                   # fixed P_D young
        "tau2":   10000.,                 # fixed old mean age
        "pd2":    0.1,                    # fixed P_D old
        "f1_0":   0.3, "f1_lo": 0.01, "f1_hi": 0.95,
    },
    dic_c1 = 80., dic_c2 = 80.,
)

result2 = Inverter(n_adam=5000, n_starts=3).fit(s2)
print(f"tau_y = {result2['params']['tau']:.2f} yr")
print(f"f_y   = {result2['f1']*100:.1f}%")
```

```python
# ── BMM-EMM-PFM (Missouri River style) ───────────────────────
s3 = Sample(
    sample_id   = "River_Sample",
    sample_date = 1990.0,
    tracers = [{"name": "3H", "obs": 50., "scale": 1.0}],
    model       = "BMM",
    model1      = "EMM",   # groundwater
    model2      = "PFM",   # prompt runoff
    free_params = "Mean Age, Fraction",
    bounds      = {
        "tau1_0": 8., "tau1_lo": 1., "tau1_hi": 15.,
        "tau2":   0.5,
        "f1_0":   0.7, "f1_lo": 0.5, "f1_hi": 0.97,
    },
)
result3 = Inverter().fit(s3)
```

---

## Package structure

```
DLPMI/
├── dlpmi/
│   ├── __init__.py       ← public API
│   ├── tracers.py        ← tracer input functions (³H, SF₆, ¹⁴C, ⁴He)
│   ├── kernels.py        ← exit-age PDFs (PFM, EMM, EPM, DM, PEM)
│   ├── forward.py        ← forward_single, forward_BMM
│   ├── params.py         ← DLPMIModel, BMMModel parameter classes
│   ├── inverter.py       ← Sample dataclass + Inverter class
│   └── uncertainty.py    ← Hessian, profile chi2, Monte Carlo
├── examples/
│   ├── quickstart.py              ← Cases 1–3 from TracerLPM manual
│   └── edwards_aquifer/
│       ├── edwards_run.py         ← Edwards Aquifer application
│       └── edwards_input_config.json
├── tests/
│   └── test_kernels.py
├── setup.py
└── README.md
```

---

## Uncertainty quantification

Three methods are available after fitting:

```python
from dlpmi.uncertainty import (chi2_probability,
                                 hessian_uncertainty,
                                 profile_uncertainty,
                                 mc_uncertainty)

# Chi-square probability (TracerLPM LPM_Prob equivalent)
prob, dof = chi2_probability(result["chi2"], n_active=3, n_free=2)

# Hessian-based 1-sigma errors (exact via torch.autograd)
sigmas, cov, dof = hessian_uncertainty(loss_fn, params_opt,
                                        result["chi2"], 3, 2)

# Profile 68% CI
lo68, hi68 = profile_uncertainty(loss_fn_1d, tau_opt,
                                   result["chi2"], 1., 200.)
```

---

## Application: Edwards Aquifer

The Edwards Aquifer (Balcones Fault Zone, Texas) application
processes 65 groundwater samples from Musgrove et al. (2023) and
compares DLPMI results with TracerLPM. See `examples/edwards_aquifer/`.

```bash
cd examples/edwards_aquifer
python edwards_run.py --adam 5000 --starts 3 --mc 50
```

---

## References

- Jurgens et al. (2012) TracerLPM, USGS TM 4-F3
- Jurgens et al. (2020) DGMETA, USGS TM 4-F5
- Bullister et al. (2002) Deep-Sea Research I, 49, 175–187
- Musgrove et al. (2023) Geophysical Research Letters, 50, e2023GL102853
- Maloszewski & Zuber (1982) J. Hydrology, 57, 207–231
- Reimer et al. (2009) Radiocarbon, 51(4), 1111–1150
