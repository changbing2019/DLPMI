"""
DLPMI — Differentiable LPM Inversion
=====================================
An open-source Python package for groundwater age dating using
Lumped Parameter Models (LPMs) and automatic differentiation.

Supported single models
-----------------------
  PFM   Piston Flow Model
  EMM   Exponential Mixing Model
  EPM   Exponential-Piston Flow Model
  DM    Dispersion Model
  PEM   Partial Exponential Model  (generalised EPM)

Binary Mixing Models
--------------------
Any two single models can be combined as a BMM:
  BMM(model1, model2)
  e.g. BMM-DM-DM, BMM-EMM-PFM, BMM-DM-EMM ...

Quick start
-----------
    from dlpmi import Inverter, Sample

    s = Sample(
        sample_id   = "My_Well",
        sample_date = 2020.0,
        tracers     = [
            {"name": "3H",  "obs": 5.2,  "scale": 1.0},
            {"name": "SF6", "obs": 3.1,  "scale": 1.0},
            {"name": "14C", "obs": 72.0, "scale": 0.85},
        ],
        model = "DM",
        bounds = {"tau_lo": 1, "tau_hi": 200, "pd_lo": 0.001, "pd_hi": 2.0},
    )

    inv = Inverter(n_adam=5000, n_starts=3)
    result = inv.fit(s)
    print(result)

References
----------
Jurgens et al. (2012) TracerLPM, USGS TM 4-F3
Bullister et al. (2002) SF6 solubility
Reimer et al. (2009) IntCal09
"""

from .tracers   import (input_3H, input_SF6, input_14C, input_4He,
                         LAMBDA_3H, LAMBDA_14C)
from .kernels   import (g_PFM, g_EMM, g_EPM, g_DM, g_PEM,
                         AGES_YOUNG, AGES_OLD, choose_ages)
from .forward   import forward_single, forward_BMM
from .params    import DLPMIModel, BMMModel
from .inverter  import Inverter, Sample
from .uncertainty import (chi2_probability, hessian_uncertainty,
                           profile_uncertainty, mc_uncertainty)
from .metrics   import (young_fraction, mixture_mean_age, delta_method_sigma,
                         gaussian_propagate, mc_metric, exit_age_grid)

__version__ = "2.0.0"
__all__ = [
    # Tracer functions
    "input_3H", "input_SF6", "input_14C", "input_4He",
    "LAMBDA_3H", "LAMBDA_14C",
    # LPM kernels
    "g_PFM", "g_EMM", "g_EPM", "g_DM", "g_PEM",
    "AGES_YOUNG", "AGES_OLD", "choose_ages",
    # Forward models
    "forward_single", "forward_BMM",
    # Parameter / model classes
    "DLPMIModel", "BMMModel",
    # Top-level API
    "Inverter", "Sample",
    # Uncertainty
    "chi2_probability", "hessian_uncertainty",
    "profile_uncertainty", "mc_uncertainty",
    # Derived metrics
    "young_fraction", "mixture_mean_age", "delta_method_sigma",
    "gaussian_propagate", "mc_metric", "exit_age_grid",
]

from .inverter import print_device_info, DEVICE
