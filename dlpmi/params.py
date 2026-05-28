"""
dlpmi/params.py
===============
PyTorch parameter classes for DLPMI inversion.

DLPMIModel  : single-component model (PFM / EMM / EPM / DM / PEM)
BMMModel    : binary mixing of any two single-component models

All learnable parameters are sigmoid-bounded to enforce physical
constraints without a constrained solver.
"""

import torch
import torch.nn as nn
import numpy as np
from .kernels import VALID_MODELS


# ── sigmoid helpers ───────────────────────────────────────────

def _to_b(r: torch.Tensor,
          lo: float, hi: float) -> torch.Tensor:
    return lo + (hi - lo) * torch.sigmoid(r)


def _logit(v: float, lo: float, hi: float) -> float:
    p = float(np.clip((v - lo) / (hi - lo), 1e-4, 1. - 1e-4))
    return float(np.log(p / (1. - p)))


# ════════════════════════════════════════════════════════════════
# SINGLE-COMPONENT MODEL PARAMETER CLASS
# ════════════════════════════════════════════════════════════════

class DLPMIModel(nn.Module):
    """
    Learnable parameter class for a single-component LPM.

    Supported models and their free parameters
    -------------------------------------------
    PFM  : tau
    EMM  : tau
    EPM  : tau, eta   (eta = mixing fraction, 0 < η ≤ 1)
    DM   : tau, PD    (PD = dispersion parameter)
    PEM  : tau, eta   (alias for EPM)

    All parameters are sigmoid-reparameterised so that physical bounds
    are guaranteed throughout optimisation.

    Parameters
    ----------
    model_name  : "PFM", "EMM", "EPM", "DM", or "PEM"
    tau0        : initial mean age (yr)
    tau_lo      : lower bound for τ (yr)
    tau_hi      : upper bound for τ (yr)
    PD0         : initial dispersion parameter  [DM only]
    pd_lo       : lower bound for P_D           [DM only]
    pd_hi       : upper bound for P_D           [DM only]
    eta0        : initial η                     [EPM / PEM only]
    eta_lo      : lower bound for η             [EPM / PEM only]
    eta_hi      : upper bound for η             [EPM / PEM only]

    Examples
    --------
    # DM model
    m = DLPMIModel("DM", tau0=30., tau_lo=1., tau_hi=200.,
                   PD0=0.1, pd_lo=0.001, pd_hi=2.0)
    params = m.get_params()   # returns {"tau": ..., "PD": ...}

    # EMM model
    m = DLPMIModel("EMM", tau0=10., tau_lo=0.5, tau_hi=100.)
    params = m.get_params()   # returns {"tau": ...}

    # EPM model
    m = DLPMIModel("EPM", tau0=20., tau_lo=1., tau_hi=200.,
                   eta0=0.7, eta_lo=0.05, eta_hi=1.0)
    params = m.get_params()   # returns {"tau": ..., "eta": ...}
    """

    def __init__(self,
                 model_name: str,
                 tau0:  float = 20.,
                 tau_lo: float = 0.5,
                 tau_hi: float = 1e5,
                 PD0:   float = 0.1,
                 pd_lo: float = 0.001,
                 pd_hi: float = 2.0,
                 eta0:  float = 0.7,
                 eta_lo: float = 0.05,
                 eta_hi: float = 1.0):
        super().__init__()

        if model_name not in VALID_MODELS:
            raise ValueError(
                f"Unknown model '{model_name}'. "
                f"Valid: {VALID_MODELS}")

        self.model_name = model_name

        # τ  — always learnable
        self.tau_lo = float(tau_lo); self.tau_hi = float(tau_hi)
        self.r_tau  = nn.Parameter(torch.tensor(
            _logit(float(np.clip(tau0, tau_lo*1.01, tau_hi*0.99)),
                   tau_lo, tau_hi)))

        # P_D — DM / PEM*  (*PEM uses eta, but keep field for safety)
        if model_name in ("DM",):
            self.pd_lo = float(pd_lo); self.pd_hi = float(pd_hi)
            self.r_pd  = nn.Parameter(torch.tensor(
                _logit(float(np.clip(PD0, pd_lo*1.01, pd_hi*0.99)),
                       pd_lo, pd_hi)))
        else:
            self.r_pd = None

        # η — EPM / PEM
        if model_name in ("EPM", "PEM"):
            self.eta_lo = float(eta_lo); self.eta_hi = float(eta_hi)
            self.r_eta  = nn.Parameter(torch.tensor(
                _logit(float(np.clip(eta0, eta_lo*1.01, eta_hi*0.99)),
                       eta_lo, eta_hi)))
        else:
            self.r_eta = None

    def get_params(self) -> dict:
        """Return dict of current physical parameter values as Tensors."""
        out = {"tau": _to_b(self.r_tau, self.tau_lo, self.tau_hi)}
        if self.r_pd  is not None:
            out["PD"]  = _to_b(self.r_pd,  self.pd_lo,  self.pd_hi)
        if self.r_eta is not None:
            out["eta"] = _to_b(self.r_eta, self.eta_lo, self.eta_hi)
        return out


# ════════════════════════════════════════════════════════════════
# BMM PARAMETER CLASS
# ════════════════════════════════════════════════════════════════

class BMMModel(nn.Module):
    """
    Learnable parameter class for a Binary Mixing Model.

    Any two single-component models (model1, model2) can be mixed.
    The free parameters are controlled by the `free_params` argument,
    which mirrors the TracerLPM Step 2 optimisation modes:

    Mode A  "Mean Age, Fraction"  — optimise τ₁ and f₁; τ₂ fixed
    Mode B  "Fraction"            — optimise f₁ only; τ₁ and τ₂ fixed
    Mode C  "Fraction, 2nd Mean Age" — optimise f₁ and τ₂; τ₁ fixed

    The shape parameters (P_D, η) of both components are always fixed
    at their initial values, matching TracerLPM convention.

    Parameters
    ----------
    model1, model2  : model name strings for the two components
    free_params     : which parameters to optimise (see modes above)
    params1_init    : dict of initial values for component 1
                      (must include "tau"; optionally "PD" or "eta")
    params2_init    : dict of initial values for component 2
    f1_0            : initial mixing fraction of component 1
    tau1_lo/hi      : bounds for τ₁
    f1_lo/hi        : bounds for f₁
    tau2_lo/hi      : bounds for τ₂ (used when τ₂ is free)

    Examples
    --------
    # BMM-DM-DM, free params = tau1 and f1 (Mode A)
    m = BMMModel("DM", "DM",
                 free_params    = "Mean Age, Fraction",
                 params1_init   = {"tau": 20., "PD": 0.04},
                 params2_init   = {"tau": 12100., "PD": 0.1},
                 f1_0           = 0.12,
                 tau1_lo=1.,    tau1_hi=100.,
                 tau2_lo=100.,  tau2_hi=60000.)

    # BMM-EMM-PFM, free params = tau1 and f1
    m = BMMModel("EMM", "PFM",
                 free_params    = "Mean Age, Fraction",
                 params1_init   = {"tau": 8.},
                 params2_init   = {"tau": 0.1},
                 f1_0           = 0.70)
    """

    def __init__(self,
                 model1: str,
                 model2: str,
                 free_params: str = "Mean Age, Fraction",
                 params1_init: dict = None,
                 params2_init: dict = None,
                 f1_0:   float = 0.5,
                 tau1_lo: float = 0.5,
                 tau1_hi: float = 500.,
                 f1_lo:   float = 0.005,
                 f1_hi:   float = 0.997,
                 tau2_lo: float = 1.,
                 tau2_hi: float = None):
        super().__init__()

        for mn in (model1, model2):
            if mn not in VALID_MODELS:
                raise ValueError(f"Unknown model '{mn}'. Valid: {VALID_MODELS}")

        self.model1 = model1
        self.model2 = model2
        self._free  = str(free_params or "Mean Age, Fraction").lower()

        if params1_init is None: params1_init = {"tau": 20., "PD": 0.1}
        if params2_init is None: params2_init = {"tau": 1000., "PD": 0.1}

        # Fixed shape parameters (always kept at init values)
        self._fixed1 = {k: v for k, v in params1_init.items() if k != "tau"}
        self._fixed2 = {k: v for k, v in params2_init.items() if k != "tau"}

        tau1_0 = float(params1_init.get("tau", 20.))
        tau2_0 = float(params2_init.get("tau", 1000.))

        # ── τ₁ ─────────────────────────────────────────────────
        self._tau1_free = "mean age" in self._free
        if self._tau1_free:
            self.tau1_lo = float(tau1_lo); self.tau1_hi = float(tau1_hi)
            self.r_tau1  = nn.Parameter(torch.tensor(
                _logit(float(np.clip(tau1_0, tau1_lo*1.01, tau1_hi*0.99)),
                       tau1_lo, tau1_hi)))
        else:
            self._tau1_val = tau1_0

        # ── f₁ ─────────────────────────────────────────────────
        self._f1_free = "fraction" in self._free
        if self._f1_free:
            self.f1_lo = float(f1_lo); self.f1_hi = float(f1_hi)
            self.r_f1  = nn.Parameter(torch.tensor(
                _logit(float(np.clip(f1_0, f1_lo*1.01, f1_hi*0.99)),
                       f1_lo, f1_hi)))
        else:
            self._f1_val = float(f1_0)

        # ── τ₂ ─────────────────────────────────────────────────
        self._tau2_free = "2nd" in self._free or "second" in self._free
        if self._tau2_free:
            if tau2_hi is None:
                tau2_hi = max(tau2_0 * 5., 200.)
            self.tau2_lo = float(tau2_lo); self.tau2_hi = float(tau2_hi)
            self.r_tau2  = nn.Parameter(torch.tensor(
                _logit(float(np.clip(tau2_0, tau2_lo*1.01, tau2_hi*0.99)),
                       tau2_lo, tau2_hi)))
        else:
            self._tau2_val = tau2_0

    def get_params(self) -> tuple:
        """
        Returns (params1, f1, params2) as (dict, Tensor, dict).

        params1 and params2 are suitable for passing directly to
        forward_BMM as `params1` and `params2`.
        """
        # τ₁
        tau1 = (_to_b(self.r_tau1, self.tau1_lo, self.tau1_hi)
                if self._tau1_free else torch.tensor(self._tau1_val))
        # f₁
        f1   = (_to_b(self.r_f1, self.f1_lo, self.f1_hi)
                if self._f1_free else torch.tensor(self._f1_val))
        # τ₂
        tau2 = (_to_b(self.r_tau2, self.tau2_lo, self.tau2_hi)
                if self._tau2_free else torch.tensor(self._tau2_val))

        params1 = {"tau": tau1, **{k: torch.tensor(float(v))
                                   for k, v in self._fixed1.items()}}
        params2 = {"tau": tau2, **{k: torch.tensor(float(v))
                                   for k, v in self._fixed2.items()}}
        return params1, f1, params2
