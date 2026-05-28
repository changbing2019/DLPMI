"""
dlpmi/kernels.py
================
Exit-age probability density functions (PDFs) for all five
single-component Lumped Parameter Models supported by DLPMI.

Each function returns a normalised PDF g(a) on the supplied age grid,
compatible with the convolution integral in forward.py.

Models
------
  PFM  Piston Flow Model         — g(a) = δ(a − τ)
  EMM  Exponential Mixing Model  — g(a) = (1/τ) exp(−a/τ)
  EPM  Exponential-Piston Model  — η fraction: PFM; (1−η): EMM
  DM   Dispersion Model          — Maloszewski & Zuber (1982)
  PEM  Partial Exponential Model — generalised EPM with shape param η

Age grids
---------
AGES_YOUNG : 480 nodes, 0.05–500 yr   (for τ ≤ 500 yr)
AGES_OLD   : 600 nodes, 0.1–60,000 yr (for τ > 500 yr; PreModern samples)

References
----------
Maloszewski & Zuber (1982) J. Hydrology, 57, 207–231
Cook & Solomon (1997) J. Hydrology, 191, 245–265
Jurgens et al. (2012) USGS TM 4-F3
"""

import torch
import numpy as np

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

# ════════════════════════════════════════════════════════════════
# AGE GRIDS
# ════════════════════════════════════════════════════════════════
AGES_YOUNG = torch.unique(torch.cat([
    torch.linspace(0.05,   10.,  80),
    torch.linspace(10.,   100., 200),
    torch.linspace(100.,  500., 200),
]))

AGES_OLD = torch.unique(torch.cat([
    torch.linspace(0.1,    1000.,  200),
    torch.linspace(1000., 20000.,  300),
    torch.linspace(20000.,60000.,  100),
]))


def choose_ages(tau: float) -> torch.Tensor:
    """Return AGES_OLD for τ > 500 yr, AGES_YOUNG otherwise."""
    return AGES_OLD if float(tau) > 500. else AGES_YOUNG


def _normalise(g: torch.Tensor, ages: torch.Tensor) -> torch.Tensor:
    """Normalise a PDF to integrate to 1 on the given age grid."""
    area = torch.trapz(g, ages)
    return g / (area + 1e-12)


# ════════════════════════════════════════════════════════════════
# PFM — Piston Flow Model
# ════════════════════════════════════════════════════════════════
def g_PFM(ages: torch.Tensor,
          tau: torch.Tensor,
          **kwargs) -> torch.Tensor:
    """
    Piston Flow Model exit-age PDF.

    Approximated as a narrow Gaussian centred on τ with σ = 0.005τ,
    normalised to unit area.  This is numerically equivalent to a
    delta function δ(a − τ) on a discrete grid.

    Parameters
    ----------
    ages : age grid (yr)
    tau  : mean transit time (yr)

    Returns
    -------
    Normalised PDF g(a)
    """
    sigma = torch.clamp(tau * 0.005, min=1e-3)
    g = torch.exp(-0.5 * ((ages - tau) / sigma) ** 2)
    return _normalise(g, ages)


# ════════════════════════════════════════════════════════════════
# EMM — Exponential Mixing Model
# ════════════════════════════════════════════════════════════════
def g_EMM(ages: torch.Tensor,
          tau: torch.Tensor,
          **kwargs) -> torch.Tensor:
    """
    Exponential Mixing Model exit-age PDF.

    g(a; τ) = (1/τ) exp(−a/τ)

    Represents a fully mixed (stirred tank) system where all ages
    from 0 to ∞ contribute according to an exponential weighting.

    Parameters
    ----------
    ages : age grid (yr)
    tau  : mean transit time (yr)

    Returns
    -------
    Normalised PDF g(a)
    """
    g = (1. / tau) * torch.exp(-ages / tau)
    return _normalise(g, ages)


# ════════════════════════════════════════════════════════════════
# EPM — Exponential-Piston Flow Model
# ════════════════════════════════════════════════════════════════
def g_EPM(ages: torch.Tensor,
          tau: torch.Tensor,
          eta: torch.Tensor,
          **kwargs) -> torch.Tensor:
    """
    Exponential-Piston Flow Model exit-age PDF.

    Combines a piston-flow (unsaturated zone / pipe) component with
    an exponential mixing component:

      g_EPM(a; τ, η) = g_EMM(a − τ(1−η); τη)   for a ≥ τ(1−η)
                     = 0                          for a < τ(1−η)

    where η (0 < η ≤ 1) is the ratio of the exponential volume to
    total volume. When η = 1 the EPM reduces to the EMM; when η → 0
    it approaches the PFM.

    Parameters
    ----------
    ages : age grid (yr)
    tau  : mean transit time of the full system (yr)
    eta  : mixing fraction (0 < η ≤ 1)

    Returns
    -------
    Normalised PDF g(a)
    """
    tau_emm = tau * eta                 # mean age of the EMM part
    t_shift = tau * (1. - eta)         # PFM delay
    # Shift the EMM: only ages > t_shift contribute
    a_shifted = ages - t_shift
    valid = (a_shifted > 0).float()
    g = valid * (1. / (tau_emm + 1e-9)) * torch.exp(
        -torch.clamp(a_shifted, min=0.) / (tau_emm + 1e-9))
    return _normalise(g, ages)


# ════════════════════════════════════════════════════════════════
# DM — Dispersion Model
# ════════════════════════════════════════════════════════════════
def g_DM(ages: torch.Tensor,
         tau: torch.Tensor,
         PD: torch.Tensor,
         **kwargs) -> torch.Tensor:
    """
    Dispersion Model exit-age PDF.

    g(a; τ, P_D) = (4π P_D a/τ)^{-½} · a^{-1} · exp[−(1 − a/τ)² / (4 P_D a/τ)]

    Describes 1-D advective-dispersive transport in a homogeneous medium.
    P_D = D/(vL) is the dimensionless dispersion parameter.

    Parameters
    ----------
    ages : age grid (yr)
    tau  : mean transit time (yr)
    PD   : dispersion parameter (dimensionless; 0.001 – 2.0)

    Returns
    -------
    Normalised PDF g(a)

    References
    ----------
    Maloszewski & Zuber (1982) J. Hydrology 57, 207–231
    """
    eps   = 1e-9
    z     = torch.clamp(ages / tau, min=eps)
    log_g = (-torch.log(ages + eps)
             - 0.5 * torch.log(4. * np.pi * PD * z + eps)
             - (1. - z) ** 2 / (4. * PD * z + eps))
    g = torch.exp(log_g)
    return _normalise(g, ages)


# ════════════════════════════════════════════════════════════════
# PEM — Partial Exponential Model  (generalised EPM)
# ════════════════════════════════════════════════════════════════
def g_PEM(ages: torch.Tensor,
          tau: torch.Tensor,
          eta: torch.Tensor,
          **kwargs) -> torch.Tensor:
    """
    Partial Exponential Model exit-age PDF.

    The PEM is the generalised form of the EPM and is equivalent to it
    in the TracerLPM implementation.  The alias is provided for
    compatibility with TracerLPM naming conventions.

    See g_EPM for full documentation.
    """
    return g_EPM(ages, tau, eta, **kwargs)


# ════════════════════════════════════════════════════════════════
# MODEL REGISTRY  — used by forward.py and params.py
# ════════════════════════════════════════════════════════════════
#: Maps model name string → (kernel_function, required_params)
#: required_params: list of parameter names that the kernel expects
#:   beyond 'tau' (all kernels receive tau; the extra params listed here
#:   are the shape/dispersion parameters).
MODEL_REGISTRY: dict = {
    "PFM": (g_PFM, []),
    "EMM": (g_EMM, []),
    "EPM": (g_EPM, ["eta"]),
    "DM":  (g_DM,  ["PD"]),
    "PEM": (g_PEM, ["eta"]),
}

VALID_MODELS = list(MODEL_REGISTRY.keys())
