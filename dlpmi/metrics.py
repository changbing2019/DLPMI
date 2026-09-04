"""
dlpmi/metrics.py
================
Derived vulnerability metrics computed from a fitted lumped parameter model.

Both metrics are smooth, differentiable functionals of the LPM parameters, so
their gradients are available from automatic differentiation and their
uncertainty follows from the same exact Hessian used for parameter
uncertainty -- no separate machinery and no refitting.

Metrics
-------
young_fraction(...)      F(a < T), the fraction of a sample younger than a
                         threshold age T.  This is the cumulative exit-age
                         distribution used by Musgrove et al. (2023,
                         Geophys. Res. Lett., Figs. 2e-2h) as their primary
                         indicator of vulnerability to land-surface
                         contamination, there reported without uncertainty.

mixture_mean_age(...)    The mixture-weighted mean age.  For a single-component
                         model this is tau; for a binary mixture it is
                         f1*tau1 + (1-f1)*tau2.  This -- not tau1 -- is the
                         quantity comparable to the "mean age" reported by
                         Musgrove et al. (2023).

Uncertainty
-----------
delta_method_sigma(...)  Propagates the parameter covariance to any scalar
                         metric via sigma^2 = grad^T C grad, with grad obtained
                         by autograd.  Exact to first order; adequate where the
                         metric is locally smooth in the parameters.

mc_metric(...)           Evaluates a metric over stored Monte Carlo parameter
                         draws.  Preferred where draws are available because it
                         inherits the full nonlinear propagation, but it
                         requires the JOINT draws (tau1, f1, tau2) -- marginal
                         summary statistics are not sufficient, since the
                         parameters are correlated.
"""

from __future__ import annotations
import numpy as np
import torch

from .kernels import g_PFM, g_EMM, g_EPM, g_DM, g_PEM

_KERNEL = {"PFM": g_PFM, "EMM": g_EMM, "EPM": g_EPM,
           "DM": g_DM, "PEM": g_PEM}

# Thresholds matching those discussed by Musgrove et al. (2023).
DEFAULT_THRESHOLDS = (10.0, 25.0, 40.0)


def exit_age_grid(tau_max: float, n: int = 4000,
                  a_min: float = 1e-3) -> torch.Tensor:
    """Logarithmic age grid spanning a_min to well beyond tau_max.

    A dedicated log grid is used rather than AGES_YOUNG / AGES_OLD because
    those are linearly spaced and resolve the first decade of ages too coarsely
    for an accurate F(a < 10 yr) when tau2 reaches tens of thousands of years.
    """
    a_max = max(float(tau_max) * 50.0, 1.0e5)
    return torch.logspace(np.log10(a_min), np.log10(a_max), n,
                          dtype=torch.float64)


def _pdf(model: str, params: dict, ages: torch.Tensor) -> torch.Tensor:
    """Normalised exit-age density on `ages`."""
    if model == "PFM":
        # Dirac delta: represent as a narrow spike at tau for quadrature.
        tau = params["tau"]
        w = torch.exp(-0.5 * ((ages - tau) / (0.01 * tau)) ** 2)
        return w / torch.trapz(w, ages)

    kern = _KERNEL[model]
    if model == "DM":
        g = kern(ages, params["tau"], params["PD"])
    elif model in ("EPM", "PEM"):
        g = kern(ages, params["tau"], params["eta"])
    else:
        g = kern(ages, params["tau"])
    g = torch.nan_to_num(g, nan=0.0, posinf=0.0, neginf=0.0)
    area = torch.trapz(g, ages)
    return g / torch.clamp(area, min=1e-300)


def _cdf_at(model: str, params: dict, T: float,
            ages: torch.Tensor) -> torch.Tensor:
    g = _pdf(model, params, ages)
    mask = (ages <= float(T)).to(g.dtype)
    return torch.trapz(g * mask, ages)


def young_fraction(model: str, params: dict, T: float,
                   f1=None, model2: str = None, params2: dict = None,
                   ages: torch.Tensor = None) -> torch.Tensor:
    """Fraction of the sample younger than T years, F(a < T).

    Single component:  F = int_0^T g(a; theta) da
    Binary mixture:    F = f1 * F_1 + (1 - f1) * F_2

    All arguments may be tensors with requires_grad set; the result is
    differentiable with respect to every parameter.
    """
    if ages is None:
        tmax = float(params["tau"].detach() if torch.is_tensor(params["tau"])
                     else params["tau"])
        if params2 is not None:
            t2 = params2["tau"]
            tmax = max(tmax, float(t2.detach() if torch.is_tensor(t2) else t2))
        ages = exit_age_grid(tmax)

    F1 = _cdf_at(model, params, T, ages)
    if f1 is None or params2 is None:
        return F1
    F2 = _cdf_at(model2 or model, params2, T, ages)
    return f1 * F1 + (1.0 - f1) * F2


def mixture_mean_age(params: dict, f1=None, params2: dict = None):
    """Mixture-weighted mean age.  Comparable to the 'mean age' of
    Musgrove et al. (2023); NOT the same as tau1 for a binary mixture."""
    if f1 is None or params2 is None:
        return params["tau"]
    return f1 * params["tau"] + (1.0 - f1) * params2["tau"]


# ── uncertainty ───────────────────────────────────────────────────────────

def delta_method_sigma(metric_fn, theta: torch.Tensor,
                       cov: np.ndarray) -> tuple:
    """First-order uncertainty of a scalar metric.

    Parameters
    ----------
    metric_fn : callable(theta_tensor) -> scalar tensor
    theta     : 1-D tensor of free parameters at the optimum
    cov       : parameter covariance matrix (from the exact Hessian)

    Returns
    -------
    (value, sigma) : float, float
    """
    t = theta.detach().clone().double().requires_grad_(True)
    val = metric_fn(t)
    grad, = torch.autograd.grad(val, t)
    g = grad.detach().numpy().reshape(-1)
    C = np.atleast_2d(np.asarray(cov, dtype=float))
    var = float(g @ C @ g)
    return float(val.detach()), float(np.sqrt(var)) if var > 0 else np.nan


def gaussian_propagate(metric_fn, theta: torch.Tensor, cov: np.ndarray,
                       n: int = 20000, seed: int = 0,
                       lo: float = None, hi: float = None) -> dict:
    """Nonlinear propagation of parameter uncertainty to a scalar metric.

    Draws JOINT samples from N(theta*, C) -- preserving parameter correlation --
    and pushes each through the metric.  Unlike the delta method this remains
    valid where the metric approaches a bound: F(a<T) saturating at 0 or 1 has
    a vanishing gradient, so a first-order expansion reports sigma -> 0 and
    understates the uncertainty badly.  The push-forward instead yields a
    correctly asymmetric interval that respects the bound.

    Still assumes the PARAMETERS are locally Gaussian; where that is doubtful,
    use mc_metric() on stored joint Monte Carlo draws.
    """
    rng = np.random.default_rng(seed)
    mu = theta.detach().numpy().astype(float).reshape(-1)
    C = np.atleast_2d(np.asarray(cov, dtype=float))
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.clip(w, 0.0, None)                 # project to nearest PSD
    L = V @ np.diag(np.sqrt(w))
    S = mu + rng.standard_normal((n, mu.size)) @ L.T
    if lo is not None or hi is not None:
        pass                                   # bounds applied to the metric
    vals = []
    for row in S:
        try:
            v = float(metric_fn(torch.tensor(row, dtype=torch.float64)))
        except Exception:                                        # noqa: BLE001
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {}
    v = np.array(vals)
    if lo is not None or hi is not None:
        v = np.clip(v, lo, hi)
    return {"p16": float(np.percentile(v, 16)),
            "p50": float(np.percentile(v, 50)),
            "p84": float(np.percentile(v, 84)),
            "std": float(v.std()), "n": int(v.size)}


def mc_metric(metric_fn, draws: np.ndarray) -> dict:
    """Evaluate a metric across joint Monte Carlo parameter draws.

    `draws` must be an (n_draws, n_params) array of JOINT samples in the same
    parameter order as metric_fn expects.  Marginal percentiles cannot be used
    here: the parameters are correlated, and recombining marginals would
    misstate the spread of the derived metric.
    """
    vals = []
    for row in np.asarray(draws, dtype=float):
        try:
            v = float(metric_fn(torch.tensor(row, dtype=torch.float64)))
        except Exception:                                        # noqa: BLE001
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {}
    v = np.array(vals)
    return {"mean": float(v.mean()), "std": float(v.std()),
            "p16": float(np.percentile(v, 16)),
            "p50": float(np.percentile(v, 50)),
            "p84": float(np.percentile(v, 84)),
            "n": int(v.size)}
