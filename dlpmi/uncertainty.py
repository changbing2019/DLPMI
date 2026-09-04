"""
dlpmi/uncertainty.py
====================
Three complementary uncertainty quantification methods for DLPMI,
all evaluated at the best-fit parameter vector θ*.

Methods
-------
1. Hessian-based (analytical)
   cov = inv(H) × χ²/dof   where H = ∂²χ²/∂θ²
   Equivalent to the Levenberg-Marquardt covariance used in TracerLPM.
   Fast — requires one autograd Hessian call.

2. Profile chi-square
   68% CI: region where χ² ≤ χ²_min + Δχ² (default Δχ² = 1.0)
   Asymmetric intervals valid even when surface is non-parabolic.

3. Monte Carlo
   Perturb observed concentrations by Gaussian noise (1σ = obs_err),
   re-optimise N times from warm-start; report std and P16/P50/P84.

Chi-square probability
   P = χ².sf(χ²*, dof)  — TracerLPM's LPM_Prob field.

References
----------
Jurgens et al. (2012) USGS TM 4-F3
"""

import torch
import numpy as np
from scipy import stats as _sp


def chi2_probability(chi2_val: float,
                     n_active: int,
                     n_free: int) -> tuple:
    """
    Chi-square goodness-of-fit probability (TracerLPM LPM_Prob).

    Parameters
    ----------
    chi2_val : float   optimised chi-square value
    n_active : int     number of tracers in the loss
    n_free   : int     number of free model parameters

    Returns
    -------
    (prob, dof) : float, int
    """
    dof = max(n_active - n_free, 1)
    return float(_sp.chi2.sf(chi2_val, dof)), dof


def hessian_uncertainty(loss_fn_direct,
                        params_opt: torch.Tensor,
                        chi2_val: float,
                        n_active: int,
                        n_free: int) -> tuple:
    """
    Hessian-based 1-sigma parameter uncertainties.

    Parameters
    ----------
    loss_fn_direct : callable(params_tensor) → scalar Tensor
                     chi2 as a function of a flat parameter vector
                     in natural (un-transformed) units
    params_opt     : 1-D Tensor at the optimum
    chi2_val       : float  chi2 at the optimum
    n_active, n_free : int

    Returns
    -------
    sigmas : np.ndarray   1-sigma errors for each parameter
    cov    : np.ndarray   full covariance matrix
    dof    : int
    """
    dof  = max(n_active - n_free, 1)
    p64  = params_opt.double()

    H = torch.autograd.functional.hessian(
        lambda p: loss_fn_direct(p.float()).double(), p64
    ).detach().numpy()

    # The factor of 2 is required and was missing before 2026-08-24.
    #
    # H here is the Hessian of chi2 itself, d2(chi2)/dtheta_i dtheta_j. For
    # chi2 = sum(r^2) the Gauss-Newton Hessian is H ~= 2 J^T J, so the
    # curvature matrix that enters the covariance is H/2, not H:
    #
    #     Cov = s^2 (J^T J)^-1 = 2 s^2 H^-1,    s^2 = chi2/dof
    #
    # Equivalently, Delta-chi2 = 1 on a quadratic gives a half-width of
    # sqrt(2/H), not sqrt(1/H). Omitting the 2 made every sigma low by
    # sqrt(2) = 1.4142. Verified against a closed-form OLS problem in
    # tests/test_hessian_covariance.py, which reproduces the exact factor.
    scale = 2.0 * (chi2_val / dof)
    try:
        Hinv = np.linalg.inv(H)
    except np.linalg.LinAlgError:
        Hinv = np.linalg.pinv(H)
    cov    = Hinv * scale
    sigmas = np.sqrt(np.abs(np.diag(cov)))

    return sigmas, cov, dof


def profile_uncertainty(loss_fn_1d,
                        param_opt: float,
                        chi2_min: float,
                        param_lo: float,
                        param_hi: float,
                        delta_chi2: float = 1.0,
                        n_bisect: int = 30) -> tuple:
    """
    Profile chi-square confidence interval for ONE parameter.

    Fixes the target parameter at a series of values (other params
    held at optimum) and finds where chi2 rises by delta_chi2.

    Parameters
    ----------
    loss_fn_1d : callable(float) → float
    param_opt  : optimal value
    chi2_min   : chi2 at the optimum
    param_lo   : hard lower bound
    param_hi   : hard upper bound
    delta_chi2 : chi2 increase defining the CI (default 1.0 → 68%)
    n_bisect   : bisection iterations

    Returns
    -------
    (lo_68, hi_68) : lower and upper CI bounds
    """
    threshold = chi2_min + delta_chi2

    def _bisect(inside, outside):
        """Crossing of `threshold` between `inside` (chi2 <= threshold) and
        `outside` (chi2 > threshold). Either may be numerically larger; only
        the roles matter.

        The roles are asserted because getting them backwards fails silently:
        the loop marches one endpoint onto the other and returns it. That is
        what the lower-bound call did before 2026-08-23, quantising every
        lower bound to an exact multiple of `step`."""
        assert loss_fn_1d(inside) <= threshold < loss_fn_1d(outside), (
            "_bisect called with its endpoints reversed")
        a, b = inside, outside
        for _ in range(n_bisect):
            mid = 0.5 * (a + b)
            if loss_fn_1d(mid) > threshold: b = mid
            else: a = mid
        return 0.5 * (a + b)

    # Upper bound
    hi68  = param_opt
    step  = max(abs(param_opt) * 0.05, 0.1)
    found = False
    while hi68 + step < param_hi:
        hi68 += step
        if loss_fn_1d(hi68) > threshold:
            hi68 = _bisect(hi68 - step, hi68); found = True; break
    if not found:
        hi68 = min(param_opt * 3., param_hi)

    # Lower bound
    lo68  = param_opt
    found = False
    while lo68 - step > param_lo:
        lo68 -= step
        if loss_fn_1d(lo68) > threshold:
            lo68 = _bisect(lo68 + step, lo68); found = True; break
    if not found:
        lo68 = max(param_lo, param_opt * 0.1)

    return float(lo68), float(hi68)


def mc_uncertainty(fit_fn,
                   obs_vals: list,
                   obs_errs: list,
                   n_mc: int = 100,
                   seed: int = 42) -> tuple:
    """
    Monte Carlo uncertainty via observation perturbation.

    Parameters
    ----------
    fit_fn    : callable(obs_perturbed_list) → dict
                Must return a dict with at least keys matching the
                parameter names you want statistics on.
    obs_vals  : list[float]   nominal observed concentrations
    obs_errs  : list[float]   1-sigma measurement errors
    n_mc      : int           number of MC realisations (default 100)
    seed      : int           random seed

    Returns
    -------
    mc_draws : dict  param_name → list of n_mc sampled values
    mc_stats : dict  param_name → {mean, std, p16, p50, p84}
    """
    rng      = np.random.default_rng(seed)
    mc_draws: dict = {}

    for _ in range(n_mc):
        obs_pert = [max(float(o) + rng.normal(0., float(e)), 1e-20)
                    for o, e in zip(obs_vals, obs_errs)]
        try:
            res = fit_fn(obs_pert)
            for key, val in res.items():
                if val is not None:
                    mc_draws.setdefault(key, []).append(float(val))
        except Exception:
            pass

    mc_stats: dict = {}
    for key, vals in mc_draws.items():
        arr = np.array(vals)
        mc_stats[key] = {
            "mean": float(np.mean(arr)),
            "std":  float(np.std(arr)),
            "p16":  float(np.percentile(arr, 16)),
            "p50":  float(np.percentile(arr, 50)),
            "p84":  float(np.percentile(arr, 84)),
        }

    return mc_draws, mc_stats
