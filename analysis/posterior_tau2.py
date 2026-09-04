#!/usr/bin/env python3
"""
posterior_tau2.py -- Bayesian counterpart to the tau2 profile analysis.

The profile result (mean_age_uncertainty.py, profile_tau2.py) is that the
premodern transit time tau2 is not constrained: the Delta chi2 = 1 interval
runs to the edge of the search window in every binary mixture. A referee's
natural question is whether that is an artefact of profiling and of an
uncalibrated Delta chi2 threshold, and whether sampling the posterior would
say something different. This script answers it by an independent route.

WHY A GRID AND NOT MCMC
-----------------------
Only two parameters are free (f1 and tau2, with tau1 pinned -- TracerLPM's
two-parameter cap, the same mode C setup mean_age_uncertainty.py uses). At
0.8 ms per forward evaluation a dense 2-D grid is affordable, and it removes
every question a sampler would invite: no chains, no burn-in, no proposal
tuning, no convergence diagnostic. The posterior is evaluated exactly on the
grid and normalised by quadrature.

THE LIKELIHOOD IS A REAL LIKELIHOOD
-----------------------------------
The manuscript's primary objective, chi2_rel, is a relative-error sum of
squares and is NOT proportional to a Gaussian log-likelihood -- which is
exactly why Delta chi2 = 1 is described there as operational rather than
calibrated. It therefore CANNOT be exponentiated into a posterior. This
script uses chi2_sigma instead,

    log L(theta) = -chi2_sigma(theta) / 2,

with sigma the 1-sigma values the published inversion assigned, read from the
config (they come verbatim from Table_4_LPM.txt in the data release). That is
a genuine Gaussian likelihood, and it is the same error model the AICc
structure comparison already uses.

Because those sigma are a uniform 10% of the observation for 164 of the 218
active pairs, the posterior width is conditional on that error model. The run
is therefore repeated with the reported ANALYTICAL sigma substituted
(--sigma reported) as a sensitivity. Read that second run with its own
goodness of fit in hand: where the analytical sigma reject the model outright,
a narrow posterior is a narrow posterior around a rejected model, not
evidence that the age is determined.

PRIORS
------
  tau2 ~ log-uniform on [max(1, 0.05*tau2_src), 5*tau2_src]
  f1   ~ uniform on [0.005, 0.997]

Both ranges are the bounds the optimiser and the profiler already enforce
(dlpmi_sweeps.free_param_bounds), so the posterior is directly comparable to
the profile interval rather than to a differently-bounded problem. Log-uniform
is the scale-invariant choice for tau2, which spans orders of magnitude.

The headline statistic is prior-to-posterior CONTRACTION in log tau2,

    contraction = 1 - (posterior 95% width) / (prior 95% width),

which is 0 when the data add nothing and approaches 1 when the parameter is
pinned down. It needs no threshold and no calibration.

Usage:
  python posterior_tau2.py --out sweeps_trueprofile
  python posterior_tau2.py --sigma reported --out sweeps_trueprofile
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd
from scipy.integrate import simpson

from dlpmi.uncertainty import chi2_probability
from profile_true import BASE, CFG, SW, _HERE

MODE_C = "Fraction, 2nd Mean Age"
F_LO, F_HI = 0.005, 0.997          # free_param_bounds' hardcoded f1 range
BND_LO, BND_HI = 0.05, 5.0         # free_param_bounds' tau2 multipliers


def study_set():
    p = os.path.join(_HERE, "sweeps_trueprofile",
                     "sweep_identifiability_trueprofile.csv")
    return set(pd.read_csv(p).SampleID)


def loglike(chi2, nobs, mode, scale_range):
    """Log-likelihood on the chi2 grid.

    mode 'inversion'/'reported' : fixed sigma, log L = -chi2/2.

    mode 'marginalised' : an unknown COMMON error scale s multiplying every
    sigma, with a Jeffreys prior p(s) ∝ 1/s truncated to
    [1/scale_range, scale_range]. Integrating s out has a closed form,

        L ∝ chi2^(-n/2) [ P(n/2, chi2/(2 s_lo^2)) - P(n/2, chi2/(2 s_hi^2)) ]

    with P the regularized lower incomplete gamma. Truncation is not
    cosmetic. With an UNBOUNDED Jeffreys prior the bracket becomes 1 and
    L ∝ chi2^(-n/2), which is improper wherever the model can interpolate the
    data: near an exact fit chi2 ~ |dtheta|^2, so the integral behaves as
    ∫ r^(1-n) dr and diverges for n >= 2. EDTRPAS1-46 (n=3, dof=1,
    chi2_sigma min 0.004) demonstrates it -- its contraction climbed from
    +0.50 to +0.98 between 161 and 321 grid points in f and would keep
    climbing. Truncating restores propriety: as chi2 -> 0 the bracket goes as
    chi2^(n/2) and cancels the divergence exactly.

    scale_range is a stated assumption -- the assigned sigma are taken to be
    right to within that factor either way -- and is varied as a sensitivity.
    """
    if mode != "marginalised":
        return -0.5 * chi2
    from scipy.special import gammainc

    a = 0.5 * nobs
    s_lo, s_hi = 1.0 / scale_range, scale_range
    c = np.maximum(chi2, 0.0) * 0.5
    bracket = gammainc(a, c / s_lo ** 2) - gammainc(a, c / s_hi ** 2)
    # log form, with the chi2^(-n/2) prefactor kept in logs for conditioning
    tiny = np.finfo(float).tiny
    with np.errstate(divide="ignore", invalid="ignore"):
        out = -a * np.log(np.maximum(chi2, tiny)) + np.log(
            np.maximum(bracket, tiny))
    # chi2 == 0 exactly: the limit is finite, take the neighbouring maximum
    return np.where(np.isfinite(out), out, np.nanmax(out[np.isfinite(out)]))


def credible(x, w, q):
    """Interval containing mass q from a weighted 1-D marginal, taken as the
    central (equal-tail) interval so it is well defined for a flat posterior."""
    c = np.cumsum(w)
    c /= c[-1]
    lo = np.interp((1.0 - q) / 2.0, c, x)
    hi = np.interp(1.0 - (1.0 - q) / 2.0, c, x)
    return float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-tau2", type=int, default=160)
    ap.add_argument("--n-f1", type=int, default=160)
    ap.add_argument("--sigma",
                    choices=["marginalised", "inversion", "reported"],
                    default="marginalised")
    ap.add_argument("--prior-window", choices=["bounds", "profile"],
                    default="bounds",
                    help="'bounds' (default): tau2 prior spans the optimiser's "
                         "parameter bounds, 0.05x-5x the source value. "
                         "'profile': the narrower window true_profile walks, "
                         "0.1x-3x the fitted optimum, for a like-for-like "
                         "span comparison against the profile interval.")
    ap.add_argument("--scale-range", type=float, default=5.0,
                    help="error-scale prior spans [1/S, S] about the assigned "
                         "sigma; only used by --sigma marginalised")
    ap.add_argument("--data-dir", default="../Musgrove_2023Data")
    ap.add_argument("--out", default="")
    A = ap.parse_args()

    SW.bootstrap(_HERE)
    torch = SW._TORCH
    recs = json.load(open(CFG))["samples"]
    baseline = SW.load_baseline(BASE)
    study = study_set()

    if A.sigma == "reported":
        # load_reported_rel_sigma only RETURNS the table; the module global
        # reported_errs() reads is assigned separately in dlpmi_sweeps.main.
        # Calling the loader and discarding the result leaves that global
        # empty, reported_errs returns (None, 0), and every sigma stays at its
        # config value -- a silent no-op that makes this run duplicate the
        # fixed-inversion-sigma run exactly. Assign it, then assert.
        SW._REPORTED_REL_SIGMA = SW.load_reported_rel_sigma(
            os.path.join(_HERE, A.data_dir))
        assert SW._REPORTED_REL_SIGMA, "reported sigma table came back empty"
        print("reported analytical sigma loaded for %d samples"
              % len(SW._REPORTED_REL_SIGMA))

    bmm = [r for r in recs
           if SW.unpack(r)["lpm"] != "DM"
           and SW.unpack(r)["sid"] in baseline
           and SW.unpack(r)["sid"] in study]
    print("binary mixtures in the 60-sample study set: %d" % len(bmm))
    print("error model: %s sigma\n" % A.sigma)

    rows, marg = [], []
    for i, rec in enumerate(bmm, 1):
        u0 = SW.unpack(rec)
        sid = u0["sid"]
        row = baseline[sid]
        already = MODE_C in str(u0["free_params"])

        r = copy.deepcopy(rec)
        r["lpm"]["free_params"] = MODE_C
        t1 = float(row["tau1_PINN"])
        r["lpm"]["tracerlpm_result"]["tau1_yr"] = t1
        r["lpm"]["init"]["tau1_yr"] = t1
        u = SW.unpack(r)

        errs = list(u["errs"])
        n_sub = 0
        if A.sigma == "reported":
            sub, n_sub = SW.reported_errs(sid, u["names"], u["obs"], errs)
            if sub is not None:
                errs = sub

        lf, pnames = SW.make_loss(u, r, errs=errs)
        assert pnames == ["f1", "tau2"], (sid, pnames)

        lr, ini = rec["lpm"]["tracerlpm_result"], rec["lpm"]["init"]
        t2_src = float(lr.get("tau2_yr") or ini["tau2_yr"] or 100.)
        lo_b, hi_b = max(1.0, t2_src * BND_LO), t2_src * BND_HI
        if A.prior_window == "profile":
            # the window true_profile() actually walks -- 0.1x to 3x the
            # FITTED optimum, clipped to the parameter bounds. Anchored on the
            # optimum rather than on the source value, so it is both narrower
            # and differently centred than the parameter bounds. Only for a
            # like-for-like span comparison against the profile interval;
            # using a data-derived range as a prior is mildly circular, which
            # is why it is not the primary.
            opt2 = float(row["tau2_PINN"])
            lo_b, hi_b = max(lo_b, 0.1 * opt2), min(3.0 * opt2, hi_b)

        t2_grid = np.exp(np.linspace(np.log(lo_b), np.log(hi_b), A.n_tau2))
        f_grid = np.linspace(F_LO, F_HI, A.n_f1)

        t0 = time.perf_counter()
        chi2 = np.empty((A.n_f1, A.n_tau2))
        with torch.no_grad():
            for a, fv in enumerate(f_grid):
                for b, tv in enumerate(t2_grid):
                    chi2[a, b] = float(lf(torch.tensor([fv, tv],
                                                       dtype=torch.float32)))
        dt = time.perf_counter() - t0

        # posterior on the grid. log-uniform in tau2 and uniform in f1 are both
        # constant on (log tau2, f1), so with the grid uniform in log tau2 the
        # prior contributes no shape and the posterior is the likelihood.
        # Marginalising a bounded error scale removes the dependence on the
        # assumed uniform 10% level (though not on the relative weighting
        # between tracers, which is the release's own). See loglike().
        ll = loglike(chi2, len(u["names"]), A.sigma, A.scale_range)
        post = np.exp(ll - ll.max())
        post /= post.sum()

        # Marginal over f1 by Simpson's rule rather than a plain sum. Where a
        # sample fits almost exactly the ridge in f1 is narrow, and a Riemann
        # sum over it converges slowly -- EDTRPAS1-46's contraction wobbled by
        # 0.03 between f grids of 161, 321 and 641 points under summation.
        m_t2 = simpson(post, x=f_grid, axis=0)
        m_t2 = np.maximum(m_t2, 0.0)
        m_t2 /= m_t2.sum()
        x = np.log(t2_grid)
        lo95, hi95 = credible(x, m_t2, 0.95)
        lo68, hi68 = credible(x, m_t2, 0.68)

        # prior is flat in x, so its equal-tail 95% width is 0.95 of the range
        prior_w95 = 0.95 * (x[-1] - x[0])
        contraction = 1.0 - (hi95 - lo95) / prior_w95

        # mass pressed against each bound: outer tenth of the prior range
        edge = 0.10 * (x[-1] - x[0])
        mass_lo = float(m_t2[x <= x[0] + edge].sum())
        mass_hi = float(m_t2[x >= x[-1] - edge].sum())
        mode_at_edge = bool(np.argmax(m_t2) in (0, A.n_tau2 - 1))

        # mean age carried through the same grid
        FF, TT = np.meshgrid(f_grid, t2_grid, indexing="ij")
        ma = FF * t1 + (1.0 - FF) * TT
        o = np.argsort(ma.ravel())
        ma_s, w_s = ma.ravel()[o], post.ravel()[o]
        ma_lo, ma_hi = credible(ma_s, w_s, 0.95)

        c2min = float(chi2.min())
        dof = max(len(u["names"]) - 2, 0)
        # the shipped helper, so the p-value convention matches the rest of
        # the analysis (it floors dof at 1, hence the explicit dof guard here)
        pval = (float(chi2_probability(c2min, len(u["names"]), 2)[0])
                if dof > 0 else float("nan"))

        # Marginalising the error scale costs one degree of freedom: f1, tau2
        # and s are then estimated from n observations. With n = 3 that is
        # saturated, and it shows -- the likelihood ridge in f1 is narrower
        # than any uniform grid resolves, the located minimum keeps falling as
        # the grid refines (EDTRPAS1-46: chi2 0.630 at 160x160, 0.004 at
        # 193x641), and the contraction does not converge. This is the same
        # situation Section 3.4 already excludes from the AICc comparison at
        # dof = 0, one degree of freedom up. Such samples are reported but
        # excluded from the marginalised summary; the fixed-sigma posteriors
        # have no such pathology and are computed for all of them.
        assessable = (A.sigma != "marginalised") or (dof >= 2)
        rows.append(dict(
            SampleID=sid, sigma_model=A.sigma, tau2_was_free=already,
            n_active=len(u["names"]), dof=dof, assessable=assessable,
            n_sigma_substituted=n_sub,
            tau1_pinned=t1, tau2_src=t2_src, tau2_lo_b=lo_b, tau2_hi_b=hi_b,
            chi2_sigma_min=c2min, p_at_min=pval,
            prior_window=A.prior_window,
            scale_range=(A.scale_range if A.sigma == "marginalised"
                         else float("nan")),
            tau2_map=float(t2_grid[np.unravel_index(post.argmax(),
                                                    post.shape)[1]]),
            tau2_ci68_lo=float(np.exp(lo68)), tau2_ci68_hi=float(np.exp(hi68)),
            tau2_ci95_lo=float(np.exp(lo95)), tau2_ci95_hi=float(np.exp(hi95)),
            tau2_ci95_span=float(np.exp(hi95) / np.exp(lo95)),
            contraction_log_tau2=float(contraction),
            post_mass_lower_decile=mass_lo, post_mass_upper_decile=mass_hi,
            marginal_mode_at_bound=mode_at_edge,
            mean_age_ci95_lo=ma_lo, mean_age_ci95_hi=ma_hi,
            mean_age_ci95_span=float(ma_hi / ma_lo) if ma_lo > 0 else np.nan,
            seconds=round(dt, 1)))
        for b in range(A.n_tau2):
            marg.append(dict(SampleID=sid, sigma_model=A.sigma,
                             tau2=float(t2_grid[b]),
                             posterior_density=float(m_t2[b]),
                             chi2_sigma_min_at_tau2=float(chi2[:, b].min())))

        print("[%2d/%d] %-26s dof %d  contraction %+.3f  95%% CI "
              "%8.0f-%9.0f yr (x%-6.1f)  edge mass %.2f/%.2f  %s%s"
              % (i, len(bmm), sid, dof, contraction, np.exp(lo95),
                 np.exp(hi95), np.exp(hi95) / np.exp(lo95), mass_lo, mass_hi,
                 "" if assessable else "EXCLUDED(saturated) ",
                 "MODE AT BOUND" if mode_at_edge else ""))

    df = pd.DataFrame(rows)
    if A.sigma == "reported":
        n = int(df.n_sigma_substituted.sum())
        assert n > 0, ("no sigma was substituted -- the reported-sigma run is "
                       "silently identical to the fixed-inversion-sigma run")
        print("\nsigma substituted in %d tracer-sample pairs "
              "(%d of %d samples affected)"
              % (n, int((df.n_sigma_substituted > 0).sum()), len(df)))
    print()
    if not df.assessable.all():
        ex = df[~df.assessable]
        print("EXCLUDED from the summary (saturated once the error scale is "
              "marginalised, dof < 2): %d of %d" % (len(ex), len(df)))
        for _, e in ex.iterrows():
            print("   %-26s n_active=%d" % (e.SampleID, e.n_active))
        print()
    d = df[df.assessable]
    print("SUMMARY over %d assessable samples" % len(d))
    print("contraction in log tau2: median %+.3f, range %+.3f to %+.3f"
          % (d.contraction_log_tau2.median(), d.contraction_log_tau2.min(),
             d.contraction_log_tau2.max()))
    print("  samples with contraction > 0.5: %d of %d"
          % (int((d.contraction_log_tau2 > 0.5).sum()), len(d)))
    print("95%% credible span of tau2: median x%.1f, min x%.1f"
          % (d.tau2_ci95_span.median(), d.tau2_ci95_span.min()))
    print("marginal mode at a prior bound: %d of %d"
          % (int(d.marginal_mode_at_bound.sum()), len(d)))
    print("mean-age 95%% span: median x%.1f, max x%.1f"
          % (d.mean_age_ci95_span.median(), d.mean_age_ci95_span.max()))
    if d.dof.gt(0).any():
        ok = d[d.dof > 0]
        print("fits with p > 0.05 at the posterior mode: %d of %d"
              % (int((ok.p_at_min > 0.05).sum()), len(ok)))

    if A.out:
        d = os.path.join(_HERE, A.out)
        os.makedirs(d, exist_ok=True)
        tag = {"marginalised": "", "inversion": "_fixed_inversion_sigma",
               "reported": "_fixed_reported_sigma"}[A.sigma]
        # The tag used to depend on --sigma ALONE, so a --prior-window profile
        # run or a non-default --scale-range overwrote the primary output
        # instead of sitting beside it. That is why no file existed for the
        # like-for-like span comparison in Section 3.3 even though the number
        # was quoted. Both non-default settings now earn their own filename.
        if A.prior_window != "bounds":
            tag += "_window_%s" % A.prior_window
        if A.sigma == "marginalised" and abs(A.scale_range - 5.0) > 1e-9:
            tag += "_scale%g" % A.scale_range
        p1 = os.path.join(d, "posterior_tau2%s.csv" % tag)
        p2 = os.path.join(d, "posterior_tau2%s_marginals.csv" % tag)
        df.to_csv(p1, index=False)
        pd.DataFrame(marg).to_csv(p2, index=False)
        print("\nwrote %s\n      %s" % (p1, p2))


if __name__ == "__main__":
    main()
