#!/usr/bin/env python3
"""Grid-convergence study for posterior_tau2.py.

The 80x80 and 160x160 runs disagreed (median contraction +0.13 vs +0.21), so
the grid was not converged and the resolution has to be chosen from evidence
rather than assumed. Under the scale-marginalised likelihood
chi2^(-n/2) the ridge in f is narrow -- f is well determined, tau2 is not --
so the f axis is the one that plausibly needs refinement.

Method: evaluate chi2 ONCE on a deliberately fine grid per sample, then
subsample that array by striding to emulate every coarser grid. Striding a
uniform grid by k gives exactly the grid with (N-1)/k+1 points over the same
range, so each coarser answer is exact, not interpolated, and the whole study
costs one fine evaluation per sample.
"""
from __future__ import annotations

import copy
import json
import os

import numpy as np
from scipy.integrate import simpson

from posterior_tau2 import (BND_HI, BND_LO, F_HI, F_LO, MODE_C, credible,
                            loglike, study_set)
from profile_true import BASE, CFG, SW, _HERE

N_T2, N_F = 193, 641          # 193 = 64*3+1, 641 = 64*10+1: many divisors
SAMPLES = ["EDTRPAS1-30", "SCTXSUS1-09/EDTRVFPS1-14", "EDTRPAS1-46"]
SCALE_RANGE = 5.0


def contraction_from(chi2, t2_grid, nobs, f_grid):
    ll = loglike(chi2, nobs, "marginalised", SCALE_RANGE)
    post = np.exp(ll - ll.max())
    post /= post.sum()
    m = np.maximum(simpson(post, x=f_grid, axis=0), 0.0)
    m /= m.sum()
    x = np.log(t2_grid)
    lo, hi = credible(x, m, 0.95)
    return 1.0 - (hi - lo) / (0.95 * (x[-1] - x[0])), float(np.exp(hi - lo))


def main():
    SW.bootstrap(_HERE)
    torch = SW._TORCH
    recs = json.load(open(CFG))["samples"]
    baseline = SW.load_baseline(BASE)
    study = study_set()

    # The study used to print only, so the evidence for the production grid
    # resolution lived in a terminal scrollback and the SI could not cite it.
    rows = []

    for sid_want in SAMPLES:
        rec = next(r for r in recs if SW.unpack(r)["sid"] == sid_want)
        u0 = SW.unpack(rec)
        assert u0["sid"] in baseline and u0["sid"] in study
        row = baseline[u0["sid"]]

        r = copy.deepcopy(rec)
        r["lpm"]["free_params"] = MODE_C
        t1 = float(row["tau1_PINN"])
        r["lpm"]["tracerlpm_result"]["tau1_yr"] = t1
        r["lpm"]["init"]["tau1_yr"] = t1
        u = SW.unpack(r)
        lf, pnames = SW.make_loss(u, r, errs=list(u["errs"]))
        assert pnames == ["f1", "tau2"]

        lr, ini = rec["lpm"]["tracerlpm_result"], rec["lpm"]["init"]
        t2s = float(lr.get("tau2_yr") or ini["tau2_yr"] or 100.)
        lo_b, hi_b = max(1.0, t2s * BND_LO), t2s * BND_HI
        t2f = np.exp(np.linspace(np.log(lo_b), np.log(hi_b), N_T2))
        ff = np.linspace(F_LO, F_HI, N_F)

        chi2 = np.empty((N_F, N_T2))
        with torch.no_grad():
            for a, fv in enumerate(ff):
                for b, tv in enumerate(t2f):
                    chi2[a, b] = float(lf(torch.tensor([fv, tv],
                                                       dtype=torch.float32)))
        nobs = len(u["names"])
        print("=" * 74)
        print("%s   n_active=%d   chi2_sigma min %.4g"
              % (u0["sid"], nobs, chi2.min()))

        print("  refine f  (tau2 fixed at %d points):" % N_T2)
        for k in (16, 8, 4, 2, 1):
            sub = chi2[::k, :]
            c, span = contraction_from(sub, t2f, nobs, ff[::k])
            print("    n_f = %4d   contraction %+.4f   95%% span x%.1f"
                  % (sub.shape[0], c, span))
            rows.append(dict(SampleID=u0["sid"], n_active=nobs,
                             chi2_sigma_min=float(chi2.min()), axis="f1",
                             n_f1=sub.shape[0], n_tau2=N_T2,
                             contraction_log_tau2=c, tau2_ci95_span=span))

        print("  refine tau2  (f fixed at %d points):" % N_F)
        for k in (16, 8, 4, 2, 1):
            sub = chi2[:, ::k]
            c, span = contraction_from(sub, t2f[::k], nobs, ff)
            print("    n_tau2 = %4d   contraction %+.4f   95%% span x%.1f"
                  % (sub.shape[1], c, span))
            rows.append(dict(SampleID=u0["sid"], n_active=nobs,
                             chi2_sigma_min=float(chi2.min()), axis="tau2",
                             n_f1=N_F, n_tau2=sub.shape[1],
                             contraction_log_tau2=c, tau2_ci95_span=span))

    import pandas as pd
    p = os.path.join(_HERE, "sweeps_trueprofile", "posterior_grid_probe.csv")
    pd.DataFrame(rows).to_csv(p, index=False)
    print("\nwrote %s (%d rows)" % (p, len(rows)))


if __name__ == "__main__":
    main()
