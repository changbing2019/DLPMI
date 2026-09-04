#!/usr/bin/env python3
"""
refit_free_tau1.py -- what happens if tau1 is NOT prescribed?

Nine of the 60 samples never had tau1 as a free parameter, and are reported as
"unconstrained" because no profile interval exists for them. This asks whether
that label is an artefact of the parameterisation or a property of the data.

What can and cannot be tested
-----------------------------
`dlpmi/params.py::BMMModel` implements exactly TracerLPM's three Step 2 modes,
each with at most TWO free parameters (the shape parameters P_D are never
free):

    A  "Mean Age, Fraction"       tau1, f1 free; tau2 fixed
    B  "Fraction"                 f1 free; tau1, tau2 fixed
    C  "Fraction, 2nd Mean Age"   f1, tau2 free; tau1 fixed

So "free tau1 as well" is not available: with n_active = 3 or 4 tracers,
a third free parameter leaves dof = n_active - 3 = 0 or 1, and at dof = 0 the
system is exactly determined -- chi2 = 0 by construction, no goodness of fit
and no identifiability to measure. The cap is not arbitrary.

What IS testable is the swap: refit the nine in mode A, so tau1 is estimated
and tau2 is prescribed instead. tau2 is pinned at the value the current
analysis reports for that sample, so the only change is which of the two
transit times is free. Then profile tau1 exactly as Section 2.4 does.

Fitting is delegated to `dlpmi_sweeps.run_fit` -> `edwards_run._run_fit`, so
bounds, seeds, optimiser schedule and tokenisation are identical to baseline.

Usage:  python refit_free_tau1.py --out sweeps_trueprofile
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd

import profile_true as PT
from profile_true import BASE, CFG, _HERE, SW, true_profile
from profile_tau2 import classify

MODE_A = "Mean Age, Fraction"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adam", type=int, default=5000)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--out", default="")
    A = ap.parse_args()

    SW.bootstrap(_HERE)
    recs = json.load(open(CFG))["samples"]
    baseline = SW.load_baseline(BASE)

    rows = []
    targets = []
    for rec in recs:
        u = SW.unpack(rec)
        if u["sid"] not in baseline:
            continue
        if u["lpm"] == "DM" or MODE_A in str(u["free_params"]):
            continue                       # tau1 already free
        targets.append(rec)
    print("samples with tau1 prescribed: %d\n" % len(targets))

    for i, rec in enumerate(targets, 1):
        u0 = SW.unpack(rec)
        sid = u0["sid"]
        row = baseline[sid]
        t1_pre = float(row["tau1_PINN"])
        t2_pre = float(row["tau2_PINN"])
        c2_pre = float(row["chi2_PINN"])

        # mode C/B -> mode A, with tau2 pinned at the value now reported
        r = copy.deepcopy(rec)
        r["lpm"]["free_params"] = MODE_A
        r["lpm"]["tracerlpm_result"]["tau2_yr"] = t2_pre
        r["lpm"]["init"]["tau2_yr"] = t2_pre

        uu = SW.unpack(r)
        t0 = time.perf_counter()
        try:
            p, c2_new, _sims = SW.run_fit(r, uu["names"], uu["obs"],
                                          uu["scales"], A.adam, A.starts)
        except Exception as exc:                                  # noqa: BLE001
            print("[%d/%d] %-26s FIT FAILED: %s" % (i, len(targets), sid, exc))
            continue
        dt = time.perf_counter() - t0

        # profile the now-free tau1, same criterion as Section 2.4
        lo, hi, hit, ref, nev = true_profile(uu, r, p, pname="tau1",
                                             chi2min=float(c2_new))
        t1_new = float(p["tau1"])
        w = (hi - lo) / t1_new if np.isfinite(lo) and np.isfinite(hi) else np.nan
        cls = classify(w, hit)
        rows.append(dict(
            SampleID=sid, n_active=len(uu["names"]),
            free_params_orig=u0["free_params"], free_params_new=MODE_A,
            tau1_prescribed=t1_pre, tau1_refit=t1_new,
            tau1_shift_pct=100 * (t1_new - t1_pre) / t1_pre,
            tau2_pinned=t2_pre, f1_orig=float(row["f1_PINN"]), f1_refit=float(p["f1"]),
            chi2_orig=c2_pre, chi2_refit=float(c2_new),
            tau1_lo=lo, tau1_hi=hi, w_tau1=w, hit_bound=hit,
            cls_tau1=cls, n_eval=nev, seconds=round(dt, 1)))
        print("[%d/%d] %-26s tau1 %6.1f -> %7.2f (%+6.1f%%)  chi2 %.4f -> %.4f  "
              "w=%s  %s" % (i, len(targets), sid, t1_pre, t1_new,
                            rows[-1]["tau1_shift_pct"], c2_pre, c2_new,
                            "%.2f" % w if np.isfinite(w) else " nan", cls),
              flush=True)

    if not rows:
        print("nothing refit")
        return
    df = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("tau1 identifiability once freed:", df.cls_tau1.value_counts().to_dict())
    print("intervals limited by a bound: %d of %d"
          % (int(df.hit_bound.sum()), len(df)))
    print("median |tau1 shift| from the prescribed value: %.1f%%"
          % df.tau1_shift_pct.abs().median())
    worse = int((df.chi2_refit > df.chi2_orig).sum())
    print("fit quality: chi2 worse in %d of %d, better in %d"
          % (worse, len(df), len(df) - worse))
    print("median chi2 %.4f -> %.4f" % (df.chi2_orig.median(), df.chi2_refit.median()))

    if A.out:
        os.makedirs(A.out, exist_ok=True)
        p = os.path.join(A.out, "refit_free_tau1.csv")
        df.to_csv(p, index=False)
        print("wrote", p)


if __name__ == "__main__":
    main()
