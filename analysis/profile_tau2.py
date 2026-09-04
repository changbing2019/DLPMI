#!/usr/bin/env python3
"""
profile_tau2.py -- true profile-likelihood intervals for the OLDER component,
tau2, in the binary-mixture samples where it is a free parameter.

Why this exists
---------------
The manuscript assesses identifiability for tau1 (the modern component) only.
tau2 exists in 18 of the 60 samples and is actually FREE in 8 of them --
exactly the 8 whose `free_params` is "Fraction, 2nd Mean Age", i.e. the ones
where tau1 is held fixed and f1/tau2 are fitted. Those 8 are reported as
"tau1 unconstrained" (w_tau1 is NaN by construction), so the only samples
carrying information about the older water are the ones the tau1 analysis
discards. This closes that gap.

Method is identical to profile_true.py --mode ident, with pname="tau2":
tau2 is stepped and the single remaining free parameter (f1) is re-minimised
at every step, referenced to the stored chi2 with dchi2 = 1. Nothing about
the forward model, bounds, seeds or objective changes -- `true_profile` and
`free_param_bounds` are imported from the existing code rather than reworked.

Usage:  python profile_tau2.py --all --out sweeps_trueprofile
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

import profile_true as PT
from profile_true import BASE, CFG, _HERE, SW, true_profile


def run_tau2(recs, baseline):
    rows = []
    for i, rec in enumerate(recs, 1):
        u = SW.unpack(rec)
        sid = u["sid"]
        row = baseline.get(sid)
        if row is None:
            continue
        _, pnames = SW.make_loss(u, rec)
        if "tau2" not in pnames:
            continue                      # tau2 fixed or absent -> nothing to profile

        params = {"tau1": row["tau1_PINN"], "pd1": row["pd1_PINN"],
                  "f1": row["f1_PINN"], "tau2": row["tau2_PINN"]}
        c2 = float(row["chi2_PINN"])
        t0 = time.perf_counter()
        lo_t, hi_t, hit_t, ref, nev = true_profile(u, rec, params,
                                                   pname="tau2", chi2min=c2)
        dt = time.perf_counter() - t0

        # conditional interval too, so the two are comparable the same way
        # profile_true.py compares them for tau1
        bnds = SW.free_param_bounds(rec, u, pnames)
        lo_c, hi_c = SW.profile_width(u, rec, params, u["names"], u["obs"],
                                      u["scales"], c2, "tau2", *bnds["tau2"])
        t2 = float(params["tau2"])
        f = np.isfinite
        rows.append(dict(
            SampleID=sid, LPM=u["lpm"], free_params=u["free_params"],
            n_active=len(u["names"]), tau1=float(params["tau1"]),
            f1=float(params["f1"]), tau2=t2,
            chi2_stored=c2, chi2_profile_ref=ref,
            tau2_lo_cond=lo_c, tau2_hi_cond=hi_c,
            w_tau2_cond=(hi_c - lo_c) / t2 if f(lo_c) and f(hi_c) else np.nan,
            tau2_lo_true=lo_t, tau2_hi_true=hi_t,
            w_tau2_true=(hi_t - lo_t) / t2 if f(lo_t) and f(hi_t) else np.nan,
            true_hit_bound=hit_t, n_eval=nev, seconds=round(dt, 2)))
        r = rows[-1]
        print("[%d/%d] %-26s tau2=%9.1f  w_cond=%s  w_true=%s  bound=%s (%ss)"
              % (i, len(recs), sid, t2,
                 "%.3f" % r["w_tau2_cond"] if f(r["w_tau2_cond"]) else "   nan",
                 "%.3f" % r["w_tau2_true"] if f(r["w_tau2_true"]) else "   nan",
                 r["true_hit_bound"], r["seconds"]), flush=True)
    return pd.DataFrame(rows)


# same thresholds the tau1 classification uses (Section 2.4)
W_WELL, W_WEAK = 0.5, 1.5


def classify(w, hit):
    if not np.isfinite(w) or hit:
        return "unconstrained"
    if w <= W_WELL:
        return "well constrained"
    if w <= W_WEAK:
        return "weakly constrained"
    return "unconstrained"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--out", default="")
    A = ap.parse_args()

    SW.bootstrap(_HERE)
    recs = json.load(open(CFG))["samples"]
    keep = {x for x in A.samples.split(",") if x}
    if keep:
        recs = [r for r in recs if r["sample_id"] in keep]
    elif not A.all:
        recs = recs[:8]

    df = run_tau2(recs, SW.load_baseline(BASE))
    if df.empty:
        print("no sample has tau2 as a free parameter")
        return

    df["cls_tau2_true"] = [classify(w, h)
                           for w, h in zip(df.w_tau2_true, df.true_hit_bound)]
    df["cls_tau2_cond"] = [classify(w, False) for w in df.w_tau2_cond]

    ok = df.dropna(subset=["w_tau2_cond", "w_tau2_true"])
    bad = ok[ok.w_tau2_true < ok.w_tau2_cond - 1e-9]
    print("\nn=%d with both widths" % len(ok))
    if len(ok):
        print("median w_cond=%.3f  median w_true=%.3f"
              % (ok.w_tau2_cond.median(), ok.w_tau2_true.median()))
    # a profile interval can only be wider than the conditional slice through
    # the same optimum; a violation means the inner minimisation failed
    print("monotonicity violations (true < cond): %d" % len(bad))
    if len(bad):
        print(bad[["SampleID", "w_tau2_cond", "w_tau2_true"]].to_string(index=False))
    print("\nclassification of tau2 (true profile):",
          df.cls_tau2_true.value_counts().to_dict())
    print("intervals limited by a bound: %d of %d"
          % (int(df.true_hit_bound.sum()), len(df)))

    if A.out:
        os.makedirs(A.out, exist_ok=True)
        p = os.path.join(A.out, "profile_true_tau2.csv")
        df.to_csv(p, index=False)
        print("wrote", p)


if __name__ == "__main__":
    main()
