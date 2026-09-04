#!/usr/bin/env python3
"""
profile_tau2_sigma.py -- is the older component unidentifiable under the
measurement-error-weighted objective too?

The tau1 classification turned out to depend strongly on the objective
(profile_sigma.py: 8 well constrained under relative error, 44-46 under the
weighted form). The claim that tau2 never closes was established under the
relative-error objective only, so it needs the same test before it can carry
any weight.

Same construction as mean_age_uncertainty.py -- every binary mixture, with the
mode-C exchange applied where tau2 is prescribed -- but fitted AND profiled
under chi2_sigma, at both threshold choices.

Usage:  python profile_tau2_sigma.py --out sweeps_trueprofile
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd

from profile_true import BASE, CFG, _HERE, SW, true_profile
from profile_tau2 import classify

MODE_C = "Fraction, 2nd Mean Age"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adam", type=int, default=5000)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--out", default="")
    A = ap.parse_args()

    SW.bootstrap(_HERE)
    recs = json.load(open(CFG))["samples"]
    baseline = SW.load_baseline(BASE)
    study = pd.read_csv(os.path.join(
        _HERE, "sweeps_trueprofile",
        "sweep_identifiability_trueprofile.csv")).set_index("SampleID")
    bmm = [r for r in recs
           if r["sample_id"] in study.index
           and study.loc[r["sample_id"], "LPM"] != "DM"]
    print("binary mixtures: %d" % len(bmm), flush=True)

    rows = []
    for i, rec in enumerate(bmm, 1):
        sid = rec["sample_id"]
        b = baseline[sid]
        already = MODE_C in str(SW.unpack(rec)["free_params"])
        r = rec if already else copy.deepcopy(rec)
        if not already:                       # free tau2, pin tau1 as reported
            r["lpm"]["free_params"] = MODE_C
            r["lpm"]["tracerlpm_result"]["tau1_yr"] = float(b["tau1_PINN"])
            r["lpm"]["init"]["tau1_yr"] = float(b["tau1_PINN"])
        u = SW.unpack(r)
        errs = list(u["errs"])
        t0 = time.perf_counter()
        try:
            with SW.sigma_objective(errs):
                p, _c2rel, sims = SW.run_fit(r, u["names"], u["obs"],
                                             u["scales"], A.adam, A.starts)
            c2s = float(SW.chi2_sigma(sims, u["obs"], errs))
        except Exception as exc:                                # noqa: BLE001
            print("[%d/%d] %-26s FIT FAILED: %s" % (i, len(bmm), sid, exc),
                  flush=True)
            continue

        _, pn = SW.make_loss(u, r)
        dof = max(len(u["names"]) - len(pn), 1)
        lo, hi, hit, _, _ = true_profile(u, r, p, pname="tau2",
                                         chi2min=c2s, errs=errs)
        lo2, hi2, hit2, _, _ = true_profile(u, r, p, pname="tau2",
                                            chi2min=c2s, errs=errs,
                                            dchi2=c2s / dof)
        t2 = float(p["tau2"])
        f = np.isfinite
        w = (hi - lo) / t2 if f(lo) and f(hi) else np.nan
        w2 = (hi2 - lo2) / t2 if f(lo2) and f(hi2) else np.nan
        rows.append(dict(
            SampleID=sid, tau2_was_free=already, tau2=t2, chi2_sigma=c2s,
            dof=dof, tau2_lo=lo, tau2_hi=hi, w_tau2=w, hit_bound=hit,
            cls_tau2=classify(w, hit),
            w_tau2_scaled=w2, hit_bound_scaled=hit2,
            cls_tau2_scaled=classify(w2, hit2),
            seconds=round(time.perf_counter() - t0, 1)))
        x = rows[-1]
        print("[%2d/%d] %-26s tau2=%9.1f  w=%s (%s)  w_scaled=%s (%s)"
              % (i, len(bmm), sid, t2,
                 "%.2f" % w if f(w) else " nan", x["cls_tau2"][:12],
                 "%.2f" % w2 if f(w2) else " nan", x["cls_tau2_scaled"][:12]),
              flush=True)

    if not rows:
        print("nothing fitted")
        return
    df = pd.DataFrame(rows)
    O = ["well constrained", "weakly constrained", "unconstrained"]
    print("\n" + "=" * 74)
    for col, lbl in (("cls_tau2", "dchi2 = 1"),
                     ("cls_tau2_scaled", "dchi2 = chi2/dof")):
        v = df[col].value_counts()
        print("tau2 under chi2_sigma, %-16s : %s"
              % (lbl, " / ".join("%d %s" % (int(v.get(c, 0)), c) for c in O)))
    print("intervals limited by the search window: %d of %d (dchi2=1), "
          "%d of %d (scaled)"
          % (int(df.hit_bound.sum()), len(df),
             int(df.hit_bound_scaled.sum()), len(df)))
    ok = df.w_tau2.dropna()
    if len(ok):
        print("median relative width: %.2f (dchi2=1), %.2f (scaled)"
              % (ok.median(), df.w_tau2_scaled.dropna().median()))
    if A.out:
        os.makedirs(A.out, exist_ok=True)
        p_ = os.path.join(A.out, "profile_tau2_sigma.csv")
        df.to_csv(p_, index=False)
        print("\nwrote", p_)


if __name__ == "__main__":
    main()
