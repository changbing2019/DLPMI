#!/usr/bin/env python3
"""
mean_age_uncertainty.py -- how well is the mixture-weighted mean age determined?

The reported mean age is f1*tau1 + (1-f1)*tau2. In these data tau2 supplies a
median 98% of it, so the mean age is essentially a restatement of tau2. But
tau2 is a free parameter in only 8 of the 18 binary mixtures; in the other 10
it is prescribed, contributes no uncertainty, and the reported mean_age_sigma
therefore describes only the wobble in f1 and tau1 while the dominant term is
held still. EDTRPAS1-46 is the clearest case: mean age 18,136 yr with a
reported sigma of 141 yr (0.8%), of which 100% comes from a tau2 fixed at
23,500 yr.

This script puts every binary mixture on the same footing:

  * samples where tau2 is already free  -> profile tau2 directly
  * samples where tau2 is prescribed    -> refit in TracerLPM mode C
                                           (f1, tau2 free; tau1 pinned at the
                                           currently reported value), then
                                           profile tau2

and propagates the tau2 profile into a mean-age interval. Because the tau2
profile usually does not close inside the search window, the resulting
mean-age spread is a LOWER BOUND on the true uncertainty, not an estimate of
it. f1 is held at its fitted value in the propagation, so co-variation with f1
is excluded as well -- again making the answer conservative.

Fitting goes through edwards_run._run_fit; nothing is reimplemented.

Usage:  python mean_age_uncertainty.py --out sweeps_trueprofile
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

    # the baseline table carries all 65 samples; the study set is the 60 in
    # the identifiability sweep (five lie outside the San Antonio segment)
    study = set(pd.read_csv(os.path.join(
        _HERE, "sweeps_trueprofile",
        "sweep_identifiability_trueprofile.csv")).SampleID)
    rows = []
    bmm = [r for r in recs
           if SW.unpack(r)["lpm"] != "DM"
           and SW.unpack(r)["sid"] in baseline
           and SW.unpack(r)["sid"] in study]
    print("binary-mixture samples in the 60-sample study set: %d\n" % len(bmm))

    for i, rec in enumerate(bmm, 1):
        u0 = SW.unpack(rec)
        sid = u0["sid"]
        row = baseline[sid]
        t1_0 = float(row["tau1_PINN"])
        t2_0 = float(row["tau2_PINN"])
        f1_0 = float(row["f1_PINN"])
        c2_0 = float(row["chi2_PINN"])
        already = MODE_C in str(u0["free_params"])

        if already:
            r, params, c2 = rec, {"tau1": t1_0, "pd1": row["pd1_PINN"],
                                  "f1": f1_0, "tau2": t2_0}, c2_0
            note = "tau2 already free"
        else:
            # mode A/B -> mode C, tau1 pinned at the value now reported
            r = copy.deepcopy(rec)
            r["lpm"]["free_params"] = MODE_C
            r["lpm"]["tracerlpm_result"]["tau1_yr"] = t1_0
            r["lpm"]["init"]["tau1_yr"] = t1_0
            uu = SW.unpack(r)
            try:
                params, c2, _ = SW.run_fit(r, uu["names"], uu["obs"],
                                           uu["scales"], A.adam, A.starts)
                c2 = float(c2)
            except Exception as exc:                              # noqa: BLE001
                print("[%d/%d] %-26s FIT FAILED: %s" % (i, len(bmm), sid, exc))
                continue
            note = "refit with tau2 freed"

        uu = SW.unpack(r)
        t0 = time.perf_counter()
        lo, hi, hit, ref, nev = true_profile(uu, r, params, pname="tau2",
                                             chi2min=c2)
        dt = time.perf_counter() - t0

        t1 = float(params["tau1"])
        f1 = float(params["f1"])
        t2 = float(params["tau2"])
        w = (hi - lo) / t2 if np.isfinite(lo) and np.isfinite(hi) else np.nan
        ma = f1 * t1 + (1 - f1) * t2
        ma_lo = f1 * t1 + (1 - f1) * lo
        ma_hi = f1 * t1 + (1 - f1) * hi
        rows.append(dict(
            SampleID=sid, tau2_was_free=already, note=note,
            free_params_orig=u0["free_params"],
            tau1=t1, f1=f1, frac_old=1 - f1, tau2=t2,
            tau2_lo=lo, tau2_hi=hi, w_tau2=w, hit_bound=hit,
            cls_tau2=classify(w, hit),
            mean_age=ma, mean_age_lo=ma_lo, mean_age_hi=ma_hi,
            mean_age_span_factor=ma_hi / ma_lo if ma_lo > 0 else np.nan,
            pct_from_tau2=100 * (1 - f1) * t2 / ma if ma > 0 else np.nan,
            chi2=c2, seconds=round(dt, 1)))
        print("[%2d/%d] %-26s meanage %8.0f  [%8.0f, %9.0f]  x%-5.1f  %s"
              % (i, len(bmm), sid, ma, ma_lo, ma_hi,
                 rows[-1]["mean_age_span_factor"], note), flush=True)

    df = pd.DataFrame(rows).sort_values("mean_age", ascending=False)
    print("\n" + "=" * 74)
    print("tau2 identifiability across all binary mixtures:",
          df.cls_tau2.value_counts().to_dict())
    print("intervals limited by the search window: %d of %d"
          % (int(df.hit_bound.sum()), len(df)))
    print("median share of the mean age coming from tau2: %.1f%%"
          % df.pct_from_tau2.median())
    print("median mean-age span (hi/lo): x%.1f" % df.mean_age_span_factor.median())
    old = df[df.mean_age > 10000]
    if len(old):
        print("\nsamples with mean age > 10,000 yr:")
        for _, r in old.iterrows():
            print("   %-26s %7.0f yr  ->  %.0f to %.0f yr  (x%.1f)"
                  % (r.SampleID, r.mean_age, r.mean_age_lo, r.mean_age_hi,
                     r.mean_age_span_factor))

    if A.out:
        os.makedirs(A.out, exist_ok=True)
        p = os.path.join(A.out, "mean_age_uncertainty.csv")
        df.to_csv(p, index=False)
        print("\nwrote", p)


if __name__ == "__main__":
    main()
