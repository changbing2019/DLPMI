#!/usr/bin/env python3
"""
profile_sigma.py -- does the identifiability classification survive a change of
objective function?

Why
---
Parameters are fitted, and tau1 profiled, under the relative-error sum of
squares conventional in lumped-parameter practice. That objective is not
proportional to a Gaussian log-likelihood, so Delta-chi2 = 1 is an operational
threshold rather than a calibrated 68% level (Section 2.3). The obvious
question, raised in review, is whether the reported classification is an
artefact of that choice.

This refits every sample under the measurement-error-weighted objective
chi2_sigma = sum[((C_sim - C_obs)/sigma)^2] and re-profiles tau1 under the
same objective, then compares the resulting widths and classes against the
reported ones. Delta-chi2 = 1 IS a calibrated 68% level for this objective, so
the comparison also indicates how much the operational threshold costs.

Nothing here replaces a reported number: the output is a separate file for the
Supporting Information.

Implementation notes
--------------------
  * fitting: sigma_objective() patches edwards_run.chi2_loss, which _run_fit
    calls by name, so the fit itself is genuinely driven by chi2_sigma.
  * profiling: make_loss builds its objective inline and is NOT affected by
    that patch, so it is passed errs= explicitly. Both halves therefore use
    the same objective.

Usage:  python profile_sigma.py --all --out sweeps_trueprofile
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import pandas as pd

from profile_true import BASE, CFG, _HERE, SW, true_profile
from profile_tau2 import W_WEAK, W_WELL, classify


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--adam", type=int, default=5000)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--samples", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    A = ap.parse_args()

    SW.bootstrap(_HERE)
    recs = json.load(open(CFG))["samples"]
    baseline = SW.load_baseline(BASE)
    rep = pd.read_csv(os.path.join(_HERE, "sweeps_trueprofile",
                                   "sweep_identifiability_trueprofile.csv")
                      ).set_index("SampleID")
    keep = {x for x in A.samples.split(",") if x}
    recs = [r for r in recs if r["sample_id"] in rep.index]
    if keep:
        recs = [r for r in recs if r["sample_id"] in keep]
    elif not A.all:
        recs = recs[:4]
    if A.limit:
        recs = recs[:A.limit]
    print("samples: %d" % len(recs), flush=True)

    rows = []
    for i, rec in enumerate(recs, 1):
        u = SW.unpack(rec)
        sid = u["sid"]
        errs = list(u["errs"])
        t0 = time.perf_counter()
        try:
            with SW.sigma_objective(errs):
                p, _c2rel, sims = SW.run_fit(rec, u["names"], u["obs"],
                                             u["scales"], A.adam, A.starts)
            c2s = float(SW.chi2_sigma(sims, u["obs"], errs))
        except Exception as exc:                                # noqa: BLE001
            print("[%d/%d] %-26s FIT FAILED: %s" % (i, len(recs), sid, exc),
                  flush=True)
            continue

        # Two thresholds, because they answer different questions.
        #   dchi2 = 1        formally the 68% level for a Gaussian likelihood,
        #                    but only if the model fits within the stated
        #                    sigmas -- which it often does not here.
        #   dchi2 = chi2/dof rescales the step by the reduced chi2, the same
        #                    misfit correction hessian_uncertainty already
        #                    applies via cov = inv(H) * chi2/dof. This is the
        #                    comparison that is actually like-for-like against
        #                    the reported relative-error intervals.
        dof = max(len(u["names"]) - len(SW.make_loss(u, rec)[1]), 1)
        lo, hi, hit, ref, nev = true_profile(u, rec, p, pname="tau1",
                                             chi2min=c2s, errs=errs)
        lo2, hi2, hit2, _, nev2 = true_profile(u, rec, p, pname="tau1",
                                               chi2min=c2s, errs=errs,
                                               dchi2=c2s / dof)
        dt = time.perf_counter() - t0
        t1 = float(p["tau1"])
        w = (hi - lo) / t1 if np.isfinite(lo) and np.isfinite(hi) else np.nan
        cls = classify(w, hit)
        w2 = (hi2 - lo2) / t1 if np.isfinite(lo2) and np.isfinite(hi2) else np.nan
        cls2 = classify(w2, hit2)
        rows.append(dict(
            SampleID=sid, LPM=u["lpm"], free_params=u["free_params"],
            n_active=len(u["names"]),
            tau1_reported=float(rep.loc[sid, "tau1"]),
            tau1_sigma=t1,
            tau1_shift_pct=100 * (t1 - float(rep.loc[sid, "tau1"]))
            / float(rep.loc[sid, "tau1"]),
            chi2_sigma=c2s,
            w_reported=float(rep.loc[sid, "w_tau1_true"]),
            w_sigma=w, hit_bound=hit,
            w_sigma_scaled=w2, hit_bound_scaled=hit2, dof=dof,
            dchi2_scaled=c2s / dof,
            cls_reported=str(rep.loc[sid, "cls_true"]), cls_sigma=cls,
            cls_sigma_scaled=cls2,
            n_eval=nev + nev2, seconds=round(dt, 1)))
        r = rows[-1]
        print("[%d/%d] %-26s tau1 %7.2f->%7.2f  w %s->%s  %s -> %s (%ss)"
              % (i, len(recs), sid, r["tau1_reported"], t1,
                 "%.2f" % r["w_reported"] if np.isfinite(r["w_reported"]) else " nan",
                 "%.2f" % w if np.isfinite(w) else " nan",
                 r["cls_reported"][:12], cls[:12], r["seconds"]), flush=True)

    if not rows:
        print("nothing fitted")
        return
    df = pd.DataFrame(rows)
    print("\n" + "=" * 76)
    vc = df.cls_sigma.value_counts()
    O = ["well constrained", "weakly constrained", "unconstrained"]
    print("class counts under chi2_sigma : %s"
          % " / ".join("%d %s" % (int(vc.get(c, 0)), c) for c in O))
    vr = df.cls_reported.value_counts()
    print("class counts as reported      : %s"
          % " / ".join("%d %s" % (int(vr.get(c, 0)), c) for c in O))
    vs = df.cls_sigma_scaled.value_counts()
    print("class counts, chi2_sigma misfit-scaled : %s"
          % " / ".join("%d %s" % (int(vs.get(c, 0)), c) for c in O))
    for col, lbl in (("cls_sigma", "dchi2 = 1"),
                     ("cls_sigma_scaled", "dchi2 = chi2/dof")):
        same = int((df[col] == df.cls_reported).sum())
        print("  same class as reported (%-16s): %d of %d (%.0f%%)"
              % (lbl, same, len(df), 100 * same / len(df)))
    ok = df.dropna(subset=["w_reported", "w_sigma"])
    if len(ok):
        print("median relative width: reported %.2f | chi2_sigma %.2f | "
              "chi2_sigma scaled %.2f"
              % (ok.w_reported.median(), ok.w_sigma.median(),
                 df.w_sigma_scaled.dropna().median()))
    print("median |tau1 shift|: %.1f%%   max %.1f%%"
          % (df.tau1_shift_pct.abs().median(), df.tau1_shift_pct.abs().max()))
    print("intervals limited by a bound: %d of %d"
          % (int(df.hit_bound.sum()), len(df)))

    if A.out:
        os.makedirs(A.out, exist_ok=True)
        pth = os.path.join(A.out, "profile_sigma.csv")
        df.to_csv(pth, index=False)
        print("\nwrote", pth)


if __name__ == "__main__":
    main()
