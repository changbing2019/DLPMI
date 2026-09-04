#!/usr/bin/env python3
"""
rebuild_ident_trueprofile.py -- assemble sweep_identifiability_trueprofile.csv
from a profile_true.py run.

That file had no generator: it was assembled by hand in an earlier session,
which is why the bisection fix of 2026-08-23 could not simply be re-run. This
script makes the assembly reproducible.

Columns taken from the profile run (they change when the profiler changes):
    w_tau1_cond, w_tau1_true, tau1_lo_true, tau1_hi_true, true_hit_bound,
    cls_true
Everything else -- Hessian diagnostics, bound activity, zone and well class --
is carried over from the existing file, because the bisection fix does not
touch it.

Classification (Section 2.4), verified to reproduce the previous file exactly:
    unconstrained if the width is not finite, or a profile bound reached a
        prescribed optimization limit, or (where no parameter bound is active)
        the Hessian is singular or not positive definite
    otherwise  w <= 0.5 well constrained, w <= 1.5 weakly constrained,
               else unconstrained

Usage:
  python rebuild_ident_trueprofile.py \
      --profile sweeps_trueprofile_fixed/profile_true.csv \
      --base    sweeps_trueprofile/sweep_identifiability_trueprofile.csv \
      --out     sweeps_trueprofile_fixed/sweep_identifiability_trueprofile.csv
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd

W_WELL, W_WEAK = 0.5, 1.5
FROM_PROFILE = ["w_tau1_cond", "w_tau1_true", "tau1_lo_true", "tau1_hi_true",
                "true_hit_bound"]


def classify(w, hit, bound_active, singular, pos_def):
    if not np.isfinite(w) or bool(hit):
        return "unconstrained"
    if not bool(bound_active) and (bool(singular) or not bool(pos_def)):
        return "unconstrained"
    if w <= W_WELL:
        return "well constrained"
    if w <= W_WEAK:
        return "weakly constrained"
    return "unconstrained"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--out", required=True)
    A = ap.parse_args()

    pr = pd.read_csv(A.profile).set_index("SampleID")
    base = pd.read_csv(A.base)
    keep = base.SampleID.tolist()                 # the 60-sample study set
    missing = [s for s in keep if s not in pr.index]
    assert not missing, "profile run is missing %d study samples: %s" % (
        len(missing), missing[:5])

    out = base.copy()
    for c in FROM_PROFILE:
        out[c] = out.SampleID.map(pr[c])
    # tau1 must agree: the fix changes intervals, not the optimum
    d = (out.SampleID.map(pr["tau1"]) - out["tau1"]).abs()
    assert float(np.nanmax(d.values)) < 1e-6, (
        "tau1 moved between runs (max |diff| %.3g) -- the profile run used a "
        "different optimum, not just a different interval" % np.nanmax(d.values))

    out["cls_true"] = [classify(r.w_tau1_true, r.true_hit_bound,
                                r.bound_active, r.singular, r.hessian_pos_def)
                       for r in out.itertuples()]

    os.makedirs(os.path.dirname(os.path.abspath(A.out)), exist_ok=True)
    out.to_csv(A.out, index=False)

    O = ["well constrained", "weakly constrained", "unconstrained"]
    vn = out.cls_true.value_counts()
    print("wrote %s\n" % A.out)
    print("%-22s %s" % ("classification", " / ".join("%5s" % c[:5] for c in O)))
    print("%-22s %s" % ("after the fix",
                        " / ".join("%5d" % int(vn.get(c, 0)) for c in O)))

    # The before/after comparison only applies when --base is a previous
    # trueprofile file. For a from-scratch build the base is
    # sweeps_final/sweep_identifiability.csv, which carries the Hessian and
    # bound columns but no cls_true, and there is nothing to compare against.
    if "cls_true" not in base.columns:
        print("\nbase carries no cls_true (from-scratch build): "
              "no before/after comparison")
        return
    vo = base.cls_true.value_counts()
    print("%-22s %s" % ("before the fix",
                        " / ".join("%5d" % int(vo.get(c, 0)) for c in O)))
    moved = out.SampleID[out.cls_true.values != base.cls_true.values].tolist()
    print("\nsamples changing class: %d" % len(moved))
    for s in moved:
        a = base.loc[base.SampleID == s].iloc[0]
        b = out.loc[out.SampleID == s].iloc[0]
        print("   %-26s w %6.3f -> %6.3f   %s -> %s"
              % (s, a.w_tau1_true, b.w_tau1_true, a.cls_true, b.cls_true))
    ok = out.dropna(subset=["w_tau1_cond", "w_tau1_true"])
    bad = ok[ok.w_tau1_true < ok.w_tau1_cond - 1e-9]
    print("\nmedian width: %.3f -> %.3f" % (base.w_tau1_true.median(),
                                            out.w_tau1_true.median()))
    print("monotonicity violations (true < cond): %d of %d" % (len(bad), len(ok)))
    print("bound-limited intervals: %d -> %d"
          % (int(base.true_hit_bound.sum()), int(out.true_hit_bound.sum())))


if __name__ == "__main__":
    main()
