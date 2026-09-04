#!/usr/bin/env python3
"""
make_gof_diagnostic.py
=======================
Goodness-of-fit diagnostic table for Sec 3.7 (Task 2 of REWORK_BRIEF.md).

The manuscript's existing GOF claim (60/60 samples acceptable) uses chi2
probabilities computed from the relative-error objective (Eq. 9), which is not
proportional to a log-likelihood, so those p-values are not really chi2
statistics. This script builds, for all 60 samples, the DM-structure fit under
both objectives (from dlpmi_sweeps.py's structure sweep) and reports:

  - chi2_rel and chi2_sigma at the Eq. 9 optimum
  - chi2_rel and chi2_sigma at the chi2_sigma optimum
  - the parameter shift |tau1_sigma - tau1_rel| / tau1_rel

The DM structure is used uniformly (regardless of each sample's a priori
DM/BMM assignment) because sweep_structure fits DM for every sample and it
reduces to one comparable free parameter (tau1), giving a single shift number
across all 60 samples rather than a harder-to-compare multi-parameter BMM
shift.

Usage
-----
  python make_gof_diagnostic.py \
      --rel-csv sweeps/sweep_structure_eq9_sensitivity.csv \
      --sigma-csv sweeps_sigma/sweep_structure.csv \
      --out sweeps_sigma/gof_diagnostic.csv
"""

from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rel-csv", default="sweeps/sweep_structure_eq9_sensitivity.csv")
    ap.add_argument("--sigma-csv", default="sweeps_sigma/sweep_structure.csv")
    ap.add_argument("--out", default="sweeps_sigma/gof_diagnostic.csv")
    args = ap.parse_args()

    rel = pd.read_csv(args.rel_csv)
    sig = pd.read_csv(args.sigma_csv)

    keep = ["SampleID", "assigned_LPM", "n_active", "tau1_DM",
            "chi2rel_DM", "chi2sig_DM"]
    m = rel[keep].merge(sig[keep], on=["SampleID", "assigned_LPM", "n_active"],
                        suffixes=("_eq9opt", "_sigmaopt"))

    m["tau1_shift_rel"] = (
        (m["tau1_DM_sigmaopt"] - m["tau1_DM_eq9opt"]).abs()
        / m["tau1_DM_eq9opt"])

    out = m.rename(columns={
        "tau1_DM_eq9opt": "tau1_DM_at_eq9opt",
        "tau1_DM_sigmaopt": "tau1_DM_at_sigmaopt",
        "chi2rel_DM_eq9opt": "chi2rel_DM_at_eq9opt",
        "chi2sig_DM_eq9opt": "chi2sig_DM_at_eq9opt",
        "chi2rel_DM_sigmaopt": "chi2rel_DM_at_sigmaopt",
        "chi2sig_DM_sigmaopt": "chi2sig_DM_at_sigmaopt",
    })
    out.to_csv(args.out, index=False)
    print(f"-> {args.out}  ({len(out)} samples)")

    print(f"\nn samples: {len(out)}")
    print(f"median chi2sig_DM at Eq.9 optimum   (wrong point): "
          f"{out['chi2sig_DM_at_eq9opt'].median():.2f}")
    print(f"median chi2sig_DM at sigma optimum  (correct point): "
          f"{out['chi2sig_DM_at_sigmaopt'].median():.2f}")
    print()
    shift = out["tau1_shift_rel"]
    print(f"tau1 shift |tau1_sigma - tau1_rel| / tau1_rel:")
    print(f"  median: {shift.median():.4f}")
    print(f"  p90:    {shift.quantile(0.90):.4f}")
    print(f"  max:    {shift.max():.4f}")
    n_material = int((shift > 0.05).sum())
    print(f"\n  samples with >5% tau1 shift: {n_material}/{len(out)}")
    print(out.sort_values("tau1_shift_rel", ascending=False)
          [["SampleID", "assigned_LPM", "tau1_DM_at_eq9opt",
            "tau1_DM_at_sigmaopt", "tau1_shift_rel"]].head(10)
          .to_string(index=False))


if __name__ == "__main__":
    main()
