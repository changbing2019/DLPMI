#!/usr/bin/env python3
"""threshold_sensitivity.csv -- the data behind SI Table S6.

Table S6 had **no generator** and was a HYBRID of two eras. Its
well-constrained counts at the two tightest threshold pairs (1 at 0.3, 2 at
0.4) reproduce the retired *conditional* width column `w_tau1`, while its
middle rows reproduce the true-profile column `w_tau1_true`. The published
range "1 to 13" came from that mixture. Under the classification rule the
manuscript actually states -- Section 2.4 and Eq. (S13) -- the range is
**0 to 12**, and three of the six rows change. Fixed 2026-08-24.

The rule, which reproduces main-text Table 2 exactly (8 / 16 / 36 and all four
zone-by-bound rows):

    a profile with no Delta-chi2 = 1 crossing inside the search window is
    UNCONSTRAINED, whatever its sentinel width;
    otherwise  w <= w_well            -> well constrained
               w_well < w <= w_weak   -> weakly constrained
               w > w_weak             -> unconstrained

`true_hit_bound` marks the non-closing profiles; `w_tau1_true` is NaN for the
nine samples whose tau1 was never a free parameter, and those are unconstrained
by construction rather than by measurement (Section 2.4).

The zone columns exist because the Table S6 caption used to assert that "the
zone contrast persists throughout". It does not, uniformly: see the printout.

Usage:  python make_threshold_sensitivity.py
"""
from __future__ import annotations

import argparse
import os

import numpy as np
import pandas as pd
from scipy.stats import fisher_exact

R = r"D:\projects\carbon isotopic\DLMPI_reanalysis"

#: the six pairs Table S6 reports, tightest first
PAIRS = [(0.3, 1.0), (0.4, 1.2), (0.5, 1.5), (0.6, 1.8), (0.75, 2.0),
         (1.0, 2.5)]
REPORTED = (0.5, 1.5)          # the pair the manuscript uses


def classify(w, no_crossing, w_well, w_weak):
    if (not np.isfinite(w)) or no_crossing:
        return "unconstrained"
    if w <= w_well:
        return "well constrained"
    if w <= w_weak:
        return "weakly constrained"
    return "unconstrained"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ident", default=os.path.join(
        R, "DLPMI_edwards_handoff", "sweeps_trueprofile",
        "sweep_identifiability_trueprofile.csv"))
    ap.add_argument("--out", default=os.path.join(
        R, "From_Claude_chat", "supplementary_data",
        "threshold_sensitivity.csv"))
    A = ap.parse_args()

    d = pd.read_csv(A.ident)
    nc = d.true_hit_bound.fillna(True).astype(bool)

    rows = []
    for lo, hi in PAIRS:
        cls = pd.Series(
            [classify(w, b, lo, hi)
             for w, b in zip(d.w_tau1_true, nc)], index=d.index)
        rec = {"w_well": lo, "w_weak": hi,
               "well": int((cls == "well constrained").sum()),
               "weakly": int((cls == "weakly constrained").sum()),
               "unconstrained": int((cls == "unconstrained").sum())}
        # zone detail: the caption's claim needs it
        for z in ("unconfined", "confined"):
            m = d.Aq_class == z
            rec["%s_well" % z[:5]] = int((cls[m] == "well constrained").sum())
            rec["%s_weak" % z[:5]] = int((cls[m] == "weakly constrained").sum())
            rec["%s_unc" % z[:5]] = int((cls[m] == "unconstrained").sum())
        # dispersion test of Section 3.2: extreme classes vs the middle one
        mid = cls == "weakly constrained"
        tab = [[int(((d.Aq_class == "confined") & ~mid).sum()),
                int(((d.Aq_class == "confined") & mid).sum())],
               [int(((d.Aq_class == "unconfined") & ~mid).sum()),
                int(((d.Aq_class == "unconfined") & mid).sum())]]
        orr, p = fisher_exact(tab)
        rec["disp_OR"] = round(float(orr), 2)
        rec["disp_p"] = round(float(p), 4)
        rows.append(rec)

    out = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    out.to_csv(A.out, index=False)
    print("wrote", A.out)

    print("\n%-12s %5s %6s %6s   %-16s %-16s %s"
          % ("thresholds", "well", "weakly", "unconst",
             "unconfined w/wk/u", "confined w/wk/u", "dispersion"))
    for r in out.itertuples():
        star = "  <- reported" if (r.w_well, r.w_weak) == REPORTED else ""
        print("(%-4g,%-4g) %5d %6d %6d   %-16s %-16s OR %5.2f p %.3f%s"
              % (r.w_well, r.w_weak, r.well, r.weakly, r.unconstrained,
                 "%d/%d/%d" % (r.uncon_well, r.uncon_weak, r.uncon_unc),
                 "%d/%d/%d" % (r.confi_well, r.confi_weak, r.confi_unc),
                 r.disp_OR, r.disp_p, star))

    print("\nwell-constrained count ranges %d to %d across the six pairs"
          % (out.well.min(), out.well.max()))
    n_big = int((out.unconstrained
                 >= out[["well", "weakly"]].max(axis=1)).sum())
    print("unconstrained is the largest class in %d of %d pairs"
          % (n_big, len(out)))
    n_loc = int((out.confi_well > out.uncon_well).sum())
    print("confined holds more well-constrained estimates in %d of %d pairs"
          % (n_loc, len(out)))
    print("the dispersion contrast reaches p < 0.05 in %d of %d pairs"
          % (int((out.disp_p < 0.05).sum()), len(out)))


if __name__ == "__main__":
    main()
