#!/usr/bin/env python3
"""objective_sensitivity.csv -- the data behind SI Table S10.

Model selection uses the measurement-error-weighted misfit of Eq. (S18),
because only that quantity is proportional to a Gaussian log-likelihood. This
script pairs it with the ranking obtained from fits driven by the relative-error
objective of Eq. (S9), and reports which samples change classification.

Both inputs come from dlpmi_sweeps.py (see DLPMI_edwards_handoff/REPRODUCE.md
section 4); nothing is refitted here.

    --objective sigma  ->  data_current/sweep_structure_sigma.csv   (Eq. S18)
    --objective rel    ->  sweeps_rel/sweep_structure.csv           (Eq. S9)

Assessability is dof >= 1, i.e. n_active >= 3: both structures carry k = 2, so
the AICc correction term is identical and cancels in the difference. Two of the
54 flip. Requiring n_active >= 4 gave 44 and one flip until 2026-08-24.

Usage:  python make_objective_sensitivity.py
"""
from __future__ import annotations

import argparse
import os

import pandas as pd

R = r"D:\projects\carbon isotopic\DLMPI_reanalysis"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sigma-csv", default=os.path.join(
        R, "From_Claude_chat", "data_current", "sweep_structure_sigma.csv"))
    ap.add_argument("--rel-csv", default=os.path.join(
        R, "DLPMI_edwards_handoff", "sweeps_rel", "sweep_structure.csv"))
    ap.add_argument("--out", default=os.path.join(
        R, "From_Claude_chat", "supplementary_data",
        "objective_sensitivity.csv"))
    A = ap.parse_args()

    sig = pd.read_csv(A.sigma_csv).set_index("SampleID")
    rel = pd.read_csv(A.rel_csv).set_index("SampleID").reindex(sig.index)

    d = pd.DataFrame({
        "SampleID": sig.index,
        "n_active": sig.n_active,
        "assigned_LPM": sig.assigned_LPM,
        "dAICc_eqS18": sig.dAICc,
        "dAICc_eqS9": rel.dAICc,
        "class_eqS18": sig.vs_assigned,
        "class_eqS9": rel.vs_assigned,
    }).reset_index(drop=True)
    d = d[d.n_active >= 3].copy()
    d["flips"] = d.class_eqS18 != d.class_eqS9

    os.makedirs(os.path.dirname(A.out), exist_ok=True)
    d.to_csv(A.out, index=False)
    print("wrote %s" % A.out)
    print("  assessable            %d" % len(d))
    print("  classification flips  %d" % int(d.flips.sum()))
    for r in d[d.flips].sort_values("dAICc_eqS18").itertuples():
        print("    %-26s S18 %8.1f %-18s   S9 %8.1f %s"
              % (r.SampleID, r.dAICc_eqS18, r.class_eqS18,
                 r.dAICc_eqS9, r.class_eqS9))
    print("\n  under Eq. S18: %s" % d.class_eqS18.value_counts().to_dict())
    print("  under Eq. S9 : %s" % d.class_eqS9.value_counts().to_dict())


if __name__ == "__main__":
    main()
