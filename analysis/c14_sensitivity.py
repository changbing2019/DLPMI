#!/usr/bin/env python3
"""
c14_sensitivity.py -- how much do tau2 and the mixture mean age move when the
14C dead-carbon dilution factor is perturbed?

Context
-------
The 14C atmospheric input is scaled by a dead-carbon correction factor (the
`scale` field of the 14C tracer entry; dlpmi/tracers.py documents a typical
range of 0.40-0.90). In this dataset the factor is already SAMPLE-SPECIFIC,
taking seven distinct values from 0.40 to 0.90 across the 60 samples -- it is
not one assumed constant. What is missing is any propagation of uncertainty in
that factor, and the premodern component depends on it directly.

This perturbs each binary mixture's own factor by +/-0.10, clipped to
[0.40, 0.90], refits in the sample's ORIGINAL free-parameter mode so the
comparison is against the reported values, and reports how tau2 and the
mixture mean age move. Fitting goes through edwards_run._run_fit; nothing is
reimplemented and no reported value is altered.

Usage:  python c14_sensitivity.py --out sweeps_trueprofile
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time

import numpy as np
import pandas as pd

from profile_true import BASE, CFG, _HERE, SW

LO, HI = 0.40, 0.90
DELTAS = (-0.10, +0.10)


def c14_scale(rec):
    for t in rec["tracers"]:
        if t["name"] == "14C":
            return float(t["scale"])
    return None


def set_c14_scale(rec, v):
    for t in rec["tracers"]:
        if t["name"] == "14C":
            t["scale"] = float(v)


def mean_age(p):
    f1, t1, t2 = float(p["f1"]), float(p["tau1"]), float(p["tau2"])
    if not np.isfinite(f1) or not np.isfinite(t2):
        return float(t1)
    return f1 * t1 + (1.0 - f1) * t2


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
    print("binary mixtures in the study set: %d" % len(bmm), flush=True)

    rows = []
    for i, rec in enumerate(bmm, 1):
        sid = rec["sample_id"]
        b = baseline[sid]
        q0 = c14_scale(rec)
        t2_0 = float(b["tau2_PINN"])
        ma_0 = mean_age({"tau1": b["tau1_PINN"], "f1": b["f1_PINN"],
                         "tau2": b["tau2_PINN"]})
        for d in DELTAS:
            q = float(np.clip(q0 + d, LO, HI))
            if abs(q - q0) < 1e-9:                 # already at the limit
                continue
            r = copy.deepcopy(rec)
            set_c14_scale(r, q)
            u = SW.unpack(r)
            t0 = time.perf_counter()
            try:
                p, c2, _ = SW.run_fit(r, u["names"], u["obs"], u["scales"],
                                      A.adam, A.starts)
            except Exception as exc:                            # noqa: BLE001
                print("[%d/%d] %-26s q=%.2f FAILED: %s"
                      % (i, len(bmm), sid, q, exc), flush=True)
                continue
            t2 = float(p["tau2"])
            ma = mean_age(p)
            rows.append(dict(
                SampleID=sid, q_baseline=q0, q_perturbed=q, delta_q=q - q0,
                tau2_baseline=t2_0, tau2_perturbed=t2,
                tau2_shift_pct=100 * (t2 - t2_0) / t2_0 if t2_0 else np.nan,
                mean_age_baseline=ma_0, mean_age_perturbed=ma,
                mean_age_shift_pct=100 * (ma - ma_0) / ma_0 if ma_0 else np.nan,
                chi2=float(c2), seconds=round(time.perf_counter() - t0, 1)))
            r_ = rows[-1]
            print("[%d/%d] %-26s q %.2f->%.2f  tau2 %8.0f->%8.0f (%+7.1f%%)  "
                  "mean age %+7.1f%%" % (i, len(bmm), sid, q0, q,
                                         t2_0, t2, r_["tau2_shift_pct"],
                                         r_["mean_age_shift_pct"]), flush=True)

    if not rows:
        print("no perturbations run")
        return
    df = pd.DataFrame(rows)
    print("\n" + "=" * 72)
    print("perturbations: %d over %d samples" % (len(df), df.SampleID.nunique()))
    for col, lbl in (("tau2_shift_pct", "tau2"),
                     ("mean_age_shift_pct", "mixture mean age")):
        s = df[col].abs().dropna()
        print("  %-18s median |shift| %6.1f%%   p90 %6.1f%%   max %7.1f%%"
              % (lbl, s.median(), s.quantile(.9), s.max()))
    if A.out:
        os.makedirs(A.out, exist_ok=True)
        p = os.path.join(A.out, "c14_sensitivity.csv")
        df.to_csv(p, index=False)
        print("\nwrote", p)


if __name__ == "__main__":
    main()
