#!/usr/bin/env python3
"""
make_tau2free_config.py -- build a config in which every binary mixture has
tau2 as a free parameter, for recomputing mean_age_sigma.

Why
---
mean_age = f1*tau1 + (1-f1)*tau2, and tau2 supplies a median 98% of it. The
Monte Carlo draws only contain the FREE parameters, so in the ten mixtures
where tau2 is prescribed it is identical across all 500 draws and contributes
nothing to mean_age_sigma. EDTRPAS1-46 is the extreme: its draws hold only
['tau1','f1'], tau2 sits at 23,500 yr in every one, and the reported sigma of
141 yr on an 18,136 yr mean age describes the wrong thing.

BMMModel caps free parameters at two, so tau2 can only be freed by pinning
tau1 (TracerLPM mode C). tau1 is pinned at the value currently reported for
that sample, so the only change is which transit time varies. The recomputed
sigma therefore captures the dominant term but excludes tau1 -- it is not
strictly larger in every case, and both should be reported.

Writes: edwards_input_config_tau2free.json  (+ prints the ids to pass to
run_mc_draws.py --ids)
"""
from __future__ import annotations

import json
import os

import pandas as pd

from profile_true import BASE, CFG, _HERE, SW

MODE_C = "Fraction, 2nd Mean Age"
OUT = os.path.join(_HERE, "examples", "edwards_aquifer",
                   "edwards_input_config_tau2free.json")


def main():
    SW.bootstrap(_HERE)
    blob = json.load(open(CFG))
    baseline = SW.load_baseline(BASE)
    study = set(pd.read_csv(os.path.join(
        _HERE, "sweeps_trueprofile",
        "sweep_identifiability_trueprofile.csv")).SampleID)

    changed = []
    for rec in blob["samples"]:
        u = SW.unpack(rec)
        sid = u["sid"]
        if u["lpm"] == "DM" or sid not in study or sid not in baseline:
            continue
        if MODE_C in str(u["free_params"]):
            continue                                   # tau2 already free
        row = baseline[sid]
        rec["lpm"]["free_params"] = MODE_C
        rec["lpm"]["tracerlpm_result"]["tau1_yr"] = float(row["tau1_PINN"])
        rec["lpm"]["init"]["tau1_yr"] = float(row["tau1_PINN"])
        if not rec["lpm"]["init"].get("tau2_yr"):
            rec["lpm"]["init"]["tau2_yr"] = float(row["tau2_PINN"])
        changed.append(sid)

    json.dump(blob, open(OUT, "w"), indent=1)
    print("wrote", OUT)
    print("switched to mode C (tau2 free, tau1 pinned): %d samples" % len(changed))
    for s in changed:
        print("   ", s)
    print("\nids for run_mc_draws.py --ids:")
    print(" ".join('"%s"' % s for s in changed))


if __name__ == "__main__":
    main()
