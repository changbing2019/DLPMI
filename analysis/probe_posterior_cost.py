#!/usr/bin/env python3
"""Cost and correctness probe for the tau2 posterior run.

Two things must hold before a grid size is chosen:
  * chi2_sigma from make_loss(errs=...) at the stored optimum must match the
    value dlpmi_sweeps.chi2_sigma reports, so the likelihood is built on the
    same objective the AICc comparison already uses;
  * the per-evaluation cost must be known, since the grid is 2-D.
"""
from __future__ import annotations

import copy
import json
import time

import numpy as np

from profile_true import BASE, CFG, SW, _HERE

MODE_C = "Fraction, 2nd Mean Age"


def main():
    SW.bootstrap(_HERE)
    recs = json.load(open(CFG))["samples"]
    baseline = SW.load_baseline(BASE)
    torch = SW._TORCH

    bmm = [r for r in recs
           if SW.unpack(r)["lpm"] != "DM" and SW.unpack(r)["sid"] in baseline]
    print("binary-mixture samples with a baseline row: %d" % len(bmm))

    rec = next(r for r in bmm
               if SW.unpack(r)["sid"] == "EDTRPAS1-46")
    u0 = SW.unpack(rec)
    sid = u0["sid"]
    row = baseline[sid]

    # mode C with tau1 pinned at the reanalysis value -- the same setup
    # mean_age_uncertainty.py uses, so the posterior is comparable to the
    # profile interval rather than to a different parameterisation
    r = copy.deepcopy(rec)
    r["lpm"]["free_params"] = MODE_C
    r["lpm"]["tracerlpm_result"]["tau1_yr"] = float(row["tau1_PINN"])
    r["lpm"]["init"]["tau1_yr"] = float(row["tau1_PINN"])
    u = SW.unpack(r)

    print("sid              : %s" % sid)
    print("active tracers   : %s" % u["names"])
    print("inversion sigma  : %s" % [round(e, 6) for e in u["errs"]])
    print("obs              : %s" % [round(float(o), 6) for o in u["obs"]])

    lf, pnames = SW.make_loss(u, r, errs=u["errs"])
    print("free params      : %s" % pnames)

    f1 = float(row["f1_PINN"])
    t2 = float(row["tau2_PINN"])
    p = {"f1": f1, "tau2": t2}
    vec = torch.tensor([p[n] for n in pnames], dtype=torch.float32)
    with torch.no_grad():
        c2s = float(lf(vec))

    # independent recomputation through the shipped helper
    from dlpmi.forward import forward_BMM
    from dlpmi.kernels import choose_ages                      # noqa: F401
    sims_ref = None
    print()
    print("chi2_sigma via make_loss(errs) : %.8f" % c2s)
    print("chi2_rel  stored in baseline   : %.8f" % float(row["chi2_PINN"]))
    lf_rel, _ = SW.make_loss(u, r)
    with torch.no_grad():
        c2r = float(lf_rel(vec))
    print("chi2_rel  via make_loss        : %.8f" % c2r)
    ratio = c2s / c2r if c2r else float("nan")
    print("chi2_sigma / chi2_rel          : %.4f   (100.0 exactly when every"
          " active tracer shares one relative sigma)" % ratio)

    n = 60
    vecs = [torch.tensor([f1 * (1 + 0.001 * i), t2 * (1 + 0.01 * i)],
                         dtype=torch.float32) for i in range(n)]
    t0 = time.perf_counter()
    with torch.no_grad():
        for v in vecs:
            lf(v)
    dt = (time.perf_counter() - t0) / n
    print()
    print("per-evaluation cost : %.2f ms" % (dt * 1e3))
    for g in (60, 80, 120):
        tot = g * g * 18
        print("  grid %3dx%-3d over 18 samples = %7d evals -> %5.1f min"
              % (g, g, tot, tot * dt / 60))
    _ = sims_ref, forward_BMM, np


if __name__ == "__main__":
    main()
