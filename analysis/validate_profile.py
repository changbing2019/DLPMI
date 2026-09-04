"""Independent brute-force check of true_profile().

true_profile walks outward in 5%-of-opt steps and bisects the first crossing.
This recomputes the SAME profile curve on a dense uniform grid over the whole
parameter box and reads the crossings off it directly -- a different algorithm
against the same objective, so a disagreement localises the bug rather than
reproducing it. It also reports whether the profile curve is monotonic away
from the optimum, since the outward walk returns the FIRST crossing and would
silently miss a re-entrant interval.
"""
import json
import os
import sys

import numpy as np
import pandas as pd

HO = r"D:\projects\carbon isotopic\DLMPI_reanalysis\DLPMI_edwards_handoff"
sys.path.insert(0, HO)
os.chdir(HO)

import dlpmi_sweeps as SW                                        # noqa: E402
from profile_true import (BASE, CFG, _nuisance_min, true_profile)  # noqa: E402

NGRID = 240

SW.bootstrap(HO)
torch = SW._TORCH
recs = json.load(open(CFG))["samples"]
baseline = SW.load_baseline(BASE)
study = pd.read_csv(os.path.join(
    "sweeps_trueprofile", "sweep_identifiability_trueprofile.csv")).set_index("SampleID")

# a spread: closed intervals and hit-bound ones, DM and BMM
want = [s for s in study.index if pd.notna(study.loc[s, "w_tau1_true"])]
pick = ([s for s in want if not study.loc[s, "true_hit_bound"]][:4]
        + [s for s in want if study.loc[s, "true_hit_bound"]][:2])

print("dense-grid check of the tau1 profile, %d points per sample\n" % NGRID)
print("%-26s %9s %9s %9s %9s %9s %s"
      % ("SampleID", "walk_lo", "grid_lo", "walk_hi", "grid_hi", "max|d|", "monotonic"))

bad = []
for sid in pick:
    rec = next(r for r in recs if r["sample_id"] == sid)
    u = SW.unpack(rec)
    b = baseline[sid]
    params = {"tau1": float(b["tau1_PINN"])}
    lf, pnames = SW.make_loss(u, rec)
    if "tau1" not in pnames:
        continue
    params = {n: float(b.get(
        {"tau1": "tau1_PINN", "f1": "f1_PINN", "tau2": "tau2_PINN",
         "pd1": "pd1_PINN"}.get(n, n + "_PINN"), np.nan)) for n in pnames}
    if any(not np.isfinite(v) for v in params.values()):
        continue

    lo_w, hi_w, hit, ref, _ = true_profile(u, rec, params, pname="tau1")

    # ---- the same profile curve, read off a dense grid -----------------
    idx = pnames.index("tau1")
    base = [params[n] for n in pnames]
    bnds = SW.free_param_bounds(rec, u, pnames)
    lo_b, hi_b = bnds["tau1"]
    nuis = [j for j in range(len(pnames)) if j != idx]

    def chi2_at(v):
        if not nuis:
            p = list(base); p[idx] = v
            with torch.no_grad():
                return float(lf(torch.tensor(p, dtype=torch.float32)))
        j = nuis[0]
        nlo, nhi = bnds[pnames[j]]
        return _nuisance_min(lf, base, idx, j, v, nlo, nhi, torch)

    opt = params["tau1"]
    thr = ref + 1.0
    g = np.linspace(max(lo_b, opt * 0.02), min(hi_b, opt * 6.0), NGRID)
    g = np.unique(np.append(g, opt))
    c = np.array([chi2_at(float(x)) for x in g])

    def cross(side):
        """linear-interpolated first crossing of thr walking away from opt"""
        m = g > opt if side == "hi" else g < opt
        xs, cs = (g[m], c[m]) if side == "hi" else (g[m][::-1], c[m][::-1])
        for k in range(len(xs)):
            if cs[k] > thr:
                x0, c0 = (opt, c[np.argmin(np.abs(g - opt))]) if k == 0 else (xs[k - 1], cs[k - 1])
                if cs[k] == c0:
                    return float(xs[k])
                return float(x0 + (thr - c0) * (xs[k] - x0) / (cs[k] - c0))
        return np.nan

    gl, gh = cross("lo"), cross("hi")
    # monotonic away from opt? (a dip back below thr past the first crossing)
    hi_side = c[g > opt]
    lo_side = c[g < opt][::-1]
    def reentrant(arr):
        above = arr > thr
        return bool(above.any() and (~above[np.argmax(above):]).any())
    mono = not (reentrant(hi_side) or reentrant(lo_side))

    d_lo = abs(lo_w - gl) / opt if np.isfinite(gl) else np.nan
    d_hi = abs(hi_w - gh) / opt if np.isfinite(gh) else np.nan
    mx = np.nanmax([d_lo, d_hi])
    print("%-26s %9.3f %9s %9.3f %9s %9s %s%s"
          % (sid, lo_w, "%.3f" % gl if np.isfinite(gl) else "none",
             hi_w, "%.3f" % gh if np.isfinite(gh) else "none",
             "%.4f" % mx if np.isfinite(mx) else "  n/a",
             "yes" if mono else "NO -- re-entrant",
             "   [hit_bound]" if hit else ""))
    if np.isfinite(mx) and mx > 0.01:
        bad.append((sid, mx))

print()
print("agreement: %s" % ("all within 1%% of opt"
                         if not bad else "DISAGREEMENT " + str(bad)))
