"""What does correcting sigma do to the weighted objective?

Evaluates chi2_sigma at the STORED optimum (no refitting) under
  (a) the sigma values the config carries -- what every chi2_sigma result in
      the manuscript actually used, and
  (b) the 1-sigma analytical uncertainties Musgrove's data release reports,
      applied as a RELATIVE uncertainty to the config's own observation so
      that no unit conversion is involved.

The goodness-of-fit probability under each tells us whether (b) is usable as
the paper's error model at all.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2 as X2

HO = r"D:\projects\carbon isotopic\DLMPI_reanalysis\DLPMI_edwards_handoff"
sys.path.insert(0, HO)
os.chdir(HO)
import dlpmi_sweeps as SW                              # noqa: E402
from profile_true import BASE, CFG                      # noqa: E402

N = lambda v: pd.to_numeric(pd.Series([v]), errors="coerce")[0]
D = r"../Musgrove_2023Data/"

# ---- reported relative 1-sigma, per sample and tracer ----------------------
t3 = pd.read_csv(D + "Table_3_Tracers.txt", sep="\t")
t5 = pd.read_csv(D + "Table_5_Carbon14.txt", sep="\t")
rel = {}
for _, r in t3.iterrows():
    k = str(r["SampleID"]).strip()
    d = rel.setdefault(k, {})
    for nm, (a, b) in (("3H", ("TRC_3H_TU", "TRC_3H_Err_TU")),
                       ("SF6", ("TRC_SF6_pptv", "TRC_SF6_Err_pptv")),
                       ("3He(trit)", ("TRC_3HeTrit_TU", "TRC_3HeTrit_Err_TU"))):
        o, e = N(r[a]), N(r[b])
        if np.isfinite(o) and np.isfinite(e) and o != 0:
            d[nm] = abs(e / o)
for _, r in t5.iterrows():
    k = str(r["SampleID"]).strip()
    o, e = N(r["14C_sample_pM"]), N(r["14C_Err_sample_pM"])
    if np.isfinite(o) and np.isfinite(e) and o != 0:
        rel.setdefault(k, {})["14C"] = abs(e / o)

SW.bootstrap(HO)
recs = json.load(open(CFG))["samples"]
baseline = SW.load_baseline(BASE)
study = pd.read_csv(os.path.join(
    "sweeps_trueprofile", "sweep_identifiability_trueprofile.csv")).set_index("SampleID")

MAP = {"tau1": "tau1_PINN", "f1": "f1_PINN", "tau2": "tau2_PINN", "pd1": "pd1_PINN"}
rows = []
for rec in recs:
    sid = rec["sample_id"]
    if sid not in study.index:
        continue
    u = SW.unpack(rec)
    b = baseline[sid]
    _, pn = SW.make_loss(u, rec)
    params = {n: float(b.get(MAP.get(n, n + "_PINN"), np.nan)) for n in pn}
    if any(not np.isfinite(v) for v in params.values()):
        continue
    try:
        sims = SW.simulate(u, rec, params, u["names"], u["scales"])
    except Exception:
        continue
    src = {}
    for k in [sid] + [x.strip() for x in sid.split("/")]:
        if k in rel:
            src = rel[k]
            break
    errs_cfg = list(u["errs"])
    errs_rep = [abs(o) * src[n] if n in src else e
                for n, o, e in zip(u["names"], u["obs"], errs_cfg)]
    n_sub = sum(1 for n in u["names"] if n in src)
    dof = max(len(u["names"]) - len(pn), 1)
    c_cfg = SW.chi2_sigma(sims, u["obs"], errs_cfg)
    c_rep = SW.chi2_sigma(sims, u["obs"], errs_rep)
    rows.append(dict(SampleID=sid, dof=dof, n_sub=n_sub,
                     chi2_cfg=c_cfg, chi2_rep=c_rep,
                     p_cfg=1 - X2.cdf(c_cfg, dof), p_rep=1 - X2.cdf(c_rep, dof)))

d = pd.DataFrame(rows)
print("evaluated at the stored optimum, n = %d samples\n" % len(d))
print("%-34s %12s %12s" % ("", "config sigma", "reported sigma"))
print("%-34s %12.2f %12.2f" % ("median chi2_sigma", d.chi2_cfg.median(), d.chi2_rep.median()))
print("%-34s %12.2f %12.2f" % ("max chi2_sigma", d.chi2_cfg.max(), d.chi2_rep.max()))
print("%-34s %12d %12d" % ("samples with p > 0.05 (fit OK)",
                           int((d.p_cfg > 0.05).sum()), int((d.p_rep > 0.05).sum())))
print("%-34s %12d %12d" % ("samples with chi2 > 100",
                           int((d.chi2_cfg > 100).sum()), int((d.chi2_rep > 100).sum())))
print()
print("Delta-chi2 = 1 as a fraction of the objective's own value:")
for lbl, c in (("config sigma", d.chi2_cfg), ("reported sigma", d.chi2_rep)):
    g = c[c > 0]
    print("   %-16s median 1/chi2 = %.4g" % (lbl, (1.0 / g).median()))
d.to_csv(os.path.join(HO, "sweeps_trueprofile", "sigma_impact.csv"), index=False)
print("\nwrote sweeps_trueprofile/sigma_impact.csv")
