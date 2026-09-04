#!/usr/bin/env python3
"""
recompute_mean_age_sigma.py -- mean_age_sigma with tau2 free.

mean_age = f1*tau1 + (1-f1)*tau2, and tau2 supplies a median 98% of it. But
tau2 is a free parameter in only 8 of the 18 binary mixtures; in the other 10
it is prescribed, so nothing that varies the free parameters can move it.
EDTRPAS1-46 reports 18,136 +/- 141 yr on a tau2 pinned at 23,500 yr that
supplies 100% of the value.

Three sigmas are reported side by side, because they are different estimators
and the distinction matters:

  sigma_delta_published  the shipped young_fraction.csv column. A FIRST-ORDER
                         DELTA-METHOD sigma off the Hessian covariance
                         (run_young_fraction.py:141) -- NOT a Monte Carlo
                         spread, which is why it differs from sigma_mc_orig
                         even though both exclude a prescribed tau2.
  sigma_mc_orig          Monte Carlo over the existing mc/draws.
  sigma_mc_tau2free      Monte Carlo over mc_tau2free/draws, the same 500-draw
                         scheme rerun with tau2 freed and tau1 pinned
                         (TracerLPM mode C).

None is complete: BMMModel allows only two free parameters, so tau1 and tau2
can never vary together. Mode A omits tau2, mode C omits tau1. Since tau2
dominates the mean age, mode C is the better single answer, but for a few
samples tau1 contributed more and the mode-A sigma is the larger one --
`sigma_recommended` is the envelope of all three.

Usage:  python recompute_mean_age_sigma.py --out sweeps_trueprofile
"""
from __future__ import annotations

import argparse
import glob
import os

import numpy as np
import pandas as pd

from profile_true import BASE, _HERE, SW


def load_draws(d, sid):
    """Return (names, draws) for a sample, or (None, None)."""
    safe = sid.replace("/", "_")
    p = os.path.join(d, "draws", safe + ".npz")
    if not os.path.exists(p):
        hits = [f for f in glob.glob(os.path.join(d, "draws", "*.npz"))
                if os.path.basename(f)[:-4] == safe]
        if not hits:
            return None, None
        p = hits[0]
    z = np.load(p, allow_pickle=True)
    return [str(x) for x in z["names"]], np.asarray(z["draws"], dtype=float)


def mean_age_stats(names, draws, fixed):
    """Mean-age distribution from joint draws; `fixed` supplies whatever the
    draw array does not carry."""
    if draws is None:
        return {}
    idx = {n: i for i, n in enumerate(names)}

    def get(row, key):
        return row[idx[key]] if key in idx else fixed[key]

    vals = []
    for row in draws:
        try:
            t1, f1, t2 = get(row, "tau1"), get(row, "f1"), get(row, "tau2")
            v = f1 * t1 + (1.0 - f1) * t2
        except (KeyError, TypeError):
            continue
        if np.isfinite(v):
            vals.append(v)
    if not vals:
        return {}
    v = np.array(vals)
    return {"n": len(v), "mean": v.mean(), "std": v.std(),
            "p16": np.percentile(v, 16), "p50": np.percentile(v, 50),
            "p84": np.percentile(v, 84)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc-orig", default="mc")
    ap.add_argument("--mc-tau2free", default="mc_tau2free")
    ap.add_argument("--out", default="")
    A = ap.parse_args()

    SW.bootstrap(_HERE)
    baseline = SW.load_baseline(BASE)
    yfc = pd.read_csv(os.path.join(
        r"D:\projects\carbon isotopic\DLMPI_reanalysis\From_Claude_chat",
        "data_current", "young_fraction.csv")).set_index("SampleID")
    ident = pd.read_csv(os.path.join(_HERE, "sweeps_trueprofile",
                                     "sweep_identifiability_trueprofile.csv"))
    bmm = ident[ident.LPM != "DM"]

    rows = []
    for _, r in bmm.iterrows():
        sid = r.SampleID
        b = baseline[sid]
        fixed = {"tau1": float(b["tau1_PINN"]), "f1": float(b["f1_PINN"]),
                 "tau2": float(b["tau2_PINN"])}
        ma_pt = fixed["f1"] * fixed["tau1"] + (1 - fixed["f1"]) * fixed["tau2"]

        n_a, d_a = load_draws(A.mc_orig, sid)
        s_a = mean_age_stats(n_a, d_a, fixed) if d_a is not None else {}
        n_c, d_c = load_draws(A.mc_tau2free, sid)
        s_c = mean_age_stats(n_c, d_c, fixed) if d_c is not None else {}

        tau2_in_a = bool(n_a and "tau2" in n_a)
        # The comparison baseline must be the FIRST-ORDER DELTA-METHOD sigma off
        # the Hessian covariance, not a Monte Carlo spread -- which is why it
        # differs from sigma_mc_orig below even though both exclude tau2 where
        # tau2 is prescribed.
        #
        # Read mean_age_sigma_DELTA, not mean_age_sigma. run_young_fraction.py
        # now sets mean_age_sigma to the ENVELOPE of the delta, MC and
        # tau2-free-MC estimates, which already incorporates this script's own
        # output. Reading it made the comparison circular: EDTRPAS1-46 came back
        # with sigma_delta_published = 19374 (its own tau2-free MC value) and
        # ratio_vs_published = 1.0, instead of 199.6 and 97.1. Fixed 2026-08-24;
        # the fallback keeps older young_fraction.csv files working.
        if sid in yfc.index:
            _c = ("mean_age_sigma_delta" if "mean_age_sigma_delta" in yfc.columns
                  else "mean_age_sigma")
            pub = float(yfc.loc[sid, _c])
        else:
            pub = np.nan
        rows.append(dict(
            SampleID=sid, mean_age=ma_pt,
            tau2_free_originally=tau2_in_a,
            sigma_delta_published=pub,
            sigma_delta_pct=100 * pub / ma_pt,
            sigma_mc_orig=s_a.get("std", np.nan),
            sigma_mc_orig_pct=100 * s_a.get("std", np.nan) / ma_pt if s_a else np.nan,
            sigma_mc_tau2free=s_c.get("std", np.nan),
            sigma_mc_tau2free_pct=100 * s_c.get("std", np.nan) / ma_pt if s_c else np.nan,
            p16_tau2free=s_c.get("p16", np.nan), p84_tau2free=s_c.get("p84", np.nan),
            n_draws_tau2free=s_c.get("n", 0),
            sigma_recommended=np.nanmax([s_a.get("std", np.nan),
                                         s_c.get("std", np.nan), pub]),
        ))

    df = pd.DataFrame(rows).sort_values("mean_age", ascending=False)
    df["sigma_recommended_pct"] = 100 * df.sigma_recommended / df.mean_age
    df["ratio_vs_published"] = df.sigma_recommended / df.sigma_delta_published

    pd.set_option("display.width", 250)
    print(df[["SampleID", "tau2_free_originally", "mean_age",
              "sigma_delta_published", "sigma_delta_pct",
              "sigma_mc_orig_pct", "sigma_mc_tau2free", "sigma_mc_tau2free_pct",
              "sigma_recommended_pct", "ratio_vs_published"]].round(
        {"mean_age": 0, "sigma_delta_published": 0, "sigma_delta_pct": 1,
         "sigma_mc_orig_pct": 1, "sigma_mc_tau2free": 0,
         "sigma_mc_tau2free_pct": 1, "sigma_recommended_pct": 1,
         "ratio_vs_published": 1}).to_string(index=False))

    upd = df[~df.tau2_free_originally & df.sigma_mc_tau2free.notna()]
    print("\nrecomputed for %d samples where tau2 was prescribed" % len(upd))
    if len(upd):
        print("  median published (delta) sigma  %7.1f%%" % upd.sigma_delta_pct.median())
        print("  median MC sigma, tau2 free      %7.1f%%" % upd.sigma_mc_tau2free_pct.median())
        print("  median growth factor            x%.1f" % upd.ratio_vs_published.median())
        print("  largest growth                  x%.0f  (%s)"
              % (upd.ratio_vs_published.max(),
                 upd.loc[upd.ratio_vs_published.idxmax(), "SampleID"]))

    if A.out:
        os.makedirs(A.out, exist_ok=True)
        p = os.path.join(A.out, "mean_age_sigma_recomputed.csv")
        df.to_csv(p, index=False)
        print("\nwrote", p)


if __name__ == "__main__":
    main()
