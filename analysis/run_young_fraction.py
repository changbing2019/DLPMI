#!/usr/bin/env python3
"""
run_young_fraction.py
=====================
Compute the young-water fraction F(a < T) and the mixture-weighted mean age
for every Edwards aquifer sample, with uncertainty derived from the exact
Hessian covariance -- as the delta-method sigma always, and as the interval
(p16/p50/p84) via either the joint Gaussian push-forward or, when
--mc-dir points at a run_mc_draws.py output, the joint Monte Carlo draws
themselves (dlpmi.metrics.mc_metric). The MC draws are preferred where
available because they carry the true (non-Gaussian, correlated) parameter
distribution rather than assuming local Gaussianity.

No refitting: parameters are read from the completed baseline summary table.

Usage
-----
  python run_young_fraction.py \
      --baseline-csv results/edwards_pinn/00_summary_table.csv \
      --config       examples/edwards_aquifer/edwards_input_config.json \
      --sites-xlsx   Table_1_Siteswsum_ages.xlsx \
      --mc-dir       mc \
      --out          sweeps/young_fraction.csv
"""

from __future__ import annotations
import argparse, json, os, sys, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

EXCLUDE = {"EDTRPAS1-38", "EDTRPAS1-39", "EDTRPAS1-40",
           "EDTRPAS1-47", "EDTRPAS1-48"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=".")
    ap.add_argument("--baseline-csv", required=True)
    ap.add_argument("--config", required=True)
    ap.add_argument("--sites-xlsx", default=None)
    ap.add_argument("--out", default="young_fraction.csv")
    ap.add_argument("--thresholds", default="10,25,40")
    ap.add_argument("--mc-dir-tau2free", default=None,
                    help="second run_mc_draws.py output, produced from a "
                         "config where tau2 is free and tau1 pinned "
                         "(make_tau2free_config.py). Used only for samples "
                         "whose primary draws cannot move a parameter the "
                         "mean age depends on.")
    ap.add_argument("--mc-dir", default=None,
                    help="output dir from run_mc_draws.py (contains "
                         "draws/<SampleID>.npz); when a sample has a draws "
                         "file, F(a<T) uncertainty is propagated through the "
                         "joint MC draws (mc_metric) instead of the Gaussian "
                         "push-forward, since the draws carry the true "
                         "parameter correlation rather than assuming it")
    args = ap.parse_args()

    pkg = os.path.abspath(args.package)
    for p in (pkg, os.path.join(pkg, "examples", "edwards_aquifer")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    from dlpmi.metrics import young_fraction, mixture_mean_age, \
        delta_method_sigma, gaussian_propagate, mc_metric, exit_age_grid
    import dlpmi_sweeps as sw

    sw.bootstrap(pkg)
    Ts = [float(x) for x in args.thresholds.split(",")]

    alt_draws_dir = args.mc_dir_tau2free
    base = pd.read_csv(args.baseline_csv).set_index("SampleID")
    recs = [r for r in json.load(open(args.config))["samples"]
            if r["sample_id"] not in EXCLUDE]

    rows = []
    for rec in recs:
        sid = rec["sample_id"]
        if sid not in base.index:
            continue
        b = base.loc[sid]
        u = sw.unpack(rec)
        is_bmm = u["lpm"] != "DM"
        nf, na = sw.n_free_of(rec), len(u["names"])
        dof = max(na - nf, 1)
        pd1 = float(rec["lpm"]["init"]["pd1"] or 0.01)
        pd2 = float(rec["lpm"]["init"]["pd2"] or 0.01)

        # exact Hessian covariance at the stored optimum
        params = {"tau1": b["tau1_PINN"], "pd1": b["pd1_PINN"],
                  "f1": b["f1_PINN"], "tau2": b["tau2_PINN"]}
        lf, pnames = sw.make_loss(u, rec)
        theta = torch.tensor([params[n] for n in pnames], dtype=torch.float64)
        H = np.atleast_2d(torch.autograd.functional.hessian(
            lambda p: lf(p.float()).double(), theta).detach().numpy())
        # factor of 2: H is d2(chi2)/dtheta2 and for chi2 = sum(r^2)
        # the Gauss-Newton Hessian is 2 J^T J, so Cov = 2 H^-1 chi2/dof.
        # Omitting it made every sigma low by sqrt(2) (fixed 2026-08-24;
        # see dlpmi/uncertainty.py and tests/test_hessian_covariance.py).
        _s = 2.0 * (float(b["chi2_PINN"]) / dof)
        try:
            cov = np.linalg.inv(H) * _s
        except np.linalg.LinAlgError:
            cov = np.linalg.pinv(H) * _s

        tmax = params["tau2"] if is_bmm and np.isfinite(params["tau2"]) \
            else params["tau1"]
        ages = exit_age_grid(float(tmax))

        def make_builder(names_list):
            """Map a free-parameter vector (in names_list order) onto full
            model parameters. Looked up by name, so this works unchanged for
            both the Hessian-based pnames and the MC draws' own names."""
            def build(p_vec):
                d = dict(zip(names_list, list(p_vec)))
                t1 = d.get("tau1", torch.tensor(float(params["tau1"]),
                                                dtype=torch.float64))
                if not is_bmm:
                    p1 = {"tau": t1, "PD": d.get("pd1", torch.tensor(
                        float(params["pd1"]), dtype=torch.float64))}
                    return "DM", p1, None, None
                f1 = d.get("f1", torch.tensor(float(params["f1"]),
                                              dtype=torch.float64))
                t2 = d.get("tau2", torch.tensor(float(params["tau2"]),
                                                dtype=torch.float64))
                p1 = {"tau": t1, "PD": torch.tensor(pd1, dtype=torch.float64)}
                p2 = {"tau": t2, "PD": torch.tensor(pd2, dtype=torch.float64)}
                return "DM", p1, f1, p2
            return build

        build = make_builder(pnames)

        mc_draws, mc_names = None, None
        if args.mc_dir:
            draws_path = os.path.join(args.mc_dir, "draws",
                                      f"{sid.replace('/', '_')}.npz")
            if os.path.exists(draws_path):
                with np.load(draws_path, allow_pickle=True) as npz:
                    mc_draws = npz["draws"]
                    mc_names = list(npz["names"])
        build_mc = make_builder(mc_names) if mc_draws is not None else None

        row = dict(SampleID=sid, Network=u["network"], AgeCat=u["age_cat"],
                   LPM=u["lpm"], n_active=na, n_free=nf,
                   tau1=params["tau1"], f1=params["f1"], tau2=params["tau2"])

        # ── mixture-weighted mean age ────────────────────────────────────
        # tau2 supplies a median 98% of this quantity but is a free parameter
        # in only 8 of the 18 binary mixtures. Where it is prescribed, neither
        # the Hessian covariance nor the primary MC draws can move it, so a
        # sigma taken from either is conditional on a fixed value that IS
        # essentially the answer -- EDTRPAS1-46 came out at 141 yr on an
        # 18,136 yr age. Compute every sigma available, record which
        # contributing parameters were held fixed, and report the envelope.
        def mean_fn(p_vec):
            m, p1, f1, p2 = build(p_vec)
            return mixture_mean_age(p1, f1, p2)

        val, sig_delta = delta_method_sigma(mean_fn, theta, cov)
        row["mean_age_mixture"] = val
        row["mean_age_sigma_delta"] = sig_delta

        sig_mc = np.nan
        if build_mc is not None:
            def mean_fn_mc(p_vec):
                m, p1, f1, p2 = build_mc(p_vec)
                return mixture_mean_age(p1, f1, p2)
            mcm = mc_metric(mean_fn_mc, mc_draws)
            sig_mc = mcm.get("std", np.nan)
            row["mean_age_p16"] = mcm.get("p16", np.nan)
            row["mean_age_p50"] = mcm.get("p50", np.nan)
            row["mean_age_p84"] = mcm.get("p84", np.nan)
        row["mean_age_sigma_mc"] = sig_mc

        free_here = set(pnames) | set(mc_names or [])
        contributors = ("tau1", "f1", "tau2") if is_bmm else ("tau1",)
        fixed = [p for p in contributors if p not in free_here]
        row["mean_age_fixed_contributors"] = ";".join(fixed)

        # a second draws directory (run_mc_draws.py over a config that frees
        # tau2 by pinning tau1) covers what the primary one structurally cannot
        sig_alt = np.nan
        if fixed and alt_draws_dir:
            alt_p = os.path.join(alt_draws_dir, "draws",
                                 f"{sid.replace('/', '_')}.npz")
            if os.path.exists(alt_p):
                with np.load(alt_p, allow_pickle=True) as npz:
                    alt_d, alt_n = npz["draws"], list(npz["names"])
                build_alt = make_builder(alt_n)

                def mean_fn_alt(p_vec):
                    m, p1, f1, p2 = build_alt(p_vec)
                    return mixture_mean_age(p1, f1, p2)
                sig_alt = mc_metric(mean_fn_alt, alt_d).get("std", np.nan)
        row["mean_age_sigma_alt"] = sig_alt

        # Envelope ONLY where a contributing parameter was actually held
        # fixed. With nothing fixed the delta-method sigma is already complete,
        # and switching estimator would silently move values that never had
        # the defect (all 42 DM samples fall here).
        if not fixed:
            row["mean_age_sigma"] = sig_delta
            row["mean_age_sigma_source"] = "delta"
        else:
            cands = {"delta": sig_delta, "mc": sig_mc, "alt": sig_alt}
            finite = {k: v for k, v in cands.items() if np.isfinite(v)}
            if finite:
                best = max(finite, key=finite.get)
                row["mean_age_sigma"] = finite[best]
                row["mean_age_sigma_source"] = best
            else:
                row["mean_age_sigma"] = np.nan
                row["mean_age_sigma_source"] = "none"

        for T in Ts:
            def f_fn(p_vec, T=T):
                m, p1, f1, p2 = build(p_vec)
                return young_fraction(m, p1, T, f1, m, p2, ages=ages)
            v, s = delta_method_sigma(f_fn, theta, cov)
            v = float(np.clip(v, 0.0, 1.0))
            row[f"F_lt{int(T)}yr"] = v
            row[f"F_lt{int(T)}yr_delta_sigma"] = s

            if build_mc is not None:
                def f_fn_mc(p_vec, T=T):
                    m, p1, f1, p2 = build_mc(p_vec)
                    return young_fraction(m, p1, T, f1, m, p2, ages=ages)
                mc = mc_metric(f_fn_mc, mc_draws)
                row[f"F_lt{int(T)}yr_p16"] = mc.get("p16", np.nan)
                row[f"F_lt{int(T)}yr_p50"] = mc.get("p50", np.nan)
                row[f"F_lt{int(T)}yr_p84"] = mc.get("p84", np.nan)
                row[f"F_lt{int(T)}yr_width"] = (mc.get("p84", np.nan)
                                                - mc.get("p16", np.nan))
                row[f"F_lt{int(T)}yr_source"] = "mc"
                row[f"F_lt{int(T)}yr_n"] = mc.get("n", 0)
            else:
                gp = gaussian_propagate(f_fn, theta, cov, n=4000,
                                        lo=0.0, hi=1.0)
                row[f"F_lt{int(T)}yr_p16"] = gp.get("p16", np.nan)
                row[f"F_lt{int(T)}yr_p50"] = gp.get("p50", np.nan)
                row[f"F_lt{int(T)}yr_p84"] = gp.get("p84", np.nan)
                row[f"F_lt{int(T)}yr_width"] = (gp.get("p84", np.nan)
                                                - gp.get("p16", np.nan))
                row[f"F_lt{int(T)}yr_source"] = "gaussian"
                row[f"F_lt{int(T)}yr_n"] = gp.get("n", 0)
        row["mc_draws_used"] = build_mc is not None
        rows.append(row)

    df = pd.DataFrame(rows)
    if args.sites_xlsx and os.path.exists(args.sites_xlsx):
        sites = (pd.read_excel(args.sites_xlsx)
                   [["SampleID", "Aq_class", "Well_class",
                     "LPM_MeanAgeFinal_yrs"]].drop_duplicates("SampleID"))
        df = df.merge(sites, on="SampleID", how="left")
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"-> {args.out}  ({len(df)} samples)")

    if args.mc_dir:
        n_mc = int(df["mc_draws_used"].sum())
        print(f"\n  joint MC draws used for {n_mc}/{len(df)} samples "
              f"(from {args.mc_dir}); {len(df) - n_mc} fell back to the "
              f"Gaussian push-forward (no draws file found)")

    print("\n=== young-water fraction by aquifer zone (median) ===")
    if "Aq_class" in df:
        cols = [f"F_lt{int(T)}yr" for T in Ts]
        print(df.groupby("Aq_class")[cols].median().round(3).to_string())
        print("\n=== median P16-P84 width (MC draws where available, "
              "else joint Gaussian push-forward) ===")
        print(df.groupby("Aq_class")[[f"{c}_width" for c in cols]]
                .median().round(3).to_string())
    print("\n=== mixture mean age vs published ===")
    if "LPM_MeanAgeFinal_yrs" in df:
        pub = pd.to_numeric(df["LPM_MeanAgeFinal_yrs"], errors="coerce")
        print(f"  DLPMI mixture mean : median {df.mean_age_mixture.median():.1f} yr")
        print(f"  Musgrove published : median {pub.median():.1f} yr")


if __name__ == "__main__":
    main()
