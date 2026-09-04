#!/usr/bin/env python3
"""
verify_scale_invariance.py
===========================
Verification 1 (decisive) for TASK_kappa_diagnostic.md: recompute the
identifiability Hessian for a handful of samples with tau parameters
expressed in days instead of years, and confirm cond_number changes by
~365.25^2 while cond_number_scaled is unchanged to several significant
figures. If cond_number_scaled moves, the scaling in sweep_ident is wrong.

This re-derives the Hessian end-to-end through the real autograd/forward-model
path (dlpmi_sweeps.make_loss), not an analytic shortcut: the loss is wrapped
so its tau-type arguments are interpreted as days and converted back to years
before calling the actual forward model, and theta/the Hessian are recomputed
fresh at the day-scaled optimum.

Usage
-----
  python verify_scale_invariance.py --baseline-csv reference/00_summary_table.csv \
      --ids EDTRPAS1-36 EDTRPAS1-42 SCTXLUSRC1-11 SCTXSUS1-13/EDTRVFPS1-08 EDTRVFPS1-11
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")

DAYS_PER_YEAR = 365.25
TAU_PARAMS = {"tau1", "tau2"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=".")
    ap.add_argument("--config", default=None)
    ap.add_argument("--baseline-csv", required=True)
    ap.add_argument("--ids", nargs="*", required=True)
    args = ap.parse_args()

    pkg = os.path.abspath(args.package)
    for p in (pkg, os.path.join(pkg, "examples", "edwards_aquifer")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    import dlpmi_sweeps as sw
    sw.bootstrap(pkg)

    import pandas as pd
    base = pd.read_csv(args.baseline_csv).set_index("SampleID")
    cfg = args.config or os.path.join(pkg, "examples", "edwards_aquifer",
                                       "edwards_input_config.json")
    recs = {r["sample_id"]: r for r in json.load(open(cfg))["samples"]}

    for sid in args.ids:
        rec = recs[sid]
        row = base.loc[sid]
        u = sw.unpack(rec)
        params = {"tau1": row["tau1_PINN"], "pd1": row["pd1_PINN"],
                  "f1": row["f1_PINN"], "tau2": row["tau2_PINN"]}

        lf, pnames = sw.make_loss(u, rec)
        theta_yr = torch.tensor([params[n] for n in pnames], dtype=torch.float64)

        H_yr = torch.autograd.functional.hessian(
            lambda p: lf(p.float()).double(), theta_yr).detach().numpy()
        H_yr = np.atleast_2d(H_yr)
        ev_yr = np.linalg.eigvalsh(H_yr)
        kappa_yr = abs(ev_yr.max() / ev_yr.min()) if ev_yr.min() != 0 else np.inf
        theta_yr_np = theta_yr.detach().numpy()
        theta_yr_g = theta_yr.clone().requires_grad_(True)
        g_yr_t, = torch.autograd.grad(lf(theta_yr_g.float()).double(), theta_yr_g)
        g_yr = g_yr_t.detach().numpy()
        H_yr_scaled = (H_yr * np.outer(theta_yr_np, theta_yr_np)
                      + np.diag(theta_yr_np * g_yr))
        ev_yr_s = np.linalg.eigvalsh(H_yr_scaled)
        kappa_yr_s = (abs(ev_yr_s.max() / ev_yr_s.min())
                     if ev_yr_s.min() != 0 else np.inf)

        tau_idx = [i for i, n in enumerate(pnames) if n in TAU_PARAMS]
        if not tau_idx:
            print(f"{sid}: no tau-type free parameter (pnames={pnames}), skipping")
            continue

        def lf_days(p_days, lf=lf, tau_idx=tau_idx):
            p_yr = p_days.clone()
            for i in tau_idx:
                p_yr = p_yr.clone()
                p_yr[i] = p_days[i] / DAYS_PER_YEAR
            return lf(p_yr)

        theta_days = theta_yr.clone()
        for i in tau_idx:
            theta_days[i] = theta_days[i] * DAYS_PER_YEAR

        H_days = torch.autograd.functional.hessian(
            lambda p: lf_days(p.float()).double(), theta_days).detach().numpy()
        H_days = np.atleast_2d(H_days)
        ev_days = np.linalg.eigvalsh(H_days)
        kappa_days = (abs(ev_days.max() / ev_days.min())
                     if ev_days.min() != 0 else np.inf)
        theta_days_np = theta_days.detach().numpy()
        theta_days_g = theta_days.clone().requires_grad_(True)
        g_days_t, = torch.autograd.grad(
            lf_days(theta_days_g.float()).double(), theta_days_g)
        g_days = g_days_t.detach().numpy()
        H_days_scaled = (H_days * np.outer(theta_days_np, theta_days_np)
                         + np.diag(theta_days_np * g_days))
        ev_days_s = np.linalg.eigvalsh(H_days_scaled)
        kappa_days_s = (abs(ev_days_s.max() / ev_days_s.min())
                       if ev_days_s.min() != 0 else np.inf)

        ratio = kappa_days / kappa_yr if np.isfinite(kappa_yr) and kappa_yr != 0 else np.nan
        rel_diff_scaled = (abs(kappa_days_s - kappa_yr_s) / kappa_yr_s
                          if np.isfinite(kappa_yr_s) and kappa_yr_s != 0 else np.nan)

        print(f"=== {sid}  (pnames={pnames}, tau_idx={tau_idx}) ===")
        print(f"  raw   kappa: years={kappa_yr:.6g}  days={kappa_days:.6g}  "
              f"ratio={ratio:.4g}  (365.25^2={DAYS_PER_YEAR**2:.4g})")
        print(f"  scaled kappa: years={kappa_yr_s:.6g}  days={kappa_days_s:.6g}  "
              f"rel_diff={rel_diff_scaled:.3e}")
        print()


if __name__ == "__main__":
    main()
