#!/usr/bin/env python3
"""
verify_hlog_exact.py
=====================
Independent cross-check of dlpmi_sweeps.py::sweep_ident's scaled-Hessian
shortcut (H_scaled = H * outer(theta,theta), diagonal gradient-correction term
omitted and only reported via gradient_check) against the TRUE log-parameter
Hessian, computed by directly reparametrizing the loss as a function of
ln(theta) and differentiating through autograd. This exercises a completely
independent code path (torch.autograd.functional.hessian on a genuinely
different function, not an algebraic manipulation of the same H), so it
catches transcription bugs the shortcut derivation itself could not.

If the two agree (within the correction term's own reported size), the
shortcut is correct and the "negligible" claim in sweep_ident is justified
rather than assumed.

Usage
-----
  python verify_hlog_exact.py --baseline-csv reference/00_summary_table.csv \
      --ids "SCTXSUS1-04/EDTRVFPS1-20" EDTRPAS1-42
"""
from __future__ import annotations
import argparse, json, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")


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
        theta = torch.tensor([params[n] for n in pnames], dtype=torch.float64)
        theta_np = theta.detach().numpy()

        # --- shortcut path (exactly what sweep_ident does) ---
        H = torch.autograd.functional.hessian(
            lambda p: lf(p.float()).double(), theta).detach().numpy()
        H = np.atleast_2d(H)
        theta_g = theta.clone().requires_grad_(True)
        grad_t, = torch.autograd.grad(lf(theta_g.float()).double(), theta_g)
        g = grad_t.detach().numpy()
        H_scaled_shortcut = H * np.outer(theta_np, theta_np)
        H_scaled_with_correction = H_scaled_shortcut + np.diag(theta_np * g)

        # --- independent path: reparametrize as a function of phi = ln(theta),
        # differentiate through autograd directly. theta = exp(phi) is
        # substituted before calling the SAME lf used above, so this exercises
        # a genuinely different computational graph, not an algebraic
        # rearrangement of the same numbers. ---
        phi0 = torch.log(theta).detach().clone()

        def lf_log(phi, lf=lf):
            th = torch.exp(phi)
            return lf(th.float())

        H_log_true = torch.autograd.functional.hessian(
            lambda p: lf_log(p).double(), phi0).detach().numpy()
        H_log_true = np.atleast_2d(H_log_true)

        diff_shortcut = np.abs(H_scaled_shortcut - H_log_true)
        diff_corrected = np.abs(H_scaled_with_correction - H_log_true)
        scale = np.abs(H_log_true) + 1e-30

        print(f"=== {sid}  (pnames={pnames}) ===")
        print("H_log_true (direct ln(theta) autograd):")
        print(H_log_true)
        print("H_scaled shortcut (no diagonal correction):")
        print(H_scaled_shortcut)
        print(f"max relative diff, shortcut vs true:   "
              f"{(diff_shortcut / scale).max():.3e}")
        print(f"max relative diff, corrected vs true:  "
              f"{(diff_corrected / scale).max():.3e}")

        ev_shortcut = np.linalg.eigvalsh(H_scaled_shortcut)
        ev_true = np.linalg.eigvalsh(H_log_true)
        k_shortcut = (abs(ev_shortcut.max() / ev_shortcut.min())
                     if ev_shortcut.min() != 0 else np.inf)
        k_true = (abs(ev_true.max() / ev_true.min())
                 if ev_true.min() != 0 else np.inf)
        print(f"cond_number  shortcut={k_shortcut:.6g}  true={k_true:.6g}  "
              f"rel_diff={abs(k_shortcut-k_true)/max(abs(k_true),1e-30):.3e}")
        print()


if __name__ == "__main__":
    main()
