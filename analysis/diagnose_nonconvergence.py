#!/usr/bin/env python3
"""
diagnose_nonconvergence.py
============================
TASK_nonconvergence.md Part 1: diagnose and (if warranted) refit the three
samples flagged with projected_grad_norm > 1.0 and no active bound
(EDTRPAS1-46, SCTXSUS1-10/EDTRVFPS1-15, SCTXSUS1-04/EDTRVFPS1-20).

For each of --starts starts:
  - captures edwards_run._run_fit's own per-start stdout (already prints
    tau/PD or tau1/f1/tau2 and chi2 for every start; parsed via regex rather
    than reimplementing the fit loop)
  - records final chi2 and parameter vector

For the winning (returned) model:
  - Adam-final vs post-LBFGS loss, from _run_fit's own returned history
    (best_hist), to see whether LBFGS improved on Adam or the line search
    failed and the Adam iterate was effectively retained
  - projected gradient norm in BOTH natural parameter space and the
    sigmoid-reparametrised (r) space the optimiser actually works in,
    via the chain-rule factor dtheta/dr = (theta-lo)*(hi-theta)/(hi-lo)

Usage
-----
  python diagnose_nonconvergence.py --adam 20000 --starts 10 \
      --ids EDTRPAS1-46 "SCTXSUS1-10/EDTRVFPS1-15" "SCTXSUS1-04/EDTRVFPS1-20" \
      --out sweeps_reconv
"""
from __future__ import annotations
import argparse, copy, io, json, os, re, sys, warnings
from contextlib import redirect_stdout
import numpy as np

warnings.filterwarnings("ignore")

# _run_fit's own _seed_tau/_seed_f are hardcoded 3-element lists
# ([t0, t0*0.5, t0*2.][:n_starts]) -- passing --starts > 3 silently reuses
# just those 3 regardless. To genuinely explore more basins without
# reimplementing the fitter, call _run_fit repeatedly with different seed
# BASES (overriding tracerlpm_result.tau1_yr/fraction1, which is what
# tau1_0/f1_0 are read from), each call still using _run_fit's own real
# 3-point spread around that base.
SEED_BASE_MULTIPLIERS = [0.3, 0.6, 1.0, 1.5, 2.5, 4.0, 6.0]

START_RE_DM = re.compile(
    r"Start (\d+)/(\d+) done: .*?τ=([\d.eE+-]+) PD=([\d.eE+-]+) χ²=([\d.eE+-]+)")
START_RE_BMM = re.compile(
    r"Start (\d+)/(\d+) done: .*?τ₁=([\d.eE+-]+) f₁=([\d.eE+-]+) τ₂=([\d.eE+-]+) "
    r"χ²=([\d.eE+-]+)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=".")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ids", nargs="*", required=True)
    ap.add_argument("--adam", type=int, default=20000)
    ap.add_argument("--starts", type=int, default=10)
    ap.add_argument("--out", default="sweeps_reconv")
    args = ap.parse_args()

    pkg = os.path.abspath(args.package)
    for p in (pkg, os.path.join(pkg, "examples", "edwards_aquifer")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import torch
    import dlpmi_sweeps as sw
    sw.bootstrap(pkg)
    er = sw._ER

    cfg = args.config or os.path.join(pkg, "examples", "edwards_aquifer",
                                       "edwards_input_config.json")
    recs = {r["sample_id"]: r for r in json.load(open(cfg))["samples"]}

    os.makedirs(args.out, exist_ok=True)
    import pandas as pd
    all_starts_rows, summary_rows = [], []

    for sid in args.ids:
        rec0 = recs[sid]
        u = sw.unpack(rec0)
        is_bmm = u["lpm"] != "DM"
        print(f"\n=== {sid} (adam={args.adam} per call, "
              f"{len(SEED_BASE_MULTIPLIERS)} seed bases x 3 real starts each "
              f"= {len(SEED_BASE_MULTIPLIERS)*3} total starting points) ===",
              flush=True)

        lr0 = rec0["lpm"]["tracerlpm_result"]
        tau1_base = float(lr0.get("tau1_yr") or rec0["lpm"]["init"]["tau1_yr"] or 20.)
        f1_base = float(lr0.get("fraction1") or rec0["lpm"]["init"].get("fraction1") or 0.5) \
            if is_bmm else None

        best_overall = dict(chi2=float("inf"))
        pattern = START_RE_BMM if is_bmm else START_RE_DM

        for mult in SEED_BASE_MULTIPLIERS:
            rec = copy.deepcopy(rec0)
            rec["lpm"]["tracerlpm_result"]["tau1_yr"] = tau1_base * mult
            if is_bmm:
                f1_try = min(max(f1_base * mult, 0.01), 0.99)
                rec["lpm"]["tracerlpm_result"]["fraction1"] = f1_try

            buf = io.StringIO()
            with redirect_stdout(buf):
                model, hist, best_loss = er._run_fit(
                    rec, u["names"], u["obs"], u["scales"], u["he4r"],
                    u["dic_c1"], u["dic_c2"], args.adam, args.starts, True,
                    u["dgmeta"])
            captured = buf.getvalue()

            for m in pattern.finditer(captured):
                if is_bmm:
                    i, n, t1, f1, t2, c2 = m.groups()
                    all_starts_rows.append(dict(
                        SampleID=sid, seed_base_mult=mult, start=int(i),
                        tau1=float(t1), f1=float(f1), tau2=float(t2),
                        chi2=float(c2)))
                else:
                    i, n, tau, pd_, c2 = m.groups()
                    all_starts_rows.append(dict(
                        SampleID=sid, seed_base_mult=mult, start=int(i),
                        tau1=float(tau), pd1=float(pd_), chi2=float(c2)))

            if best_loss < best_overall["chi2"]:
                adam_final = hist[args.adam - 1] if len(hist) >= args.adam else np.nan
                best_overall = dict(chi2=best_loss, model=model, hist=hist,
                                    adam_final=adam_final, mult=mult)
            print(f"  seed_base_mult={mult}: best_chi2_this_base={best_loss:.6g}",
                  flush=True)

        model, hist, best_loss = (best_overall["model"], best_overall["hist"],
                                  best_overall["chi2"])
        adam_final = best_overall["adam_final"]
        post_lbfgs = hist[-1]
        n_lbfgs_evals = len(hist) - args.adam

        with torch.no_grad():
            if is_bmm:
                t1, p1, f1, t2, p2 = model.get_params()
                params = {"tau1": float(t1), "pd1": float(p1),
                          "f1": float(f1), "tau2": float(t2)}
            else:
                t1, p1 = model.get_params()
                params = {"tau1": float(t1), "pd1": float(p1)}

        lf, pnames = sw.make_loss(u, rec0)
        theta = torch.tensor([params[n] for n in pnames], dtype=torch.float64)
        theta_g = theta.clone().requires_grad_(True)
        g_t, = torch.autograd.grad(lf(theta_g.float()).double(), theta_g)
        g = g_t.detach().numpy()
        theta_np = theta.detach().numpy()
        bounds = sw.free_param_bounds(rec0, u, pnames)

        g_r = np.zeros_like(g)
        for i, name in enumerate(pnames):
            lo, hi = bounds[name]
            th = theta_np[i]
            dtheta_dr = (th - lo) * (hi - th) / (hi - lo)
            g_r[i] = g[i] * dtheta_dr

        starts_here = [r for r in all_starts_rows if r["SampleID"] == sid]
        chi2_vals = sorted(set(round(r["chi2"], 3) for r in starts_here))

        summary_rows.append(dict(
            SampleID=sid, is_bmm=is_bmm, best_chi2=best_loss,
            best_seed_mult=best_overall["mult"],
            adam_final_chi2=adam_final, post_lbfgs_chi2=post_lbfgs,
            lbfgs_improved=bool(post_lbfgs < adam_final) if np.isfinite(adam_final) else None,
            n_lbfgs_evals=n_lbfgs_evals,
            grad_norm_natural=float(np.linalg.norm(g)),
            grad_norm_rspace=float(np.linalg.norm(g_r)),
            n_total_starts=len(starts_here),
            n_distinct_chi2_basins=len(chi2_vals),
            distinct_chi2_values=str(chi2_vals),
            **{f"final_{n}": params[n] for n in pnames}))
        print(f"  -> OVERALL best_chi2={best_loss:.6g} (seed_mult={best_overall['mult']})  "
              f"adam_final={adam_final:.6g}  post_lbfgs={post_lbfgs:.6g}  "
              f"n_lbfgs_evals={n_lbfgs_evals}  "
              f"grad_natural={np.linalg.norm(g):.4g}  "
              f"grad_rspace={np.linalg.norm(g_r):.4g}  "
              f"n_starts={len(starts_here)} n_basins={len(chi2_vals)} {chi2_vals}",
              flush=True)

    pd.DataFrame(all_starts_rows).to_csv(
        os.path.join(args.out, "all_starts.csv"), index=False)
    pd.DataFrame(summary_rows).to_csv(
        os.path.join(args.out, "summary.csv"), index=False)
    print(f"\n-> {args.out}/all_starts.csv, {args.out}/summary.csv")


if __name__ == "__main__":
    main()
