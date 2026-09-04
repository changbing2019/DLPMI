#!/usr/bin/env python3
"""
run_mc_draws.py
===============
Monte Carlo uncertainty for every Edwards aquifer sample, storing the JOINT
parameter draws rather than marginal summary statistics.

Why this exists
---------------
edwards_run.py::_compute_mc accumulates tau1, f1 and tau2 into three separate
lists and returns only percentiles of each.  The per-draw correspondence
between parameters is therefore lost, and correlated parameters cannot be
recombined afterwards.  Any derived quantity that depends on more than one
parameter at once -- the young-water fraction F(a<T), the mixture-weighted
mean age -- cannot be propagated correctly from that output.

This script replicates the same perturbation scheme (observations perturbed by
their analytical errors, warm-started refit, rng seed 42) but writes an aligned
(n_draws x n_params) array per sample, so downstream metrics inherit the full
joint distribution.

Outputs
-------
  <out>/draws/<SampleID>.npz    arrays: draws, names, theta_star, chi2
  <out>/mc_summary.csv          per-sample percentiles (for cross-checking
                                against the original 00_summary_table.csv)

Usage
-----
  python run_mc_draws.py --n-mc 500 --processes 8
  python run_mc_draws.py --n-mc 100 --ids EDTRPAS1-36 --verbose
"""

from __future__ import annotations
import argparse, json, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")

EXCLUDE = {"EDTRPAS1-38", "EDTRPAS1-39", "EDTRPAS1-40",
           "EDTRPAS1-47", "EDTRPAS1-48"}

_ER = None
_TORCH = None


def bootstrap(pkg_root: str):
    global _ER, _TORCH
    if _ER is not None:
        return _ER
    pkg = os.path.abspath(pkg_root)
    for p in (pkg, os.path.join(pkg, "examples", "edwards_aquifer")):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    torch.set_num_threads(1)
    import edwards_run as er
    _TORCH, _ER = torch, er
    return er


def _prep_sample(rec: dict) -> dict:
    """Shared per-sample setup used by both the warm-start and cold-start
    paths: active-tracer lists, forward-model constants, and the free-
    parameter name order. Kept in one place so the two paths cannot silently
    diverge in how they build these."""
    er = _ER
    sid = rec["sample_id"]
    lpm = rec["lpm"]["name"]
    is_bmm = lpm != "DM"
    sd = float(rec["sample_date"])
    fp = rec["lpm"].get("free_params") or "Mean Age"

    tns, obs_v, obs_e, scls, act = er.build_tracer_lists(rec)
    keep = [i for i, a in enumerate(act) if a]
    tns_a = [tns[i] for i in keep]
    obs_a = [obs_v[i] for i in keep]
    err_a = [obs_e[i] for i in keep]
    scl_a = [scls[i] for i in keep]

    he4r = float(rec["he4_params"]["solution_rate_ccpgpyr"] or 1e-12)
    dic1 = float(rec["carbon14_params"]["dic_c1_mmolL"] or 100.)
    dic2 = float(rec["carbon14_params"]["dic_c2_mmolL"] or 100.)
    dg = rec.get("dgmeta_params", {}) or {}

    t1f, f1f, t2f = er._bmm_free_flags(fp) if is_bmm else (True, False, False)
    names = (["tau1", "pd1"] if not is_bmm else
             [n for n, f in zip(["tau1", "f1", "tau2"], [t1f, f1f, t2f]) if f])

    return dict(sid=sid, lpm=lpm, is_bmm=is_bmm, sd=sd, fp=fp,
                tns_a=tns_a, obs_a=obs_a, err_a=err_a, scl_a=scl_a,
                he4r=he4r, dic1=dic1, dic2=dic2, dg=dg, names=names)


def mc_one(rec: dict, n_mc: int, n_adam: int, n_starts: int,
           seed: int, out_dir: str, verbose: bool = False,
           cold_start: bool = False) -> dict:
    """Run MC for one sample, returning summary and writing joint draws."""
    er, torch = _ER, _TORCH
    from dlpmi.params import PINN_DM, PINN_BMM
    from dlpmi.forward import forward_BMM as _fwd_bmm

    ctx = _prep_sample(rec)
    sid, lpm, is_bmm = ctx["sid"], ctx["lpm"], ctx["is_bmm"]
    sd, fp = ctx["sd"], ctx["fp"]
    tns_a, obs_a, err_a, scl_a = (ctx["tns_a"], ctx["obs_a"],
                                  ctx["err_a"], ctx["scl_a"])
    he4r, dic1, dic2, dg = ctx["he4r"], ctx["dic1"], ctx["dic2"], ctx["dg"]

    # ── nominal fit ──────────────────────────────────────────────────────
    model, _, chi2_star = er._run_fit(rec, tns_a, obs_a, scl_a, he4r,
                                      dic1, dic2, n_adam, n_starts, False, dg)
    with torch.no_grad():
        if is_bmm:
            t1, p1, f1, t2, p2 = model.get_params()
            star = dict(tau1=float(t1), pd1=float(p1), f1=float(f1),
                        tau2=float(t2), pd2=float(p2))
        else:
            t1, p1 = model.get_params()
            star = dict(tau1=float(t1), pd1=float(p1))

    b = rec["pinn_bounds"]
    tau1_lo, pd1_lo = float(b["tau1_lo"]), float(b["pd1_lo"])
    pd1_hi = float(b["pd1_hi"])
    tau1_hi = float(b["tau1_hi"])
    if is_bmm:
        tau1_hi = min(tau1_hi, star["tau2"] * 0.9)

    names = ctx["names"]

    rng = np.random.default_rng(seed)
    rows, n_fail = [], 0

    for _ in range(n_mc):
        pert = [max(o + rng.normal(0., e), 1e-20)
                for o, e in zip(obs_a, err_a)]
        try:
            if cold_start:
                # Cold start (Task 3, REWORK_BRIEF.md): reuse er._run_fit
                # itself -- the same multi-start fit used for the nominal
                # optimum -- on the perturbed observations. It seeds from the
                # config's own lpm.init/tracerlpm_result values and bounds,
                # never from `star`, and runs the full n_adam with n_starts
                # restarts, unlike the warm-started single-shot refit below.
                m_draw, _, _ = er._run_fit(rec, tns_a, pert, scl_a, he4r,
                                           dic1, dic2, n_adam, n_starts,
                                           False, dg)
                with torch.no_grad():
                    if is_bmm:
                        a1, _, ff, a2, _ = m_draw.get_params()
                        vals = {"tau1": float(a1), "f1": float(ff),
                                "tau2": float(a2)}
                        rows.append([vals[n] for n in names])
                    else:
                        tv, pv = m_draw.get_params()
                        rows.append([float(tv), float(pv)])
            elif not is_bmm:
                m = PINN_DM(star["tau1"], star["pd1"],
                            tau1_lo, tau1_hi, pd1_lo, pd1_hi)

                def lfn(m=m, op=pert):
                    tau, pd_ = m.get_params()
                    ag = (er.AGES_OLD if float(tau.detach()) > 500.
                          else er.AGES_YOUNG)
                    return er.chi2_loss(
                        er.forward_DM(tau, pd_, sd, tns_a, scl_a, he4r,
                                      ages=ag, dgmeta=dg), op)
                er.train_pinn(m, lfn, min(n_adam // 5, 800))
                tv, pv = m.get_params()
                rows.append([float(tv), float(pv)])
            else:
                m = PINN_BMM(fp, star["tau1"], star["pd1"], star["f1"],
                             star["tau2"], star["pd2"],
                             tau1_lo, tau1_hi, 0.005, 0.997,
                             tau2_lo=max(1., star["tau2"] * 0.05),
                             tau2_hi=star["tau2"] * 5.)

                uz = float(rec["lpm"].get("uz_tt_yr") or 0.)

                def lfn(m=m, op=pert):
                    # NOTE: edwards_run.py imports the package function as
                    # _forward_BMM_pkg; the bare name forward_BMM is unbound
                    # inside its _compute_mc, so every BMM draw there raises
                    # NameError and is swallowed by the bare except.
                    a1, q1, ff, a2, q2 = m.get_params()
                    return er.chi2_loss(
                        _fwd_bmm("DM", {"tau": a1, "PD": q1},
                                 "DM", {"tau": a2, "PD": q2}, ff,
                                 sd, tns_a, scl_a, he4r, dic1, dic2, uz, dg),
                        op)
                er.train_pinn(m, lfn, min(n_adam // 5, 800))
                a1, _, ff, a2, _ = m.get_params()
                vals = {"tau1": float(a1), "f1": float(ff), "tau2": float(a2)}
                rows.append([vals[n] for n in names])
        except Exception:                                        # noqa: BLE001
            n_fail += 1

    draws = np.array(rows, dtype=float) if rows else np.zeros((0, len(names)))

    os.makedirs(os.path.join(out_dir, "draws"), exist_ok=True)
    safe = sid.replace("/", "_")
    np.savez_compressed(
        os.path.join(out_dir, "draws", f"{safe}.npz"),
        draws=draws, names=np.array(names, dtype=object),
        theta_star=np.array([star[n] for n in names], dtype=float),
        chi2=float(chi2_star))

    # A draw can converge to NaN without raising (torch propagates NaN
    # silently rather than throwing), so n_fail == 0 does not imply every row
    # is finite. Filter before any mean/std/percentile, exactly as
    # dlpmi/metrics.py::mc_metric() does -- otherwise a handful of NaN rows
    # poison every summary statistic for the whole sample.
    finite = draws[np.isfinite(draws).all(axis=1)] if draws.shape[0] else draws
    n_nonfinite = int(draws.shape[0] - finite.shape[0])

    out = dict(SampleID=sid, LPM=lpm, n_draws=int(draws.shape[0]),
               n_fail=n_fail, n_nonfinite=n_nonfinite, chi2=float(chi2_star))
    for j, n in enumerate(names):
        if finite.shape[0]:
            c = finite[:, j]
            out[f"{n}_star"] = star[n]
            out[f"{n}_mean"] = float(c.mean())
            out[f"{n}_std"] = float(c.std())
            out[f"{n}_p16"] = float(np.percentile(c, 16))
            out[f"{n}_p84"] = float(np.percentile(c, 84))
    # correlation between the first two free parameters, the quantity the
    # original marginal-only output discarded
    if finite.shape[0] > 2 and finite.shape[1] > 1:
        C = np.corrcoef(finite.T)
        out["corr_p1_p2"] = float(C[0, 1])
    if verbose:
        print(f"  {sid}: {finite.shape[0]}/{n_mc} ok "
              f"({n_nonfinite} non-finite)", flush=True)
    return out


def _worker(payload):
    rec, args = payload
    bootstrap(args.package)
    try:
        return mc_one(rec, args.n_mc, args.adam, args.starts,
                      args.seed, args.out, args.verbose,
                      cold_start=args.cold_start)
    except Exception as e:                                       # noqa: BLE001
        return dict(SampleID=rec["sample_id"], error=str(e)[:200])


def _cold_draw_worker(payload):
    """One cold-start draw: a full er._run_fit on perturbed observations.
    Flattened to (sample, draw) granularity -- not (sample,) -- so a small
    --ids list still uses all --processes workers instead of leaving most
    cores idle behind a handful of long sequential per-sample loops."""
    rec, ctx, pert, n_adam, n_starts, package = payload
    bootstrap(package)
    er, torch = _ER, _TORCH
    try:
        m, _, _ = er._run_fit(rec, ctx["tns_a"], pert, ctx["scl_a"],
                              ctx["he4r"], ctx["dic1"], ctx["dic2"],
                              n_adam, n_starts, False, ctx["dg"])
        with torch.no_grad():
            if ctx["is_bmm"]:
                a1, _, ff, a2, _ = m.get_params()
                vals = {"tau1": float(a1), "f1": float(ff), "tau2": float(a2)}
                row = [vals[n] for n in ctx["names"]]
            else:
                tv, pv = m.get_params()
                row = [float(tv), float(pv)]
        return (ctx["sid"], row, None)
    except Exception as e:                                       # noqa: BLE001
        return (ctx["sid"], None, str(e)[:150])


def run_cold_start(recs: list, args) -> None:
    """Cold-start MC (Task 3, REWORK_BRIEF.md): flat (sample, draw)
    parallelism across all --processes workers, rather than one process per
    sample serially looping its own draws -- with only a handful of --ids,
    the latter would leave most cores idle for hours."""
    bootstrap(args.package)
    er, torch = _ER, _TORCH

    sample_ctx, tasks = {}, []
    for rec in recs:
        ctx = _prep_sample(rec)
        sid = ctx["sid"]
        model, _, chi2_star = er._run_fit(
            rec, ctx["tns_a"], ctx["obs_a"], ctx["scl_a"], ctx["he4r"],
            ctx["dic1"], ctx["dic2"], args.adam, args.starts, False, ctx["dg"])
        with torch.no_grad():
            if ctx["is_bmm"]:
                t1, p1, f1, t2, p2 = model.get_params()
                star = dict(tau1=float(t1), pd1=float(p1), f1=float(f1),
                            tau2=float(t2), pd2=float(p2))
            else:
                t1, p1 = model.get_params()
                star = dict(tau1=float(t1), pd1=float(p1))
        sample_ctx[sid] = (rec, ctx, star, chi2_star)

        rng = np.random.default_rng(args.seed)   # same per-sample seeding
        for _ in range(args.n_mc):                # convention as mc_one
            pert = [max(o + rng.normal(0., e), 1e-20)
                    for o, e in zip(ctx["obs_a"], ctx["err_a"])]
            tasks.append((rec, ctx, pert, args.adam, args.starts,
                         args.package))

    print(f"Cold-start MC: {len(recs)} samples x {args.n_mc} draws "
          f"| processes={args.processes} | total draws={len(tasks)}",
          flush=True)

    results = {sid: [] for sid in sample_ctx}
    n_fail = {sid: 0 for sid in sample_ctx}
    if args.processes > 1:
        import multiprocessing as mp
        with mp.Pool(args.processes) as pool:
            for i, (sid, row, err) in enumerate(
                    pool.imap_unordered(_cold_draw_worker, tasks), 1):
                (results[sid].append(row) if row is not None
                 else n_fail.__setitem__(sid, n_fail[sid] + 1))
                if i % 20 == 0 or i == len(tasks):
                    print(f"  [{i}/{len(tasks)}] draws done", flush=True)
    else:
        for i, t in enumerate(tasks, 1):
            sid, row, err = _cold_draw_worker(t)
            (results[sid].append(row) if row is not None
             else n_fail.__setitem__(sid, n_fail[sid] + 1))
            print(f"  [{i}/{len(tasks)}] draws done", flush=True)

    os.makedirs(os.path.join(args.out, "draws"), exist_ok=True)
    summary_rows = []
    for sid, (rec, ctx, star, chi2_star) in sample_ctx.items():
        names = ctx["names"]
        draws = (np.array(results[sid], dtype=float) if results[sid]
                 else np.zeros((0, len(names))))
        safe = sid.replace("/", "_")
        np.savez_compressed(
            os.path.join(args.out, "draws", f"{safe}.npz"),
            draws=draws, names=np.array(names, dtype=object),
            theta_star=np.array([star[n] for n in names], dtype=float),
            chi2=float(chi2_star))
        finite = (draws[np.isfinite(draws).all(axis=1)]
                 if draws.shape[0] else draws)
        n_nonfinite = int(draws.shape[0] - finite.shape[0])
        row = dict(SampleID=sid, LPM=ctx["lpm"], n_draws=int(draws.shape[0]),
                   n_fail=n_fail[sid], n_nonfinite=n_nonfinite,
                   chi2=float(chi2_star))
        for j, n in enumerate(names):
            if finite.shape[0]:
                c = finite[:, j]
                row[f"{n}_star"] = star[n]
                row[f"{n}_mean"] = float(c.mean())
                row[f"{n}_std"] = float(c.std())
                row[f"{n}_p16"] = float(np.percentile(c, 16))
                row[f"{n}_p84"] = float(np.percentile(c, 84))
        if finite.shape[0] > 2 and finite.shape[1] > 1:
            C = np.corrcoef(finite.T)
            row["corr_p1_p2"] = float(C[0, 1])
        summary_rows.append(row)

    import pandas as pd
    df = pd.DataFrame(summary_rows)
    path = os.path.join(args.out, "mc_summary.csv")
    df.to_csv(path, index=False)
    print(f"\n-> {path}")
    print(f"-> {os.path.join(args.out, 'draws')}/  ({len(summary_rows)} npz files)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=".")
    ap.add_argument("--config", default=None)
    ap.add_argument("--out", default="mc")
    ap.add_argument("--n-mc", type=int, default=500)
    ap.add_argument("--adam", type=int, default=5000)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--keep-excluded", action="store_true")
    ap.add_argument("--verbose", action="store_true")
    ap.add_argument("--cold-start", action="store_true",
                    help="seed each draw from the config's own lpm.init/"
                         "tracerlpm_result values and bounds via a full "
                         "er._run_fit (same multi-start schedule as the "
                         "nominal fit), instead of warm-starting a single "
                         "refit from the nominal optimum with n_adam//5 "
                         "epochs. Much more expensive per draw -- intended "
                         "for a small --ids spot check, not the full run.")
    args = ap.parse_args()

    bootstrap(args.package)
    cfg = args.config or os.path.join(
        os.path.abspath(args.package), "examples", "edwards_aquifer",
        "edwards_input_config.json")
    recs = json.load(open(cfg))["samples"]
    if not args.keep_excluded:
        recs = [r for r in recs if r["sample_id"] not in EXCLUDE]
    if args.ids:
        recs = [r for r in recs if r["sample_id"] in set(args.ids)]
    if args.limit:
        recs = recs[:args.limit]

    if args.cold_start:
        os.makedirs(args.out, exist_ok=True)
        run_cold_start(recs, args)
        return

    os.makedirs(args.out, exist_ok=True)
    print(f"Monte Carlo: {len(recs)} samples x {args.n_mc} draws "
          f"| processes={args.processes}", flush=True)

    payloads = [(r, args) for r in recs]
    rows = []
    if args.processes > 1:
        import multiprocessing as mp
        with mp.Pool(args.processes) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, payloads), 1):
                rows.append(r)
                print(f"  [{i}/{len(payloads)}] {r.get('SampleID')}", flush=True)
    else:
        for i, p in enumerate(payloads, 1):
            r = _worker(p)
            rows.append(r)
            print(f"  [{i}/{len(payloads)}] {r.get('SampleID')}", flush=True)

    import pandas as pd
    df = pd.DataFrame(rows)
    path = os.path.join(args.out, "mc_summary.csv")
    df.to_csv(path, index=False)
    print(f"\n-> {path}")
    print(f"-> {os.path.join(args.out, 'draws')}/  ({len(rows)} npz files)")
    if "corr_p1_p2" in df:
        print(f"\nmedian |correlation| between first two free parameters: "
              f"{df.corr_p1_p2.abs().median():.3f}")
        print("(this is the information the marginal-only output discarded)")


if __name__ == "__main__":
    main()
