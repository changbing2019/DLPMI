#!/usr/bin/env python3
"""
profile_true.py -- true profile-likelihood intervals for tau1.

`dlpmi_sweeps.profile_width` fixes the profiled parameter and holds every other
parameter at its optimum: a CONDITIONAL chi2 interval, not a profile likelihood.
This recomputes the interval with the nuisance parameter re-optimised at each
step, which is what the manuscript claims.

Profiling tau1 always leaves at most ONE free nuisance parameter here (k = 2 for
both DM and BMM-DM-DM; dlpmi/params.py never frees three), so the inner problem
is a bounded scalar minimisation -- a dense grid plus golden-section refine,
which is more reliable than gradient descent and cannot hit the
sigmoid-saturation trap in HANDOFF.md because it never uses the sigmoid
reparametrisation. The forward model and objective are unchanged: both come from
`dlpmi_sweeps.make_loss`, the same path the Hessians use.

Two details matter for the interval to be comparable with the published one:

  * the Delta-chi2 = 1 threshold is referenced to the STORED fitted chi2 -- the
    same global minimum the published conditional intervals used -- so the only
    thing that differs is the nuisance re-optimisation;
  * the stored optimum's own nuisance value is forced into the inner search
    grid. Without it the grid can miss that value, the profile curve can sit
    slightly ABOVE the conditional curve, and the interval comes out narrower
    than the conditional one -- which a profile interval can never legitimately
    be. With it, w_true >= w_cond holds for every sample.

Modes
  --mode ident   full-suite tau1 profile for every sample
  --mode loto    leave-one-tracer-out: re-profile full and reduced suites and
                 recompute the information ratio
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
sys.path.insert(0, os.path.join(_HERE, "examples", "edwards_aquifer"))

import dlpmi_sweeps as SW

CFG = os.path.join("examples", "edwards_aquifer", "edwards_input_config.json")
BASE = os.path.join("reference", "00_summary_table.csv")
LOTO_IN = os.path.join("sweeps", "sweep_loto.csv")
N_GRID, N_REFINE, N_BISECT = 48, 24, 30
W_WEAK = SW.W_WEAK


def _nuisance_min(lf, base, idx, j, v, nlo, nhi, torch):
    """min over the single nuisance parameter j with parameter idx fixed at v."""
    def g(x):
        p = list(base); p[idx] = v; p[j] = x
        with torch.no_grad():
            return float(lf(torch.tensor(p, dtype=torch.float32)))

    if nlo > 0 and nhi / max(nlo, 1e-12) > 50:
        xs = np.exp(np.linspace(np.log(nlo), np.log(nhi), N_GRID))
    else:
        xs = np.linspace(nlo, nhi, N_GRID)
    # The stored optimum's own nuisance value must be a candidate: it is the
    # value the conditional interval holds fixed, so including it guarantees
    # the profile curve lies at or below the conditional curve pointwise, and
    # hence that the profile interval is never narrower than the conditional
    # one -- which a profile interval can never legitimately be.
    # inserted in sorted position, not appended: the golden-section bracket
    # below is xs[k-1]..xs[k+1], so appending out of order gives a reversed
    # bracket that does not straddle the winning point whenever the appended
    # value is itself the grid minimum.
    xs = np.sort(np.append(xs, np.clip(float(base[j]), nlo, nhi)))
    vals = [g(float(x)) for x in xs]
    k = int(np.argmin(vals)); best = vals[k]
    a = float(xs[max(k - 1, 0)]); b = float(xs[min(k + 1, len(xs) - 1)])
    phi = 0.5 * (np.sqrt(5.0) - 1.0)
    c, dd = b - phi * (b - a), a + phi * (b - a)
    fc, fd = g(c), g(dd)
    for _ in range(N_REFINE):
        if fc < fd:
            b, dd, fd = dd, c, fc; c = b - phi * (b - a); fc = g(c)
        else:
            a, c, fc = c, dd, fd; dd = a + phi * (b - a); fd = g(dd)
    return min(best, fc, fd)


def true_profile(u, rec, params, pname="tau1", names=None, obs=None,
                 scales=None, dchi2=1.0, chi2min=None, errs=None):
    """(lo, hi, hit, chi2_ref, n_eval) for a true profile interval on `pname`."""
    torch = SW._TORCH
    # errs, when given, switches make_loss to the measurement-error-weighted
    # objective so that profiling matches the objective the fit used
    lf, pnames = SW.make_loss(u, rec, names, obs, scales, errs=errs)
    if pname not in pnames:
        return np.nan, np.nan, False, np.nan, 0
    idx = pnames.index(pname)
    base = [params[n] for n in pnames]
    bnds = SW.free_param_bounds(rec, u, pnames)
    lo_b, hi_b = bnds[pname]
    nuis = [j for j in range(len(pnames)) if j != idx]
    assert len(nuis) <= 1, f"{u['sid']}: {len(nuis)} nuisance params"
    calls = [0]

    def chi2_at(v):
        calls[0] += 1
        if not nuis:
            p = list(base); p[idx] = v
            with torch.no_grad():
                return float(lf(torch.tensor(p, dtype=torch.float32)))
        j = nuis[0]
        nlo, nhi = bnds[pnames[j]]
        return _nuisance_min(lf, base, idx, j, v, nlo, nhi, torch)

    opt = float(params[pname])
    # Reference chi2 = the fitted global minimum, the same value the published
    # conditional intervals used. Using the stored minimum (rather than a
    # re-minimised neighbourhood value) keeps the comparison apples-to-apples:
    # the ONLY thing that differs from the published interval is that the
    # nuisance parameter is now re-optimised at each step.
    chi2_ref = float(chi2min) if chi2min is not None else chi2_at(opt)
    thr = chi2_ref + dchi2

    def bisect(inside, outside):
        """Crossing of `thr` between `inside` (chi2 <= thr) and `outside`
        (chi2 > thr). Either may be numerically larger -- only the roles
        matter, because this is plain interval halving.

        The roles are asserted. Passing them the wrong way round does not
        fail loudly: the loop simply marches one endpoint onto the other and
        returns it. That is what happened on the lower side before
        2026-08-23, where bisect(lo, lo + step) had `lo` (which exceeds thr)
        in the `inside` slot, quantising every lower bound to an exact
        multiple of the coarse step."""
        assert chi2_at(inside) <= thr < chi2_at(outside), (
            "bisect called with its endpoints reversed")
        a, b = inside, outside
        for _ in range(N_BISECT):
            mid = 0.5 * (a + b)
            if chi2_at(mid) > thr:
                b = mid
            else:
                a = mid
        return 0.5 * (a + b)

    step = max(abs(opt) * 0.05, 0.1)
    hit = False

    hi, found = opt, False
    while hi + step < hi_b:
        hi += step
        if chi2_at(hi) > thr:
            hi = bisect(hi - step, hi); found = True; break
    if not found:
        hi = min(opt * 3.0, hi_b); hit = True

    lo, found = opt, False
    while lo - step > lo_b:
        lo -= step
        if chi2_at(lo) > thr:
            lo = bisect(lo + step, lo); found = True; break
    if not found:
        lo = max(lo_b, opt * 0.1); hit = True

    return float(lo), float(hi), bool(hit), float(chi2_ref), calls[0]


# ── mode: ident ─────────────────────────────────────────────────────────
def run_ident(recs, baseline):
    rows = []
    for i, rec in enumerate(recs, 1):
        u = SW.unpack(rec); sid = u["sid"]
        row = baseline.get(sid)
        if row is None:
            continue
        params = {"tau1": row["tau1_PINN"], "pd1": row["pd1_PINN"],
                  "f1": row["f1_PINN"], "tau2": row["tau2_PINN"]}
        c2 = float(row["chi2_PINN"])
        t0 = time.perf_counter()
        lo_t, hi_t, hit_t, ref, nev = true_profile(u, rec, params, chi2min=c2)
        dt = time.perf_counter() - t0

        lf, pnames = SW.make_loss(u, rec)
        bnds = SW.free_param_bounds(rec, u, pnames)
        if "tau1" in pnames:
            lo_c, hi_c = SW.profile_width(u, rec, params, u["names"], u["obs"],
                                          u["scales"], c2, "tau1", *bnds["tau1"])
        else:
            lo_c = hi_c = np.nan
        t1 = float(params["tau1"])
        f = np.isfinite
        rows.append(dict(
            SampleID=sid, LPM=u["lpm"], free_params=u["free_params"],
            n_active=len(u["names"]), tau1=t1, chi2_stored=c2, chi2_profile_ref=ref,
            tau1_lo_cond=lo_c, tau1_hi_cond=hi_c,
            w_tau1_cond=(hi_c - lo_c) / t1 if f(lo_c) and f(hi_c) else np.nan,
            tau1_lo_true=lo_t, tau1_hi_true=hi_t,
            w_tau1_true=(hi_t - lo_t) / t1 if f(lo_t) and f(hi_t) else np.nan,
            true_hit_bound=hit_t, n_eval=nev, seconds=round(dt, 2)))
        r = rows[-1]
        print(f"[{i}/{len(recs)}] {sid:26s} w_cond={r['w_tau1_cond']:.3f} "
              f"w_true={r['w_tau1_true']:.3f} ({r['seconds']}s)", flush=True)
    return pd.DataFrame(rows)


# ── mode: loto ──────────────────────────────────────────────────────────
def run_loto(recs, loto):
    by_sid = {r["sample_id"]: r for r in recs}
    rows = []
    sids = [s for s in loto.SampleID.unique() if s in by_sid]
    for i, sid in enumerate(sids, 1):
        rec = by_sid[sid]
        u = SW.unpack(rec)
        sub = loto[loto.SampleID == sid]
        t0 = time.perf_counter()

        # full suite, at the optimum sweep_loto itself refitted
        tf = float(sub.tau1_full.iloc[0])
        pf = {"tau1": tf, "pd1": np.nan, "f1": np.nan, "tau2": np.nan}
        # nuisance is minimised over, so only tau1 must be right; fill the rest
        # from the config initial values so make_loss's fixed slots are sane
        init = rec["lpm"]["init"]
        pf["pd1"] = float(init["pd1"] or 0.01)
        pf["f1"] = float(init["fraction1"] or 0.5)
        pf["tau2"] = float(init["tau2_yr"] or 1000.)
        lo_f, hi_f, hit_f, ref_f, _ = true_profile(
            u, rec, pf, chi2min=float(sub.chi2_full.iloc[0]))
        w_full = (hi_f - lo_f) / tf if np.isfinite(lo_f) and np.isfinite(hi_f) else np.nan

        _, pnames = SW.make_loss(u, rec)
        bnds = SW.free_param_bounds(rec, u, pnames)
        lo_b, hi_b = bnds.get("tau1", (np.nan, np.nan))

        for _, r in sub.iterrows():
            nm = r.omitted_tracer
            if nm not in u["names"]:
                continue
            keep = [k for k, x in enumerate(u["names"]) if x != nm]
            nk = [u["names"][k] for k in keep]
            ok = [u["obs"][k] for k in keep]
            sk = [u["scales"][k] for k in keep]
            tr = float(r.tau1_reduced) if np.isfinite(r.tau1_reduced) else np.nan
            if not np.isfinite(tr):
                continue
            pj = dict(pf); pj["tau1"] = tr
            loj, hij, hitj, refj, _ = true_profile(
                u, rec, pj, "tau1", nk, ok, sk,
                chi2min=float(r.chi2_reduced))
            wj = (hij - loj) / tr if np.isfinite(loj) and np.isfinite(hij) else np.nan
            cens = bool(hitj or (np.isfinite(wj) and wj > W_WEAK))
            I = wj / w_full if np.isfinite(w_full) and w_full > 0 else np.nan
            rows.append(dict(
                SampleID=sid, omitted_tracer=nm, Aq_class=r.Aq_class,
                LPM=u["lpm"], n_active_full=int(r.n_active_full),
                tau1_full=tf, tau1_reduced=tr,
                delta_tau1_rel=float(r.delta_tau1_rel),
                w_full_cond=float(r.w_full), w_reduced_cond=float(r.w_reduced),
                information_ratio_cond=float(r.information_ratio),
                censored_cond=bool(r.censored),
                w_full_true=w_full, w_reduced_true=wj,
                information_ratio_true=I, censored_true=cens))
        dt = time.perf_counter() - t0
        print(f"[{i}/{len(sids)}] {sid:26s} w_full {sub.w_full.iloc[0]:.3f}->"
              f"{w_full:.3f}  ({dt:.1f}s)", flush=True)
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["ident", "loto"], default="ident")
    ap.add_argument("--samples", default="")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="")
    A = ap.parse_args()

    SW.bootstrap(_HERE)
    recs = json.load(open(CFG))["samples"]
    keep = set(x for x in A.samples.split(",") if x)
    if keep:
        recs = [r for r in recs if r["sample_id"] in keep]
    elif not A.all:
        recs = recs[:4]
    if A.limit:
        recs = recs[:A.limit]

    if A.mode == "ident":
        df = run_ident(recs, SW.load_baseline(BASE))
        name = "profile_true.csv"
        ok = df.dropna(subset=["w_tau1_cond", "w_tau1_true"])
        bad = ok[ok.w_tau1_true < ok.w_tau1_cond - 1e-9]
        print(f"\nn={len(ok)}  median w_cond={ok.w_tau1_cond.median():.3f}  "
              f"median w_true={ok.w_tau1_true.median():.3f}")
        print(f"monotonicity violations (true < cond): {len(bad)}")
    else:
        loto = pd.read_csv(LOTO_IN)
        keepids = {r["sample_id"] for r in recs}
        loto = loto[loto.SampleID.isin(keepids)]
        df = run_loto(recs, loto)
        name = "profile_true_loto.csv"
        ok = df.dropna(subset=["information_ratio_cond", "information_ratio_true"])
        print(f"\nrows={len(df)}  with both ratios={len(ok)}")
        print(f"median I_cond={ok.information_ratio_cond.median():.3f}  "
              f"median I_true={ok.information_ratio_true.median():.3f}")

    if A.out:
        os.makedirs(A.out, exist_ok=True)
        pth = os.path.join(A.out, name)
        df.to_csv(pth, index=False)
        print("wrote", pth)


if __name__ == "__main__":
    main()
