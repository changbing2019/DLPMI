#!/usr/bin/env python3
"""
dlpmi_sweeps.py
===============
Identifiability, leave-one-tracer-out, and model-structure sweeps for the
Edwards aquifer DLPMI analysis (Methods sections 3.4-3.6).

This script reuses the fit machinery in examples/edwards_aquifer/edwards_run.py
so that every fit produced here follows exactly the same conventions -- bounds,
seeds, optimiser schedule, free-parameter tokenisation -- as the baseline run
that produced 00_summary_table.csv.  Nothing is re-implemented.

Three sweeps
------------
  ident      Identifiability diagnostics (Methods 3.4).  Exact Hessian,
             eigendecomposition, condition number, parameter correlation, and
             profile-width classification.  Requires NO refitting when a
             baseline summary CSV is supplied -- the Hessian is evaluated at
             the stored optimum.  Seconds per sample.

  loto       Leave-one-tracer-out (Methods 3.5).  Refits each sample once per
             active tracer, omitting that tracer.  Reports the information
             ratio I_j = w(-j) / w(full) for tau1.  ~n_active refits/sample.

  structure  Model-structure support (Methods 3.6).  Refits every sample under
             both DM and BMM-DM-DM, computes the measurement-error-weighted
             chi2_sigma and AICc, and reports dAICc.  ~2 refits/sample.

Usage
-----
  # fast, no refitting -- run this first
  python dlpmi_sweeps.py --sweeps ident \
      --baseline-csv results/edwards_pinn/00_summary_table.csv

  # full run on 8 cores
  python dlpmi_sweeps.py --sweeps ident,loto,structure --processes 8

  # quick smoke test
  python dlpmi_sweeps.py --sweeps loto --limit 3 --adam 500 --starts 1

Outputs (to --out)
------------------
  sweep_identifiability.csv
  sweep_loto.csv
  sweep_structure.csv
"""

from __future__ import annotations
import argparse, contextlib, copy, json, os, sys, warnings, itertools
import numpy as np

warnings.filterwarnings("ignore")

# Samples outside the San Antonio Segment of the Edwards aquifer.
EXCLUDE_DEFAULT = {"EDTRPAS1-38", "EDTRPAS1-39", "EDTRPAS1-40",
                   "EDTRPAS1-47", "EDTRPAS1-48"}

# Identifiability thresholds on the relative profile width w = (hi-lo)/theta*.
W_WELL, W_WEAK = 0.5, 1.5
KAPPA_MAX = 1e8          # condition number above which H is treated as singular
R_STRONG = 0.8           # |correlation| above which profile CIs understate sigma
BOUND_DIST_MAX = 0.01    # range-normalised distance within which a param is "at" its bound
BOUND_RATIO_MIN = 0.1    # min abs(theta*g) / abs(H_uncorrected[i,i]) to call a bound "active"
                         # (empirical gap in the data: 9 samples < 0.05, 18 > 0.49; see sweep_ident)

# Defaults used when a DM sample is refitted as a BMM (Methods 3.6).
# BMM_TAU2_DEFAULT is the median tau2 across the 19 baseline BMM samples in
# 00_summary_table.csv (range 38-23,500 yr; 12,000 yr sat above the 75th
# percentile and biased the counterfactual refit toward an old-water seed).
BMM_TAU2_DEFAULT = 1000.0
BMM_PD2_DEFAULT = 0.10
BMM_F1_DEFAULT = 0.90


# ══════════════════════════════════════════════════════════════════════════
# PACKAGE WIRING
# ══════════════════════════════════════════════════════════════════════════

_ER = None          # edwards_run module
_TORCH = None


def bootstrap(package_root: str):
    """Import torch, the dlpmi package, and edwards_run from package_root."""
    global _ER, _TORCH
    if _ER is not None:
        return _ER
    pkg = os.path.abspath(package_root)
    ex = os.path.join(pkg, "examples", "edwards_aquifer")
    for p in (pkg, ex):
        if p not in sys.path:
            sys.path.insert(0, p)
    import torch
    torch.set_num_threads(1)          # avoid oversubscription under --processes
    import edwards_run as er
    _TORCH, _ER = torch, er
    return er


# ══════════════════════════════════════════════════════════════════════════
# RECORD HANDLING
# ══════════════════════════════════════════════════════════════════════════

def unpack(rec: dict) -> dict:
    """Extract fit arguments from a config record, matching fit_sample()."""
    er = _ER
    tns, obs_v, obs_e, scls, act = er.build_tracer_lists(rec)
    keep = [i for i, a in enumerate(act) if a]
    return dict(
        sid=rec["sample_id"],
        lpm=rec["lpm"]["name"],
        free_params=rec["lpm"].get("free_params") or "Mean Age",
        sample_date=float(rec["sample_date"]),
        age_cat=rec.get("age_category", ""),
        network=rec.get("network", ""),
        names=[tns[i] for i in keep],
        obs=[obs_v[i] for i in keep],
        errs=[obs_e[i] for i in keep],
        scales=[scls[i] for i in keep],
        he4r=float(rec["he4_params"]["solution_rate_ccpgpyr"] or 1e-12),
        dic_c1=float(rec["carbon14_params"]["dic_c1_mmolL"] or 100.),
        dic_c2=float(rec["carbon14_params"]["dic_c2_mmolL"] or 100.),
        dgmeta=rec.get("dgmeta_params", {}) or {},
    )


def n_free_of(rec: dict) -> int:
    return _ER._count_free_params(rec)


def chi2_rel(sims, obs) -> float:
    return float(sum(((float(s) - float(o)) / (abs(float(o)) + 1e-30)) ** 2
                     for s, o in zip(sims, obs)))


def chi2_sigma(sims, obs, errs) -> float:
    """Measurement-error-weighted misfit, Eq. (19).  Supports AICc; the
    relative-error objective of Eq. (9) does not.

    A non-positive sigma is rejected rather than clamped. Clamping it to a
    floor (the behaviour before 2026-08-23) turned a negative obs_err into a
    weight of ~1e60, which silently dominates the whole sum instead of
    failing. Two config entries carry a negative obs_err -- EDTRPAS1-38 3H
    and SCTXLUSRC1-08 4He -- neither of which is an active tracer in the
    60-sample study set, so nothing published was affected."""
    out = 0.0
    for s, o, e in zip(sims, obs, errs):
        e = float(e)
        if not (e > 0.0) or not np.isfinite(e):
            raise ValueError("chi2_sigma: non-positive or non-finite sigma "
                             "%r for obs %r" % (e, o))
        out += ((float(s) - float(o)) / e) ** 2
    return float(out)


def aicc(chi2s: float, k: int, n: int):
    """Small-sample corrected AIC.  Undefined when n - k - 1 <= 0.

    NOTE for the structure comparison: k is 2 under BOTH structures here, so
    the correction term is identical for the two and cancels exactly in
    ΔAICc, which therefore equals Δχ²σ. sweep_structure() relies on that to
    stay well posed at dof >= 1, where this function returns NaN. Do not
    generalise the cancellation to comparisons with unequal k."""
    if n - k - 1 <= 0:
        return np.nan
    return chi2s + 2 * k + 2 * k * (k + 1) / (n - k - 1)


_REPORTED_REL_SIGMA: dict | None = None


def load_reported_rel_sigma(data_dir: str) -> dict:
    """Per-sample, per-tracer reported analytical 1-sigma, AS A FRACTION of
    the tabulated concentration, from Musgrove's Tables 3 and 5.

    Applied relatively so no unit conversion enters -- the same convention
    sigma_impact.py uses. These are NOT the sigmas the published inversion
    used (Table_4_LPM.txt supplies those, a uniform 10% for 164 of the 218
    active pairs); they exist only to support the error-scale sensitivity
    reported in Text S6.
    """
    import pandas as pd

    def num(v):
        return pd.to_numeric(pd.Series([v]), errors="coerce")[0]

    rel: dict = {}
    t3 = pd.read_csv(os.path.join(data_dir, "Table_3_Tracers.txt"), sep="\t")
    for _, r in t3.iterrows():
        d = rel.setdefault(str(r["SampleID"]).strip(), {})
        for nm, (a, b) in (("3H", ("TRC_3H_TU", "TRC_3H_Err_TU")),
                           ("SF6", ("TRC_SF6_pptv", "TRC_SF6_Err_pptv")),
                           ("3He(trit)", ("TRC_3HeTrit_TU",
                                          "TRC_3HeTrit_Err_TU"))):
            o, e = num(r[a]), num(r[b])
            if np.isfinite(o) and np.isfinite(e) and o != 0:
                d[nm] = abs(e / o)
    t5 = pd.read_csv(os.path.join(data_dir, "Table_5_Carbon14.txt"), sep="\t")
    for _, r in t5.iterrows():
        o, e = num(r["14C_sample_pM"]), num(r["14C_Err_sample_pM"])
        if np.isfinite(o) and np.isfinite(e) and o != 0:
            rel.setdefault(str(r["SampleID"]).strip(), {})["14C"] = abs(e / o)
    return rel


def reported_errs(sid, names, obs, errs_cfg):
    """errs_cfg with each entry replaced by the reported analytical sigma
    where one exists for that sample/tracer; unchanged where it does not.
    Returns (errs, n_substituted)."""
    if not _REPORTED_REL_SIGMA:
        return None, 0
    src = {}
    for k in [sid] + [x.strip() for x in str(sid).split("/")]:
        if k in _REPORTED_REL_SIGMA:
            src = _REPORTED_REL_SIGMA[k]
            break
    out = [abs(float(o)) * src[n] if (n in src and float(o) != 0.0) else e
           for n, o, e in zip(names, obs, errs_cfg)]
    return out, sum(1 for n in names if n in src)


@contextlib.contextmanager
def sigma_objective(errs):
    """Temporarily replace edwards_run.chi2_loss with a sigma-weighted form.

    _run_fit calls the bare name chi2_loss(sims, obs) -- looked up in
    edwards_run's module globals at call time, not captured as a local alias
    at import -- so patching er.chi2_loss here redirects it correctly. errs
    must be in the same active-tracer order as the obs passed to _run_fit.
    """
    er = _ER
    orig = er.chi2_loss

    def _loss(sims, obs_vals, _e=list(errs)):
        assert len(sims) == len(obs_vals) == len(_e), "tracer/err misalignment"
        return sum(((s - float(o)) / max(float(e), 1e-30)) ** 2
                   for s, o, e in zip(sims, obs_vals, _e))
    er.chi2_loss = _loss
    try:
        yield
    finally:
        er.chi2_loss = orig


# ══════════════════════════════════════════════════════════════════════════
# FORWARD EVALUATION AT ARBITRARY PARAMETERS
# ══════════════════════════════════════════════════════════════════════════

def make_loss(u: dict, rec: dict, names=None, obs=None, scales=None,
              errs=None):
    """Return (loss_fn, pnames) where loss_fn maps a flat tensor of natural-unit
    free parameters to a chi2.  Used for exact Hessians and for profiling
    without refitting.

    errs : optional per-tracer measurement sigmas, ordered as `names`. When
        given, the objective is the measurement-error-weighted sum of squares
        instead of the relative-error form. sigma_objective() patches
        edwards_run.chi2_loss, which _run_fit calls, but it does NOT reach here
        because this builds its objective inline -- passing errs is the only
        way to profile under the weighted objective."""
    torch = _TORCH
    from dlpmi.forward import forward_single, forward_BMM
    from dlpmi.kernels import choose_ages

    names = names if names is not None else u["names"]
    obs = obs if obs is not None else u["obs"]
    scales = scales if scales is not None else u["scales"]
    init = rec["lpm"]["init"]

    def _chi2(sims):
        if errs is None:
            return sum(((sm - float(o)) / (abs(float(o)) + 1e-30)) ** 2
                       for sm, o in zip(sims, obs))
        assert len(errs) == len(obs), "err/obs misalignment in make_loss"
        return sum(((sm - float(o)) / max(float(e), 1e-30)) ** 2
                   for sm, o, e in zip(sims, obs, errs))

    if u["lpm"] == "DM":
        pnames = ["tau1", "pd1"]

        def loss(p):
            pp = {"tau": p[0], "PD": p[1]}
            ages = choose_ages(float(p[0].detach()))
            sims = forward_single("DM", pp, u["sample_date"], names, scales,
                                  u["he4r"], ages, u["dgmeta"])
            return _chi2(sims)
        return loss, pnames

    t1f, f1f, t2f = _ER._bmm_free_flags(u["free_params"])
    pnames = [n for n, f in zip(["tau1", "f1", "tau2"], [t1f, f1f, t2f]) if f]
    lr = rec["lpm"]["tracerlpm_result"]
    fix_t1 = float(lr.get("tau1_yr") or init["tau1_yr"] or 20.)
    fix_f1 = float(lr.get("fraction1") or init["fraction1"] or 0.5)
    fix_t2 = float(lr.get("tau2_yr") or init["tau2_yr"] or 1000.)
    pd1 = float(init["pd1"] or 0.01)
    pd2 = float(init["pd2"] or 0.01)

    def loss(p):
        i = 0
        if t1f:
            t1 = p[i]; i += 1
        else:
            t1 = torch.tensor(fix_t1)
        if f1f:
            f1 = p[i]; i += 1
        else:
            f1 = torch.tensor(fix_f1)
        if t2f:
            t2 = p[i]; i += 1
        else:
            t2 = torch.tensor(fix_t2)
        sims = forward_BMM("DM", {"tau": t1, "PD": torch.tensor(pd1)},
                           "DM", {"tau": t2, "PD": torch.tensor(pd2)}, f1,
                           u["sample_date"], names, scales, u["he4r"],
                           u["dic_c1"], u["dic_c2"],
                           float(rec["lpm"].get("uz_tt_yr") or 0.), u["dgmeta"])
        return _chi2(sims)
    return loss, pnames


# ══════════════════════════════════════════════════════════════════════════
# FITTING
# ══════════════════════════════════════════════════════════════════════════

def run_fit(rec: dict, names, obs, scales, n_adam, n_starts):
    """Fit via edwards_run._run_fit.  Returns (params dict, chi2_rel, sims)."""
    torch = _TORCH
    u = unpack(rec)
    model, _, loss = _ER._run_fit(rec, names, obs, scales, u["he4r"],
                                  u["dic_c1"], u["dic_c2"],
                                  n_adam, n_starts, False, u["dgmeta"])
    with torch.no_grad():
        if rec["lpm"]["name"] == "DM":
            tau, pd = model.get_params()
            out = {"tau1": float(tau), "pd1": float(pd),
                   "f1": np.nan, "tau2": np.nan}
        else:
            t1, p1, f1, t2, p2 = model.get_params()
            out = {"tau1": float(t1), "pd1": float(p1),
                   "f1": float(f1), "tau2": float(t2)}
    lf, _ = make_loss(u, rec, names, obs, scales)
    theta = torch.tensor([out[n] for n in
                          (["tau1", "pd1"] if rec["lpm"]["name"] == "DM"
                           else [n for n, f in zip(["tau1", "f1", "tau2"],
                                                   _ER._bmm_free_flags(u["free_params"])) if f])],
                         dtype=torch.float32)
    from dlpmi.forward import forward_single, forward_BMM  # noqa: F401
    sims = simulate(u, rec, out, names, scales)
    return out, float(loss), sims


def simulate(u, rec, params, names, scales):
    """Simulated concentrations at a given parameter set."""
    torch = _TORCH
    from dlpmi.forward import forward_single, forward_BMM
    from dlpmi.kernels import choose_ages
    with torch.no_grad():
        if rec["lpm"]["name"] == "DM":
            pp = {"tau": torch.tensor(params["tau1"]),
                  "PD": torch.tensor(params["pd1"])}
            ages = choose_ages(params["tau1"])
            sims = forward_single("DM", pp, u["sample_date"], names, scales,
                                  u["he4r"], ages, u["dgmeta"])
        else:
            init = rec["lpm"]["init"]
            # pd1 must match what _run_fit actually held fixed during
            # training -- lpm_r.get("pd1") or init["pd1"] or 0.01, per its
            # own pd1_0 computation -- not init["pd1"] alone. params["pd1"]
            # already carries the correct value via model.get_params().
            sims = forward_BMM(
                "DM", {"tau": torch.tensor(params["tau1"]),
                       "PD": torch.tensor(float(params["pd1"]))},
                "DM", {"tau": torch.tensor(params["tau2"]),
                       "PD": torch.tensor(float(init["pd2"] or 0.01))},
                torch.tensor(params["f1"]), u["sample_date"], names, scales,
                u["he4r"], u["dic_c1"], u["dic_c2"],
                float(rec["lpm"].get("uz_tt_yr") or 0.), u["dgmeta"])
    return [float(s) for s in sims]


def profile_width(u, rec, params, names, obs, scales, chi2min, pname,
                  lo_b, hi_b, dchi2=1.0):
    """Profile chi2 68% interval for one parameter, others held at optimum."""
    torch = _TORCH
    lf, pnames = make_loss(u, rec, names, obs, scales)
    if pname not in pnames:
        return np.nan, np.nan
    idx = pnames.index(pname)
    base = [params[n] for n in pnames]

    def f1d(v):
        p = list(base); p[idx] = v
        with torch.no_grad():
            return float(lf(torch.tensor(p, dtype=torch.float32)))

    from dlpmi.uncertainty import profile_uncertainty
    return profile_uncertainty(f1d, params[pname], chi2min, lo_b, hi_b, dchi2)


def free_param_bounds(rec: dict, u: dict, pnames) -> dict:
    """Bounds actually enforced by edwards_run._run_fit for each pname --
    matched exactly to its internal computation (edwards_run.py lines ~200-251),
    not re-derived independently, since a mismatch here would silently make
    active-bound detection meaningless. f1's 0.005/0.997 is hardcoded in
    _run_fit itself (it does not read pinn_bounds' f1_lo/f1_hi), and tau1's
    upper bound is narrowed relative to tau2 for BMM samples exactly as
    _run_fit narrows it before fitting.
    """
    b = rec["pinn_bounds"]
    lr = rec["lpm"]["tracerlpm_result"]
    init = rec["lpm"]["init"]
    is_bmm = u["lpm"] != "DM"
    out = {}
    if "tau1" in pnames:
        tau1_lo, tau1_hi = float(b["tau1_lo"]), float(b["tau1_hi"])
        if is_bmm:
            tau2_0 = float(lr.get("tau2_yr") or init["tau2_yr"] or 100.)
            tau2_v = float(lr.get("tau2_yr") or tau2_0)
            t1_hi_b = min(tau1_hi, tau2_v * 0.9)
            t1_hi_b = max(t1_hi_b, tau1_lo * 2.)
            out["tau1"] = (tau1_lo, t1_hi_b)
        else:
            out["tau1"] = (tau1_lo, tau1_hi)
    if "pd1" in pnames:
        out["pd1"] = (float(b["pd1_lo"]), float(b["pd1_hi"]))
    if "f1" in pnames:
        out["f1"] = (0.005, 0.997)
    if "tau2" in pnames:
        tau2_0 = float(lr.get("tau2_yr") or init["tau2_yr"] or 100.)
        tau2_v = float(lr.get("tau2_yr") or tau2_0)
        out["tau2"] = (max(1., tau2_v * 0.05), tau2_v * 5.)
    return out


def projected_grad_info(rec: dict, u: dict, params: dict,
                        names=None, obs=None, scales=None) -> dict:
    """Gradient / active-bound diagnostics at a fitted point (TASK_active_bounds.md,
    TASK_nonconvergence.md). Shared by sweep_ident, sweep_loto and sweep_structure
    so every sweep reports the same KKT check (projected_grad_norm) rather than
    silently assuming convergence -- the gap that let three non-converged
    samples through as far as headline identifiability counts.

    names/obs/scales let a caller pass a reduced tracer set (sweep_loto) or a
    coerced structure (sweep_structure's as_structure output); default to
    u's own full active set.
    """
    torch = _TORCH
    lf, pnames = make_loss(u, rec, names, obs, scales)
    theta = torch.tensor([params[n] for n in pnames], dtype=torch.float64)
    theta_np = theta.detach().numpy()
    theta_g = theta.clone().requires_grad_(True)
    grad_t, = torch.autograd.grad(lf(theta_g.float()).double(), theta_g)
    g = grad_t.detach().numpy()

    H = torch.autograd.functional.hessian(
        lambda p: lf(p.float()).double(), theta).detach().numpy()
    H = np.atleast_2d(H)
    H_unc = H * np.outer(theta_np, theta_np)

    bounds = free_param_bounds(rec, u, pnames)
    active_flags, active_names = [], []
    for i, name in enumerate(pnames):
        lo_i, hi_i = bounds[name]
        rng = hi_i - lo_i
        d_i = min(theta_np[i] - lo_i, hi_i - theta_np[i]) / rng
        at_bound_i = d_i < BOUND_DIST_MAX
        nearer_is_lower = (theta_np[i] - lo_i) <= (hi_i - theta_np[i])
        outward_i = (g[i] > 0) if nearer_is_lower else (g[i] < 0)
        local_ref = abs(H_unc[i, i])
        active_i = bool(at_bound_i and outward_i
                       and abs(theta_np[i] * g[i]) > BOUND_RATIO_MIN * local_ref)
        active_flags.append(active_i)
        if active_i:
            active_names.append(name)

    g_projected = np.array([0.0 if a else g[i]
                            for i, a in enumerate(active_flags)])
    return dict(
        grad_norm=float(np.linalg.norm(g)),
        projected_grad_norm=float(np.linalg.norm(g_projected)),
        bound_active=bool(any(active_flags)),
        bound_active_params=",".join(active_names))


# ══════════════════════════════════════════════════════════════════════════
# SWEEP 1 — IDENTIFIABILITY  (Methods 3.4)
# ══════════════════════════════════════════════════════════════════════════

def sweep_ident(rec, baseline, args):
    torch = _TORCH
    u = unpack(rec)
    sid = u["sid"]
    b = rec["pinn_bounds"]
    nf, na = n_free_of(rec), len(u["names"])
    dof = max(na - nf, 1)

    if baseline is not None and sid in baseline:
        row = baseline[sid]
        params = {"tau1": row["tau1_PINN"], "pd1": row["pd1_PINN"],
                  "f1": row["f1_PINN"], "tau2": row["tau2_PINN"]}
        c2 = float(row["chi2_PINN"])
    else:
        params, c2, _ = run_fit(rec, u["names"], u["obs"], u["scales"],
                                args.adam, args.starts)

    lf, pnames = make_loss(u, rec)
    theta = torch.tensor([params[n] for n in pnames], dtype=torch.float64)

    H = torch.autograd.functional.hessian(
        lambda p: lf(p.float()).double(), theta).detach().numpy()
    H = np.atleast_2d(H)
    ev = np.linalg.eigvalsh(H)
    lam_min, lam_max = float(ev.min()), float(ev.max())
    kappa = abs(lam_max / lam_min) if lam_min != 0 else np.inf
    singular = (lam_min <= 0) or (kappa > KAPPA_MAX)

    # ── scale-invariant (log-parameter) Hessian diagnostics ─────────────────
    # The raw Hessian's eigendecomposition is not scale-invariant: tau1 (yr,
    # median ~21) and PD (dimensionless, median ~0.01) differ by ~6 orders of
    # magnitude, so its eigenvectors just recover the coordinate axes.
    # d^2chi2/d(ln theta_i)d(ln theta_j) = theta_i*theta_j*H_ij + delta_ij*theta_i*g_i
    # (the diagonal term vanishes only at an EXACT optimum). Cross-validated
    # against an independently reparametrized-and-autograd'd log(theta)
    # Hessian (verify_hlog_exact.py): the diagonal term is NOT always
    # negligible -- up to 8-10% relative effect on cond_number for some
    # samples -- so it is included here, not just reported. Covariance/
    # correlation stay on the raw H (already scale-invariant, and sigma is
    # needed in physical units) -- only the eigen-diagnostics are duplicated.
    theta_star_vec = theta.detach().numpy().astype(float)
    theta_g = theta.clone().requires_grad_(True)
    grad_t, = torch.autograd.grad(lf(theta_g.float()).double(), theta_g)
    g_vec = grad_t.detach().numpy().astype(float)

    H_scaled_uncorrected = H * np.outer(theta_star_vec, theta_star_vec)
    gradient_check = float(np.linalg.norm(g_vec * theta_star_vec)
                          / np.linalg.norm(H_scaled_uncorrected))
    H_scaled = H_scaled_uncorrected + np.diag(theta_star_vec * g_vec)

    ev_s = np.linalg.eigvalsh(H_scaled)
    lam_min_s, lam_max_s = float(ev_s.min()), float(ev_s.max())
    kappa_s = abs(lam_max_s / lam_min_s) if lam_min_s != 0 else np.inf
    singular_scaled = (lam_min_s <= 0) or (kappa_s > KAPPA_MAX)

    w_s, V_s = np.linalg.eigh(H_scaled)
    vmin_s = V_s[:, 0]
    dom_s = pnames[int(np.argmax(np.abs(vmin_s)))]
    load_s = float(np.max(np.abs(vmin_s)))

    # ── active-bound detection (TASK_active_bounds.md) ──────────────────────
    # At an interior optimum g=0 exactly, so H_scaled = D^T H D and the two
    # Hessians' eigendecompositions agree (Sylvester's law, pure congruence).
    # They can only disagree when g != 0, which happens when a parameter is
    # pinned at its bound: the KKT condition there is g pointing outward, not
    # g=0, and the resulting diag(theta*g) term can flip lambda_min's sign --
    # not a scale-invariance failure, but the correct signal that the
    # "unconstrained minimum" this Hessian is supposed to describe doesn't
    # exist in any parametrisation for that sample.
    # Magnitude gate: TASK_active_bounds.md's own pseudocode compares
    # abs(theta_i*g_i) against 1e-3*norm(H_log) using the FULL matrix norm.
    # Tested directly against the task's own worked example (SCTXLUSRC1-11:
    # theta_pd1=0.001 exactly at its bound, g_pd1=+107.5, unambiguous) that
    # formula does NOT flag it -- norm(H_scaled)=134.6 is dominated entirely
    # by the unrelated tau1-tau1 entry (134.58), diluting the pd1 correction
    # (0.108) below the 1e-3 threshold (0.135) by ~20%. This is the same
    # cross-parameter-scale dilution the whole scaled-Hessian exercise exists
    # to fix, recurring in the gate meant to detect it. Switching the
    # reference to the same parameter's own uncorrected diagonal entry (a
    # local, dimensionally consistent quantity) fixes the false negative, but
    # reusing the SAME 1e-3 constant against a much smaller denominator then
    # over-flags: the ratio abs(theta_i*g_i)/abs(H_uncorrected[i,i]) across
    # all 42 pd1-bearing samples shows a clean empirical gap (9 samples below
    # 0.05, 18 above 0.49 -- no sample falls between), so BOUND_RATIO_MIN=0.1
    # (an order of magnitude inside that gap) separates a genuinely dominant
    # correction from a negligible one, verified against all three of the
    # task's worked examples (SCTXLUSRC1-11, EDTRPAS1-50, EDTRPAS1-32).
    bounds = free_param_bounds(rec, u, pnames)
    active_flags, active_names = [], []
    for i, name in enumerate(pnames):
        lo_i, hi_i = bounds[name]
        rng = hi_i - lo_i
        d_i = min(theta_star_vec[i] - lo_i, hi_i - theta_star_vec[i]) / rng
        at_bound_i = d_i < BOUND_DIST_MAX
        nearer_is_lower = (theta_star_vec[i] - lo_i) <= (hi_i - theta_star_vec[i])
        outward_i = (g_vec[i] > 0) if nearer_is_lower else (g_vec[i] < 0)
        local_ref = abs(H_scaled_uncorrected[i, i])
        active_i = bool(at_bound_i and outward_i
                       and abs(theta_star_vec[i] * g_vec[i])
                           > BOUND_RATIO_MIN * local_ref)
        active_flags.append(active_i)
        if active_i:
            active_names.append(name)

    bound_active = any(active_flags)
    bound_active_params = ",".join(active_names)
    g_projected = np.array([0.0 if a else g_vec[i]
                            for i, a in enumerate(active_flags)])
    projected_grad_norm = float(np.linalg.norm(g_projected))

    # The factor of 2 belongs here for the same reason as in
    # dlpmi/uncertainty.py. It changes nothing downstream: cov is used only
    # for the correlation matrix and the var > 0 test, both invariant to a
    # positive scalar. Corrected anyway so the four copies agree.
    _s = 2.0 * (c2 / dof)
    try:
        cov = np.linalg.inv(H) * _s
    except np.linalg.LinAlgError:
        cov = np.linalg.pinv(H) * _s

    # A correlation coefficient is only meaningful when the covariance is
    # positive definite.  If any variance is non-positive the Hessian is
    # indefinite at theta*, sqrt(|var|) would silently mask the sign, and the
    # resulting "correlation" can exceed 1.  Report NaN instead.
    var = np.diag(cov)
    pos_def = bool(np.all(var > 0)) and lam_min > 0
    if pos_def:
        sig = np.sqrt(var)
        with np.errstate(invalid="ignore", divide="ignore"):
            R = cov / np.outer(sig, sig)
        off = [abs(R[i, j]) for i in range(len(pnames))
               for j in range(i + 1, len(pnames))]
        rmax = float(max(off)) if off else np.nan
    else:
        rmax = np.nan

    # least-constrained direction
    w_, V_ = np.linalg.eigh(H)
    vmin = V_[:, 0]
    dom = pnames[int(np.argmax(np.abs(vmin)))]
    load = float(np.max(np.abs(vmin)))

    lo, hi = profile_width(u, rec, params, u["names"], u["obs"], u["scales"],
                           c2, "tau1", float(b["tau1_lo"]), float(b["tau1_hi"]))
    w = (hi - lo) / params["tau1"] if np.isfinite(lo) and np.isfinite(hi) else np.nan
    hit = (np.isfinite(lo) and abs(lo - b["tau1_lo"]) / b["tau1_lo"] < 1e-3) or \
          (np.isfinite(hi) and abs(hi - b["tau1_hi"]) / b["tau1_hi"] < 1e-3)

    if singular or hit or not np.isfinite(w) or w > W_WEAK:
        cls = "unconstrained"
    elif w > W_WELL:
        cls = "weakly constrained"
    else:
        cls = "well constrained"

    # Same classification, but with the scale-invariant singular flag. `hit`
    # and `w` (profile width) are already scale-invariant, so they carry over
    # unchanged -- only the Hessian-derived singularity test differs.
    if singular_scaled or hit or not np.isfinite(w) or w > W_WEAK:
        cls_scaled = "unconstrained"
    elif w > W_WELL:
        cls_scaled = "weakly constrained"
    else:
        cls_scaled = "well constrained"

    # Final classification (TASK_active_bounds.md): tau1's profile width (w,
    # hit) is primary regardless of bound-active status -- it stays
    # interpretable because the profile already holds every other parameter,
    # including any bound-pinned one, fixed. For bound-active samples the
    # Hessian singularity test is suppressed entirely (neither `singular` nor
    # `singular_scaled` describes an unconstrained minimum in any
    # parametrisation there). For interior samples this reduces to the raw
    # `cls` by construction -- at g=0 exactly, singular_scaled == singular
    # (Sylvester's law on the pure congruence transform), so using the raw
    # flag here is equivalent to the scaled one and matches verification 2's
    # requirement (identifiability_final == identifiability off-bound) exactly
    # rather than relying on that equivalence to hold to floating-point
    # precision at the classification thresholds.
    if bound_active:
        if hit or not np.isfinite(w) or w > W_WEAK:
            cls_final = "unconstrained"
        elif w > W_WELL:
            cls_final = "weakly constrained"
        else:
            cls_final = "well constrained"
    else:
        cls_final = cls

    return dict(SampleID=sid, Network=u["network"], AgeCat=u["age_cat"],
                LPM=u["lpm"], free_params=u["free_params"],
                n_active=na, n_free=nf, dof=dof, chi2=c2,
                tau1=params["tau1"], tau1_profile_lo=lo, tau1_profile_hi=hi,
                w_tau1=w, lambda_min=lam_min, lambda_max=lam_max,
                cond_number=kappa, singular=bool(singular),
                hessian_pos_def=bool(pos_def),
                profile_hit_bound=bool(hit), max_abs_corr=rmax,
                strong_corr=bool(np.isfinite(rmax) and rmax > R_STRONG),
                sloppy_direction=dom, sloppy_loading=load,
                identifiability=cls,
                lambda_min_scaled=lam_min_s, lambda_max_scaled=lam_max_s,
                cond_number_scaled=kappa_s,
                sloppy_direction_scaled=dom_s, sloppy_loading_scaled=load_s,
                gradient_check=gradient_check,
                singular_scaled=bool(singular_scaled),
                identifiability_scaled=cls_scaled,
                bound_active=bool(bound_active),
                bound_active_params=bound_active_params,
                projected_grad_norm=projected_grad_norm,
                identifiability_final=cls_final)


# ══════════════════════════════════════════════════════════════════════════
# SWEEP 2 — LEAVE ONE TRACER OUT  (Methods 3.5)
# ══════════════════════════════════════════════════════════════════════════

def sweep_loto(rec, baseline, args):
    u = unpack(rec)
    sid = u["sid"]
    b = rec["pinn_bounds"]
    lo_b, hi_b = float(b["tau1_lo"]), float(b["tau1_hi"])
    rows = []

    pf, c2f, _ = run_fit(rec, u["names"], u["obs"], u["scales"],
                         args.adam, args.starts)
    lo, hi = profile_width(u, rec, pf, u["names"], u["obs"], u["scales"],
                           c2f, "tau1", lo_b, hi_b)
    w_full = (hi - lo) / pf["tau1"]
    pg_full = projected_grad_info(rec, u, pf, u["names"], u["obs"], u["scales"])

    for j, nm in enumerate(u["names"]):
        if len(u["names"]) - 1 < 1:
            continue
        keep = [i for i in range(len(u["names"])) if i != j]
        nk = [u["names"][i] for i in keep]
        ok = [u["obs"][i] for i in keep]
        sk = [u["scales"][i] for i in keep]
        try:
            pj, c2j, _ = run_fit(rec, nk, ok, sk, args.adam, args.starts)
            loj, hij = profile_width(u, rec, pj, nk, ok, sk, c2j,
                                     "tau1", lo_b, hi_b)
            wj = (hij - loj) / pj["tau1"]
            censored = (abs(loj - lo_b) / lo_b < 1e-3 or
                        abs(hij - hi_b) / hi_b < 1e-3 or wj > W_WEAK)
            I = wj / w_full if np.isfinite(w_full) and w_full > 0 else np.nan
            pg_reduced = projected_grad_info(rec, u, pj, nk, ok, sk)
            rows.append(dict(
                SampleID=sid, Network=u["network"], AgeCat=u["age_cat"],
                LPM=u["lpm"], omitted_tracer=nm,
                n_active_full=len(u["names"]), n_active_reduced=len(nk),
                tau1_full=pf["tau1"], tau1_reduced=pj["tau1"],
                delta_tau1_rel=abs(pj["tau1"] - pf["tau1"]) / pf["tau1"],
                w_full=w_full, w_reduced=wj,
                information_ratio=I, censored=bool(censored),
                chi2_full=c2f, chi2_reduced=c2j,
                projected_grad_norm_full=pg_full["projected_grad_norm"],
                projected_grad_norm_reduced=pg_reduced["projected_grad_norm"],
                bound_active_reduced=pg_reduced["bound_active"]))
        except Exception as e:                                  # noqa: BLE001
            rows.append(dict(SampleID=sid, omitted_tracer=nm,
                             error=str(e)[:120]))
    return rows


# ══════════════════════════════════════════════════════════════════════════
# SWEEP 3 — MODEL STRUCTURE  (Methods 3.6)
# ══════════════════════════════════════════════════════════════════════════

def as_structure(rec: dict, target: str) -> dict:
    """Return a copy of rec coerced to 'DM' or 'BMM-DM-DM'."""
    r = copy.deepcopy(rec)
    lr = r["lpm"]["tracerlpm_result"]
    init = r["lpm"]["init"]
    if target == "DM":
        r["lpm"]["name"] = "DM"
        r["lpm"]["free_params"] = "Mean Age, Model Parm 1"
        if not init.get("pd1"):
            init["pd1"] = 0.01
    else:
        r["lpm"]["name"] = "BMM-DM-DM"
        r["lpm"]["free_params"] = "Mean Age, Fraction"
        if not init.get("tau2_yr"):
            init["tau2_yr"] = BMM_TAU2_DEFAULT
            lr["tau2_yr"] = None
        if not init.get("pd2"):
            init["pd2"] = BMM_PD2_DEFAULT
        if not init.get("fraction1"):
            init["fraction1"] = BMM_F1_DEFAULT
    return r


def sweep_structure(rec, baseline, args):
    u = unpack(rec)
    sid, na = u["sid"], len(u["names"])
    objective = getattr(args, "objective", "rel")
    out = dict(SampleID=sid, Network=u["network"], AgeCat=u["age_cat"],
               assigned_LPM=u["lpm"], n_active=na, objective=objective)
    store = {}
    b = rec["pinn_bounds"]
    for tgt in ("DM", "BMM-DM-DM"):
        r = as_structure(rec, tgt)
        uu = unpack(r)
        # --fit-sigma-source reported: REFIT under the reported analytical
        # sigmas rather than evaluating them at the inversion-sigma optimum.
        # The distinction matters. A uniform rescaling leaves the optimum
        # invariant, so a no-refit evaluation is exact; the analytical sigmas
        # are per-tracer and span roughly 0.2% to 30%, so reweighting by
        # factors of ~40 moves the optimum substantially and a no-refit
        # evaluation is not a usable approximation to the analytical-sigma
        # analysis. Swapping errs here makes the objective, the misfit and
        # the classification all consistently analytical-sigma.
        if getattr(args, "fit_sigma_source", "inversion") == "reported":
            er, n_sub = reported_errs(sid, uu["names"], uu["obs"], uu["errs"])
            if er is None:
                raise SystemExit("--fit-sigma-source reported needs "
                                 "--reported-sigma-dir")
            uu["errs"] = er
            out["n_sigma_substituted"] = n_sub
        try:
            if objective == "sigma":
                with sigma_objective(uu["errs"]):
                    p, _fit_loss, sims = run_fit(r, uu["names"], uu["obs"],
                                                 uu["scales"], args.adam,
                                                 args.starts)
            else:
                p, _fit_loss, sims = run_fit(r, uu["names"], uu["obs"],
                                             uu["scales"], args.adam,
                                             args.starts)
            # Always report both misfits at whatever optimum the fit above
            # landed on -- unambiguous regardless of which objective drove it,
            # and what Task 2's goodness-of-fit diagnostic needs.
            c2r = chi2_rel(sims, uu["obs"])
            c2s = chi2_sigma(sims, uu["obs"], uu["errs"])
            k = n_free_of(r)
            a = aicc(c2s, k, na)
            store[tgt] = a
            tag = "DM" if tgt == "DM" else "BMM"
            # Error-scale sensitivity for Text S6, evaluated AT THIS OPTIMUM
            # without refitting. Exact for a uniform rescaling of sigma (the
            # optimum is invariant); approximate for the per-tracer reported
            # analytical sigmas, which do move it. That limitation is stated
            # where the number is used.
            er, n_sub = reported_errs(sid, uu["names"], uu["obs"], uu["errs"])
            if er is not None:
                out[f"chi2sig_rep_{tag}"] = chi2_sigma(sims, uu["obs"], er)
                out[f"n_sigma_substituted_{tag}"] = n_sub
            pg = projected_grad_info(r, uu, p, uu["names"], uu["obs"], uu["scales"])
            out.update({f"tau1_{tag}": p["tau1"], f"f1_{tag}": p["f1"],
                        f"chi2rel_{tag}": c2r, f"chi2sig_{tag}": c2s,
                        f"k_{tag}": k, f"AICc_{tag}": a,
                        f"projected_grad_norm_{tag}": pg["projected_grad_norm"]})
        except Exception as e:                                   # noqa: BLE001
            out[f"error_{tgt}"] = str(e)[:120]
            store[tgt] = np.nan

    # Bound-collapse check (Task 1, verification 2): does the BMM refit's
    # tau1/f1 land on the optimisation bound rather than an interior optimum?
    tau1_bmm, f1_bmm = out.get("tau1_BMM"), out.get("f1_BMM")
    tau1_lo, f1_hi = float(b["tau1_lo"]), 0.997
    out["BMM_tau1_at_lower_bound"] = bool(
        tau1_bmm is not None and np.isfinite(tau1_bmm)
        and abs(tau1_bmm - tau1_lo) / tau1_lo < 0.01)
    out["BMM_f1_at_upper_bound"] = bool(
        f1_bmm is not None and np.isfinite(f1_bmm)
        and abs(f1_bmm - f1_hi) / f1_hi < 0.01)

    d = store.get("BMM-DM-DM", np.nan) - store.get("DM", np.nan)

    # The small-sample correction CANCELS in the difference whenever the two
    # structures carry the same parameter count, which they always do here
    # (k = 2 for both; verified across all 60 samples, and dAICc equals
    # chi2_sigma(BMM) - chi2_sigma(DM) to 1.4e-14). So the comparison is well
    # posed at any dof >= 1, even where aicc() itself is undefined because
    # n_a - k - 1 <= 0. Before 2026-08-24 those samples were dropped as "not
    # assessable" on the strength of an undefined correction, discarding the
    # ten three-tracer samples for no reason and holding Section 3.4 to 44 of
    # 60 rather than 54.
    #
    # Two-tracer samples (dof = 0) stay excluded, but for the real reason: both
    # structures are saturated, so a misfit comparison carries no information.
    # Note they do NOT all fit exactly -- |dchi2_sigma| runs to 2.95 -- so
    # "both fit exactly" is not the justification either.
    kd, kb = out.get("k_DM"), out.get("k_BMM")
    dof_here = na - (kd if kd is not None else 0)
    if (not np.isfinite(d)) and kd is not None and kd == kb and dof_here >= 1:
        cs_d, cs_b = out.get("chi2sig_DM"), out.get("chi2sig_BMM")
        if cs_d is not None and cs_b is not None \
                and np.isfinite(cs_d) and np.isfinite(cs_b):
            d = float(cs_b) - float(cs_d)
            out["dAICc_from_chi2sigma"] = True

    out["dAICc"] = d
    if not np.isfinite(d):
        out["preferred"], out["support"] = "not assessable", "not assessable"
    else:
        out["preferred"] = "BMM-DM-DM" if d < 0 else "DM"
        ad = abs(d)
        out["support"] = ("indistinguishable" if ad < 2
                          else "positive" if ad < 10 else "strong")
    if out["support"] in ("indistinguishable", "not assessable"):
        out["vs_assigned"] = out["support"]
    else:
        out["vs_assigned"] = ("supported" if out["preferred"] == u["lpm"]
                              else "contradicted")
    return out


# ══════════════════════════════════════════════════════════════════════════
# DRIVER
# ══════════════════════════════════════════════════════════════════════════

def worker(payload):
    rec, baseline, args, which = payload
    bootstrap(args.package)
    # Windows spawns fresh interpreters, so main()'s module-level load does
    # not reach the workers -- rehydrate here, once per worker process. Skip
    # this and every chi2sig_rep_* column silently comes back empty under
    # --processes > 1 while looking fine in serial.
    global _REPORTED_REL_SIGMA
    if getattr(args, "reported_sigma_dir", None) and not _REPORTED_REL_SIGMA:
        _REPORTED_REL_SIGMA = load_reported_rel_sigma(args.reported_sigma_dir)
    try:
        if which == "ident":
            return [sweep_ident(rec, baseline, args)]
        if which == "loto":
            return sweep_loto(rec, baseline, args)
        return [sweep_structure(rec, baseline, args)]
    except Exception as e:                                       # noqa: BLE001
        import traceback
        return [dict(SampleID=rec["sample_id"], error=str(e)[:200],
                     traceback=traceback.format_exc()[-400:])]


def load_baseline(path):
    if not path or not os.path.exists(path):
        return None
    import pandas as pd
    df = pd.read_csv(path)
    return {r["SampleID"]: r for _, r in df.iterrows()}


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--package", default=".",
                    help="DLPMI package root (contains dlpmi/ and examples/)")
    ap.add_argument("--config", default=None,
                    help="edwards_input_config.json (default: inside package)")
    ap.add_argument("--baseline-csv", default=None,
                    help="00_summary_table.csv; lets 'ident' skip refitting")
    ap.add_argument("--out", default="sweeps")
    ap.add_argument("--sweeps", default="ident",
                    help="comma list of: ident,loto,structure")
    ap.add_argument("--objective", choices=["rel", "sigma"], default="rel",
                    help="structure sweep only: 'rel' fits under the "
                         "relative-error chi2 (Eq. 9, legacy behaviour); "
                         "'sigma' fits under the measurement-error-weighted "
                         "chi2_sigma (Eq. 19) so AICc compares each structure "
                         "at its own maximum-likelihood optimum")
    ap.add_argument("--adam", type=int, default=5000)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--processes", type=int, default=1)
    ap.add_argument("--ids", nargs="*", default=None)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--sites-xlsx", default=None,
                    help="Table_1_Sites xlsx; merges Aq_class (zone), Aq_seg, "
                         "Well_class onto every output table")
    ap.add_argument("--keep-excluded", action="store_true",
                    help="retain the 5 samples outside the San Antonio Segment")
    ap.add_argument("--reported-sigma-dir", default=None,
                    help="Musgrove data-release directory holding "
                         "Table_3_Tracers.txt and Table_5_Carbon14.txt. When "
                         "given, the structure sweep also reports chi2_sigma "
                         "at each optimum under the reported analytical "
                         "sigmas (Text S6 error-scale sensitivity).")
    ap.add_argument("--fit-sigma-source", choices=["inversion", "reported"],
                    default="inversion",
                    help="which sigmas drive the fit itself. 'reported' "
                         "REFITS under the analytical sigmas -- required for "
                         "a valid analytical-sigma sensitivity, since those "
                         "sigmas are per-tracer and move the optimum.")
    args = ap.parse_args()

    if args.reported_sigma_dir:
        global _REPORTED_REL_SIGMA
        _REPORTED_REL_SIGMA = load_reported_rel_sigma(args.reported_sigma_dir)
        print("  reported analytical sigma loaded for %d samples"
              % len(_REPORTED_REL_SIGMA))

    bootstrap(args.package)
    cfg = args.config or os.path.join(
        os.path.abspath(args.package), "examples", "edwards_aquifer",
        "edwards_input_config.json")
    recs = json.load(open(cfg))["samples"]
    if not args.keep_excluded:
        recs = [r for r in recs if r["sample_id"] not in EXCLUDE_DEFAULT]
    if args.ids:
        recs = [r for r in recs if r["sample_id"] in set(args.ids)]
    if args.limit:
        recs = recs[:args.limit]

    baseline = load_baseline(args.baseline_csv)
    os.makedirs(args.out, exist_ok=True)
    import pandas as pd

    for which in [s.strip() for s in args.sweeps.split(",") if s.strip()]:
        obj_tag = f" | objective={args.objective}" if which == "structure" else ""
        print(f"\n=== sweep: {which} | {len(recs)} samples "
              f"| adam={args.adam} starts={args.starts} "
              f"| processes={args.processes}{obj_tag} ===", flush=True)
        payloads = [(r, baseline, args, which) for r in recs]
        rows = []
        if args.processes > 1:
            import multiprocessing as mp
            with mp.Pool(args.processes) as pool:
                for i, res in enumerate(pool.imap_unordered(worker, payloads), 1):
                    rows.extend(res)
                    print(f"  [{i}/{len(payloads)}] done", flush=True)
        else:
            for i, p in enumerate(payloads, 1):
                res = worker(p)
                rows.extend(res)
                sid = res[0].get("SampleID", "?")
                print(f"  [{i}/{len(payloads)}] {sid}", flush=True)

        df = pd.DataFrame(rows)
        if args.sites_xlsx and os.path.exists(args.sites_xlsx):
            sites = (pd.read_excel(args.sites_xlsx)
                       [["SampleID", "Aq_class", "Aq_seg", "Well_class"]]
                       .drop_duplicates("SampleID"))
            df = df.merge(sites, on="SampleID", how="left")
        path = os.path.join(args.out, {
            "ident": "sweep_identifiability.csv",
            "loto": "sweep_loto.csv",
            "structure": "sweep_structure.csv"}[which])
        df.to_csv(path, index=False)
        print(f"  -> {path}  ({len(df)} rows)")

        if which == "ident" and "identifiability" in df:
            print("\n  tau1 identifiability (raw Hessian):")
            print(df["identifiability"].value_counts().to_string())
            if "Aq_class" in df:
                print("\n  by aquifer zone (raw):")
                print(pd.crosstab(df["Aq_class"], df["identifiability"]).to_string())
            if "identifiability_scaled" in df:
                print("\n  tau1 identifiability (scale-invariant, log-theta Hessian):")
                print(df["identifiability_scaled"].value_counts().to_string())
                if "Aq_class" in df:
                    print("\n  by aquifer zone (scaled):")
                    print(pd.crosstab(df["Aq_class"], df["identifiability_scaled"]).to_string())
                changed = df[df["identifiability"] != df["identifiability_scaled"]]
                print(f"\n  samples changing classification: {len(changed)}/{len(df)}")
                if len(changed):
                    print(changed[["SampleID", "identifiability",
                                   "identifiability_scaled", "cond_number",
                                   "cond_number_scaled"]].to_string(index=False))
                print(f"\n  sloppy_loading_scaled distribution:")
                print(df["sloppy_loading_scaled"].describe().to_string())
                print(f"\n  cond_number_scaled distribution:")
                print(df["cond_number_scaled"].describe().to_string())
                print(f"\n  gradient_check (norm(g*theta*)/norm(H_scaled)) distribution:")
                print(df["gradient_check"].describe().to_string())
            if "bound_active" in df:
                print(f"\n  projected_grad_norm distribution (should be ~0 for all -- KKT check):")
                print(df["projected_grad_norm"].describe().to_string())
                not_conv = df[df["projected_grad_norm"] > 1.0]
                if len(not_conv):
                    print(f"\n  WARNING: {len(not_conv)} samples with "
                          f"projected_grad_norm > 1.0 (not converged):")
                    print(not_conv[["SampleID", "projected_grad_norm"]].to_string(index=False))
                interior = df[~df["bound_active"]]
                mismatch = interior[interior["identifiability"] != interior["identifiability_final"]]
                print(f"\n  interior samples where identifiability_final != "
                      f"identifiability (should be 0): {len(mismatch)}/{len(interior)}")
                if len(mismatch):
                    print(mismatch[["SampleID", "identifiability",
                                    "identifiability_final"]].to_string(index=False))
                print(f"\n  n bound_active: {int(df['bound_active'].sum())}/{len(df)}")
                print(df[df["bound_active"]][["SampleID", "bound_active_params"]]
                      .to_string(index=False))
                print("\n  identifiability_final counts:")
                print(df["identifiability_final"].value_counts().to_string())
                print("\n  identifiability_final stratified by bound_active:")
                print(pd.crosstab(df["bound_active"], df["identifiability_final"]).to_string())
                if "Aq_class" in df:
                    print("\n  identifiability_final by Aq_class:")
                    print(pd.crosstab(df["Aq_class"], df["identifiability_final"]).to_string())
                    print("\n  identifiability_final by Aq_class, bound-active only:")
                    ba = df[df["bound_active"]]
                    if len(ba):
                        print(pd.crosstab(ba["Aq_class"], ba["identifiability_final"]).to_string())
        if which == "structure" and "vs_assigned" in df:
            print(f"\n  a priori assignment vs data (objective={args.objective}):")
            print(df["vs_assigned"].value_counts().to_string())
            if "n_active" in df:
                # Assessability needs dof >= 1, not n_active >= 4: the AICc
                # correction term is identical for the two structures (k = 2
                # each) and cancels in the difference, so a sample with three
                # active tracers is comparable even though aicc() itself is
                # undefined there. See the dAICc block above.
                n_ok = int((df["n_active"] >= 3).sum())
                n_bad = len(df) - n_ok
                print(f"\n  assessable (dof>=1, i.e. n_active>=3): "
                      f"{n_ok}/{len(df)} samples; {n_bad} not assessable "
                      f"(n_active=2, both structures saturated) -- "
                      f"Sec 3.4 restricted to the {n_ok}-sample subset")
            if "BMM_tau1_at_lower_bound" in df:
                n_tau1_bound = int(df["BMM_tau1_at_lower_bound"].sum())
                n_f1_bound = int(df["BMM_f1_at_upper_bound"].sum())
                print(f"\n  BMM refits landing on a bound: tau1 at lower "
                      f"bound in {n_tau1_bound}/{len(df)}, f1 at upper bound "
                      f"in {n_f1_bound}/{len(df)}")
        if which == "loto" and "information_ratio" in df:
            print("\n  median information ratio by tracer:")
            print(df.groupby("omitted_tracer")["information_ratio"]
                    .median().round(3).to_string())


if __name__ == "__main__":
    main()
