"""
edwards_pinn_estimate.py  (with uncertainty analysis)
======================================================
Estimate groundwater ages + parameter uncertainties for all 65 samples
in the Edwards Aquifer Table 4 dataset using Physics-Informed Neural
Networks (PINNs).

Three-file architecture
-----------------------
  edwards_input_config.json   ← sample data, TracerLPM params, scale factors
  edwards_pinn_physics.py     ← tracer input functions, DM/BMM forward models,
                                 PINN classes, loss, training, uncertainty fns
  edwards_pinn_estimate.py    ← THIS FILE: load → fit → uncertainty → plot → CSV

Uncertainty methods (all at the PINN optimum)
----------------------------------------------
  Hessian (H)  : sigma = sqrt(chi2/dof * diag(inv(H)))
                 Equivalent to LM covariance used by TracerLPM.
                 Fast, assumes locally parabolic chi2 surface.

  Profile (P)  : 68 % CI where Δchi2 <= 1  (other params held fixed).
                 Asymmetric intervals. Most robust for non-parabolic surfaces.

  Monte Carlo (MC) : Perturb obs by measurement errors, re-fit N times.
                     sigma = std of fitted params over N realisations.
                     Captures full nonlinear error propagation.

  Chi2 probability : prob = chi2.sf(chi2_val, dof)  — TracerLPM's LPM_Prob.

Usage
-----
  python edwards_pinn_estimate.py                   # all 65 samples, all methods
  python edwards_pinn_estimate.py --ids EDTRPAS1-29 EDTRPAS1-36
  python edwards_pinn_estimate.py --starts 3 --adam 5000 --mc 100
  python edwards_pinn_estimate.py --no-profile      # skip slow profile CI
  python edwards_pinn_estimate.py --no-mc           # skip Monte Carlo

Outputs (./outputs/edwards_pinn/ by default)
--------------------------------------------
  {SampleID}_pinn.png          4-panel figure + uncertainty panel
  00_summary_all_samples.png   τ₁ & χ² bar chart with error bars
  00_summary_table.csv         full comparison + all uncertainty estimates
"""

import argparse, json, os, csv as _csv, sys, warnings, traceback as _tb
import torch
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats as sp_stats

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _HERE)
from edwards_pinn_physics import (
    LAMBDA_3H, LAMBDA_14C,
    AGES_YOUNG, AGES_OLD,
    g_DM, forward_DM, forward_BMM,
    chi2_loss, train_pinn,
    PINN_DM, PINN_BMM,
    chi2_probability,
    hessian_uncertainty,
    profile_uncertainty,
    mc_uncertainty,
)

warnings.filterwarnings("ignore")
torch.set_default_dtype(torch.float32)
torch.manual_seed(42)

try:
    _trapz = np.trapezoid
except AttributeError:
    _trapz = np.trapz

_CONFIG_DEFAULT = os.path.join(_HERE, "edwards_input_config.json")
_OUT_DIR        = os.path.join(_HERE, "edwards_pinn")

BLUE="#2b6cb0"; RED="#c0392b"; ORG="#e07b00"
TEAL="#2c7a6e"; PUR="#6a3fa1"; GRY="#555555"; GRN="#276221"


# ════════════════════════════════════════════════════════════
# DATA HELPERS
# ════════════════════════════════════════════════════════════
def load_config(path):
    with open(path) as f:
        return json.load(f)["samples"]


def build_tracer_lists(s):
    names, obs, errs, scales, active = [], [], [], [], []
    for t in s["tracers"]:
        nm = t["name"]; sc = float(t.get("scale", 1.0))
        ov = t.get("obs")
        if ov is None: continue
        ov = float(ov)
        ev = t.get("obs_err")
        ev = float(ev) if ev is not None else abs(ov)*0.10
        eff_sc = float(t["sf6_effective_scale"]) if nm=="SF6" and "sf6_effective_scale" in t else sc
        names.append(nm); obs.append(ov); errs.append(ev)
        scales.append(eff_sc); active.append(sc != 0.0)
    return names, obs, errs, scales, active


def _parse_free_params(s):
    return str(s["lpm"].get("free_params") or "Mean Age").lower()


def _count_free_params(s):
    lpm = s["lpm"]["name"]
    if lpm == "DM":
        return 2
    fp = _parse_free_params(s)
    return sum(["mean age" in fp, "fraction" in fp,
                "2nd" in fp or "second" in fp])


# ════════════════════════════════════════════════════════════
# CORE FIT  (unchanged logic)
# ════════════════════════════════════════════════════════════
def _run_fit(s, tns_a, obs_a, scls_a, he4r, dic_c1, dic_c2,
             n_adam, n_starts, verbose, dgmeta=None):
    """Inner fit returning best_model, best_hist, best_loss."""
    if dgmeta is None:
        dgmeta = {}
    lpm_name = s["lpm"]["name"]
    init     = s["lpm"]["init"]
    bounds   = s["pinn_bounds"]
    lpm_r    = s["lpm"]["tracerlpm_result"]
    sd       = float(s["sample_date"])
    fp       = _parse_free_params(s)

    tau1_0 = float(lpm_r.get("tau1_yr")  or init["tau1_yr"]  or 20.)
    pd1_0  = float(lpm_r.get("pd1")      or init["pd1"]      or 0.01)
    f1_0   = float(lpm_r.get("fraction1")or init["fraction1"]or 0.5)
    tau2_0 = float(lpm_r.get("tau2_yr")  or init["tau2_yr"]  or 100.)
    pd2_0  = float(init["pd2"] or 0.01)

    tau1_lo = float(bounds["tau1_lo"]); tau1_hi = float(bounds["tau1_hi"])
    pd1_lo  = float(bounds["pd1_lo"]);  pd1_hi  = float(bounds["pd1_hi"])
    f1_lo   = 0.005; f1_hi = 0.997

    def _seed_tau(t0):
        return [np.clip(v, tau1_lo*1.02, tau1_hi*0.98)
                for v in [t0, t0*0.5, t0*2.]][:n_starts]
    def _seed_f(f0):
        return [np.clip(v, f1_lo*1.02, f1_hi*0.98)
                for v in [f0, f0*0.7, f0*1.3]][:n_starts]

    best_loss = float("inf"); best_model = None; best_hist = []

    if lpm_name == "DM":
        for tau0 in _seed_tau(tau1_0):
            m  = PINN_DM(tau0, pd1_0, tau1_lo, tau1_hi, pd1_lo, pd1_hi)
            def lfn(m=m):
                tau, pd = m.get_params()
                ag = AGES_OLD if float(tau.detach()) > 500. else AGES_YOUNG
                return chi2_loss(forward_DM(tau, pd, sd, tns_a, scls_a, he4r,
                                            ages=ag, dgmeta=dgmeta), obs_a)
            h  = train_pinn(m, lfn, n_adam)
            fl = lfn().item()
            if verbose:
                tv, pv = m.get_params()
                print(f"      τ₀={tau0:.1f} → τ={tv.item():.3f} PD={pv.item():.5f} χ²={fl:.5f}")
            if fl < best_loss: best_loss=fl; best_model=m; best_hist=h

    else:
        tau2_v  = float(lpm_r.get("tau2_yr") or tau2_0)
        t1_hi_b = min(tau1_hi, tau2_v*0.9); t1_hi_b = max(t1_hi_b, tau1_lo*2.)
        for tau0, f0 in zip(_seed_tau(tau1_0), _seed_f(f1_0)):
            m = PINN_BMM(fp,
                         float(np.clip(tau1_0, tau1_lo*1.02, t1_hi_b*0.98)),
                         pd1_0, float(f0), tau2_v, pd2_0,
                         tau1_lo, t1_hi_b, f1_lo, f1_hi,
                         tau2_lo=max(1., tau2_v*0.05),
                         tau2_hi=tau2_v*5.)
            def lfn(m=m):
                t1,p1,f1,t2,p2 = m.get_params()
                return chi2_loss(forward_BMM(t1,p1,f1,t2,p2,sd,tns_a,scls_a,he4r,dic_c1,dic_c2), obs_a)
            h  = train_pinn(m, lfn, n_adam)
            fl = lfn().item()
            if verbose:
                t1,_,f1,t2,_ = m.get_params()
                print(f"      τ₁₀={tau0:.1f} f₀={f0:.3f} → τ₁={t1.item():.3f} f₁={f1.item():.4f} χ²={fl:.5f}")
            if fl < best_loss: best_loss=fl; best_model=m; best_hist=h

    return best_model, best_hist, best_loss


# ════════════════════════════════════════════════════════════
# UNCERTAINTY HELPERS
# ════════════════════════════════════════════════════════════
def _compute_hessian(s, model, tns_a, obs_a, scls_a, he4r, dic_c1, dic_c2, chi2_val):
    """Compute Hessian-based 1-sigma uncertainties."""
    sd = float(s["sample_date"])
    lpm = s["lpm"]["name"]
    n_free = _count_free_params(s)
    n_act  = sum(1 for t in s["tracers"] if t.get("scale", 0) != 0)

    if lpm == "DM":
        tau_v, pd_v = [p.item() for p in model.get_params()]
        params_opt  = torch.tensor([tau_v, pd_v], dtype=torch.float32)
        bounds_lo   = [float(s["pinn_bounds"]["tau1_lo"]),
                       float(s["pinn_bounds"]["pd1_lo"])]
        bounds_hi   = [float(s["pinn_bounds"]["tau1_hi"]),
                       float(s["pinn_bounds"]["pd1_hi"])]

        def lfn_direct(p):
            ag = AGES_OLD if float(p[0].detach()) > 500. else AGES_YOUNG
            return chi2_loss(forward_DM(p[0], p[1], sd, tns_a, scls_a, he4r, ages=ag), obs_a)

        sigmas, cov, dof = hessian_uncertainty(lfn_direct, params_opt,
                                                chi2_val, n_act, n_free)
        param_names = ["tau1", "pd1"]
        param_vals  = [tau_v, pd_v]
        lo_bounds   = bounds_lo; hi_bounds = bounds_hi

    else:
        t1v, p1v, f1v, t2v, p2v = [p.item() for p in model.get_params()]
        params_opt = torch.tensor([t1v, f1v], dtype=torch.float32)
        tau1_hi_b  = min(float(s["pinn_bounds"]["tau1_hi"]), t2v*0.9)
        bounds_lo  = [float(s["pinn_bounds"]["tau1_lo"]), 0.005]
        bounds_hi  = [tau1_hi_b, 0.997]
        fp = _parse_free_params(s)

        def lfn_direct(p):
            # Use the free params that were actually optimised
            tau1 = p[0] if "mean age" in fp else torch.tensor(t1v)
            f1   = p[1] if "fraction" in fp else torch.tensor(f1v)
            return chi2_loss(
                forward_BMM(tau1, torch.tensor(p1v), f1,
                            torch.tensor(t2v), torch.tensor(p2v),
                            sd, tns_a, scls_a, he4r, dic_c1, dic_c2), obs_a)

        sigmas, cov, dof = hessian_uncertainty(lfn_direct, params_opt,
                                                chi2_val, n_act, n_free)
        param_names = ["tau1", "f1"]
        param_vals  = [t1v, f1v]
        lo_bounds   = bounds_lo; hi_bounds = bounds_hi

    return {
        "param_names"  : param_names,
        "param_vals"   : param_vals,
        "sigmas_hess"  : sigmas.tolist(),
        "cov"          : cov.tolist(),
        "dof"          : dof,
        "lo_bounds"    : lo_bounds,
        "hi_bounds"    : hi_bounds,
        "lfn_direct"   : lfn_direct,   # for profile re-use
    }


def _compute_profile(hess_info, chi2_min):
    """Profile 68 % CI for each free parameter."""
    profile_ci = {}
    for i, (pname, pval, lo, hi) in enumerate(zip(
            hess_info["param_names"],
            hess_info["param_vals"],
            hess_info["lo_bounds"],
            hess_info["hi_bounds"])):

        params_fixed = list(hess_info["param_vals"])

        def lfn_1d(val, i=i, pf=params_fixed[:]):
            pf[i] = val
            return hess_info["lfn_direct"](torch.tensor(pf, dtype=torch.float32)).item()

        lo68, hi68 = profile_uncertainty(lfn_1d, pval, chi2_min, lo, hi)
        profile_ci[pname] = (lo68, hi68)

    return profile_ci


def _compute_mc(s, model, tns, tns_a, obs_all, obs_errs_all,
                scls, scls_a, he4r, dic_c1, dic_c2, n_mc, n_adam):
    """Monte Carlo uncertainty via observation perturbation (warm-start)."""
    sd   = float(s["sample_date"])
    lpm  = s["lpm"]["name"]
    act  = [t.get("scale",0)!=0 for t in s["tracers"]]
    fp   = _parse_free_params(s)
    tau1_v, pd1_v = (model.get_params()[0].item(),
                     model.get_params()[1].item())

    # Gather active obs/errs
    obs_a_nom = [o for o,a in zip(obs_all, act) if a]
    err_a     = [e for e,a in zip(obs_errs_all, act) if a]

    if lpm == "DM":
        t2v = None; f1v = None
        tau1_lo = float(s["pinn_bounds"]["tau1_lo"])
        tau1_hi = float(s["pinn_bounds"]["tau1_hi"])
        pd1_lo  = float(s["pinn_bounds"]["pd1_lo"])
        pd1_hi  = float(s["pinn_bounds"]["pd1_hi"])
    else:
        pars     = model.get_params()
        f1v      = pars[2].item(); t2v = pars[3].item(); pd2v = pars[4].item()
        tau1_lo  = float(s["pinn_bounds"]["tau1_lo"])
        tau1_hi  = min(float(s["pinn_bounds"]["tau1_hi"]), t2v*0.9)
        pd1_lo   = float(s["pinn_bounds"]["pd1_lo"])
        pd1_hi   = float(s["pinn_bounds"]["pd1_hi"])

    rng   = np.random.default_rng(42)
    tau1s = []; f1s = []; pd1s = []; tau2s = []
    n_fail = 0

    for _ in range(n_mc):
        obs_pert = [max(o + rng.normal(0., e), 1e-20)
                    for o, e in zip(obs_a_nom, err_a)]

        try:
            if lpm == "DM":
                m = PINN_DM(tau1_v, pd1_v, tau1_lo, tau1_hi, pd1_lo, pd1_hi)
                def lfn(m=m, op=obs_pert):
                    tau, pd = m.get_params()
                    ag = AGES_OLD if float(tau.detach())>500. else AGES_YOUNG
                    return chi2_loss(forward_DM(tau,pd,sd,tns_a,scls_a,he4r,ages=ag), op)
                train_pinn(m, lfn, min(n_adam//5, 800))    # warm-start: fewer epochs
                tv, pv = m.get_params()
                tau1s.append(tv.item()); pd1s.append(pv.item())
            else:
                t1_hi_b = min(tau1_hi, (t2v or 100.)*0.9)
                m = PINN_BMM(fp, tau1_v, pd1_v, f1v or 0.5,
                              t2v or 100., pd2v if t2v else 0.01,
                              tau1_lo, t1_hi_b, 0.005, 0.997,
                              tau2_lo=max(1., (t2v or 100.)*0.05),
                              tau2_hi=(t2v or 100.)*5.)
                def lfn(m=m, op=obs_pert):
                    t1,p1,f1,t2,p2 = m.get_params()
                    return chi2_loss(
                        forward_BMM(t1,p1,f1,t2,p2,sd,tns_a,scls_a,he4r,dic_c1,dic_c2), op)
                train_pinn(m, lfn, min(n_adam//5, 800))
                t1,_,f1,t2,_ = m.get_params()
                tau1s.append(t1.item()); f1s.append(f1.item())
                if "2nd" in fp or "second" in fp: tau2s.append(t2.item())
        except Exception:
            n_fail += 1

    def _stat(arr):
        a = np.array(arr)
        return {"mean":float(np.mean(a)), "std":float(np.std(a)),
                "p16":float(np.percentile(a,16)),
                "p50":float(np.percentile(a,50)),
                "p84":float(np.percentile(a,84))}

    mc = {}
    if tau1s: mc["tau1"] = _stat(tau1s)
    if pd1s:  mc["pd1"]  = _stat(pd1s)
    if f1s:   mc["f1"]   = _stat(f1s)
    if tau2s: mc["tau2"] = _stat(tau2s)
    mc["n_ok"]   = n_mc - n_fail
    mc["n_fail"] = n_fail
    return mc


# ════════════════════════════════════════════════════════════
# FIT ONE SAMPLE  (main entry point)
# ════════════════════════════════════════════════════════════
def fit_sample(s, n_adam=5000, n_starts=3, n_mc=100,
               do_profile=True, do_mc=True, verbose=False):
    """
    Fit one sample and compute three uncertainty estimates.

    Returns a results dict with keys:
      sid, lpm, chi2, chi2_lpm, chi2_prob, dof,
      tau1, pd1, f1, tau2, pd2,
      tau1_lpm, pd1_lpm, f1_lpm, tau2_lpm,
      tracer_names, obs_vals, obs_errs, eff_scales, active,
      sim_vals, sim_lpm, hist, sample_date, age_cat,
      unc_hess   : {'param_names', 'param_vals', 'sigmas_hess'}
      unc_profile: {'tau1': (lo,hi), 'f1': (lo,hi), ...}
      unc_mc     : {'tau1': {mean,std,p16,p50,p84}, ...}
    """
    sd     = float(s["sample_date"])
    lpm_n  = s["lpm"]["name"]
    lpm_r  = s["lpm"]["tracerlpm_result"]
    he4r   = float(s["he4_params"]["solution_rate_ccpgpyr"] or 1e-12)
    dic_c1 = float(s["carbon14_params"]["dic_c1_mmolL"] or 100.)
    dic_c2 = float(s["carbon14_params"]["dic_c2_mmolL"] or 100.)
    dgmeta = s.get("dgmeta_params", {})

    tns, obs_v, obs_e, scls, act = build_tracer_lists(s)
    tns_a   = [t  for t,a  in zip(tns,  act) if a]
    obs_a   = [o  for o,a  in zip(obs_v, act) if a]
    scls_a  = [sc for sc,a in zip(scls, act)  if a]

    # ── core fit ──────────────────────────────────────────────
    best_model, best_hist, best_loss = _run_fit(
        s, tns_a, obs_a, scls_a, he4r, dic_c1, dic_c2,
        n_adam, n_starts, verbose,
                              dgmeta=dgmeta)

    # ── extract optimal params ────────────────────────────────
    if lpm_n == "DM":
        tau_f, pd_f = best_model.get_params()
        tau1_v=tau_f.item(); pd1_v=pd_f.item(); f1_v=None; tau2_v=None; pd2_v=None
        ag_f = AGES_OLD if tau1_v>500. else AGES_YOUNG
        sims_f = forward_DM(tau1_v, pd1_v, sd, tns, scls, he4r, ages=ag_f)
    else:
        t1f,p1f,f1f,t2f,p2f = best_model.get_params()
        tau1_v=t1f.item(); pd1_v=p1f.item(); f1_v=f1f.item()
        tau2_v=t2f.item(); pd2_v=p2f.item()
        sims_f = forward_BMM(tau1_v,pd1_v,f1_v,tau2_v,pd2_v,
                              sd,tns,scls,he4r,dic_c1,dic_c2)

    # ── chi2 probability ──────────────────────────────────────
    n_act  = sum(act); n_free = _count_free_params(s)
    prob, dof = chi2_probability(best_loss, n_act, n_free)

    # ── Hessian uncertainty ───────────────────────────────────
    try:
        hess_info = _compute_hessian(s, best_model, tns_a, obs_a, scls_a,
                                     he4r, dic_c1, dic_c2, best_loss)
        unc_hess = {k: hess_info[k]
                    for k in ("param_names","param_vals","sigmas_hess","dof")}
    except Exception as e:
        unc_hess = {"error": str(e)}
        hess_info = None

    # ── Profile CI ────────────────────────────────────────────
    unc_profile = {}
    if do_profile and hess_info:
        try:
            unc_profile = _compute_profile(hess_info, best_loss)
        except Exception as e:
            unc_profile = {"error": str(e)}

    # ── Monte Carlo ───────────────────────────────────────────
    unc_mc = {}
    if do_mc:
        try:
            unc_mc = _compute_mc(s, best_model, tns, tns_a,
                                 obs_v, obs_e, scls, scls_a,
                                 he4r, dic_c1, dic_c2, n_mc, n_adam)
        except Exception as e:
            unc_mc = {"error": str(e)}

    return dict(
        sid=s["sample_id"], lpm=lpm_n,
        chi2=best_loss, chi2_lpm=lpm_r.get("chi2"),
        chi2_prob=prob, dof=dof,
        tau1=tau1_v, pd1=pd1_v, f1=f1_v, tau2=tau2_v, pd2=pd2_v,
        tau1_lpm=lpm_r.get("tau1_yr"), pd1_lpm=lpm_r.get("pd1"),
        f1_lpm=lpm_r.get("fraction1"), tau2_lpm=lpm_r.get("tau2_yr"),
        tau1_err_lpm=lpm_r.get("tau1_err"),
        f1_err_lpm  =lpm_r.get("f1_err"),
        tracer_names=tns, obs_vals=obs_v, obs_errs=obs_e,
        eff_scales=scls, active=act,
        sim_vals=[v.item() for v in sims_f],
        sim_lpm=lpm_r.get("mod_conc") or [],
        hist=best_hist, sample_date=sd,
        age_cat=s["age_category"],
        unc_hess=unc_hess,
        unc_profile=unc_profile,
        unc_mc=unc_mc,
    )


# ════════════════════════════════════════════════════════════
# FIGURE HELPERS
# ════════════════════════════════════════════════════════════
def _fmt(v, nd=4):
    if v is None or (isinstance(v, float) and np.isnan(v)): return "—"
    fv = float(v)
    if fv == 0.: return "0"
    if abs(fv) < 1e-4 or abs(fv) > 1e5: return f"{fv:.3e}"
    return f"{fv:.{nd}g}"

def _pct(s, o):
    return abs(float(s)-float(o))/abs(float(o))*100 if o else 0.


# ════════════════════════════════════════════════════════════
# PER-SAMPLE 5-PANEL FIGURE
# ════════════════════════════════════════════════════════════
def plot_sample(res, out_dir):
    """4-panel figure:  (a) Loss  (b) Tracer bars  (c) Cumulative age dist.  (d) PDF age dist.
    Plus a bottom uncertainty row spanning both columns."""
    sid   = res["sid"];    tns  = res["tracer_names"]
    obs_v = res["obs_vals"]; obs_e = res["obs_errs"]
    sim_v = res["sim_vals"]; sim_l = res["sim_lpm"]
    hist  = res["hist"];   act  = res["active"]
    lpm   = res["lpm"];    n    = len(tns)
    N_ADAM = 5000

    uh = res.get("unc_hess",   {})
    up = res.get("unc_profile",{})
    um = res.get("unc_mc",     {})

    # ── layout: 3 rows × 2 cols; row 2 spans both cols for uncertainty ────────
    fig = plt.figure(figsize=(16, 14))
    fig.patch.set_facecolor("white")
    gs  = GridSpec(3, 2, figure=fig, hspace=0.46, wspace=0.35,
                   left=0.08, right=0.97, top=0.93, bottom=0.05,
                   height_ratios=[1, 1, 0.55])

    # ── (a) Loss ──────────────────────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    ah = hist[:N_ADAM]; lh = hist[N_ADAM:]
    ax.semilogy(range(1, len(ah)+1), ah, color=BLUE, lw=1.8, label="Adam")
    if lh:
        ax.semilogy(range(N_ADAM+1, N_ADAM+len(lh)+1), lh,
                    color=ORG, lw=2.0, label="L-BFGS")
    ax.axvline(N_ADAM, color=GRY, ls=":", lw=1.2)
    ax.set_xlabel("Epoch", fontsize=10); ax.set_ylabel("χ²", fontsize=10)
    ax.set_title("(a)  Loss vs Epochs", fontsize=11, fontweight="bold", pad=7)
    ax.legend(fontsize=9)
    chi2_l = res.get("chi2_lpm"); better = res["chi2"] <= (chi2_l or 1e9)
    chi2_l_str = f"{chi2_l:.4f}" if chi2_l is not None else "n/a"
    ax.text(0.98, 0.97,
            f"χ²_PINN = {res['chi2']:.4f}\n"
            f"χ²_LPM  = {chi2_l_str}\n"
            f"Prob = {res['chi2_prob']:.4f}  (dof={res['dof']})",
            transform=ax.transAxes, ha="right", va="top", fontsize=8.5,
            color=GRN if better else ORG,
            bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.92))
    ax.grid(True, which="both", ls="--", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    # ── (b) Obs / Sim bar chart ───────────────────────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(n); w = 0.26
    norm_sim = [float(sim_v[i])/float(obs_v[i]) if obs_v[i] else 0. for i in range(n)]
    norm_lpm = [float(sim_l[i])/float(obs_v[i])
                if sim_l and i<len(sim_l) and sim_l[i] and obs_v[i] else None
                for i in range(n)]
    norm_err = [abs(float(obs_e[i])/float(obs_v[i])) if obs_v[i] else 0. for i in range(n)]

    ax.bar(x-w, [1.]*n, w, color=BLUE, alpha=0.85, label="Observed")
    bars_p = ax.bar(x, norm_sim, w, color=ORG, alpha=0.85, label="PINN")
    ax.errorbar(x-w, [1.]*n, yerr=[abs(e) for e in norm_err],
                fmt="none", ecolor=BLUE, capsize=3, lw=1.4)
    vx = [xi for xi,v in zip(x, norm_lpm) if v is not None]
    vv = [v  for v    in norm_lpm         if v is not None]
    if vv:
        ax.bar([xi+w for xi in vx], vv, w, color=RED, alpha=0.70, label="TracerLPM")
    ax.axhline(1., color=BLUE, ls="--", lw=0.8, alpha=0.45)
    for i, bar in enumerate(bars_p):
        if not act[i]:
            bar.set_hatch("///"); bar.set_edgecolor(GRY); bar.set_alpha(0.25)
        pe = _pct(sim_v[i], obs_v[i])
        col = GRN if pe < 10 else (ORG if pe < 30 else RED)
        ax.text(x[i]+0.01, max(norm_sim[i], 0.)+0.06, f"{pe:.0f}%",
                ha="center", fontsize=7.5, fontweight="bold", color=col)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{nm}\n{'●' if a else '○'}" for nm,a in zip(tns,act)], fontsize=8)
    ax.set_ylabel("Sim / Obs  (ratio)"); ax.legend(fontsize=8.5)
    ax.set_title("(b)  Tracer Fit  (● in loss,  ○ inactive)",
                 fontsize=11, fontweight="bold", pad=7)
    ax.grid(axis="y", ls="--", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    # ── helper: build PDF + CDF for one DM component ─────────────────────────
    def _pdf_cdf(tau_val, pd_val, ages_t):
        g   = g_DM(ages_t, torch.tensor(float(tau_val)),
                   torch.tensor(float(pd_val))).detach().numpy()
        a   = ages_t.numpy()
        da  = np.diff(a)
        cdf = np.insert(np.cumsum(0.5*(g[:-1]+g[1:])*da), 0, 0.)
        cdf /= max(cdf[-1], 1e-12)
        return a, g, cdf

    tau1_v = res["tau1"]; pd1_v = res["pd1"]
    ages_plt = AGES_OLD if tau1_v > 500. else AGES_YOUNG
    a1, g1, cdf1 = _pdf_cdf(tau1_v, pd1_v, ages_plt)

    tau1_lpm = res.get("tau1_lpm"); pd1_lpm = res.get("pd1_lpm") or pd1_v
    if tau1_lpm:
        _, g_lpm, cdf_lpm = _pdf_cdf(float(tau1_lpm), float(pd1_lpm), ages_plt)
    xlim = (min(max(tau1_v*5, 50.), 60000.) if tau1_v > 500.
            else min(max(tau1_v*5, 30.), 300.))

    # ── (c) Cumulative Age Distribution ──────────────────────────────────────
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(a1, cdf1, color=TEAL, lw=2.4, label=f"PINN  τ₁={tau1_v:.2f} yr")
    ax.fill_between(a1, cdf1, alpha=0.12, color=TEAL)
    if tau1_lpm:
        ax.plot(a1, cdf_lpm, color=RED, lw=1.8, ls="-.",
                label=f"LPM   τ₁={float(tau1_lpm):.2f} yr")

    # BMM: also show old component CDF and composite
    if lpm == "BMM-DM-DM" and res.get("f1") is not None:
        f1_v = res["f1"]; tau2_v = res["tau2"]
        ages_o = AGES_OLD if tau2_v > 500. else AGES_YOUNG
        a2, g2, cdf2 = _pdf_cdf(tau2_v, res["pd2"], ages_o)
        # Composite CDF on shared age axis
        a_all   = np.sort(np.concatenate([a1, a2]))
        cdf1_e  = np.interp(a_all, a1, cdf1)
        cdf2_e  = np.interp(a_all, a2, cdf2)
        cdf_comp = f1_v*cdf1_e + (1.-f1_v)*cdf2_e
        ax.plot(a_all, cdf_comp, color=PUR, lw=2.0, ls="--",
                label=f"Composite BMM")
        # x-axis in log for BMM (spans orders of magnitude)
        ax.set_xscale("log"); xlim_c = max(a2[-1]*0.9, 1000.)
        ax.set_xlim(max(a_all[0], 0.05), xlim_c)
        ax.set_xlabel("Age  (yr, log scale)", fontsize=10)
    else:
        ax.set_xlim(0, xlim)
        ax.set_xlabel("Age  (yr)", fontsize=10)

    # Percentile annotations
    for pv, lbl in [(0.10,"P10"), (0.50,"P50"), (0.90,"P90")]:
        a_at = float(np.interp(pv, cdf1, a1))
        ax.axhline(pv, color=GRY, ls=":", lw=0.9, alpha=0.55)
        ax.axvline(a_at, color=TEAL, ls=":", lw=0.9, alpha=0.55)
        ax.text(a_at, pv+0.025, f"{lbl}={a_at:.1f} yr",
                fontsize=7.5, color=TEAL, ha="left" if pv < 0.7 else "right")
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Cumulative fraction  F(a)", fontsize=10)
    ax.set_title("(c)  Cumulative Age Distribution",
                 fontsize=11, fontweight="bold", pad=7)
    ax.legend(fontsize=8.5, loc="lower right")
    ax.grid(True, ls="--", alpha=0.3)
    ax.spines[["top","right"]].set_visible(False)

    # ── (d) PDF Age Distribution ──────────────────────────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    ax.fill_between(a1, g1, alpha=0.18, color=TEAL)
    ax.plot(a1, g1, color=TEAL, lw=2.2, label=f"PINN  τ₁={tau1_v:.2f} yr")
    ax.axvline(tau1_v, color=TEAL, ls="--", lw=1.6, alpha=0.75)
    if tau1_lpm:
        ax.plot(a1, g_lpm, color=RED, lw=1.6, ls="-.",
                label=f"LPM   τ₁={float(tau1_lpm):.2f} yr")
    ax.set_xlabel("Age  (yr)", fontsize=10)
    ax.set_ylabel("PDF  g₁(a)", fontsize=10, color=TEAL)
    ax.tick_params(axis="y", labelcolor=TEAL)

    if lpm == "BMM-DM-DM" and res.get("f1") is not None:
        f1_v = res["f1"]; tau2_v = res["tau2"]
        ages_o = AGES_OLD if tau2_v > 500. else AGES_YOUNG
        a2, g2, _ = _pdf_cdf(tau2_v, res["pd2"], ages_o)
        ax2 = ax.twinx()
        ax2.fill_between(a2/1000., g2*1000., alpha=0.10, color=PUR)
        ax2.plot(a2/1000., g2*1000., color=PUR, lw=1.5, ls="--",
                 label=f"Old DM  τ₂={tau2_v:.0f} yr")
        ax2.set_ylabel("g₂(a) × 10³  (old)", fontsize=9, color=PUR)
        ax2.tick_params(axis="y", labelcolor=PUR)
        total = f1_v*tau1_v + (1.-f1_v)*tau2_v
        ax.text(0.03, 0.96,
                f"f₁={f1_v:.3f}  f₂={1-f1_v:.3f}\nTotal age ≈ {total:.0f} yr",
                transform=ax.transAxes, fontsize=8.5, va="top",
                bbox=dict(boxstyle="round", fc="white", ec="#ccc", alpha=0.92))
        l1, lb1 = ax.get_legend_handles_labels()
        l2, lb2 = ax2.get_legend_handles_labels()
        ax.legend(l1+l2, lb1+lb2, fontsize=8, loc="upper right")
    else:
        ax.legend(fontsize=8, loc="upper right")

    ax.set_xlim(0, xlim)
    ax.set_title("(d)  Age PDF", fontsize=11, fontweight="bold", pad=7)
    ax.grid(True, ls="--", alpha=0.3)
    ax.spines[["top"]].set_visible(False)

    # ── (e) Uncertainty table  (spans both columns, bottom row) ──────────────
    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")

    pnames = uh.get("param_names", [])
    pvals  = uh.get("param_vals",  [])
    sigmas = uh.get("sigmas_hess", [None]*len(pnames))
    lpm_tau1_err = res.get("tau1_err_lpm")
    lpm_f1_err   = res.get("f1_err_lpm")

    col_h2 = ["Param", "Best fit",
               "Hessian σ", "LPM σ (ref)",
               "Profile 68% lo", "Profile 68% hi", "Profile ±σ",
               "MC mean", "MC σ", "MC P16–P84", "N_MC"]
    rows2 = []
    for i, pn in enumerate(pnames):
        pv    = pvals[i]  if i < len(pvals)  else None
        sig_h = sigmas[i] if i < len(sigmas) else None
        pci   = up.get(pn)
        if isinstance(pci, tuple):
            lo68, hi68 = pci
            prof_lo = f"{lo68:.3f}"; prof_hi = f"{hi68:.3f}"
            prof_sym = f"±{(hi68-lo68)/2:.3f}"
        else:
            prof_lo = prof_hi = prof_sym = "—"
        mcs   = um.get(pn, {})
        mc_m  = f"{mcs['mean']:.3f}" if "mean" in mcs else "—"
        mc_s  = f"{mcs['std']:.3f}"  if "std"  in mcs else "—"
        mc_r  = f"[{mcs['p16']:.3f}, {mcs['p84']:.3f}]" if "p16" in mcs else "—"
        n_mc  = str(um.get("n_ok", "—"))
        lpm_err = "—"
        if pn == "tau1" and lpm_tau1_err: lpm_err = f"{float(lpm_tau1_err):.3f}"
        if pn == "f1"   and lpm_f1_err:   lpm_err = f"{float(lpm_f1_err):.4g}"
        rows2.append([pn,
                      f"{pv:.4f}" if pv is not None else "—",
                      f"{sig_h:.4f}" if sig_h is not None else "—",
                      lpm_err,
                      prof_lo, prof_hi, prof_sym,
                      mc_m, mc_s, mc_r, n_mc])

    if rows2:
        tbl2 = ax.table(cellText=rows2, colLabels=col_h2,
                        loc="center", cellLoc="center")
        tbl2.auto_set_font_size(False); tbl2.set_fontsize(9)
        tbl2.scale(1.0, 1.9)
        for (ri, ci), cell in tbl2.get_celld().items():
            if ri == 0:
                cell.set_facecolor("#2b6cb0")
                cell.set_text_props(color="white", fontweight="bold")
            elif ri % 2 == 0:
                cell.set_facecolor("#f0f4fa")
            else:
                cell.set_facecolor("white")
            if ri > 0 and ci in (2, 8):   # Hessian σ and MC σ
                cell.set_facecolor("#e8f4f8")
            if ri > 0 and ci in (4, 5, 9):  # Profile CI and MC range
                cell.set_facecolor("#eaf4e8")

    ax.set_title(
        f"(e)  Uncertainty Summary  |  "
        f"Hessian σ ≈ LM covariance  |  "
        f"Profile 68% CI (Δχ²=1)  |  "
        f"Monte Carlo  N={um.get('n_ok','—')}",
        fontsize=11, fontweight="bold", pad=6)

    # ── main title ────────────────────────────────────────────────────────────
    f1s  = f"  f₁={res['f1']:.4f}" if res.get("f1") is not None else ""
    cll  = f"{chi2_l:.3f}" if res.get("chi2_lpm") else "n/a"
    fig.suptitle(
        f"PINN Edwards Aquifer — {sid}  ({res['age_cat']})  |  {lpm}"
        f"  |  τ₁={tau1_v:.2f} yr{f1s}"
        f"  |  χ²={res['chi2']:.3f}  (LPM {cll})   Prob={res['chi2_prob']:.3f}",
        fontsize=11, fontweight="bold", color="#333333", y=0.975
    )

    out_path = os.path.join(out_dir, f"{sid.replace('/','_')}_pinn.png")
    plt.savefig(out_path, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close()
    return out_path


# ════════════════════════════════════════════════════════════
# SUMMARY FIGURE
# ════════════════════════════════════════════════════════════
def plot_summary(all_results, out_dir):
    n = len(all_results)
    fig, axes = plt.subplots(1, 2, figsize=(22, max(10, n*0.44+2.5)))
    fig.patch.set_facecolor("white")

    panels = [
        (axes[0],"tau1","τ₁  Mean Age  (yr)","Mean age τ₁  (yr)",False),
        (axes[1],"chi2","χ²  Goodness of Fit","χ²",True),
    ]
    for ax, metric, title, xlabel, log_x in panels:
        sids   = [r["sid"] for r in all_results]
        pinn_v = [r[metric] or 0. for r in all_results]
        lpm_k  = "tau1_lpm" if metric=="tau1" else "chi2_lpm"
        lpm_v  = [float(r[lpm_k]) if r.get(lpm_k) else np.nan for r in all_results]
        is_bmm = [r["lpm"]=="BMM-DM-DM" for r in all_results]
        pc     = [TEAL if b else BLUE for b in is_bmm]
        lc     = [RED  if b else ORG  for b in is_bmm]

        y=np.arange(n)
        ax.barh(y-0.20, pinn_v, 0.38, color=pc, alpha=0.87, label="PINN")
        vld=[(i,v) for i,v in enumerate(lpm_v) if not np.isnan(v)]
        if vld:
            yi,vi=zip(*vld)
            ax.barh([yy+0.20 for yy in yi],vi,0.38,
                    color=[lc[i] for i in yi],alpha=0.52,hatch="//",label="TracerLPM")

        # Error bars from Hessian uncertainty
        if metric == "tau1":
            for i,r in enumerate(all_results):
                uh = r.get("unc_hess",{})
                pn = uh.get("param_names",[])
                sg = uh.get("sigmas_hess",[])
                if "tau1" in pn:
                    idx = pn.index("tau1")
                    if idx < len(sg) and sg[idx] is not None:
                        sigma = float(sg[idx]) if sg[idx] is not None else 0.
                        sigma = abs(sigma)  # guard against negative/NaN
                        lo_err = max(min(sigma, pinn_v[i] * 0.99), 0.)
                        hi_err = max(sigma, 0.)
                        if lo_err > 0 or hi_err > 0:
                            ax.errorbar(pinn_v[i], y[i]-0.20,
                                        xerr=[[lo_err],[hi_err]],
                                        fmt="none", ecolor="black",
                                        capsize=3, lw=1.2, zorder=5)
                # MC error bars (P16-P84)
                mc = r.get("unc_mc",{}).get("tau1",{})
                if mc.get("p16") is not None and mc.get("p84") is not None:
                    mc_lo = max(pinn_v[i] - float(mc["p16"]), 0.)
                    mc_hi = max(float(mc["p84"]) - pinn_v[i], 0.)
                    if mc_lo > 0 or mc_hi > 0:
                        ax.errorbar(pinn_v[i], y[i]-0.20,
                                    xerr=[[mc_lo],[mc_hi]],
                                    fmt="none", ecolor=TEAL, capsize=2, lw=0.8,
                                    alpha=0.6, zorder=4)

        ax.set_yticks(y); ax.set_yticklabels(sids, fontsize=6.8)
        ax.set_xlabel(xlabel, fontsize=10); ax.set_title(title, fontsize=12, fontweight="bold")
        ax.legend(fontsize=9); ax.grid(axis="x", ls="--", alpha=0.3)
        if log_x: ax.set_xscale("log")
        ax.spines[["top","right"]].set_visible(False)

    legend_el=[
        Patch(color=TEAL,label="PINN — BMM-DM-DM"),
        Patch(color=BLUE,label="PINN — DM"),
        Patch(color=RED, alpha=0.52,hatch="//",label="TracerLPM — BMM-DM-DM"),
        Patch(color=ORG, alpha=0.52,hatch="//",label="TracerLPM — DM"),
    ]
    fig.legend(handles=legend_el,loc="upper center",ncol=4,fontsize=9,
               framealpha=0.9,bbox_to_anchor=(0.5,1.01))
    fig.suptitle("Edwards Aquifer PINN — All Samples Summary (error bars = Hessian σ + MC P16–P84)",
                 fontsize=12,fontweight="bold",y=1.05)
    plt.tight_layout()
    out = os.path.join(out_dir, "00_summary_all_samples.png")
    plt.savefig(out, dpi=150, bbox_inches="tight", facecolor="white")
    plt.close(); return out


# ════════════════════════════════════════════════════════════
# SUMMARY CSV  (extended with all uncertainty columns)
# ════════════════════════════════════════════════════════════

def _count_free_params_from_res(r):
    """Infer n_free_params from dof and n_active."""
    act = r.get("active", [])
    n_act = sum(act) if act else 0
    return max(n_act - r.get("dof", 1), 0)

def write_tracer_csv(all_results, path):
    """
    One row per (SampleID, Tracer) — all samples grouped in a single table.
    Columns: SampleID, AgeCat, LPM, Tracer, Active, Obs, Meas_Err,
             PINN_sim, PINN_rel_err_pct, LPM_sim, LPM_rel_err_pct
    """
    cols = ["SampleID", "AgeCat", "LPM",
            "Tracer", "Active_in_loss",
            "Obs", "Meas_Err",
            "PINN_sim", "PINN_rel_err_pct",
            "LPM_sim",  "LPM_rel_err_pct"]

    def ff(v, nd=5):
        if v is None or (isinstance(v, float) and np.isnan(v)): return ""
        fv = float(v)
        if abs(fv) < 1e-4 or abs(fv) > 1e5: return f"{fv:.3e}"
        return f"{fv:.{nd}g}"

    def pct(s, o):
        try: return f"{abs(float(s)-float(o))/abs(float(o))*100:.2f}"
        except: return ""

    with open(path, "w", newline="") as f:
        w = _csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_results:
            tns   = r["tracer_names"]
            obs_v = r["obs_vals"]
            obs_e = r["obs_errs"]
            sim_v = r["sim_vals"]
            sim_l = r.get("sim_lpm") or []
            act   = r.get("active", [True]*len(tns))
            for i, nm in enumerate(tns):
                sl = float(sim_l[i]) if sim_l and i<len(sim_l) and sim_l[i] is not None else None
                w.writerow({
                    "SampleID"        : r["sid"],
                    "AgeCat"          : r["age_cat"],
                    "LPM"             : r["lpm"],
                    "Tracer"          : nm,
                    "Active_in_loss"  : "Yes" if act[i] else "No",
                    "Obs"             : ff(obs_v[i]),
                    "Meas_Err"        : ff(obs_e[i]),
                    "PINN_sim"        : ff(sim_v[i]),
                    "PINN_rel_err_pct": pct(sim_v[i], obs_v[i]),
                    "LPM_sim"         : ff(sl),
                    "LPM_rel_err_pct" : pct(sl, obs_v[i]) if sl is not None else "",
                })


def write_summary_csv(all_results, path):
    cols = [
        "SampleID","Date","AgeCat","LPM","n_active","n_free","dof",
        # Best-fit parameters
        "tau1_PINN","pd1_PINN","f1_PINN","tau2_PINN",
        "tau1_LPM","f1_LPM","tau2_LPM",
        # Fit quality
        "chi2_PINN","chi2_LPM","chi2_delta","chi2_prob_PINN","chi2_prob_LPM","PINN_better",
        # Hessian uncertainties
        "tau1_sigma_hess","f1_sigma_hess","pd1_sigma_hess",
        "tau1_sigma_LPM","f1_sigma_LPM",
        # Profile CI
        "tau1_profile_lo","tau1_profile_hi","tau1_profile_sym",
        "f1_profile_lo","f1_profile_hi","f1_profile_sym",
        # Monte Carlo
        "tau1_MC_mean","tau1_MC_std","tau1_MC_p16","tau1_MC_p84",
        "f1_MC_mean","f1_MC_std","f1_MC_p16","f1_MC_p84",
        "MC_n_ok","MC_n_fail",
        # Tracer info
        "Tracer_names","Obs_vals","PINN_sim","LPM_sim",
        "RelErr_PINN","RelErr_LPM",
    ]

    def ff(v, nd=4):
        if v is None or (isinstance(v,float) and np.isnan(v)): return ""
        return f"{float(v):.{nd}f}"

    def rerr(sims, obs):
        return "; ".join(
            f"{abs(float(s)-float(o))/abs(float(o))*100:.1f}%" if o else "—"
            for s,o in zip(sims,obs))

    with open(path,"w",newline="") as f:
        w=_csv.DictWriter(f,fieldnames=cols); w.writeheader()
        for r in all_results:
            uh=r.get("unc_hess",{});  up=r.get("unc_profile",{});  um=r.get("unc_mc",{})
            pns=uh.get("param_names",[]); sgs=uh.get("sigmas_hess",[])
            def _sg(nm):
                return ff(sgs[pns.index(nm)]) if nm in pns and pns.index(nm)<len(sgs) else ""
            def _pci(nm,j):
                v=up.get(nm); return ff(v[j]) if isinstance(v,tuple) else ""
            def _mcs(nm,k):
                v=um.get(nm,{}); return ff(v.get(k)) if isinstance(v,dict) else ""

            chi2_l=r.get("chi2_lpm")
            prob_l=ff(sp_stats.chi2.sf(chi2_l,r["dof"])) if chi2_l else ""
            dchi2=f"{r['chi2']-chi2_l:+.4f}" if chi2_l else ""
            better="Yes" if chi2_l and r["chi2"]<=chi2_l else ""

            prof_sym_tau=""; prof_sym_f=""
            pci_t=up.get("tau1"); pci_f=up.get("f1")
            if isinstance(pci_t,tuple): prof_sym_tau=ff((pci_t[1]-pci_t[0])/2)
            if isinstance(pci_f,tuple): prof_sym_f  =ff((pci_f[1]-pci_f[0])/2)

            obs=r["obs_vals"]; sv=r["sim_vals"]; sl=r.get("sim_lpm") or []
            w.writerow({
                "SampleID"       :r["sid"],
                "Date"           :ff(r["sample_date"],3),
                "AgeCat"         :r["age_cat"],
                "LPM"            :r["lpm"],
                "n_active"       :sum(r.get("active",[True]*len(r["tracer_names"]))),
                "n_free"         :_count_free_params_from_res(r),
                "dof"            :r["dof"],
                "tau1_PINN"      :ff(r["tau1"]),
                "pd1_PINN"       :ff(r["pd1"]),
                "f1_PINN"        :ff(r.get("f1")),
                "tau2_PINN"      :ff(r.get("tau2")),
                "tau1_LPM"       :ff(r.get("tau1_lpm")),
                "f1_LPM"         :ff(r.get("f1_lpm")),
                "tau2_LPM"       :ff(r.get("tau2_lpm")),
                "chi2_PINN"      :ff(r["chi2"]),
                "chi2_LPM"       :ff(chi2_l),
                "chi2_delta"     :dchi2,
                "chi2_prob_PINN" :ff(r["chi2_prob"]),
                "chi2_prob_LPM"  :prob_l,
                "PINN_better"    :better,
                "tau1_sigma_hess":_sg("tau1"),
                "f1_sigma_hess"  :_sg("f1"),
                "pd1_sigma_hess" :_sg("pd1"),
                "tau1_sigma_LPM" :ff(r.get("tau1_err_lpm")),
                "f1_sigma_LPM"   :ff(r.get("f1_err_lpm")),
                "tau1_profile_lo":_pci("tau1",0),
                "tau1_profile_hi":_pci("tau1",1),
                "tau1_profile_sym":prof_sym_tau,
                "f1_profile_lo"  :_pci("f1",0),
                "f1_profile_hi"  :_pci("f1",1),
                "f1_profile_sym" :prof_sym_f,
                "tau1_MC_mean"   :_mcs("tau1","mean"),
                "tau1_MC_std"    :_mcs("tau1","std"),
                "tau1_MC_p16"    :_mcs("tau1","p16"),
                "tau1_MC_p84"    :_mcs("tau1","p84"),
                "f1_MC_mean"     :_mcs("f1","mean"),
                "f1_MC_std"      :_mcs("f1","std"),
                "f1_MC_p16"      :_mcs("f1","p16"),
                "f1_MC_p84"      :_mcs("f1","p84"),
                "MC_n_ok"        :str(um.get("n_ok","")),
                "MC_n_fail"      :str(um.get("n_fail","")),
                "Tracer_names"   :"; ".join(r["tracer_names"]),
                "Obs_vals"       :"; ".join(ff(v,5) for v in obs),
                "PINN_sim"       :"; ".join(ff(v,5) for v in sv),
                "LPM_sim"        :"; ".join(ff(v,5) for v in sl),
                "RelErr_PINN"    :rerr(sv,obs),
                "RelErr_LPM"     :rerr(sl,obs) if sl else "",
            })


# ════════════════════════════════════════════════════════════
# MAIN
# ════════════════════════════════════════════════════════════
def main():
    parser = argparse.ArgumentParser(
        description="Edwards Aquifer PINN — age dating + uncertainty analysis")
    parser.add_argument("--ids",       nargs="*", default=None)
    parser.add_argument("--starts",    type=int,  default=3)
    parser.add_argument("--adam",      type=int,  default=5000)
    parser.add_argument("--mc",        type=int,  default=100,
                        help="Monte Carlo realisations (default 100)")
    parser.add_argument("--no-profile",action="store_true",
                        help="Skip profile CI (faster)")
    parser.add_argument("--no-mc",    action="store_true",
                        help="Skip Monte Carlo (faster)")
    parser.add_argument("--verbose",  action="store_true")
    parser.add_argument("--config",   default=_CONFIG_DEFAULT)
    parser.add_argument("--outdir",   default=_OUT_DIR)
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    samples = load_config(args.config)
    if args.ids:
        samples = [s for s in samples if s["sample_id"] in args.ids]
        if not samples:
            sys.exit(f"No samples matched: {args.ids}")

    do_profile = not args.no_profile
    do_mc      = not args.no_mc

    print(f"\n{'='*72}")
    print(f"  Edwards Aquifer PINN  —  {len(samples)} sample(s)")
    print(f"  Adam: {args.adam}  Starts: {args.starts}  "
          f"MC: {args.mc if do_mc else 'OFF'}  "
          f"Profile: {'ON' if do_profile else 'OFF'}")
    print(f"  Output: {args.outdir}")
    print(f"{'='*72}\n")

    all_results = []; n_ok=0; n_err=0

    for i,s in enumerate(samples):
        tag=(f"[{i+1:2d}/{len(samples)}]  {s['sample_id']:<35}"
             f"  ({s['lpm']['name']}, {s['age_category']})")
        print(tag, end="  ", flush=True)

        try:
            res = fit_sample(s, n_adam=args.adam, n_starts=args.starts,
                              n_mc=args.mc, do_profile=do_profile,
                              do_mc=do_mc, verbose=args.verbose)
            plot_sample(res, args.outdir)
            all_results.append(res); n_ok+=1

            # Compact summary line
            uh=res.get("unc_hess",{}); pns=uh.get("param_names",[])
            sgs=uh.get("sigmas_hess",[])
            sig_tau=f"±{sgs[pns.index('tau1')]:.2f}" if "tau1" in pns else ""
            mc_t = res.get("unc_mc",{}).get("tau1",{})
            mc_str = f" MC±{mc_t['std']:.2f}" if mc_t.get("std") else ""
            f1s = f"  f₁={res['f1']:.4f}" if res.get("f1") is not None else ""
            tick = " ✓" if res["chi2"]<=(res.get("chi2_lpm") or 1e9) else ""
            print(f"τ₁={res['tau1']:.2f}{sig_tau} yr{f1s}  "
                  f"χ²={res['chi2']:.4f}  prob={res['chi2_prob']:.3f}"
                  f"{mc_str}{tick}")

        except Exception as exc:
            n_err+=1
            print(f"ERROR — {exc}")
            if args.verbose: _tb.print_exc()

    print(f"\n{'─'*72}")
    print(f"  {n_ok}/{len(samples)} OK  ({n_err} error(s))")
    print(f"{'─'*72}\n")

    if not all_results: return
    print("Writing summary outputs ...")
    plot_summary(all_results, args.outdir)
    csv_path = os.path.join(args.outdir, "00_summary_table.csv")
    write_summary_csv(all_results, csv_path)
    tracer_csv  = os.path.join(args.outdir, "00_tracer_detail.csv")
    write_tracer_csv(all_results, tracer_csv)
    print(f"  Tracer  → {tracer_csv}")
    print(f"  Summary → {args.outdir}/00_summary_all_samples.png")
    print(f"  CSV     → {csv_path}")
    chi2_p=[r["chi2"] for r in all_results if r["chi2"] is not None]
    chi2_l=[r["chi2_lpm"] for r in all_results if r.get("chi2_lpm")]
    n_b=sum(1 for r in all_results if r["chi2"] is not None
            and r.get("chi2_lpm") and r["chi2"]<=r["chi2_lpm"])
    print(f"\n  Median χ² PINN={np.median(chi2_p):.4f}"
          + (f"  LPM={np.median(chi2_l):.4f}" if chi2_l else ""))
    print(f"  PINN ≤ TracerLPM χ² in {n_b}/{len(all_results)} samples")
    print("Done.")


if __name__=="__main__":
    main()
