"""
dlpmi/inverter.py
=================
High-level API: Sample dataclass and Inverter class.

Typical usage
-------------
    from dlpmi import Inverter, Sample

    # ── single-component DM ──────────────────────────────────
    s = Sample(
        sample_id   = "PSW-1_Modesto",
        sample_date = 2004.6,
        tracers     = [
            {"name": "3H",        "obs": 4.57,  "scale": 1.0},
            {"name": "3He(trit)", "obs": 32.70, "scale": 1.0},
            {"name": "SF6",       "obs": 0.75,  "scale": 1.0},
        ],
        model  = "DM",
        bounds = {"tau_lo": 50., "tau_hi": 100.,
                  "pd_lo":  0.01, "pd_hi": 1.0},
    )
    result = Inverter().fit(s)

    # ── BMM-DM-DM ───────────────────────────────────────────
    s2 = Sample(
        sample_id   = "SSW_Albuquerque",
        sample_date = 2007.43,
        tracers     = [
            {"name": "3H",  "obs": 2.35,  "scale": 1.0},
            {"name": "14C", "obs": 43.98, "scale": 0.85},
        ],
        model  = "BMM",
        model1 = "DM", model2 = "DM",
        free_params = "Mean Age, Fraction",
        bounds = {
            "tau1_lo": 15., "tau1_hi": 30.,
            "pd1":     0.042,              # fixed
            "tau2":    12100.,             # fixed
            "pd2":     0.1,                # fixed
            "f1_lo":   0.01, "f1_hi": 0.30,
        },
        uz_tt = 15.,
    )
    result2 = Inverter().fit(s2)
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .forward    import forward_single, forward_BMM
from .params     import DLPMIModel, BMMModel
from .kernels    import AGES_YOUNG, AGES_OLD, choose_ages


# ════════════════════════════════════════════════════════════════
# SAMPLE DATACLASS
# ════════════════════════════════════════════════════════════════

@dataclass
class Sample:
    """
    Container for one groundwater sample.

    Attributes
    ----------
    sample_id   : identifier string
    sample_date : decimal year of sampling  (e.g. 2004.6)
    tracers     : list of dicts, each with keys:
                    "name"    : tracer name string
                    "obs"     : observed concentration (float)
                    "scale"   : input scale factor (float, default 1.0)
                    "obs_err" : measurement error (float, optional;
                                defaults to 10% of obs)
                    "active"  : include in chi2 loss? (bool, default True)
    model       : "DM" | "EMM" | "EPM" | "PFM" | "PEM" | "BMM"
    model1      : first component model name  [BMM only]
    model2      : second component model name [BMM only]
    free_params : TracerLPM optimisation mode [BMM only]
                  "Mean Age, Fraction" | "Fraction" |
                  "Fraction, 2nd Mean Age"
    bounds      : dict of parameter bounds and fixed values
                  Single models:
                    tau_lo, tau_hi  : τ bounds (yr)
                    pd_lo, pd_hi    : P_D bounds  [DM only]
                    eta_lo, eta_hi  : η bounds    [EPM/PEM only]
                  BMM:
                    tau1_lo, tau1_hi, f1_lo, f1_hi (free params)
                    tau2_lo, tau2_hi (if τ₂ is free)
                    pd1, pd2, tau2  (fixed values for non-free params)
    he4_rate    : ⁴He accumulation rate β (cc STP g⁻¹ yr⁻¹)
    dic_c1      : DIC of young component (mmol/L)  [BMM ¹⁴C only]
    dic_c2      : DIC of old  component (mmol/L)   [BMM ¹⁴C only]
    uz_tt       : unsaturated-zone travel time (yr) [BMM only]
    dgmeta      : dict of SF₆ DGMETA correction params
                  T_recharge_C, excess_air_cckg,
                  fractionation_F, elevation_m
    age_category : "Modern" | "Mixed" | "PreModern" (optional label)
    """
    sample_id   : str
    sample_date : float
    tracers     : list
    model       : str = "DM"
    model1      : Optional[str] = None
    model2      : Optional[str] = None
    free_params : str = "Mean Age, Fraction"
    bounds      : dict = field(default_factory=dict)
    he4_rate    : float = 1e-12
    dic_c1      : float = 100.
    dic_c2      : float = 100.
    uz_tt       : float = 0.
    dgmeta      : dict = field(default_factory=dict)
    age_category: str = "Unknown"

    # ── derived properties ────────────────────────────────────
    @property
    def tracer_names(self) -> list:
        return [t["name"] for t in self.tracers]

    @property
    def obs_vals(self) -> list:
        return [float(t["obs"]) for t in self.tracers]

    @property
    def obs_errs(self) -> list:
        return [float(t.get("obs_err", abs(float(t["obs"])) * 0.10))
                for t in self.tracers]

    @property
    def scales(self) -> list:
        return [float(t.get("scale", 1.0)) for t in self.tracers]

    @property
    def active(self) -> list:
        return [bool(t.get("active", True)) for t in self.tracers]

    @property
    def active_names(self) -> list:
        return [n for n, a in zip(self.tracer_names, self.active) if a]

    @property
    def active_obs(self) -> list:
        return [o for o, a in zip(self.obs_vals, self.active) if a]

    @property
    def active_scales(self) -> list:
        return [s for s, a in zip(self.scales, self.active) if a]


# ── chi2 loss ─────────────────────────────────────────────────

def _chi2_loss(sims: list, obs_vals: list) -> torch.Tensor:
    """Sum of squared relative errors (TracerLPM Relative Error mode)."""
    loss = torch.tensor(0.)
    for s, o in zip(sims, obs_vals):
        loss = loss + ((s - float(o)) / (abs(float(o)) + 1e-30)) ** 2
    return loss


# ── two-phase optimiser ───────────────────────────────────────

def _train(model: nn.Module,
           loss_fn,
           n_adam: int = 5000,
           lr_adam: float = 3e-3,
           lr_lbfgs: float = 0.02) -> list:
    """Adam warm-start + L-BFGS refinement. Returns loss history."""
    opt   = optim.Adam(model.parameters(), lr=lr_adam)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_adam, eta_min=1e-5)
    hist  = []
    for _ in range(n_adam):
        opt.zero_grad(); L = loss_fn(); L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step(); sched.step(); hist.append(L.item())

    opt2 = optim.LBFGS(model.parameters(), lr=lr_lbfgs,
                        max_iter=300, history_size=20,
                        line_search_fn="strong_wolfe")
    def closure():
        opt2.zero_grad(); Lv = loss_fn(); Lv.backward()
        hist.append(Lv.item()); return Lv
    opt2.step(closure)
    return hist


# ════════════════════════════════════════════════════════════════
# INVERTER CLASS
# ════════════════════════════════════════════════════════════════

class Inverter:
    """
    DLPMI parameter estimator.

    Fits a single-component or BMM LPM to observed tracer concentrations
    using a two-phase (Adam + L-BFGS) optimiser with multi-start.

    Parameters
    ----------
    n_adam   : Adam epochs per start (default 5000)
    n_starts : number of multi-start seeds (default 3)
    lr_adam  : Adam learning rate (default 3e-3)
    verbose  : print per-start convergence details (default False)

    Methods
    -------
    fit(sample)  → dict   fit one Sample, return results dict
    """

    def __init__(self,
                 n_adam:   int   = 5000,
                 n_starts: int   = 3,
                 lr_adam:  float = 3e-3,
                 verbose:  bool  = False):
        self.n_adam   = n_adam
        self.n_starts = n_starts
        self.lr_adam  = lr_adam
        self.verbose  = verbose

    # ── seed generation ───────────────────────────────────────

    @staticmethod
    def _seeds(tau0: float, lo: float, hi: float, n: int) -> list:
        return [float(np.clip(v, lo * 1.02, hi * 0.98))
                for v in [tau0, tau0 * 0.5, tau0 * 2.0][:n]]

    # ── single-component fit ──────────────────────────────────

    def _fit_single(self, s: Sample) -> dict:
        b   = s.bounds
        mn  = s.model

        tau0   = b.get("tau0",  b.get("tau_lo", 1.) * 3.)
        tau_lo = b.get("tau_lo", 0.5)
        tau_hi = b.get("tau_hi", 1e5)
        pd0    = b.get("pd0",   b.get("pd_lo",  0.001) * 5.)
        pd_lo  = b.get("pd_lo",  0.001)
        pd_hi  = b.get("pd_hi",  2.0)
        eta0   = b.get("eta0",   0.7)
        eta_lo = b.get("eta_lo", 0.05)
        eta_hi = b.get("eta_hi", 1.0)

        tau_seeds = self._seeds(tau0, tau_lo, tau_hi, self.n_starts)

        best_loss, best_model, best_hist = float("inf"), None, []

        for t0 in tau_seeds:
            m = DLPMIModel(mn, tau0=t0, tau_lo=tau_lo, tau_hi=tau_hi,
                           PD0=pd0, pd_lo=pd_lo, pd_hi=pd_hi,
                           eta0=eta0, eta_lo=eta_lo, eta_hi=eta_hi)

            def lfn(m=m):
                pp   = m.get_params()
                ages = choose_ages(float(pp["tau"].detach()))
                sims = forward_single(
                    mn, pp, s.sample_date,
                    s.active_names, s.active_scales,
                    s.he4_rate, ages, s.dgmeta)
                return _chi2_loss(sims, s.active_obs)

            h  = _train(m, lfn, self.n_adam, self.lr_adam)
            fl = lfn().item()
            if self.verbose:
                pp = m.get_params()
                print(f"      seed τ={t0:.1f} → τ={pp['tau'].item():.3f} "
                      f"loss={fl:.5f}")
            if fl < best_loss:
                best_loss = fl; best_model = m; best_hist = h

        # final simulation over ALL tracers (including inactive)
        pp     = best_model.get_params()
        ages_f = choose_ages(float(pp["tau"].detach()))
        sims_f = forward_single(mn, pp, s.sample_date,
                                 s.tracer_names, s.scales,
                                 s.he4_rate, ages_f, s.dgmeta)

        return dict(
            sample_id    = s.sample_id,
            sample_date  = s.sample_date,
            age_category = s.age_category,
            model        = mn,
            model1       = None, model2 = None,
            params       = {k: v.item() for k, v in pp.items()},
            f1           = None,
            chi2         = best_loss,
            tracer_names = s.tracer_names,
            obs_vals     = s.obs_vals,
            obs_errs     = s.obs_errs,
            active       = s.active,
            sim_vals     = [v.item() for v in sims_f],
            hist         = best_hist,
            _model_obj   = best_model,
            _sample      = s,
        )

    # ── BMM fit ───────────────────────────────────────────────

    def _fit_bmm(self, s: Sample) -> dict:
        b      = s.bounds
        m1, m2 = s.model1, s.model2
        fp     = s.free_params or "Mean Age, Fraction"
        fp_low = fp.lower()

        tau1_0  = b.get("tau1_0",  b.get("tau1_lo", 1.) * 3.)
        tau1_lo = b.get("tau1_lo", 0.5)
        tau1_hi = b.get("tau1_hi", 500.)
        f1_0    = b.get("f1_0",   0.5)
        f1_lo   = b.get("f1_lo",  0.005)
        f1_hi   = b.get("f1_hi",  0.997)
        tau2_0  = b.get("tau2",   b.get("tau2_0", 1000.))
        tau2_lo = b.get("tau2_lo", 1.)
        tau2_hi = b.get("tau2_hi", None)

        # Fixed shape params for component 1
        p1_fixed: dict = {}
        if m1 == "DM":
            p1_fixed["PD"] = b.get("pd1", 0.1)
        elif m1 in ("EPM", "PEM"):
            p1_fixed["eta"] = b.get("eta1", 0.7)

        # Fixed shape params for component 2
        p2_fixed: dict = {}
        if m2 == "DM":
            p2_fixed["PD"] = b.get("pd2", 0.1)
        elif m2 in ("EPM", "PEM"):
            p2_fixed["eta"] = b.get("eta2", 0.7)

        params1_init = {"tau": tau1_0, **p1_fixed}
        params2_init = {"tau": tau2_0, **p2_fixed}

        # Seeds
        tau_seeds = self._seeds(tau1_0, tau1_lo, tau1_hi, self.n_starts)
        f_seeds   = [np.clip(f1_0,          f1_lo*1.02, f1_hi*0.98),
                     np.clip(f1_0 * 0.7,    f1_lo*1.02, f1_hi*0.98),
                     np.clip(f1_0 * 1.3,    f1_lo*1.02, f1_hi*0.98)][:self.n_starts]

        best_loss, best_model, best_hist = float("inf"), None, []

        for t0, f0 in zip(tau_seeds, f_seeds):
            p1i = {"tau": t0,    **p1_fixed}
            m   = BMMModel(m1, m2, free_params=fp,
                           params1_init=p1i,
                           params2_init=params2_init,
                           f1_0=float(f0),
                           tau1_lo=tau1_lo, tau1_hi=tau1_hi,
                           f1_lo=f1_lo,    f1_hi=f1_hi,
                           tau2_lo=tau2_lo, tau2_hi=tau2_hi)

            def lfn(m=m):
                pp1, f1, pp2 = m.get_params()
                sims = forward_BMM(
                    m1, pp1, m2, pp2, f1,
                    s.sample_date, s.active_names, s.active_scales,
                    s.he4_rate, s.dic_c1, s.dic_c2, s.uz_tt, s.dgmeta)
                return _chi2_loss(sims, s.active_obs)

            h  = _train(m, lfn, self.n_adam, self.lr_adam)
            fl = lfn().item()
            if self.verbose:
                pp1, f1_, pp2 = m.get_params()
                print(f"      seed τ₁={t0:.1f} f={f0:.3f} → "
                      f"τ₁={pp1['tau'].item():.3f} f₁={f1_.item():.4f} "
                      f"loss={fl:.5f}")
            if fl < best_loss:
                best_loss = fl; best_model = m; best_hist = h

        pp1, f1, pp2 = best_model.get_params()
        sims_f = forward_BMM(
            m1, pp1, m2, pp2, f1,
            s.sample_date, s.tracer_names, s.scales,
            s.he4_rate, s.dic_c1, s.dic_c2, s.uz_tt, s.dgmeta)

        return dict(
            sample_id    = s.sample_id,
            sample_date  = s.sample_date,
            age_category = s.age_category,
            model        = "BMM",
            model1       = m1,
            model2       = m2,
            params       = {k: v.item() for k, v in pp1.items()},
            params2      = {k: v.item() for k, v in pp2.items()},
            f1           = f1.item(),
            chi2         = best_loss,
            tracer_names = s.tracer_names,
            obs_vals     = s.obs_vals,
            obs_errs     = s.obs_errs,
            active       = s.active,
            sim_vals     = [v.item() for v in sims_f],
            hist         = best_hist,
            _model_obj   = best_model,
            _sample      = s,
        )

    # ── public fit method ─────────────────────────────────────

    def fit(self, sample: Sample) -> dict:
        """
        Fit a single Sample.  Returns a results dict.

        The dict contains:
          sample_id, sample_date, age_category, model, model1, model2
          params      : {tau, PD, eta, ...} for component 1 (single or BMM)
          params2     : {tau, PD, ...} for component 2   [BMM only]
          f1          : mixing fraction                   [BMM only]
          chi2        : final chi-square value
          tracer_names, obs_vals, obs_errs, active, sim_vals
          hist        : loss history
        """
        if sample.model.upper() == "BMM":
            if sample.model1 is None or sample.model2 is None:
                raise ValueError(
                    "BMM model requires 'model1' and 'model2' to be set.")
            return self._fit_bmm(sample)
        else:
            return self._fit_single(sample)
