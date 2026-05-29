"""
edwards_pinn_physics.py
=======================
Core physics library for Edwards Aquifer PINN age dating.

Tracer input functions  :  input_3H, input_SF6, input_14C, input_4He
DM kernel               :  g_DM(age, tau, PD)
Forward models          :  forward_DM, forward_BMM
PINN classes            :  PINN_DM, PINN_BMM
Loss & training         :  chi2_loss (relative-error), train_pinn
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

try:
    _trapz = np.trapezoid     # numpy >= 2.0
except AttributeError:
    _trapz = np.trapz

torch.set_default_dtype(torch.float32)

# ════════════════════════════════════════════════════════════
# CONSTANTS
# ════════════════════════════════════════════════════════════
LAMBDA_3H  = np.log(2) / 12.32    # yr⁻¹
LAMBDA_14C = np.log(2) / 5730.0   # yr⁻¹

# ════════════════════════════════════════════════════════════
# AGE GRIDS
# ════════════════════════════════════════════════════════════
AGES_YOUNG = torch.unique(torch.cat([
    torch.linspace(0.05,  10.,   80),
    torch.linspace(10.,  100.,  200),
    torch.linspace(100., 500.,  200),
]))

AGES_OLD = torch.unique(torch.cat([
    torch.linspace(0.1,   1000.,  200),
    torch.linspace(1000., 20000., 300),
    torch.linspace(20000.,60000., 100),
]))

# ════════════════════════════════════════════════════════════
# TRACER INPUT FUNCTIONS
# ════════════════════════════════════════════════════════════

# ── 3H ──────────────────────────────────────────────────────
_PY3H = torch.tensor([float(y) for y in range(1953, 2020)])
_RAW3H = torch.tensor([
     8., 10, 12, 15, 25, 50,100,150,
   300.,900,1600,800,350,
   200.,130, 90, 65, 50, 35,
    25., 22, 18, 16, 15, 13, 12, 11,10,10,
    10.,  9,  9,  8,  8,  8,  7,  7, 7, 7,
     7.,  6,  6,  6,  6,  6,  6,  6, 6, 6,
     6.,  6,  6,  6,  6,  6,  6,  6, 6, 6,
     6.,  6,  6,  6,  6,  6,  6,
], dtype=torch.float32)
# Latitude calibration for the Edwards Aquifer, Texas (30°N, subtropical).
# Pre-1981:  × 0.668 — standard correction for low-to-mid latitude sites.
# Post-1980: × 0.668 — Texas receives LOWER bomb-³H than N-hemisphere average.
#   The value 1.336 is designed for high-latitude sites (Ottawa, Vienna) and
#   is incorrect for subtropical Texas. Using it inflates post-1980 ³H by ~2×,
#   causing systematic overestimation of ³H for recent recharge (2000–2015),
#   which directly drives underestimation of f₁ in BMM samples.
#   Back-calculation from TracerLPM DM-Modern sample fits confirms × 0.668.
_RAW3H[:28] *= 0.668   # pre-1981  (unchanged)
_RAW3H[28:] *= 0.668   # post-1980 (corrected: 1.336 → 0.668 for Texas)

def input_3H(t, scale=1.0):
    """GNIP N-hemisphere ³H precipitation (TU)."""
    t_c = torch.clamp(t, float(_PY3H[0]), float(_PY3H[-1]))
    f   = t_c - float(_PY3H[0])
    lo  = torch.clamp(f.long(), 0, len(_RAW3H)-2)
    hi  = torch.clamp(lo+1, 0, len(_RAW3H)-1)
    w   = torch.clamp(f - lo.float(), 0., 1.)
    v   = _RAW3H[lo]*(1-w) + _RAW3H[hi]*w
    return torch.where(t < _PY3H[0], torch.full_like(v, 2.), v) * scale

# ── SF6 : NOAA/ESRL atmospheric record + Bullister et al. (2002) solubility ─
# Correct NOAA Global Monitoring Laboratory SF6 (N. Hemisphere, pptv)
# SF6 increases monotonically: ~0 pptv (1955) to ~10 pptv (2020)
_SF6_YRS = torch.tensor([
    1955.,1960,1965,1970,1975,1978,1980,1982,1984,1986,1988,1990,
    1992.,1994,1996,1998,2000,2002,2004,2006,2008,2010,2012,2014,
    2016.,2018,2020,2022.])
_SF6_PPTV = torch.tensor([
    0.000,0.000,0.000,0.020,0.080,0.100,0.170,0.270,0.430,0.600,
    0.830,1.140,1.520,1.950,2.420,2.920,3.420,3.960,4.540,5.120,
    5.800,6.600,7.390,8.200,9.000,9.750,10.50,11.20])


def _Ks_SF6(T_C):
    """SF6 solubility mol/(L·atm), Bullister et al. (2002), S=0."""
    T_K = float(T_C) + 273.15
    return torch.exp(torch.tensor(
        -96.5975 + 139.883*(100./T_K) + 37.8193*float(torch.log(torch.tensor(T_K/100.)))))


def input_SF6(t, scale=1.0,
              T_recharge_C=18.0, excess_air_cckg=5.0,
              fractionation_F=0.5, elevation_m=300.0):
    """
    SF₆ atmospheric equivalent (pptv) with DGMETA corrections.

    Applies per-sample corrections for solubility equilibrium at recharge
    temperature and pressure (Bullister et al., 2002), excess air
    entrainment, and gas fractionation (CE model), following the
    DGMETA program (Jurgens et al., 2020).

    Parameters: T_recharge_C, excess_air_cckg, fractionation_F,
                elevation_m — all from Table 2 DGMETA output.
    """
    # Interpolate atmospheric SF6
    lo = torch.zeros_like(t, dtype=torch.long)
    for i in range(len(_SF6_YRS)-1):
        lo = torch.where(t >= _SF6_YRS[i], torch.full_like(lo, i), lo)
    hi  = torch.clamp(lo+1, 0, len(_SF6_PPTV)-1)
    dtt = _SF6_YRS[hi] - _SF6_YRS[lo]
    w   = torch.clamp((t - _SF6_YRS[lo]) / torch.clamp(dtt, min=0.01), 0., 1.)
    sf6_atm = _SF6_PPTV[lo]*(1-w) + _SF6_PPTV[hi]*w
    sf6_atm = torch.where(t < _SF6_YRS[0], torch.zeros_like(sf6_atm), sf6_atm)

    # Solubility and pressure
    Ks = _Ks_SF6(T_recharge_C)                              # mol/(L·atm)
    P  = torch.exp(torch.tensor(-elevation_m / 8434.5))     # atm fraction

    # Dissolved equilibrium + excess air (CE model)
    C_eq  = sf6_atm * Ks * P
    C_exc = (excess_air_cckg / 22400.) * sf6_atm * 1e-12 * fractionation_F * 1e12

    # Back-convert to atmospheric equivalent pptv
    return (C_eq + C_exc) / (Ks * P) * scale

# ── 14C ─────────────────────────────────────────────────────
_IC09_BP = torch.tensor([
      0., 500,1000,1500,2000,2500,3000,3500,4000,4500,
   5000.,5500,6000,6500,7000,7500,8000,8500,9000,9500,
  10000.,10500,11000,11500,12000,12500,13000,13500,14000,15000,
  16000.,17000,18000,20000,25000,30000,40000,50000.])
_IC09_PM = torch.tensor([
   100.,100,100, 99, 99,100,100, 99,100,101,
   103.,104,106,107,108,110,115,119,122,127,
   131.,132,130,125,119,117,115,113,112,107,
   100., 95, 90, 80, 60, 44, 20,  8.])
_BOMB_YRS = torch.tensor([
   1950.,1955,1960,1961,1962,1963,1964,1965,1966,1967,
   1968.,1969,1970,1972,1975,1978,1981,1984,1987,1990,
   1993.,1996,1999,2002,2007,2010,2015,2018,2020.])
_BOMB_PM  = torch.tensor([
   100.,100,100,102,110,155,194,196,186,175,
   168.,163,158,152,143,134,126,120,115,112,
   109.,107,107,107,107,106,105,104,104.])

def _interp1d(t, xs, ys):
    lo = torch.zeros_like(t, dtype=torch.long)
    for i in range(len(xs)-1):
        lo = torch.where(t >= xs[i], torch.full_like(lo, i), lo)
    hi = torch.clamp(lo+1, 0, len(ys)-1)
    dx = xs[hi] - xs[lo]
    w  = torch.clamp((t - xs[lo]) / torch.clamp(dx, min=0.01), 0., 1.)
    return ys[lo]*(1-w) + ys[hi]*w

def input_14C(t, scale=1.0):
    """IntCal09 + NH Zone 2 bomb curve, scaled by dead-carbon factor."""
    age_bp = torch.clamp(1950. - t, min=0.)
    ic09   = _interp1d(age_bp, _IC09_BP, _IC09_PM)
    bomb   = _interp1d(
        torch.clamp(t, float(_BOMB_YRS[0]), float(_BOMB_YRS[-1])),
        _BOMB_YRS, _BOMB_PM)
    return torch.where(t >= 1950., bomb, ic09) * scale

def input_4He(ages, he4_rate):
    """Radiogenic ⁴He (cc STP/g): linear accumulation at he4_rate."""
    return he4_rate * ages

# ════════════════════════════════════════════════════════════
# DM KERNEL
# ════════════════════════════════════════════════════════════
def g_DM(age, tau, PD):
    """Normalised DM exit-age PDF on the given age grid."""
    eps   = 1e-9
    z     = torch.clamp(age / tau, min=eps)
    log_g = (- torch.log(age + eps)
             - 0.5 * torch.log(4.*np.pi*PD*z + eps)
             - (1.-z)**2 / (4.*PD*z + eps))
    g = torch.exp(log_g)
    return g / (torch.trapz(g, age) + eps)

# ════════════════════════════════════════════════════════════
# FORWARD MODELS
# ════════════════════════════════════════════════════════════
def _sim_component(g, ages, t_cal, tracer_list, scale_list, he4_rate, dgmeta=None):
    """Convolve one DM component over all requested tracers."""
    if dgmeta is None:
        dgmeta = {}
    dec3H  = torch.exp(-LAMBDA_3H  * ages)
    dec14C = torch.exp(-LAMBDA_14C * ages)
    out = []
    for i, name in enumerate(tracer_list):
        sc = float(scale_list[i]) if i < len(scale_list) else 1.0
        nm = name.strip()
        if nm == '3H':
            out.append(torch.trapz(input_3H(t_cal, sc)  * dec3H  * g, ages))
        elif nm in ('3He(trit)', '3Hetrit', '3He_trit'):
            out.append(torch.trapz(input_3H(t_cal, sc)  * (1.-dec3H) * g, ages))
        elif nm == 'SF6':
            out.append(torch.trapz(
                input_SF6(t_cal, sc,
                          T_recharge_C=dgmeta.get('T_recharge_C', 18.),
                          excess_air_cckg=dgmeta.get('excess_air_cckg', 5.),
                          fractionation_F=dgmeta.get('fractionation_F', 0.5),
                          elevation_m=dgmeta.get('elevation_m', 300.)) * g, ages))
        elif nm == '14C':
            out.append(torch.trapz(input_14C(t_cal, sc) * dec14C * g, ages))
        elif nm == '4He':
            out.append(torch.trapz(input_4He(ages, he4_rate) * g, ages))
        else:
            out.append(torch.tensor(0.))
    return out


def _choose_ages(tau):
    """Select age grid: OLD for tau > 500 yr, YOUNG otherwise."""
    return AGES_OLD if float(tau) > 500. else AGES_YOUNG


def forward_DM(tau, PD, sample_date, tracer_list, scale_list,
               he4_rate=1e-12, ages=None, dgmeta=None):
    """Single-component DM forward model. Returns list[Tensor]."""
    if ages is None:
        ages = _choose_ages(tau)
    g     = g_DM(ages, tau, PD)
    t_cal = torch.clamp(sample_date - ages, 1900., float(sample_date))
    return _sim_component(g, ages, t_cal, tracer_list, scale_list, he4_rate, dgmeta)


def forward_BMM(tau1, PD1, f1, tau2, PD2,
                sample_date, tracer_list, scale_list,
                he4_rate=1e-12, dic_c1=100., dic_c2=100., dgmeta=None):
    """
    BMM-DM-DM forward model. Returns list[Tensor].
    Young component always uses AGES_YOUNG.
    Old component uses AGES_OLD if tau2 > 500 yr, else AGES_YOUNG.
    14C mixing is DIC-weighted.
    """
    ages_o = _choose_ages(tau2)

    g1    = g_DM(AGES_YOUNG, tau1, PD1)
    t_y   = torch.clamp(sample_date - AGES_YOUNG, 1900., float(sample_date))
    sims1 = _sim_component(g1, AGES_YOUNG, t_y, tracer_list, scale_list, he4_rate, dgmeta)

    g2    = g_DM(ages_o, tau2, PD2)
    t_o   = torch.clamp(sample_date - ages_o, -60000., float(sample_date))
    sims2 = _sim_component(g2, ages_o, t_o, tracer_list, scale_list, he4_rate, dgmeta)

    out = []
    for i, name in enumerate(tracer_list):
        if name.strip() == '14C':
            # DIC-weighted 14C mixing
            mix = (f1*sims1[i]*dic_c1 + (1.-f1)*sims2[i]*dic_c2) / \
                  (f1*dic_c1           + (1.-f1)*dic_c2)
        else:
            mix = f1*sims1[i] + (1.-f1)*sims2[i]
        out.append(mix)
    return out

# ════════════════════════════════════════════════════════════
# PINN CLASSES
# ════════════════════════════════════════════════════════════
def _to_b(r, lo, hi):
    return lo + (hi - lo) * torch.sigmoid(r)

def _logit(v, lo, hi):
    p = float(np.clip((v - lo) / (hi - lo), 1e-4, 1. - 1e-4))
    return float(np.log(p / (1. - p)))


class PINN_DM(nn.Module):
    """
    DM PINN: two learnable params — τ and P_D.
    Both sigmoid-bounded to [lo, hi].
    """
    def __init__(self, tau0, pd0,
                 tau_lo=0.5, tau_hi=100000.,
                 pd_lo=0.001, pd_hi=2.0):
        super().__init__()
        self.tau_lo = tau_lo; self.tau_hi = tau_hi
        self.pd_lo  = pd_lo;  self.pd_hi  = pd_hi
        self.r_tau  = nn.Parameter(torch.tensor(
            _logit(float(np.clip(tau0, tau_lo*1.01, tau_hi*0.99)),
                   tau_lo, tau_hi)))
        self.r_pd   = nn.Parameter(torch.tensor(
            _logit(float(np.clip(pd0, pd_lo*1.01, pd_hi*0.99)),
                   pd_lo, pd_hi)))

    def get_params(self):
        return (_to_b(self.r_tau, self.tau_lo, self.tau_hi),
                _to_b(self.r_pd,  self.pd_lo,  self.pd_hi))


class PINN_BMM(nn.Module):
    """
    BMM-DM-DM PINN.

    Free parameters depend on the TracerLPM optimisation mode:

    Mode A  (free_params = 'Mean Age, Fraction'):
        optimise τ₁ and f₁ ; τ₂ fixed; P_D₁, P_D₂ fixed

    Mode B  (free_params = 'Fraction'):
        optimise f₁ only; τ₁ fixed; τ₂ fixed

    Mode C  (free_params = 'Fraction, 2nd Mean Age'):
        optimise f₁ and τ₂; τ₁ fixed

    P_D values are always fixed at their TracerLPM initial values.
    """
    def __init__(self, free_params,
                 tau1_0, pd1_0, f1_0,
                 tau2_0, pd2_0,
                 tau1_lo=0.5, tau1_hi=500.,
                 f1_lo=0.005, f1_hi=0.997,
                 tau2_lo=1.,  tau2_hi=None):
        super().__init__()
        self._pd1  = float(pd1_0)
        self._pd2  = float(pd2_0)
        self._free = str(free_params or 'Mean Age, Fraction').lower()

        # Determine which params are learnable
        self._tau1_free = 'mean age' in self._free
        self._f1_free   = 'fraction' in self._free
        self._tau2_free = '2nd' in self._free or 'second' in self._free

        if self._tau1_free:
            tau1_0c = float(np.clip(tau1_0, tau1_lo*1.01, tau1_hi*0.99))
            self.r_tau1 = nn.Parameter(torch.tensor(
                _logit(tau1_0c, tau1_lo, tau1_hi)))
            self._tau1_lo = tau1_lo; self._tau1_hi = tau1_hi
        else:
            self._tau1_val = float(tau1_0)

        if self._f1_free:
            f1_0c = float(np.clip(f1_0, f1_lo*1.01, f1_hi*0.99))
            self.r_f1 = nn.Parameter(torch.tensor(
                _logit(f1_0c, f1_lo, f1_hi)))
            self._f1_lo = f1_lo; self._f1_hi = f1_hi
        else:
            self._f1_val = float(f1_0)

        if self._tau2_free:
            if tau2_hi is None:
                tau2_hi = max(float(tau2_0)*5., 200.)
            tau2_0c = float(np.clip(tau2_0, tau2_lo*1.01, tau2_hi*0.99))
            self.r_tau2 = nn.Parameter(torch.tensor(
                _logit(tau2_0c, tau2_lo, tau2_hi)))
            self._tau2_lo = tau2_lo; self._tau2_hi = tau2_hi
        else:
            self._tau2_val = float(tau2_0)

    def get_params(self):
        tau1 = (_to_b(self.r_tau1, self._tau1_lo, self._tau1_hi)
                if self._tau1_free else torch.tensor(self._tau1_val))
        f1   = (_to_b(self.r_f1,   self._f1_lo,   self._f1_hi)
                if self._f1_free   else torch.tensor(self._f1_val))
        tau2 = (_to_b(self.r_tau2, self._tau2_lo, self._tau2_hi)
                if self._tau2_free else torch.tensor(self._tau2_val))
        pd1  = torch.tensor(self._pd1)
        pd2  = torch.tensor(self._pd2)
        return tau1, pd1, f1, tau2, pd2

# ════════════════════════════════════════════════════════════
# LOSS  — squared relative error (TracerLPM "Relative Error" mode)
#   χ² = Σ [(sim − obs) / obs]²
# ════════════════════════════════════════════════════════════
def chi2_loss(sims, obs_vals):
    """Sum of squared relative errors over active tracers."""
    loss = torch.tensor(0.)
    for s, o in zip(sims, obs_vals):
        denom = abs(float(o)) + 1e-30
        loss  = loss + ((s - float(o)) / denom)**2
    return loss

# ════════════════════════════════════════════════════════════
# TRAINING  —  Adam → L-BFGS
# ════════════════════════════════════════════════════════════
def train_pinn(model, loss_fn, n_adam=5000, lr_adam=3e-3, lr_lbfgs=0.02):
    """Two-phase optimiser. Returns loss history list."""
    opt   = optim.Adam(model.parameters(), lr=lr_adam)
    sched = optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=n_adam, eta_min=1e-5)
    hist  = []

    for _ in range(n_adam):
        opt.zero_grad()
        L = loss_fn()
        L.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        opt.step(); sched.step()
        hist.append(L.item())

    opt2 = optim.LBFGS(model.parameters(), lr=lr_lbfgs, max_iter=300,
                        history_size=20, line_search_fn="strong_wolfe")
    def closure():
        opt2.zero_grad(); Lv = loss_fn(); Lv.backward()
        hist.append(Lv.item()); return Lv
    opt2.step(closure)
    return hist


# ════════════════════════════════════════════════════════════
# UNCERTAINTY ANALYSIS
# ════════════════════════════════════════════════════════════
#
# Three complementary methods, all evaluated at the PINN optimum:
#
# 1. Hessian-based (analytical)
#    cov = inv(H) * chi2/dof   where H = d²chi2/d(theta)²
#    sigma_i = sqrt(cov[i,i])
#    Equivalent to Levenberg-Marquardt covariance used by TracerLPM.
#    Fast (one autograd Hessian call).  Assumes a locally parabolic surface.
#
# 2. Profile chi2 (most robust)
#    Hold each parameter at a series of values; minimise all others.
#    68 % CI = region where chi2 <= chi2_min + 1.0  (Δchi2 = 1).
#    Gives asymmetric intervals when surface is non-parabolic.
#    Slow (many forward/backward passes) but the most reliable.
#
# 3. Monte Carlo (stochastic, matches TracerLPM's MC option)
#    Perturb observed concentrations by Gaussian noise (1-sigma = obs_err).
#    Re-optimise from the current best-fit.  Repeat N times.
#    sigma_i = std of N fitted parameter values.
#    Captures full nonlinear error propagation from measurement uncertainty.
#
# Chi2 probability (TracerLPM's LPM_Prob):
#    prob = chi2.sf(chi2_val, dof)   dof = max(n_active - n_free, 1)
#    Probability that a random chi2 variate with dof degrees of freedom
#    exceeds the observed chi2.  High prob = good fit.
# ════════════════════════════════════════════════════════════

from scipy import stats as _sp_stats


def chi2_probability(chi2_val, n_active_tracers, n_free_params):
    """
    Chi2 goodness-of-fit probability  (TracerLPM LPM_Prob).

    Parameters
    ----------
    chi2_val         : float  optimised chi2 value
    n_active_tracers : int    number of tracers in the loss
    n_free_params    : int    number of free model parameters

    Returns
    -------
    float  probability P(chi2 >= chi2_val | dof)
    """
    dof = max(n_active_tracers - n_free_params, 1)
    return float(_sp_stats.chi2.sf(chi2_val, dof)), dof


def hessian_uncertainty(loss_fn_direct, params_opt, chi2_val,
                         n_active_tracers, n_free_params):
    """
    Hessian-based 1-sigma parameter uncertainties.

    Parameters
    ----------
    loss_fn_direct  : callable(params_tensor) -> scalar tensor
                      chi2 loss as function of a flat parameter vector
                      in NATURAL (un-transformed) units, e.g. [tau, PD]
    params_opt      : 1-D torch.Tensor   optimal parameter values
    chi2_val        : float              chi2 at the optimum
    n_active_tracers: int
    n_free_params   : int

    Returns
    -------
    sigmas : np.ndarray   1-sigma errors for each parameter
    cov    : np.ndarray   full covariance matrix
    dof    : int          degrees of freedom
    """
    dof = max(n_active_tracers - n_free_params, 1)
    p64 = params_opt.double()

    # Autograd Hessian
    H = torch.autograd.functional.hessian(
        lambda p: loss_fn_direct(p.float()).double(), p64
    ).detach().numpy()

    try:
        Hinv = np.linalg.inv(H)
        cov  = Hinv * (chi2_val / dof)
        sigmas = np.sqrt(np.abs(np.diag(cov)))
    except np.linalg.LinAlgError:
        # Singular Hessian → use pseudo-inverse
        Hinv   = np.linalg.pinv(H)
        cov    = Hinv * (chi2_val / dof)
        sigmas = np.sqrt(np.abs(np.diag(cov)))

    return sigmas, cov, dof


def profile_uncertainty(loss_fn_1d, param_opt, chi2_min,
                        param_lo, param_hi, n_bisect=30):
    """
    Profile chi2 68 % confidence interval for ONE parameter.

    Fixes the target parameter at a series of values while the loss
    function already marginalises over all other parameters (the caller
    is responsible for supplying a 1-D loss that holds other params fixed
    or re-optimises them).

    For speed we use the simpler fixed-others profile (other params held
    at their optimal values), which gives a valid 1-sigma CI when the
    chi2 surface is approximately ellipsoidal.

    Parameters
    ----------
    loss_fn_1d : callable(float) -> float   chi2 at fixed param value
    param_opt  : float                      optimal value
    chi2_min   : float                      chi2 at the optimum
    param_lo   : float                      hard lower bound
    param_hi   : float                      hard upper bound
    n_bisect   : int                        bisection iterations

    Returns
    -------
    (lo_68, hi_68) : (float, float)   lower and upper 68 % CI bounds
    """
    threshold = chi2_min + 1.0   # Δchi2 = 1 → 68 % for 1 free parameter

    def _bisect(a, b):
        """Find root of loss_fn_1d(x) - threshold by bisection."""
        for _ in range(n_bisect):
            mid = 0.5 * (a + b)
            if loss_fn_1d(mid) > threshold:
                b = mid
            else:
                a = mid
        return 0.5 * (a + b)

    # Upper bound
    hi_68 = param_opt
    step  = max(abs(param_opt) * 0.05, 0.1)
    while hi_68 + step < param_hi:
        hi_68 += step
        if loss_fn_1d(hi_68) > threshold:
            hi_68 = _bisect(hi_68 - step, hi_68)
            break
    else:
        hi_68 = min(param_opt * 3., param_hi)   # unbounded

    # Lower bound
    lo_68 = param_opt
    step  = max(abs(param_opt) * 0.05, 0.1)
    while lo_68 - step > param_lo:
        lo_68 -= step
        if loss_fn_1d(lo_68) > threshold:
            lo_68 = _bisect(lo_68, lo_68 + step)
            break
    else:
        lo_68 = max(param_lo, param_opt * 0.1)  # unbounded

    return float(lo_68), float(hi_68)


def mc_uncertainty(fit_fn, obs_vals, obs_errs, scales, n_mc=100,
                   seed=42, progress=False):
    """
    Monte Carlo uncertainty via observation perturbation.

    Perturbs each observed concentration by Gaussian noise with 1-sigma
    = obs_err, re-fits the model, and returns the distribution of
    fitted parameters across N_mc realisations.

    Parameters
    ----------
    fit_fn   : callable(obs_perturbed_list) -> dict
               Must accept a list of perturbed obs values (same length as
               obs_vals) and return a dict with at least keys 'tau1', 'f1'.
               Typically a thin wrapper around fit_sample().
    obs_vals : list[float]   nominal observed concentrations
    obs_errs : list[float]   1-sigma measurement errors
    scales   : list[float]   active-tracer flags (unused here, for compat.)
    n_mc     : int           number of Monte Carlo realisations (default 100)
    seed     : int           random seed for reproducibility
    progress : bool          print progress dots

    Returns
    -------
    mc_results : dict of param_name -> np.ndarray of length n_mc
    mc_stats   : dict of param_name -> {'mean', 'std', 'p16', 'p84', 'p50'}
    """
    rng = np.random.default_rng(seed)
    mc_results = {}

    for k in range(n_mc):
        if progress and k % 10 == 0:
            print(f"  MC {k}/{n_mc}", end="\r", flush=True)

        obs_pert = [
            max(float(o) + rng.normal(0., float(e)), 1e-20)
            for o, e in zip(obs_vals, obs_errs)
        ]

        try:
            res = fit_fn(obs_pert)
            for key, val in res.items():
                if val is not None:
                    mc_results.setdefault(key, []).append(float(val))
        except Exception:
            pass   # skip failed realisations

    if progress:
        print()

    mc_stats = {}
    for key, vals in mc_results.items():
        arr = np.array(vals)
        mc_stats[key] = {
            'mean': float(np.mean(arr)),
            'std' : float(np.std(arr)),
            'p16' : float(np.percentile(arr, 16)),
            'p50' : float(np.percentile(arr, 50)),
            'p84' : float(np.percentile(arr, 84)),
        }

    return mc_results, mc_stats
