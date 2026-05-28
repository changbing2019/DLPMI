"""
dlpmi/forward.py
================
Generic forward models for any single-component LPM or BMM combination.

forward_single  : simulate any one of PFM / EMM / EPM / DM / PEM
forward_BMM     : simulate any BMM(model1, model2) pair

The ¹⁴C mixing in BMM uses DIC-weighted blending to account for the
dead-carbon dilution effect in carbonate aquifers.
"""

import torch
from .tracers import (input_3H, input_SF6, input_14C, input_4He,
                       LAMBDA_3H, LAMBDA_14C)
from .kernels import choose_ages, AGES_YOUNG, MODEL_REGISTRY


# ════════════════════════════════════════════════════════════════
# HELPERS
# ════════════════════════════════════════════════════════════════

def _sim_on_grid(g, ages, t_cal, tracer_list, scale_list,
                 he4_rate=1e-12, dgmeta=None):
    """
    Convolve one kernel g(a) over all requested tracers.

    Parameters
    ----------
    g           : normalised PDF on `ages` grid
    ages        : age grid tensor (yr)
    t_cal       : calendar year at each age point  (= sample_date − age)
    tracer_list : list of tracer name strings
    scale_list  : list of float scale factors (same length as tracer_list)
    he4_rate    : ⁴He accumulation rate β (cc STP g⁻¹ yr⁻¹)
    dgmeta      : dict with SF₆ DGMETA correction params
                  keys: T_recharge_C, excess_air_cckg,
                        fractionation_F, elevation_m

    Returns
    -------
    list of scalar Tensors — one per tracer
    """
    if dgmeta is None:
        dgmeta = {}
    dec3H  = torch.exp(-LAMBDA_3H  * ages)
    dec14C = torch.exp(-LAMBDA_14C * ages)
    out    = []

    for i, nm in enumerate(tracer_list):
        sc = float(scale_list[i]) if i < len(scale_list) else 1.0
        nm = nm.strip()

        if nm == "3H":
            v = torch.trapz(input_3H(t_cal, sc) * dec3H * g, ages)

        elif nm in ("3He(trit)", "3Hetrit", "3He_trit", "3He"):
            v = torch.trapz(input_3H(t_cal, sc) * (1. - dec3H) * g, ages)

        elif nm == "SF6":
            sf6_in = input_SF6(
                t_cal, sc,
                T_recharge_C    = dgmeta.get("T_recharge_C",    18.),
                excess_air_cckg = dgmeta.get("excess_air_cckg",  5.),
                fractionation_F = dgmeta.get("fractionation_F",  0.5),
                elevation_m     = dgmeta.get("elevation_m",     300.))
            v = torch.trapz(sf6_in * g, ages)

        elif nm == "14C":
            v = torch.trapz(input_14C(t_cal, sc) * dec14C * g, ages)

        elif nm == "4He":
            v = torch.trapz(input_4He(ages, he4_rate) * g, ages)

        else:
            v = torch.tensor(0.)

        out.append(v)
    return out


# ════════════════════════════════════════════════════════════════
# SINGLE-COMPONENT FORWARD MODEL
# ════════════════════════════════════════════════════════════════

def forward_single(model_name: str,
                   params: dict,
                   sample_date: float,
                   tracer_list: list,
                   scale_list: list,
                   he4_rate: float = 1e-12,
                   ages: torch.Tensor = None,
                   dgmeta: dict = None) -> list:
    """
    Single-component LPM forward model for any supported model type.

    Parameters
    ----------
    model_name  : one of "PFM", "EMM", "EPM", "DM", "PEM"
    params      : dict of model parameters, e.g.:
                    DM  → {"tau": <Tensor>, "PD": <Tensor>}
                    EMM → {"tau": <Tensor>}
                    EPM → {"tau": <Tensor>, "eta": <Tensor>}
    sample_date : decimal year of sampling
    tracer_list : list of tracer name strings (e.g. ["3H", "SF6", "14C"])
    scale_list  : list of float scale factors (one per tracer)
    he4_rate    : ⁴He accumulation rate β (cc STP g⁻¹ yr⁻¹)
    ages        : age grid; auto-selected from tau if None
    dgmeta      : dict of SF₆ DGMETA correction parameters

    Returns
    -------
    list of scalar Tensors — one per tracer in tracer_list
    """
    if model_name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{model_name}'. "
            f"Valid choices: {list(MODEL_REGISTRY.keys())}")

    kernel_fn, extra_params = MODEL_REGISTRY[model_name]
    tau = params["tau"]

    if ages is None:
        ages = choose_ages(float(tau.detach()) if hasattr(tau, 'detach') else tau)

    # Build keyword arguments for the kernel
    kw = {k: params[k] for k in extra_params if k in params}
    g  = kernel_fn(ages, tau, **kw)

    t_cal = torch.clamp(
        torch.tensor(float(sample_date)) - ages,
        -60000., float(sample_date))

    return _sim_on_grid(g, ages, t_cal, tracer_list, scale_list,
                        he4_rate, dgmeta)


# ════════════════════════════════════════════════════════════════
# BINARY MIXING MODEL FORWARD MODEL
# ════════════════════════════════════════════════════════════════

def forward_BMM(model1: str,
                params1: dict,
                model2: str,
                params2: dict,
                f1: torch.Tensor,
                sample_date: float,
                tracer_list: list,
                scale_list: list,
                he4_rate: float = 1e-12,
                dic_c1: float = 100.,
                dic_c2: float = 100.,
                uz_tt: float = 0.,
                dgmeta: dict = None) -> list:
    """
    Binary Mixing Model forward model for any two single-component models.

    The mixed concentration is:
        C_mix = f₁ · C_1 + (1−f₁) · C_2

    For ¹⁴C, DIC-weighted blending is used to account for dead-carbon:
        ¹⁴C_mix = (f₁·¹⁴C₁·DIC₁ + (1−f₁)·¹⁴C₂·DIC₂) / (f₁·DIC₁ + (1−f₁)·DIC₂)

    Parameters
    ----------
    model1      : name of the first (young) component model
    params1     : parameter dict for model1
    model2      : name of the second (old) component model
    params2     : parameter dict for model2
    f1          : mixing fraction of the first component (Tensor)
    sample_date : decimal year of sampling
    tracer_list : tracer names
    scale_list  : scale factors
    he4_rate    : ⁴He accumulation rate (cc STP g⁻¹ yr⁻¹)
    dic_c1      : DIC concentration of young component (mmol/L)
    dic_c2      : DIC concentration of old component (mmol/L)
    uz_tt       : unsaturated-zone travel time (yr) subtracted from
                  calendar time before evaluating input functions
    dgmeta      : SF₆ DGMETA correction parameters

    Returns
    -------
    list of scalar Tensors — one per tracer in tracer_list

    Examples
    --------
    # BMM-DM-DM (Edwards Aquifer)
    sims = forward_BMM("DM", {"tau": t1, "PD": p1},
                       "DM", {"tau": t2, "PD": p2},
                       f1, sample_date, tracers, scales)

    # BMM-EMM-PFM (Missouri River Case 3)
    sims = forward_BMM("EMM", {"tau": tau_emm},
                       "PFM", {"tau": tau_pfm},
                       f_emm, sample_date, tracers, scales)
    """
    if model1 not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model1 '{model1}'. Valid: {list(MODEL_REGISTRY.keys())}")
    if model2 not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model2 '{model2}'. Valid: {list(MODEL_REGISTRY.keys())}")

    tau1 = params1["tau"]
    tau2 = params2["tau"]

    # Component 1 always uses AGES_YOUNG (young recharge)
    ages1 = AGES_YOUNG
    # Component 2: use age grid appropriate to tau2
    ages2 = choose_ages(float(tau2.detach()) if hasattr(tau2, 'detach') else tau2)

    kernel1, extra1 = MODEL_REGISTRY[model1]
    kernel2, extra2 = MODEL_REGISTRY[model2]

    kw1 = {k: params1[k] for k in extra1 if k in params1}
    kw2 = {k: params2[k] for k in extra2 if k in params2}

    g1 = kernel1(ages1, tau1, **kw1)
    g2 = kernel2(ages2, tau2, **kw2)

    sd = float(sample_date)
    t1 = torch.clamp(torch.tensor(sd) - ages1 - uz_tt, -60000., sd)
    t2 = torch.clamp(torch.tensor(sd) - ages2 - uz_tt, -60000., sd)

    sims1 = _sim_on_grid(g1, ages1, t1, tracer_list, scale_list,
                         he4_rate, dgmeta)
    sims2 = _sim_on_grid(g2, ages2, t2, tracer_list, scale_list,
                         he4_rate, dgmeta)

    out = []
    for i, nm in enumerate(tracer_list):
        if nm.strip() == "14C":
            # DIC-weighted mixing for ¹⁴C
            mix = (f1 * sims1[i] * dic_c1 + (1. - f1) * sims2[i] * dic_c2) / \
                  (f1 * dic_c1            + (1. - f1) * dic_c2)
        else:
            mix = f1 * sims1[i] + (1. - f1) * sims2[i]
        out.append(mix)
    return out
