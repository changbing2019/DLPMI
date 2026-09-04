"""
dlpmi/forward.py  v1.3.0
========================
Forward model: convolve LPM kernels over tracer input functions.

v1.3.0 — uses TracerLPM-compatible input functions from Jurgens CSV.
         h3_source (column name) is passed via dgmeta dict per sample.
"""

import torch
from .kernels import MODEL_REGISTRY, choose_ages
from .tracers import (input_3H, input_SF6, input_14C, input_4He,
                      H3_DEFAULT_COL, LAMBDA_3H, LAMBDA_14C)


def _sim_on_grid(g, ages, t_cal, tracer_list, scale_list,
                 he4_rate=1e-12, dgmeta=None):
    """
    Convolve one kernel g(a) over all requested tracers.

    Parameters
    ----------
    g           : exit-age PDF tensor
    ages        : age grid (yr)
    t_cal       : calendar year of sampling (float tensor)
    tracer_list : list of tracer name strings
    scale_list  : list of float scale factors
    he4_rate    : ⁴He accumulation rate (cc STP/g/yr)
    dgmeta      : dict with optional per-sample physics parameters:
                    T_recharge_C, excess_air_cckg, fractionation_F,
                    elevation_m, h3_source (column name), h3_scale
    """
    if dgmeta is None:
        dgmeta = {}

    h3_source = dgmeta.get('h3_source', H3_DEFAULT_COL)
    h3_scale  = float(dgmeta.get('h3_scale', 1.0))

    sims = []
    for i, nm in enumerate(tracer_list):
        sc = float(scale_list[i]) if i < len(scale_list) else 1.0
        t_cal_grid = t_cal - ages

        if nm == '3H':
            c_in = input_3H(t_cal_grid, scale=h3_scale, source_col=h3_source)
            dec3H = torch.exp(-LAMBDA_3H * ages)
            v = torch.trapz(c_in * dec3H * g, ages)

        elif nm in ('3He(trit)', '3Hetrit'):
            c_in = input_3H(t_cal_grid, scale=h3_scale, source_col=h3_source)
            dec3H = torch.exp(-LAMBDA_3H * ages)
            v = torch.trapz(c_in * (1. - dec3H) * g, ages)

        elif nm == 'SF6':
            sf6_in = input_SF6(
                t_cal_grid,
                scale           = sc,
                T_recharge_C    = float(dgmeta.get('T_recharge_C',    18.0)),
                excess_air_cckg = float(dgmeta.get('excess_air_cckg',  5.0)),
                fractionation_F = float(dgmeta.get('fractionation_F',  0.5)),
                elevation_m     = float(dgmeta.get('elevation_m',    300.0)),
            )
            v = torch.trapz(sf6_in * g, ages)

        elif nm == '14C':
            c_in  = input_14C(t_cal_grid, scale=sc)
            dec14C = torch.exp(-LAMBDA_14C * ages)
            v = torch.trapz(c_in * dec14C * g, ages)

        elif nm == '4He':
            v = torch.trapz(input_4He(ages, he4_rate) * g, ages)

        else:
            continue

        sims.append(v)

    return sims


def forward_single(model_name, params, sample_date,
                   tracer_list, scale_list,
                   he4_rate=1e-12, ages=None, dgmeta=None):
    if dgmeta is None: dgmeta = {}
    if ages is None:
        ages = choose_ages(float(params['tau'].detach()))
    kernel_fn, _ = MODEL_REGISTRY[model_name.upper()]
    g = kernel_fn(ages, **params)
    t_cal = torch.tensor(float(sample_date), dtype=torch.float32)
    return _sim_on_grid(g, ages, t_cal, tracer_list, scale_list,
                        he4_rate, dgmeta)


def forward_BMM(model1, params1, model2, params2, f1,
                sample_date, tracer_list, scale_list,
                he4_rate=1e-12, dic_c1=100., dic_c2=100.,
                uz_tt=0., dgmeta=None):
    if dgmeta is None: dgmeta = {}
    tau1 = float(params1['tau'].detach())
    tau2 = float(params2['tau'].detach())
    ages1 = choose_ages(tau1)
    ages2 = choose_ages(tau2)
    t_samp = float(sample_date)
    t1 = torch.tensor(t_samp - uz_tt, dtype=torch.float32)
    t2 = torch.tensor(t_samp,         dtype=torch.float32)
    kernel1, _ = MODEL_REGISTRY[model1.upper()]
    kernel2, _ = MODEL_REGISTRY[model2.upper()]
    g1 = kernel1(ages1, **params1)
    g2 = kernel2(ages2, **params2)
    sims1 = _sim_on_grid(g1, ages1, t1, tracer_list, scale_list,
                         he4_rate, dgmeta)
    sims2 = _sim_on_grid(g2, ages2, t2, tracer_list, scale_list,
                         he4_rate, dgmeta)
    sims_mix = []
    for i, nm in enumerate(tracer_list):
        if nm == '14C':
            w1 = f1 * dic_c1; w2 = (1. - f1) * dic_c2
            v  = (w1 * sims1[i] + w2 * sims2[i]) / (w1 + w2 + 1e-30)
        else:
            v  = f1 * sims1[i] + (1. - f1) * sims2[i]
        sims_mix.append(v)
    return sims_mix
