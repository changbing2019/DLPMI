"""
examples/quickstart.py
======================
Three minimal worked examples demonstrating DLPMI on the TracerLPM
benchmark cases from Jurgens et al. (2012).

Run:
    python examples/quickstart.py
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from dlpmi import Inverter, Sample

# ════════════════════════════════════════════════════════════════
# Case 1 — PSW-1, Modesto CA  (DM model)
# Jurgens et al. (2012), TracerLPM Example 1
# TracerLPM reference: tau=75.0 yr, PD=0.800
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("Case 1 — PSW-1 Modesto CA  (DM)")
print("=" * 60)

case1 = Sample(
    sample_id   = "PSW-1_Modesto_CA",
    sample_date = 2004.6257,
    tracers     = [
        {"name": "3H",        "obs": 4.57,  "scale": 1.0},
        {"name": "3He(trit)", "obs": 32.70, "scale": 1.0},
        {"name": "SF6",       "obs": 0.75,  "scale": 0.640},  # USGS CFC lab scale
    ],
    model  = "DM",
    bounds = {
        "tau0":  75.,  "tau_lo": 50.,  "tau_hi": 100.,
        "pd0":    0.8,  "pd_lo":  0.01, "pd_hi":  1.0,
    },
)

result1 = Inverter(n_adam=5000, n_starts=5).fit(case1)

tau = result1["params"]["tau"]
PD  = result1["params"]["PD"]
print(f"  DLPMI:     tau = {tau:.2f} yr,  P_D = {PD:.3f}")
print(f"  TracerLPM: tau = 75.0 yr,  P_D = 0.800")
print(f"  chi2 = {result1['chi2']:.5f}")
print()
for nm, obs, sim in zip(result1["tracer_names"],
                        result1["obs_vals"],
                        result1["sim_vals"]):
    err = abs(sim - obs) / abs(obs) * 100 if obs else 0.
    print(f"    {nm:<15} obs={obs:.4f}  sim={sim:.4f}  err={err:.1f}%")


# ════════════════════════════════════════════════════════════════
# Case 2 — SSW, Albuquerque NM  (BMM-DM-DM)
# Jurgens et al. (2012), TracerLPM Example 2
# TracerLPM ref: tau_y=19.343 yr, PD_y=0.04209, f_y=11.746%, tau_o=12100 yr
# Optimised on 3H + 14C only (CFC-113 excluded)
# ════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Case 2 — SSW Albuquerque NM  (BMM-DM-DM)")
print("=" * 60)

case2 = Sample(
    sample_id   = "SSW_Albuquerque_NM",
    sample_date = 2007.43,
    tracers     = [
        {"name": "3H",       "obs": 2.35,  "scale": 1.778},
        {"name": "14C",      "obs": 43.98, "scale": 0.65},
        # CFC-113 measured but not used in fit:
        {"name": "CFC-113",  "obs": 5.44,  "scale": 1.0, "active": False},
    ],
    model       = "BMM",
    model1      = "DM",   # young component
    model2      = "DM",   # old component
    free_params = "Mean Age, Fraction",   # Mode A
    bounds      = {
        "tau1_0":  19.3, "tau1_lo": 15., "tau1_hi": 30.,
        "pd1":     0.042,                  # fixed P_D young
        "tau2":    12100.,                 # fixed old mean age
        "pd2":     0.1,                    # fixed P_D old
        "f1_0":    0.12, "f1_lo":  0.01, "f1_hi": 0.30,
    },
    uz_tt   = 15.,
    dic_c1  = 100.,
    dic_c2  = 100.,
)

result2 = Inverter(n_adam=5000, n_starts=3).fit(case2)

tau_y = result2["params"]["tau"]
f_y   = result2["f1"]
print(f"  DLPMI:     tau_y = {tau_y:.2f} yr,  f_y = {f_y*100:.1f}%")
print(f"  TracerLPM: tau_y = 19.343 yr, f_y = 11.7%")
print(f"  chi2 = {result2['chi2']:.5f}")
print()
for nm, obs, sim, act in zip(result2["tracer_names"],
                              result2["obs_vals"],
                              result2["sim_vals"],
                              result2["active"]):
    err = abs(sim - obs) / abs(obs) * 100 if obs else 0.
    tag = " (not in fit)" if not act else ""
    print(f"    {nm:<15} obs={obs:.4f}  sim={sim:.4f}  "
          f"err={err:.1f}%{tag}")


# ════════════════════════════════════════════════════════════════
# Case 3 — Missouri River  (BMM-EMM-PFM time series)
# Jurgens et al. (2012), TracerLPM Example 3
# 35 annual 3H observations 1963-1997
# TracerLPM ref: tau=4.3 yr, f_emm=0.84
# ════════════════════════════════════════════════════════════════
print()
print("=" * 60)
print("Case 3 — Missouri River NE  (BMM-EMM-PFM time series)")
print("=" * 60)

OBS_YEARS = list(range(1963, 1998))
OBS_3H    = [1294,1162,1025, 981, 700, 565, 437, 369, 305, 233,
              161, 155, 143, 119, 100,  57,  81,  69,  64,  61,
               44,  40,  35,  35,  29,  32,  26,  24,  25,  16,
               16,  17,  15,  14,  13]

# For time-series: create one Sample per observation year
# Each shares the same model parameters but different sample_date
# The Inverter handles this internally via a single loss over all obs
# Here we build a synthetic single-date Sample for demonstration;
# see examples/case3_timeseries.py for the full time-series treatment.
import numpy as np
import torch

# Quick scalar demonstration: fit to 1997 endpoint only
case3_single = Sample(
    sample_id   = "MissouriRiver_1997",
    sample_date = 1997.5,
    tracers     = [
        {"name": "3H", "obs": 13., "scale": 1.0},
    ],
    model       = "BMM",
    model1      = "EMM",   # groundwater component
    model2      = "PFM",   # prompt runoff component
    free_params = "Mean Age, Fraction",
    bounds      = {
        "tau1_0":  8., "tau1_lo": 1., "tau1_hi": 15.,
        "tau2":    0.5,   # PFM mean age ~ 0.5 yr
        "f1_0":    0.70, "f1_lo": 0.50, "f1_hi": 0.97,
    },
)

result3 = Inverter(n_adam=3000, n_starts=3).fit(case3_single)
tau_emm = result3["params"]["tau"]
f_emm   = result3["f1"]
print(f"  DLPMI:     tau_EMM = {tau_emm:.2f} yr,  f_EMM = {f_emm:.3f}")
print(f"  TracerLPM: tau_EMM = 4.3 yr,           f_EMM = 0.840")
print(f"  (See examples/case3_timeseries.py for full 35-obs time-series fit)")
print()
print("All cases complete.")
