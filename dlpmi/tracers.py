"""
dlpmi/tracers.py
================
Environmental tracer input functions and decay constants.

All functions accept a PyTorch tensor of calendar years and return
a tensor of tracer concentrations in the appropriate units.

Tracers
-------
  ³H        GNIP N-hemisphere precipitation (TU)
  ³He(trit) Tritiogenic helium-3 — in-situ accumulation from ³H decay
  SF₆       NOAA/ESRL atmospheric record + Bullister et al. (2002) solubility
             + CE excess-air model (per-sample DGMETA corrections)
  ¹⁴C       IntCal09 pre-bomb + NH Zone 2 bomb curve
  ⁴He       Radiogenic helium-4 — linear accumulation

³H latitude calibration note
-----------------------------
The GNIP N-hemisphere record is split at 1981 and scaled by latitude
correction factors.  The default values (pre-1981 × 0.668, post-1980 × 0.668)
are calibrated for subtropical / semi-arid sites (≈ 30°N), such as the
Edwards Aquifer, Texas, where post-bomb ³H in precipitation declined faster
and to lower values than the hemisphere-wide average.

For HIGH-LATITUDE sites (Ottawa, Vienna; > 45°N) use post-1980 factor = 1.336.
For MID-LATITUDE sites (≈ 35–45°N) use post-1980 factor ≈ 1.0.
For SUBTROPICAL sites (≈ 25–35°N) use post-1980 factor = 0.668 (default).

The factor is controlled by the module-level constant H3_SCALE_POST1980.
Override it before importing or pass a custom record via input_3H_custom().

References
----------
Bullister et al. (2002) Deep-Sea Research I, 49, 175–187
Jurgens et al. (2020) USGS TM 4-F5 (DGMETA)
Levin & Kromer (2004) Radiocarbon, 46(3), 1261–1272
Reimer et al. (2009) Radiocarbon, 51(4), 1111–1150
"""

import torch
import numpy as np

# ── radioactive decay constants (yr⁻¹) ───────────────────────────────────────
LAMBDA_3H  = float(np.log(2) / 12.32)
LAMBDA_14C = float(np.log(2) / 5730.0)

# ══════════════════════════════════════════════════════════════════════════════
# ³H — GNIP Northern Hemisphere calibrated record
# ══════════════════════════════════════════════════════════════════════════════

# ── Latitude calibration constants ───────────────────────────────────────────
# Adjust these for your study area before use:
#   Subtropical / semi-arid (≈ 30°N, e.g. Texas, Arizona):  0.668  ← default
#   Mid-latitude (≈ 35–45°N, e.g. California, mid-Europe):  1.000
#   High-latitude (> 45°N, e.g. Ottawa, Vienna):            1.336
H3_SCALE_PRE1981  = 0.668   # pre-1981  (low-latitude correction)
H3_SCALE_POST1980 = 0.668   # post-1980 (subtropical default; see note above)

_3H_START = 1953.
_3H_VALS_RAW = torch.tensor([
     8., 10, 12, 15, 25, 50, 100, 150,
   300., 900, 1600, 800, 350,
   200., 130,  90,  65,  50,  35,
    25.,  22,  18,  16,  15,  13,  12, 11, 10, 10,
    10.,   9,   9,   8,   8,   8,   7,  7,  7,  7,
     7.,   6,   6,   6,   6,   6,   6,  6,  6,  6,
     6.,   6,   6,   6,   6,   6,   6,  6,  6,  6,
     6.,   6,   6,   6,   6,   6,   6,
], dtype=torch.float32)

# Apply latitude calibration to internal record
_3H_VALS = _3H_VALS_RAW.clone()
_3H_VALS[:28] *= H3_SCALE_PRE1981
_3H_VALS[28:] *= H3_SCALE_POST1980

_3H_PRE = 2.0   # pre-1953 background TU (subtropical; use 5.0 for high-latitude)


def input_3H(t: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    ³H precipitation input (TU).

    The built-in record uses the GNIP N-hemisphere values scaled by the
    module-level latitude calibration constants H3_SCALE_PRE1981 and
    H3_SCALE_POST1980.  The additional ``scale`` parameter applies a
    multiplicative correction on top of the latitude calibration — for
    example, to account for elevation or local precipitation patterns.

    Parameters
    ----------
    t     : calendar year(s)
    scale : additional site-specific scale factor (default 1.0)

    Returns
    -------
    ³H concentration (TU)

    Notes
    -----
    For a new site, determine the correct scale by back-calculating from
    DM Modern samples whose age is well-constrained by SF₆ or ³He(trit).
    """
    t_c = torch.clamp(t, _3H_START, _3H_START + len(_3H_VALS) - 1.)
    f   = t_c - _3H_START
    lo  = torch.clamp(f.long(), 0, len(_3H_VALS) - 2)
    hi  = torch.clamp(lo + 1,   0, len(_3H_VALS) - 1)
    w   = torch.clamp(f - lo.float(), 0., 1.)
    v   = _3H_VALS[lo] * (1 - w) + _3H_VALS[hi] * w
    return torch.where(t < _3H_START,
                       torch.full_like(v, _3H_PRE), v) * scale


def input_3H_custom(t: torch.Tensor,
                    record_years: torch.Tensor,
                    record_vals:  torch.Tensor,
                    scale: float = 1.0) -> torch.Tensor:
    """
    ³H precipitation input using a user-supplied regional record.

    Use this function when local precipitation ³H monitoring data are
    available (e.g., from a nearby IAEA/GNIP station or USGS precipitation
    sampling programme).  The record is linearly interpolated onto the
    requested years.

    Parameters
    ----------
    t            : calendar year(s) to evaluate
    record_years : 1-D tensor of calendar years in the record
    record_vals  : 1-D tensor of ³H values (TU) at those years
    scale        : additional multiplicative correction (default 1.0)

    Returns
    -------
    ³H concentration (TU)

    Example
    -------
    # Albuquerque NM record from TracerLPM Stored_historic_tracer_data.xlsx
    years = torch.tensor([1960., 1965., 1970., ...])
    vals  = torch.tensor([74., 475., 119.6, ...])
    sim   = input_3H_custom(t_cal, years, vals, scale=1.0)
    """
    lo = torch.zeros_like(t, dtype=torch.long)
    for i in range(len(record_years) - 1):
        lo = torch.where(t >= record_years[i],
                         torch.full_like(lo, i), lo)
    hi  = torch.clamp(lo + 1, 0, len(record_vals) - 1)
    dx  = record_years[hi] - record_years[lo]
    w   = torch.clamp((t - record_years[lo]) /
                      torch.clamp(dx, min=0.01), 0., 1.)
    v   = record_vals[lo] * (1 - w) + record_vals[hi] * w
    # Clamp to record range
    v   = torch.where(t < record_years[0],
                      torch.full_like(v, float(record_vals[0])), v)
    v   = torch.where(t > record_years[-1],
                      torch.full_like(v, float(record_vals[-1])), v)
    return v * scale


# ══════════════════════════════════════════════════════════════════════════════
# SF₆ — NOAA/ESRL atmospheric record + DGMETA corrections
# ══════════════════════════════════════════════════════════════════════════════

# Correct NOAA/ESRL monotonically increasing SF₆ mole fraction (pptv)
_SF6_YRS  = torch.tensor([
    1955., 1960, 1965, 1970, 1975, 1978, 1980, 1982, 1984, 1986,
    1988., 1990, 1992, 1994, 1996, 1998, 2000, 2002, 2004, 2006,
    2008., 2010, 2012, 2014, 2016, 2018, 2020, 2022.])
_SF6_PPTV = torch.tensor([
    0.000, 0.000, 0.000, 0.020, 0.080, 0.100, 0.170, 0.270, 0.430, 0.600,
    0.830, 1.140, 1.520, 1.950, 2.420, 2.920, 3.420, 3.960, 4.540, 5.120,
    5.800, 6.600, 7.390, 8.200, 9.000, 9.750, 10.50, 11.20])


def _Ks_SF6(T_C: float) -> torch.Tensor:
    """
    SF₆ Henry's law solubility (mol L⁻¹ atm⁻¹), salinity = 0.
    Bullister et al. (2002) Table 1.
    """
    T_K = T_C + 273.15
    val = -96.5975 + 139.883 * (100. / T_K) + 37.8193 * np.log(T_K / 100.)
    return torch.exp(torch.tensor(val, dtype=torch.float32))


def input_SF6(t: torch.Tensor,
              scale: float         = 1.0,
              T_recharge_C: float  = 18.0,
              excess_air_cckg: float = 5.0,
              fractionation_F: float = 0.5,
              elevation_m: float   = 300.0) -> torch.Tensor:
    """
    SF₆ atmospheric equivalent (pptv) with DGMETA physical corrections.

    Converts the NOAA/ESRL atmospheric SF₆ record to an effective
    dissolved-phase atmospheric equivalent using the Bullister et al. (2002)
    solubility equation and the Closed-system Equilibration (CE) excess-air
    model (Jurgens et al., 2020).

    Input values for new samples
    -----------------------------
    Provide the raw dissolved SF₆ concentration as measured by the CFC
    laboratory (fmol/L), converted to atmospheric equivalent pptv:

      Workflow A (recommended): run DGMETA → provide corrected pptv,
        set excess_air_cckg = 0 (excess air already removed).
      Workflow B (no DGMETA): convert dissolved fmol/L to raw atmospheric
        equivalent pptv = C_fmol / (Ks(T) × P), then provide the
        DGMETA parameters below so DLPMI applies the corrections internally.

    Parameters
    ----------
    t               : calendar year(s)
    scale           : additional multiplicative factor (default 1.0)
    T_recharge_C    : recharge temperature (°C)
    excess_air_cckg : excess air volume (ccSTP kg⁻¹); 0 if DGMETA pre-corrected
    fractionation_F : CE model fractionation factor
    elevation_m     : recharge zone elevation (m a.s.l.)

    Returns
    -------
    SF₆ atmospheric equivalent (pptv)
    """
    lo = torch.zeros_like(t, dtype=torch.long)
    for i in range(len(_SF6_YRS) - 1):
        lo = torch.where(t >= _SF6_YRS[i],
                         torch.full_like(lo, i), lo)
    hi      = torch.clamp(lo + 1, 0, len(_SF6_PPTV) - 1)
    dtt     = _SF6_YRS[hi] - _SF6_YRS[lo]
    w       = torch.clamp((t - _SF6_YRS[lo]) /
                          torch.clamp(dtt, min=0.01), 0., 1.)
    sf6_atm = _SF6_PPTV[lo] * (1 - w) + _SF6_PPTV[hi] * w
    sf6_atm = torch.where(t < _SF6_YRS[0],
                          torch.zeros_like(sf6_atm), sf6_atm)

    Ks = _Ks_SF6(T_recharge_C)
    P  = torch.exp(torch.tensor(-elevation_m / 8434.5, dtype=torch.float32))

    C_eq  = sf6_atm * Ks * P
    C_exc = (excess_air_cckg / 22400.) * sf6_atm * 1e-12 * \
            fractionation_F * 1e12
    return (C_eq + C_exc) / (Ks * P) * scale


# ══════════════════════════════════════════════════════════════════════════════
# ¹⁴C — IntCal09 + NH Zone 2 bomb curve
# ══════════════════════════════════════════════════════════════════════════════

# IntCal09 pre-bomb (age BP → pmC)
_IC09_BP = torch.tensor([
      0.,  500, 1000, 1500, 2000, 2500, 3000, 3500, 4000, 4500,
   5000., 5500, 6000, 6500, 7000, 7500, 8000, 8500, 9000, 9500,
  10000.,10500,11000,11500,12000,12500,13000,13500,14000,15000,
  16000.,17000,18000,20000,25000,30000,40000,50000.])
_IC09_PM = torch.tensor([
   100., 100, 100,  99,  99, 100, 100,  99, 100, 101,
   103., 104, 106, 107, 108, 110, 115, 119, 122, 127,
   131., 132, 130, 125, 119, 117, 115, 113, 112, 107,
   100.,  95,  90,  80,  60,  44,  20,   8.])

# NH Zone 2 post-1950 bomb curve (calendar year → pmC)
_BOMB_YRS = torch.tensor([
   1950., 1955, 1960, 1961, 1962, 1963, 1964, 1965, 1966, 1967,
   1968., 1969, 1970, 1972, 1975, 1978, 1981, 1984, 1987, 1990,
   1993., 1996, 1999, 2002, 2007, 2010, 2015, 2018, 2020.])
_BOMB_PM  = torch.tensor([
   100., 100, 100, 102, 110, 155, 194, 196, 186, 175,
   168., 163, 158, 152, 143, 134, 126, 120, 115, 112,
   109., 107, 107, 107, 107, 106, 105, 104, 104.])


def _interp1d(t: torch.Tensor,
              xs: torch.Tensor,
              ys: torch.Tensor) -> torch.Tensor:
    lo = torch.zeros_like(t, dtype=torch.long)
    for i in range(len(xs) - 1):
        lo = torch.where(t >= xs[i], torch.full_like(lo, i), lo)
    hi = torch.clamp(lo + 1, 0, len(ys) - 1)
    dx = xs[hi] - xs[lo]
    w  = torch.clamp((t - xs[lo]) / torch.clamp(dx, min=0.01), 0., 1.)
    return ys[lo] * (1 - w) + ys[hi] * w


def input_14C(t: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """
    ¹⁴C atmospheric input (pmC).

    Uses IntCal09 for pre-bomb years (before 1950) and the NH Zone 2
    bomb curve for post-1950 years.

    Parameters
    ----------
    t     : calendar year(s)
    scale : dead-carbon correction factor (typically 0.40–0.90 for
            carbonate aquifers; set from the ¹⁴C of Modern groundwater)

    Returns
    -------
    ¹⁴C atmospheric input (pmC)

    Notes
    -----
    For new samples, provide the measured pmC as the JSON obs value
    (no pre-correction required).  Set the scale factor to match the
    ¹⁴C content of Modern groundwater in the study area, which accounts
    for dead-carbon dilution from dissolution of ¹⁴C-free carbonate minerals.
    """
    age_bp = torch.clamp(1950. - t, min=0.)
    ic09   = _interp1d(age_bp, _IC09_BP, _IC09_PM)
    bomb   = _interp1d(
        torch.clamp(t, float(_BOMB_YRS[0]), float(_BOMB_YRS[-1])),
        _BOMB_YRS, _BOMB_PM)
    return torch.where(t >= 1950., bomb, ic09) * scale


# ══════════════════════════════════════════════════════════════════════════════
# ⁴He — radiogenic helium
# ══════════════════════════════════════════════════════════════════════════════

def input_4He(ages: torch.Tensor, he4_rate: float) -> torch.Tensor:
    """
    Radiogenic ⁴He accumulation (cc STP g⁻¹).

    Assumes linear accumulation at a constant rate β derived from the
    U/Th content, matrix porosity, and bulk density of the aquifer.

    Parameters
    ----------
    ages     : age grid (yr)
    he4_rate : accumulation rate β (cc STP g⁻¹ yr⁻¹);
               compute from U_ppm, Th_ppm, porosity, bulk_density_gcm3
               using the DGMETA formula (Jurgens et al., 2020)

    Returns
    -------
    ⁴He concentration (cc STP g⁻¹)

    Notes
    -----
    For new samples, provide the measured dissolved ⁴He (cc STP/g) as the
    JSON obs value; no pre-correction is required.
    """
    return he4_rate * ages
