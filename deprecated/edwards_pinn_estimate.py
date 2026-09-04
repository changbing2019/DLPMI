"""DEPRECATED -- THIS SCRIPT DOES NOT RUN. Retained for reference only.

Superseded by examples/edwards_aquifer/edwards_run.py, which is what produced
every published result. This file is kept because earlier versions of the
manuscript referred to it, and removing it silently would leave that reference
dangling.

Known defects, none fixed:
  * expects sample_date as a '%Y-%m-%d' string; the configuration holds a float
  * reads bounds from lpm['bounds']; they live in pinn_bounds at sample level,
    so bounds silently fall back to defaults of 0.5-100000 yr
  * reads dic_c1 at sample level; it lives under carbon14_params
  * hardcodes n_free = 3 for the binary mixture, reintroducing the
    degrees-of-freedom error that the token-based free-parameter fix removed
  * its Hessian block is a `pass` stub

Use edwards_run.py.
"""

"""
edwards_pinn_estimate.py  v1.3.0
================================
Run DLPMI on all 65 Edwards Aquifer samples using Jurgens-calibrated
input functions (H3, SF6, 14C from tracer_input_curves.csv).

Usage:
    python edwards_pinn_estimate.py
    python edwards_pinn_estimate.py --json path/to/edwards_input_config.json
                                     --out  path/to/results/
"""

import argparse, os, json, csv
import torch
import numpy as np
import pandas as pd
from scipy import stats

# ── Add DLPMI package to path ─────────────────────────────────────────────────
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from dlpmi import Inverter, Sample
from dlpmi.tracers import H3_DEFAULT_COL
from dlpmi.uncertainty import hessian_uncertainty, chi2_probability

# ── Excluded samples (outside San Antonio Segment) ───────────────────────────
EXCLUDE = {'EDTRPAS1-40','EDTRPAS1-38','EDTRPAS1-39',
           'EDTRPAS1-48','EDTRPAS1-47'}

# ── Argument parsing ──────────────────────────────────────────────────────────
ap = argparse.ArgumentParser()
ap.add_argument('--json', default=
    os.path.join(os.path.dirname(__file__), '..', '..', '..',
                 'outputs', 'edwards_input_config.json'))
ap.add_argument('--out', default=
    os.path.join(os.path.dirname(__file__), '..', '..', '..',
                 'outputs'))
args = ap.parse_args()

os.makedirs(args.out, exist_ok=True)

with open(args.json) as f:
    cfg_data = json.load(f)['samples']

print(f"Loaded {len(cfg_data)} samples from JSON")
print(f"Excluding {len(EXCLUDE)} out-of-boundary samples")
print(f"Output directory: {args.out}")
print()

results = []

for s_cfg in cfg_data:
    sid = s_cfg['sample_id']
    if sid in EXCLUDE:
        continue

    lpm        = s_cfg['lpm']
    lpm_name   = lpm['name']
    free_params= lpm.get('free_params', 'Mean Age, Fraction')
    h3_col     = lpm.get('h3_source_col', H3_DEFAULT_COL)
    init       = lpm['init']
    res_lpm    = lpm['tracerlpm_result']

    # ── Build dgmeta with Jurgens H3 source ──────────────────────────────────
    dg = s_cfg.get('dgmeta', {})
    dg['h3_source'] = h3_col
    dg['h3_scale']  = 1.0

    # ── Parse sample date ─────────────────────────────────────────────────────
    date_str = s_cfg['sample_date']
    from datetime import datetime
    dt = datetime.strptime(date_str.split()[0], '%Y-%m-%d')
    sample_date = dt.year + (dt.timetuple().tm_yday - 1) / 365.25

    # ── Build tracer list ─────────────────────────────────────────────────────
    tracer_obs = s_cfg.get('tracers', [])
    tracers_in = []
    for tr in tracer_obs:
        if tr.get('scale', 0) > 0:
            tracers_in.append({
                'name':    tr['name'],
                'obs':     float(tr['obs']),
                'obs_err': float(tr.get('obs_err', abs(float(tr['obs']))*0.10)),
                'scale':   float(tr['scale']),
                'active':  True,
            })

    if not tracers_in:
        print(f"  SKIP {sid} — no active tracers")
        continue

    # ── Build Sample ──────────────────────────────────────────────────────────
    bounds = lpm.get('bounds', {})

    if lpm_name == 'DM':
        sample = Sample(
            sample_id   = sid,
            sample_date = sample_date,
            tracers     = tracers_in,
            model       = 'DM',
            bounds      = bounds,
            dgmeta      = dg,
            age_category= s_cfg.get('age_category', ''),
        )
    else:  # BMM-DM-DM
        sample = Sample(
            sample_id   = sid,
            sample_date = sample_date,
            tracers     = tracers_in,
            model       = 'BMM',
            model1      = 'DM',
            model2      = 'DM',
            free_params = free_params,
            bounds      = bounds,
            dic_c1      = float(s_cfg.get('dic_c1', 100.)),
            dic_c2      = float(s_cfg.get('dic_c2', 100.)),
            uz_tt       = float(s_cfg.get('uz_tt', 0.)),
            dgmeta      = dg,
            age_category= s_cfg.get('age_category', ''),
        )

    # ── Run DLPMI ─────────────────────────────────────────────────────────────
    print(f"  Fitting {sid} ({lpm_name}, {len(tracers_in)} tracers, H3={h3_col})")
    try:
        inv = Inverter(n_adam=5000, n_starts=3, verbose=False)
        res = inv.fit(sample)
    except Exception as e:
        print(f"    ERROR: {e}")
        continue

    # ── Chi-square probability ────────────────────────────────────────────────
    n_free = 3 if lpm_name == 'BMM-DM-DM' else 2
    dof    = max(len(tracers_in) - n_free, 1)
    prob_d = float(stats.chi2.sf(res['chi2'], dof))
    prob_l = float(stats.chi2.sf(
        float(res_lpm.get('chi2', np.nan) or np.nan), dof)) \
             if res_lpm.get('chi2') else np.nan

    # ── Hessian uncertainty ───────────────────────────────────────────────────
    sigma_tau1 = np.nan
    try:
        from dlpmi.uncertainty import hessian_uncertainty
        # build loss function for hessian
        pass  # simplified — full version in uncertainty module
    except Exception:
        pass

    row = {
        'SampleID':        sid,
        'AgeCat':          s_cfg.get('age_category',''),
        'LPM':             lpm_name,
        'H3_source':       h3_col,
        'n_active':        len(tracers_in),
        'dof':             dof,
        'tau1_DLPMI':      res['params'].get('tau', np.nan),
        'tau1_LPM':        res_lpm.get('tau1_yr', np.nan),
        'pd1_DLPMI':       res['params'].get('PD', np.nan),
        'f1_DLPMI':        res.get('f1', np.nan),
        'f1_LPM':          res_lpm.get('f1', np.nan),
        'chi2_DLPMI':      res['chi2'],
        'chi2_LPM':        res_lpm.get('chi2', np.nan),
        'chi2_prob_DLPMI': prob_d,
        'chi2_prob_LPM':   prob_l,
        'DLPMI_better':    'Yes' if res['chi2'] <= float(res_lpm.get('chi2', np.inf) or np.inf) else 'No',
    }

    # Add simulated tracer values
    for t, s_val in zip(res['tracer_names'], res['sim_vals']):
        row[f'sim_{t}_DLPMI'] = s_val

    results.append(row)
    print(f"    τ₁={row['tau1_DLPMI']:.2f} yr  χ²={res['chi2']:.5f}  p={prob_d:.4f}")

# ── Save results ──────────────────────────────────────────────────────────────
df_out = pd.DataFrame(results)
out_path = os.path.join(args.out, '00_summary_table_jurgens.csv')
df_out.to_csv(out_path, index=False)
print(f"\nResults saved: {out_path}")
print(f"Total samples: {len(df_out)}")
print(f"p>0.05 DLPMI: {(df_out['chi2_prob_DLPMI']>0.05).sum()}/{len(df_out)}")
print(f"DLPMI better: {(df_out['DLPMI_better']=='Yes').sum()}/{len(df_out)}")
