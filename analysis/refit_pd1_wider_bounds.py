#!/usr/bin/env python3
"""
refit_pd1_wider_bounds.py
==========================
TASK_active_bounds.md: pd1_lo=0.001 was determined to be a DLPMI package
default, not a bound inherited from TracerLPM/Musgrove's published analysis.
Per the task's decision tree, refit the pd1-bound-active samples (from
sweeps_final/sweep_identifiability.csv) with pd1_lo lowered by two orders of
magnitude (0.001 -> 1e-5) and report whether they find an interior optimum at
smaller P_D or run to zero (which would degenerate the dispersion model
toward piston flow).

Reuses edwards_run._run_fit via dlpmi_sweeps.run_fit -- same bounds-derivation
convention, seeds, optimiser schedule, and free-parameter tokenisation as the
baseline, with only pinn_bounds.pd1_lo changed. Nothing else about the sample
record is modified.

Usage
-----
  python refit_pd1_wider_bounds.py \
      --ident-csv sweeps_final/sweep_identifiability.csv \
      --baseline-csv reference/00_summary_table.csv \
      --out sweeps_final/pd1_refit_wider_bounds.csv
"""
from __future__ import annotations
import argparse, copy, json, os, sys, warnings
import numpy as np

warnings.filterwarnings("ignore")

NEW_PD1_LO = 1e-5  # two orders of magnitude below the DLPMI default 0.001


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--package", default=".")
    ap.add_argument("--config", default=None)
    ap.add_argument("--ident-csv", required=True)
    ap.add_argument("--baseline-csv", required=True)
    ap.add_argument("--adam", type=int, default=5000)
    ap.add_argument("--starts", type=int, default=3)
    ap.add_argument("--out", default="pd1_refit_wider_bounds.csv")
    ap.add_argument("--ids", nargs="*", default=None)
    args = ap.parse_args()

    pkg = os.path.abspath(args.package)
    for p in (pkg, os.path.join(pkg, "examples", "edwards_aquifer")):
        if p not in sys.path:
            sys.path.insert(0, p)

    import dlpmi_sweeps as sw
    sw.bootstrap(pkg)

    import pandas as pd
    ident = pd.read_csv(args.ident_csv)
    active = ident[(ident["bound_active"]) &
                   (ident["bound_active_params"].str.contains("pd1", na=False))]
    sids = active["SampleID"].tolist()
    if args.ids:
        sids = [s for s in sids if s in set(args.ids)]
    print(f"Refitting {len(sids)} pd1-bound-active samples with "
          f"pd1_lo={NEW_PD1_LO} (was 0.001)", flush=True)

    base = pd.read_csv(args.baseline_csv).set_index("SampleID")
    cfg = args.config or os.path.join(pkg, "examples", "edwards_aquifer",
                                       "edwards_input_config.json")
    recs = {r["sample_id"]: r for r in json.load(open(cfg))["samples"]}

    rows = []
    for i, sid in enumerate(sids, 1):
        rec = copy.deepcopy(recs[sid])
        rec["pinn_bounds"]["pd1_lo"] = NEW_PD1_LO
        u = sw.unpack(rec)
        try:
            params, c2r, sims = sw.run_fit(rec, u["names"], u["obs"],
                                           u["scales"], args.adam, args.starts)
        except Exception as e:                                   # noqa: BLE001
            rows.append(dict(SampleID=sid, error=str(e)[:150]))
            print(f"  [{i}/{len(sids)}] {sid}: ERROR {e}", flush=True)
            continue

        old_pd1 = float(base.loc[sid, "pd1_PINN"])
        new_pd1 = float(params["pd1"])
        old_tau1 = float(base.loc[sid, "tau1_PINN"])
        new_tau1 = float(params["tau1"])
        # "runs to zero": pd1 drops by >=1 order of magnitude from the old
        # bound-pinned value and does not simply re-pin at the new floor
        at_new_floor = abs(new_pd1 - NEW_PD1_LO) / NEW_PD1_LO < 0.01
        runs_to_zero = new_pd1 < old_pd1 * 0.1 and not at_new_floor
        interior = not at_new_floor and new_pd1 >= old_pd1 * 0.1

        rows.append(dict(
            SampleID=sid, old_pd1=old_pd1, new_pd1=new_pd1,
            pd1_ratio=new_pd1 / old_pd1 if old_pd1 else np.nan,
            old_tau1=old_tau1, new_tau1=new_tau1,
            tau1_shift_rel=abs(new_tau1 - old_tau1) / old_tau1,
            old_chi2=float(base.loc[sid, "chi2_PINN"]), new_chi2=c2r,
            at_new_floor=at_new_floor, runs_to_zero=runs_to_zero,
            interior=interior))
        print(f"  [{i}/{len(sids)}] {sid}: pd1 {old_pd1:.5g} -> {new_pd1:.5g}  "
              f"tau1 {old_tau1:.3f} -> {new_tau1:.3f}  "
              f"{'AT NEW FLOOR' if at_new_floor else ('RUNS TO ZERO' if runs_to_zero else 'interior')}",
              flush=True)

    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    df.to_csv(args.out, index=False)
    print(f"\n-> {args.out}  ({len(df)} samples)")

    if "at_new_floor" in df:
        print(f"\nat new floor (still pinned, {NEW_PD1_LO}): "
              f"{int(df['at_new_floor'].sum())}/{len(df)}")
        print(f"runs to zero (pd1 drops >=10x, not re-pinned): "
              f"{int(df['runs_to_zero'].sum())}/{len(df)}")
        print(f"interior (pd1 settles at a value >=10% of old, off both bounds): "
              f"{int(df['interior'].sum())}/{len(df)}")
        print(f"\nmedian tau1 shift: {df['tau1_shift_rel'].median():.4f}")
        print(f"max tau1 shift: {df['tau1_shift_rel'].max():.4f}")


if __name__ == "__main__":
    main()
