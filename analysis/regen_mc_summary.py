#!/usr/bin/env python3
"""
regen_mc_summary.py
====================
Rebuild <out>/mc_summary.csv from existing <out>/draws/*.npz, filtering
non-finite rows before computing mean/std/p16/p84 -- run_mc_draws.py's
original summary code did not do this, so any sample with a draw that
converged to NaN (torch propagates NaN silently rather than raising) had every
summary statistic silently reported as NaN even though n_fail was 0 and the
draws themselves were fine. No refitting: the npz draws already on disk are
authoritative and untouched by this fix.

Usage
-----
  python regen_mc_summary.py --dir mc
"""
from __future__ import annotations
import argparse, glob, os
import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="mc")
    args = ap.parse_args()

    rows = []
    for path in sorted(glob.glob(os.path.join(args.dir, "draws", "*.npz"))):
        d = np.load(path, allow_pickle=True)
        draws, names = d["draws"], list(d["names"])
        theta_star, chi2 = d["theta_star"], float(d["chi2"])
        sid = os.path.splitext(os.path.basename(path))[0].replace("_", "/", 1) \
            if "/" not in os.path.basename(path) else os.path.basename(path)
        # safe filenames use "_" for "/"; recover the true SampleID by
        # matching against the mc_summary.csv this replaces, not by guessing
        rows.append((path, draws, names, theta_star, chi2))

    # Recover true SampleIDs from the existing summary (safe filename -> id
    # is not always invertible, e.g. compound IDs already contain no "_", but
    # the existing summary has the authoritative SampleID/LPM per npz file).
    old = pd.read_csv(os.path.join(args.dir, "mc_summary.csv"))
    safe_to_sid = {sid.replace("/", "_"): (sid, lpm)
                  for sid, lpm in zip(old["SampleID"], old["LPM"])}

    out_rows = []
    for path, draws, names, theta_star, chi2 in rows:
        safe = os.path.splitext(os.path.basename(path))[0]
        sid, lpm = safe_to_sid.get(safe, (safe, ""))
        finite = draws[np.isfinite(draws).all(axis=1)] if draws.shape[0] else draws
        n_nonfinite = int(draws.shape[0] - finite.shape[0])
        row = dict(SampleID=sid, LPM=lpm, n_draws=int(draws.shape[0]),
                   n_nonfinite=n_nonfinite, chi2=chi2)
        for j, n in enumerate(names):
            row[f"{n}_star"] = float(theta_star[j])
            if finite.shape[0]:
                c = finite[:, j]
                row[f"{n}_mean"] = float(c.mean())
                row[f"{n}_std"] = float(c.std())
                row[f"{n}_p16"] = float(np.percentile(c, 16))
                row[f"{n}_p84"] = float(np.percentile(c, 84))
        if finite.shape[0] > 2 and finite.shape[1] > 1:
            row["corr_p1_p2"] = float(np.corrcoef(finite.T)[0, 1])
        out_rows.append(row)

    # preserve n_fail from the old summary (fit-time failures, orthogonal to
    # post-hoc non-finite filtering)
    fail_map = dict(zip(old["SampleID"], old["n_fail"]))
    for row in out_rows:
        row["n_fail"] = fail_map.get(row["SampleID"], 0)

    df = pd.DataFrame(out_rows)
    path = os.path.join(args.dir, "mc_summary.csv")
    df.to_csv(path, index=False)
    print(f"-> {path}  ({len(df)} samples)")
    print(f"samples with non-finite draws: "
          f"{int((df['n_nonfinite'] > 0).sum())}/{len(df)}")
    print(df[df["n_nonfinite"] > 0][["SampleID", "n_draws", "n_nonfinite"]]
          .to_string(index=False))


if __name__ == "__main__":
    main()
