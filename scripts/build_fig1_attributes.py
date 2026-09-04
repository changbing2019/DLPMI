#!/usr/bin/env python3
"""
build_fig1_attributes.py -- attribute table for Figure 1 (sample locations
symbolised by transit-time identifiability).

Geography (SITE_NO, latitude, longitude, coord_source, Depth_m, County, wkt,
shared_location) is carried over verbatim from the superseded file: it is site
metadata and is unaffected by any of the inversion reruns. Every result column
is regenerated, and the identifiability class is the TRUE PROFILE-LIKELIHOOD
classification (8 well / 16 weakly / 36 unconstrained), not the superseded
conditional one (10 / 29 / 21).

Output: figures/Fig1_sample_locations.csv  (+ .prj-less WKT column for direct
import; EPSG:4326, WGS84 decimal degrees)
"""
from __future__ import annotations
import os
import pandas as pd

FC = r"D:\projects\carbon isotopic\DLMPI_reanalysis\From_Claude_chat"
HO = r"D:\projects\carbon isotopic\DLMPI_reanalysis\DLPMI_edwards_handoff"
TP = os.path.join(HO, "sweeps_trueprofile")

OLD = os.path.join(FC, "figures", "Fig1_sample_locations_SUPERSEDED.csv")
# the live attribute table is a Supporting Information data file, so it lives
# with the other SI datasets rather than in figures/
OUT = os.path.join(FC, "supplementary_data", "Fig1_sample_locations.csv")

old = pd.read_csv(OLD)
ident = pd.read_csv(os.path.join(TP, "sweep_identifiability_trueprofile.csv"))
yf = pd.read_csv(os.path.join(FC, "data_current", "young_fraction.csv"))
st = pd.read_csv(os.path.join(FC, "data_current", "sweep_structure_sigma.csv"))

# ── static site metadata, reused verbatim ───────────────────────────────
STATIC = ["SampleID", "shared_location", "SITE_NO", "latitude", "longitude",
          "coord_source", "Depth_m", "County", "wkt"]
out = old[STATIC].copy()

# ── regenerated result columns ──────────────────────────────────────────
idc = ident.set_index("SampleID")
out = out.merge(
    idc[["Aq_class", "Well_class", "Aq_seg", "LPM", "n_active", "tau1",
         "tau1_lo_true", "tau1_hi_true", "w_tau1_true", "cls_true",
         "bound_active", "true_hit_bound", "max_abs_corr",
         "projected_grad_norm"]],
    on="SampleID", how="left")

out = out.rename(columns={
    "tau1_lo_true": "tau1_lo",
    "tau1_hi_true": "tau1_hi",
    "w_tau1_true": "w_tau1",
    "cls_true": "identifiability",
    "true_hit_bound": "interval_at_bound",
})

out = out.merge(
    yf.set_index("SampleID")[["f1", "tau2", "mean_age_mixture", "F_lt25yr",
                              "F_lt25yr_p16", "F_lt25yr_p84", "F_lt40yr"]],
    on="SampleID", how="left")

out = out.merge(st.set_index("SampleID")[["dAICc", "vs_assigned"]],
                on="SampleID", how="left")

# ── plotting conveniences ───────────────────────────────────────────────
# integer rank for ordered symbology / colour ramps in GIS
RANK = {"well constrained": 1, "weakly constrained": 2, "unconstrained": 3}
out["ident_rank"] = out["identifiability"].map(RANK)
out["label_tau1"] = out["tau1"].round(1)

# ── checks ──────────────────────────────────────────────────────────────
assert len(out) == 60, len(out)
assert out.latitude.notna().all() and out.longitude.notna().all()
assert set(out.Aq_seg.dropna().unique()) == {"San Antonio"}, out.Aq_seg.unique()
vc = out.identifiability.value_counts()
assert (vc.get("well constrained"), vc.get("weakly constrained"),
        vc.get("unconstrained")) == (8, 16, 36), vc.to_dict()

out.to_csv(OUT, index=False)
print("wrote", OUT)
print("rows:", len(out), " columns:", len(out.columns))
print("\nidentifiability (the symbolisation field):")
for k in ["well constrained", "weakly constrained", "unconstrained"]:
    g = out[out.identifiability == k]
    print(f"  {k:20s} n={len(g):2d}   unconfined {int((g.Aq_class=='unconfined').sum()):2d}"
          f"   confined {int((g.Aq_class=='confined').sum()):2d}")
print("\nchanged class vs the superseded file:")
chg = out[["SampleID", "identifiability"]].merge(
    old[["SampleID", "identifiability_final"]], on="SampleID")
chg = chg[chg.identifiability != chg.identifiability_final]
print(f"  {len(chg)} of 60 samples")
print("\ncolumns:", list(out.columns))
