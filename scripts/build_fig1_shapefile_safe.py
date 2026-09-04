#!/usr/bin/env python3
"""
build_fig1_shapefile_safe.py -- shapefile/dBASE-safe variant of the Figure 1
attribute table.

Three constraints of the .shp (dBASE) format that the full CSV violates:

  1. SITE_NO is a 15-digit USGS site identifier (max 295406097551201), which
     exceeds 32-bit Long. dBASE has no 64-bit integer, so ArcGIS types it
     BigInteger and Export Features refuses it -- ERROR 002809. It is an
     identifier, not a quantity, so it is written as text here.
  2. Field names are capped at 10 characters and are silently truncated
     (and de-duplicated with junk suffixes) past that. All names below are
     <= 10 characters and unique.
  3. There is no boolean type; True/False would land as text. The two flags
     are written as 1/0 integers instead.

The `wkt` column is dropped: it is redundant once XY Table To Point has run,
and it is long text.

Use the full CSV (Fig1_sample_locations.csv) for a file geodatabase, which has
none of these limits. Use this one for .shp.
"""
from __future__ import annotations
import os
import pandas as pd

FC = r"D:\projects\carbon isotopic\DLMPI_reanalysis\From_Claude_chat"
SRC = os.path.join(FC, "supplementary_data", "Fig1_sample_locations.csv")
# the .shp variant is a GIS convenience for producing Figure 1, not an SI
# deliverable, so it stays alongside the figure assets
OUT = os.path.join(FC, "figures", "Fig1_sample_locations_shp.csv")

d = pd.read_csv(SRC)

# 15-digit identifier -> text, so dBASE can hold it
d["SITE_NO"] = d["SITE_NO"].astype("Int64").astype(str)

# no boolean type in dBASE
for c in ("bound_active", "interval_at_bound"):
    d[c] = d[c].fillna(False).astype(bool).astype(int)

RENAME = {
    "shared_location":     "SharedLoc",
    "coord_source":        "CoordSrc",
    "identifiability":     "IdentClass",   # <- symbolise colour on this
    "ident_rank":          "IdentRank",
    "bound_active":        "BndActive",
    "interval_at_bound":   "IntAtBnd",
    "max_abs_corr":        "MaxAbsCorr",
    "projected_grad_norm": "ProjGrad",
    "mean_age_mixture":    "MeanAgeMix",
    "F_lt25yr":            "F_lt25yr",
    "F_lt25yr_p16":        "F25_p16",
    "F_lt25yr_p84":        "F25_p84",
    "F_lt40yr":            "F_lt40yr",
    "vs_assigned":         "VsAssigned",
    "label_tau1":          "LblTau1",
}
d = d.drop(columns=["wkt"]).rename(columns=RENAME)

over = [c for c in d.columns if len(c) > 10]
assert not over, over
assert len(set(d.columns)) == len(d.columns), "duplicate field name"
assert len(d) == 60

d.to_csv(OUT, index=False)
print("wrote", OUT)
print(f"{len(d)} rows, {len(d.columns)} fields, all names <= 10 chars\n")
print("field name mapping (full CSV -> shapefile-safe):")
for k, v in RENAME.items():
    if k != v:
        print(f"   {k:22s} -> {v}")
print("\nfields:", list(d.columns))
print("\nIdentClass:", d.IdentClass.value_counts().to_dict())
print("IntAtBnd = 1 for", int(d.IntAtBnd.sum()), "samples")
