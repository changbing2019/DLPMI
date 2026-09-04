# Figure 1 attribute table — field notes

**File:** `Fig1_sample_locations.csv` — 60 rows, one per retained sample.
**CRS:** EPSG:4326 (WGS84 decimal degrees). Import on `longitude` / `latitude`,
or use the `wkt` column directly (`POINT (lon lat)`).

Supersedes `Fig1_sample_locations_SUPERSEDED.csv`, which carried pre-dgmeta-fix
conditional classes. **17 of 60 samples have a different identifiability class
between the two files** — do not mix them.

## Symbolise on this

| field | use |
|---|---|
| `identifiability` | **primary symbol colour.** Three classes, in order: `well constrained`, `weakly constrained`, `unconstrained` |
| `ident_rank` | same thing as 1/2/3, for ordered ramps or sorting |
| `Aq_class` | **symbol shape.** `unconfined` (recharge) vs `confined` (artesian) |

Class counts, for checking the legend against the manuscript:

| class | n | unconfined | confined |
|---|---|---|---|
| well constrained | 8 | 2 | 6 |
| weakly constrained | 16 | 11 | 5 |
| unconstrained | 36 | 15 | 21 |

Suggested colours (Okabe–Ito, colourblind-safe, distinguishable in greyscale —
matches Figures 2 and 3 so the panels read as one set):

- well constrained `#0072B2` (blue)
- weakly constrained `#E69F00` (amber)
- unconstrained `#D55E00` (vermillion)

## Optional labels

`label_tau1` is τ₁ rounded to 0.1 yr, if you want the old Figure 4 style of
labelling each point with its transit time. Note the previous version of that
map labelled values up to 48 yr; the current range is **3.0–42.0 yr**.

## Fields

**Geography** (unchanged from the superseded file — site metadata, unaffected by
any inversion rerun)
`SampleID`, `shared_location`, `SITE_NO`, `latitude`, `longitude`,
`coord_source`, `Depth_m`, `County`, `wkt`

- `coord_source`: 27 samples agree with the published coordinate column; the
  other 33 are decoded from `SITE_NO` as DDMMSSDDDMMSSNN, validated to within
  55 m against those 27.
- `Depth_m` is missing for 7 of 60 wells.

**Classification and transit time** (regenerated, true profile likelihood)
`Aq_class`, `Well_class`, `Aq_seg`, `LPM`, `n_active`, `tau1`, `tau1_lo`,
`tau1_hi`, `w_tau1`, `identifiability`, `ident_rank`, `label_tau1`

- `tau1_lo` / `tau1_hi` are the 68% **profile-likelihood** bounds with the
  nuisance parameter re-optimised at each step, and `w_tau1 = (hi − lo)/τ₁` is
  the relative width the classification thresholds are applied to
  (≤ 0.5 well, 0.5–1.5 weakly, > 1.5 unconstrained).
- `w_tau1`, `tau1_lo` and `tau1_hi` are blank for **9** samples whose
  free-parameter set fixes τ₁ (`free_params = "Fraction, 2nd Mean Age"`), so no
  τ₁ profile exists. All 9 are classified `unconstrained`. Symbolise them like
  any other unconstrained sample; just don't try to draw an interval for them.

**Diagnostic flags** — useful if you want a second symbol layer, not required
`bound_active`, `interval_at_bound`, `max_abs_corr`, `projected_grad_norm`

- `interval_at_bound` is TRUE for **21 of 60** samples, where the profile
  reached a prescribed optimisation bound before Δχ² = 1 was crossed, so the
  interval is limited by the bound rather than by the likelihood. These are
  classified `unconstrained` on that basis. Worth a hollow ring or hatch if you
  want to distinguish "unconstrained because wide" from "unconstrained because
  bounded".
- `max_abs_corr` > 0.8 in 9 samples, where only a combination of the two free
  parameters is determined.

**Derived quantities, for context or alternative maps**
`f1`, `tau2`, `mean_age_mixture`, `F_lt25yr`, `F_lt25yr_p16`, `F_lt25yr_p84`,
`F_lt40yr`, `dAICc`, `vs_assigned`

- `mean_age_mixture` is the quantity comparable to Musgrove's published mean
  age (3–18,136 yr), **not** `tau1`.
- `F_lt25yr` with its P16–P84 bounds is the young-water fraction; the §5.1
  argument is that this is the better vulnerability metric in the recharge zone.
- `vs_assigned` is the model-structure verdict: `supported` 16,
  `contradicted` 16, `indistinguishable` 12, and the literal string
  `not assessable` for the 16 samples with nₐ < 4 (6 with two active tracers,
  10 with three). It is never blank, so filter on the string rather than on
  null if you map this field.
