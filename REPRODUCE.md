# Reproducing the published results

Every number, table and figure in *Information Content and Identifiability of
Groundwater Ages Estimated from Environmental Tracers in a Karst Aquifer*
(Environmental Science & Technology) is produced by the commands below, in this
order, and checked by a single harness at the end.

Run everything from this directory unless stated otherwise. On Windows set
`PYTHONIOENCODING=utf-8` first; several scripts print Unicode (τ, χ², ³He).

---

## 0. What you need

The observational data are public: Musgrove et al. (2023),
<https://doi.org/10.5066/P9CWM574>, expected at `../Musgrove_2023Data/`.

The inversion reads `examples/edwards_aquifer/edwards_input_config.json`,
distributed with this code. It carries the per-sample tracer concentrations and
their standard errors, the prescribed parameters, the parameter bounds and the
dissolved-gas metadata.

**Every tracer value and every uncertainty in it comes verbatim from the data
release** — `Table_4_LPM.txt`, columns `LPM_Meas_Tracer_NN`,
`LPM_Meas_Tracer_NN_Err`, `LPM_Tracer_Name_NN` and `LPM_ScaleFact_NN`, which
hold the values Musgrove used as inputs to their own TracerLPM inversion. All
**218 of 218** active tracer–sample observations and all **218 of 218**
standard errors match exactly (checked by the harness, section `[prov]`).
Nothing is derived by an undocumented transformation, and nothing is invented.
Note that the file is UTF-16.

Two cautions if you rebuild the config yourself:

- Use **Table 4**, not Tables 3 and 5. Tables 3 and 5 hold the *analytical*
  measurements, which differ: ³H and ¹⁴C agree, but SF₆ matches for only 20 of
  54 samples and ³He(trit) for 15 of 44, and the analytical 1σ values differ
  from the inversion's by factors of 0.5 to 41 by tracer (Table S11). Both sets
  are legitimate; they answer different questions, and the fits use Table 4 so
  that they remain comparable with the published ones.
- The tracer naming convention (`3H`, `3He(trit)`, `SF6`, `14C`, `4He`) is
  Table 4's, and `LPM_ScaleFact_NN` supplies the per-tracer weight that
  determines which tracers are active for each sample.

`Table_1_Sites.xlsx` is derived locally from
`Musgrove_2023Data/Table_1_Siteswsum&ages.txt` via pandas (same columns:
`SampleID`, `Aq_class`, `Aq_seg`, `Well_class`). It supplies only the zone
merge, so any step taking `--sites-xlsx` can be run without it at the cost of
the zone-stratified summaries.

## 1. Environment, and the check that must pass first

```bash
pip install -r requirements.txt
python verify_setup.py
```

`verify_setup.py` **must** print `tau1 = 29.57 yr` and `chi2 = 0.045` for
EDTRPAS1-36. If it does not, the package tree or the numerical environment
differs from the one the published results came from and nothing below is
comparable — the usual causes are a stale `dlpmi/params.py` (it must use
token-based free-parameter matching, not substring) or a missing
`dlpmi/data/tracer_input_curves.csv`.

Fitting is deterministic: repeated identical calls return bit-identical
parameters and χ² within a fixed environment. Low-order digits are not
guaranteed across torch versions, so `requirements.txt` records the exact
versions used.

## 2. Baseline fits

`reference/00_summary_table.csv` holds the baseline optimum for all 60 samples
and is the starting point for everything downstream. It was produced by
`examples/edwards_aquifer/edwards_run.py`. Do **not** use
`edwards_pinn_estimate.py`; it is retained for reference only and will not run.

## 3. Monte Carlo, with joint draws

```bash
python run_mc_draws.py --n-mc 500 --processes 28 --out mc
```

Writes `mc/draws/<SampleID>.npz` (aligned 500×2 arrays) and `mc/mc_summary.csv`.
The joint draws matter: `edwards_run.py::_compute_mc` accumulates each parameter
into a separate list and returns only per-parameter percentiles, which destroys
the correspondence needed for any quantity depending on more than one parameter
(the young-water fraction, the mixture mean age). Measured correlation between
free parameters is 0.4–0.7, so this is not negligible.

## 4. Identifiability, information content and model structure

```bash
python dlpmi_sweeps.py --sweeps ident,loto,structure \
    --baseline-csv reference/00_summary_table.csv --processes 28 \
    --sites-xlsx ../Musgrove_2023Data/Table_1_Sites.xlsx --out sweeps
```

Also required, under the measurement-error-weighted objective, for the AICc
model selection of Section 3.4:

```bash
python dlpmi_sweeps.py --sweeps structure --objective sigma \
    --baseline-csv reference/00_summary_table.csv --processes 28 \
    --sites-xlsx ../Musgrove_2023Data/Table_1_Sites.xlsx --out sweeps_sigma
```

`sweeps_sigma/sweep_structure.csv` is the file the manuscript reports, copied to
`From_Claude_chat/data_current/sweep_structure_sigma.csv`. `--sites-xlsx` is
required, not optional: without it the `Aq_class` column is absent and the
per-zone rows of Table 4 and the two panels of Figure 5 cannot be built.

It classifies **54 of 60** samples, those retaining at least one degree of
freedom. Both structures carry k = 2, so the AICc correction term is identical
and cancels in ΔAICc, which therefore equals Δχ²σ — the two agree to 1.4 ×
10⁻¹⁴ wherever AICc itself is defined. Runs before 2026-08-24 reported only 44,
having required nₐ ≥ 4 so that `aicc()` returned a number for each structure
separately; that discarded the ten three-tracer samples on the strength of a
correction that cancels. The six two-tracer samples remain excluded because
both structures are saturated there (nₐ = k), so a misfit difference says
nothing about structure.

The error-model sensitivity of Text S6 needs two more runs. A *uniform*
rescaling of σ leaves the optimum unchanged and divides χ²σ by the square of
the scale factor, so it is exact without refitting — compute it from the
columns of the run above. The reported analytical σ are per-tracer and span
roughly 0.2% to 30%, so they move the optimum and require a genuine refit:

```bash
python dlpmi_sweeps.py --sweeps structure --objective sigma \
    --baseline-csv reference/00_summary_table.csv --processes 28 \
    --sites-xlsx ../Musgrove_2023Data/Table_1_Sites.xlsx \
    --reported-sigma-dir ../Musgrove_2023Data --fit-sigma-source reported \
    --out sweeps_sigma_reported
```

Do **not** substitute `--reported-sigma-dir` alone for this. That flag only adds
`chi2sig_rep_*`, the analytical-σ misfit evaluated at the *inversion*-σ optimum,
which is not an approximation to the analytical-σ fit: reweighting residuals by
factors of ~40 between tracers relocates the optimum. Evaluating rather than
refitting inflates single-component support badly (24 of 36 against the refit).

For the objective-function sensitivity of Table S10, the same sweep under the
relative-error objective:

```bash
python dlpmi_sweeps.py --sweeps structure --objective rel \
    --baseline-csv reference/00_summary_table.csv --processes 28 \
    --sites-xlsx ../Musgrove_2023Data/Table_1_Sites.xlsx --out sweeps_rel
```

Then pair the two rankings — no refitting, it only joins the two CSVs:

```bash
python ../From_Claude_chat/scripts/make_objective_sensitivity.py
```

That writes `From_Claude_chat/supplementary_data/objective_sensitivity.csv` and
prints the flip list. Two of the 54 change classification between objectives
(EDTRPAS1-28 and EDTRPAS1-50); the SI listed one of 44 before 2026-08-24.

`sweeps_final/sweep_identifiability.csv` carries the Hessian diagnostics,
scaled condition numbers and active-bound columns used downstream. See
`CLAUDE.md` for how that file was built up over several passes; it is the
`--base` input to step 6.

## 5. True profile-likelihood intervals

`dlpmi_sweeps.profile_width` holds every nuisance parameter at its optimum — a
*conditional* χ² interval, not a profile. These are the reported intervals:

```bash
python profile_true.py --mode ident --all --out sweeps_trueprofile
python profile_true.py --mode loto  --all --out sweeps_trueprofile
```

The `loto` mode reads `sweeps/sweep_loto.csv` from step 4, so run step 4 first.

## 6. Assemble the identifiability table

```bash
python rebuild_ident_trueprofile.py \
    --profile sweeps_trueprofile/profile_true.csv \
    --base    sweeps_final/sweep_identifiability.csv \
    --out     sweeps_trueprofile/sweep_identifiability_trueprofile.csv
```

This file is the single source for the identifiability results and is read by
almost every step below. Verified 2026-08-23: this chain reproduces the
published classification exactly — 8 well constrained / 16 weakly / 36
unconstrained — and 43 of 46 shared columns bit-for-bit, the other three
differing only at the 6th–7th decimal from a nuisance-grid fix that does not
move any reported figure.

## 7. The older component, the parameterization swap, and mean ages

```bash
python profile_tau2.py --all --out sweeps_trueprofile
python refit_free_tau1.py     --out sweeps_trueprofile
python mean_age_uncertainty.py --out sweeps_trueprofile
```

Then the τ₂-free Monte Carlo used for the mean-age uncertainty:

```bash
python make_tau2free_config.py
```

That writes `examples/edwards_aquifer/edwards_input_config_tau2free.json`,
switching the mixtures whose τ₂ is prescribed into TracerLPM mode C (τ₂ free,
τ₁ pinned at its currently reported value), and **prints the sample ids it
changed**. Pass exactly those to the Monte Carlo — the other samples are
unchanged and do not need redrawing:

```bash
python run_mc_draws.py --n-mc 500 --processes 28 \
    --config examples/edwards_aquifer/edwards_input_config_tau2free.json \
    --ids <the ids printed above> \
    --out mc_tau2free
python recompute_mean_age_sigma.py --mc-orig mc --mc-tau2free mc_tau2free \
    --out sweeps_trueprofile
```

`recompute_mean_age_sigma.py` writes `mean_age_sigma_recomputed.csv`, holding
the delta-method, Monte Carlo and τ₂-free Monte Carlo standard deviations side
by side. They are three different estimators and must not be conflated: the
shipped `mean_age_sigma` is the first-order delta method off the Hessian
covariance, not Monte Carlo.

## 8. Young-water fraction and mixture mean ages

```bash
python run_young_fraction.py \
    --baseline-csv reference/00_summary_table.csv \
    --config examples/edwards_aquifer/edwards_input_config.json \
    --mc-dir mc --mc-dir-tau2free mc_tau2free \
    --sites-xlsx ../Musgrove_2023Data/Table_1_Sites.xlsx \
    --out sweeps/young_fraction.csv
```

## 9. Sensitivity analyses

```bash
python c14_sensitivity.py     --out sweeps_trueprofile   # 14C dead-carbon
python profile_sigma.py --all --out sweeps_trueprofile   # tau1 under chi2_sigma
python profile_tau2_sigma.py  --out sweeps_trueprofile   # tau2 under chi2_sigma
python sigma_impact.py                                   # assumed vs reported sigma
```

## 10. Independent validation (optional but recommended)

```bash
python validate_profile.py
```

Recomputes the τ₁ profile on a dense 240-point grid and reads the Δχ² crossings
off it directly — a different algorithm against the same objective, so a
disagreement localises a bug rather than reproducing it. Agreement should be
within about 0.1% of τ₁, and every profile curve should be monotonic away from
the optimum.

## 11. Figures, tables, and the harness

From the repository root:

```bash
python From_Claude_chat/scripts/regen_trueprofile.py
python From_Claude_chat/scripts/verify_manuscript_numbers.py
```

`regen_trueprofile.py` rebuilds Figures 2, 3, 4 and 8 and Tables 2 and 3 from
the true-profile outputs. It seeds `numpy.random` so Figure 4's jitter is
reproducible.

`verify_manuscript_numbers.py` is the acceptance test. It restates **266**
quantitative claims from the main text and the Supporting Information as
assertions against the file each should come from, checks Tables 2, 3, S10 and
S11 cell by cell, confirms all seven embedded figures are byte-identical to
their `figures/*.png` sources, and fails if a retired value or a retired form of
words reappears. It must print `266 of 266 claims reproduce` and exit 0.

Run it after any rerun of the inversion and before submission. If a check
fails, one of the two ends is stale — find out which before editing either.

---

## Known limits on reproducibility

These are stated in the manuscript and are not defects in the code:

- **The tracer uncertainties are those of the published inversion** (Table 4 of
  the release), not the analytical ones tabulated separately in Tables 3 and 5
  (Table S11). They are retained for comparability with the published fits, but
  they set the scale of the weighted objective, so the AICc model-selection
  counts of Section 3.4 are conditional on that choice: single-component support
  is 2 of 30 under the inherited error model and 9 of 30 under the analytical
  uncertainties. The direction is stable at every scale tested; the counts are
  not.
- **Δχ² = 1 on the relative-error objective is operational, not calibrated.**
  That objective is not proportional to a Gaussian log-likelihood. Section 3.2
  and Text S3 give the consequences.
- **Multi-start uses three fixed seeds per sample.** `_seed_tau`/`_seed_f` are
  hardcoded three-element lists, so `--starts 10` silently reuses the same
  three. Global optimality is not claimed. Note that
  `diagnose_nonconvergence.py`'s seed-base workaround is only meaningful where
  τ₁ is free; applied to a prescribed-τ₁ sample it sweeps the prescribed value
  and returns different *models*, not different basins.
- **Twenty-one of the 51 defined τ₁ intervals never close** inside the search
  window. Those are reported as unconstrained and excluded from every median
  width; the no-crossing fallback value is a sentinel, not a measurement.
- **Nine of the 60 samples never had τ₁ as a free parameter** — the source
  parameterization prescribes it. They are unconstrained by construction, and
  both denominators (60 and 51) are reported.
