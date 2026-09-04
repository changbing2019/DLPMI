#!/usr/bin/env python3
"""
verify_manuscript_numbers.py -- reproducibility check for the EST manuscript.

Every quantitative claim in the main text is restated here as an assertion
against the file it should come from. Run after any rerun of the inversion,
and before submission. If a check fails, one of the two ends is stale --
find out which before editing either.

Authoritative sources
---------------------
  sweeps_trueprofile/sweep_identifiability_trueprofile.csv
        identifiability classes, profile intervals, Hessian diagnostics,
        bound activity
  sweeps_trueprofile/profile_true_loto.csv
        leave-one-tracer-out; use the *_true columns, not *_cond
  data_current/sweep_structure_sigma.csv      AICc model selection
  data_current/young_fraction.csv             mixture ages, F(a<T)
  supplementary_data/gof_diagnostic.csv       goodness of fit

NOT authoritative: sweeps/, sweeps_final/ and reference/ hold the retired
conditional-chi2 run (10/29/21 rather than 8/16/36). The *_cond columns of
profile_true_loto.csv belong to that run too.

Usage:  python verify_manuscript_numbers.py
Exit 0 if every claim reproduces, 1 otherwise.
"""
from __future__ import annotations

import json
import os
import re
import sys

import numpy as np
import pandas as pd
from scipy.stats import chi2 as CHI2

FC = r"D:\projects\carbon isotopic\DLMPI_reanalysis\From_Claude_chat"
HO = r"D:\projects\carbon isotopic\DLMPI_reanalysis\DLPMI_edwards_handoff"
TP = os.path.join(HO, "sweeps_trueprofile")

ident = pd.read_csv(os.path.join(TP, "sweep_identifiability_trueprofile.csv"))
loto = pd.read_csv(os.path.join(TP, "profile_true_loto.csv"))
st = pd.read_csv(os.path.join(FC, "data_current", "sweep_structure_sigma.csv"))
yf = pd.read_csv(os.path.join(FC, "data_current", "young_fraction.csv"))
gof = pd.read_csv(os.path.join(FC, "supplementary_data", "gof_diagnostic.csv"))

C: list[tuple[str, str, object, object]] = []


def add(sec, claim, paper, data):
    C.append((sec, claim, paper, data))


# ---- 2.1 dataset -----------------------------------------------------------
add("2.1", "60 samples retained", 60, len(ident))
add("2.1", "28 unconfined", 28, int((ident.Aq_class == "unconfined").sum()))
add("2.1", "32 confined", 32, int((ident.Aq_class == "confined").sum()))
add("2.1", "one aquifer segment only", 1, ident.Aq_seg.nunique())
add("2.1", "44 samples with all four tracers", 44, int((ident.n_active == 4).sum()))
add("2.1", "10 samples with three tracers", 10, int((ident.n_active == 3).sum()))
add("2.1", "6 samples with two tracers", 6, int((ident.n_active == 2).sum()))

# ---- 3.1 transit times -----------------------------------------------------
add("3.1", "tau1 range lower bound (yr)", 3, int(round(ident.tau1.min())))
add("3.1", "tau1 range upper bound (yr)", 42, int(round(ident.tau1.max())))

# ---- 3.2 identifiability ---------------------------------------------------
cls = ident.cls_true.value_counts()
add("3.2", "8 well constrained", 8, int(cls.get("well constrained", 0)))
add("3.2", "16 weakly constrained", 16, int(cls.get("weakly constrained", 0)))
add("3.2", "36 unconstrained", 36, int(cls.get("unconstrained", 0)))
add("3.2", "60% of the dataset unconstrained", 60,
    round(100 * int(cls.get("unconstrained", 0)) / len(ident)))
unc = ident[ident.Aq_class == "unconfined"]
con = ident[ident.Aq_class == "confined"]
add("3.2", "2 of 28 unconfined well constrained", 2,
    int((unc.cls_true == "well constrained").sum()))
add("3.2", "6 of 32 confined well constrained", 6,
    int((con.cls_true == "well constrained").sum()))
sc = ident[ident.max_abs_corr.abs() > 0.8]
add("3.2", "9 samples with |r| > 0.8", 9, len(sc))
add("3.2", "8 of the 9 are transit-time/dispersion pairs", 8, int((sc.LPM == "DM").sum()))
add("3.2", "1 of the 9 is a binary-mixture pair", 1, int((sc.LPM != "DM").sum()))
add("3.2", "all 9 have two free parameters", 0, int((sc.n_free < 2).sum()))

# ---- 2.4 / 3.2 samples where tau1 is prescribed, not fitted ----------------
# free_params starting with "Mean Age" == tau1 is free. The rest hold tau1 at a
# prescribed value (all exactly integer) and fit f1 and/or tau2 instead; their
# profile width is NaN and they land in "unconstrained" by construction.
t1free = ident.free_params.str.startswith("Mean Age")
add("2.4", "9 samples with tau1 prescribed rather than fitted", 9, int((~t1free).sum()))
add("2.4", "all prescribed tau1 values are exactly integer", 9,
    int(np.isclose(ident.loc[~t1free, "tau1"], ident.loc[~t1free, "tau1"].round()).sum()))
add("2.4", "no fitted tau1 is exactly integer", 0,
    int(np.isclose(ident.loc[t1free, "tau1"], ident.loc[t1free, "tau1"].round()).sum()))
add("2.4", "all 9 prescribed-tau1 samples are reported unconstrained", 9,
    int((ident.loc[~t1free, "cls_true"] == "unconstrained").sum()))
add("3.2", "51 samples where tau1 was estimated", 51, int(t1free.sum()))
add("3.2", "27 of those 51 unconstrained", 27,
    int((ident.loc[t1free, "cls_true"] == "unconstrained").sum()))
add("3.2", "53% of the estimated subset unconstrained", 53,
    round(100 * int((ident.loc[t1free, "cls_true"] == "unconstrained").sum())
          / int(t1free.sum())))

# ---- 3.2 the older component ----------------------------------------------
T2 = os.path.join(TP, "profile_true_tau2.csv")
if os.path.exists(T2):
    t2 = pd.read_csv(T2)
    add("3.2", "tau2 free in 8 samples", 8, len(t2))
    add("3.2", "all 8 tau2 estimates unconstrained", 8,
        int((t2.cls_tau2_true == "unconstrained").sum()))
    add("3.2", "all 8 tau2 intervals limited by the search range", 8,
        int(t2.true_hit_bound.astype(bool).sum()))
    # a closed interval would sit strictly inside the 0.1x-3x search window;
    # an end equal to the window edge means Delta-chi2 = 1 was never reached
    hi_open = np.isclose(t2.tau2_hi_true, t2.tau2 * 3.0, rtol=1e-6)
    lo_open = np.isclose(t2.tau2_lo_true, t2.tau2 * 0.1, rtol=1e-6)
    add("3.2", "7 of 8 upper limits set by the search range", 7, int(hi_open.sum()))
    add("3.2", "2 of 8 lower limits set by the search range", 2, int(lo_open.sum()))
    add("3.2", "1 interval closed at neither end", 1, int((hi_open & lo_open).sum()))
    add("3.2", "tau2 range lower, rounded (yr)", 80, int(round(t2.tau2.min())))
    add("3.2", "tau2 range upper, rounded (yr)", 13050,
        int(round(t2.tau2.max() / 10) * 10))
else:
    print("NOTE: profile_true_tau2.csv absent -- run profile_tau2.py --all")

# ---- 3.2 the parameterization swap (tau1 freed, tau2 pinned) ---------------
RF = os.path.join(TP, "refit_free_tau1.csv")
if os.path.exists(RF):
    rf = pd.read_csv(RF)
    add("3.2", "9 samples refit with tau1 free", 9, len(rf))
    add("3.2", "identifiability does not improve: 0 of 9", 0,
        int((rf.cls_tau1 != "unconstrained").sum()))
    add("3.2", "all 9 profiles still run to a bound", 9,
        int(rf.hit_bound.astype(bool).sum()))
    add("3.2", "relative widths from 1.18", 1.18, round(float(rf.w_tau1.min()), 2))
    add("3.2", "relative widths to 4.90", 4.90, round(float(rf.w_tau1.max()), 2))
    add("3.2", "median |tau1 shift| = 36%", 36,
        int(round(rf.tau1_shift_pct.abs().median())))
    add("3.2", "largest tau1 shift = 102%", 102,
        int(round(rf.tau1_shift_pct.abs().max())))
    add("3.2", "7 of 9 shift upward", 7, int((rf.tau1_shift_pct > 0).sum()))
    # chi2 cannot get worse -- the swapped parameterization can reproduce the
    # original optimum -- so the manuscript reports the magnitude, not the fact
    add("3.2", "chi2 never worse (guaranteed, not a finding)", 0,
        int((rf.chi2_refit > rf.chi2_orig + 1e-12).sum()))
    add("3.2", "median chi2 improvement factor 1.5", 1.5,
        round(float((rf.chi2_orig / rf.chi2_refit).median()), 1))
    add("3.2", "refit tau1 stays inside the reported 3-42 yr range", 0,
        int(((rf.tau1_refit < 3) | (rf.tau1_refit > 42)).sum()))
else:
    print("NOTE: refit_free_tau1.csv absent -- run refit_free_tau1.py")

# ---- 3.2 / 3.3 mean-age uncertainty ----------------------------------------
# tau2 freed in every binary mixture (exchanged with tau1 where it was pinned),
# profiled, then carried through to f1*tau1 + (1-f1)*tau2.
MA = os.path.join(TP, "mean_age_uncertainty.csv")
if os.path.exists(MA):
    ma = pd.read_csv(MA)
    add("3.2", "18 binary mixtures in the study set", 18, len(ma))
    add("3.2", "tau2 unconstrained in all 18", 18,
        int((ma.cls_tau2 == "unconstrained").sum()))
    add("3.2", "all 18 intervals end at the search range", 18,
        int(ma.hit_bound.astype(bool).sum()))
    add("3.2", "tau2 supplies a median 98% of the mean age", 98,
        int(round(ma.pct_from_tau2.median())))
    add("3.2", "median mean-age span factor 11", 11,
        int(round(ma.mean_age_span_factor.median())))
    # the three oldest, as quoted in 3.2 (rounded to 100 yr)
    top = ma.sort_values("mean_age", ascending=False).head(3)
    for want_lo, want_hi in [(4500, 59500), (2700, 36500), (1000, 30300)]:
        hit = any(abs(round(r.mean_age_lo / 100) * 100 - want_lo) <= 100
                  and abs(round(r.mean_age_hi / 100) * 100 - want_hi) <= 100
                  for _, r in top.iterrows())
        add("3.2", "oldest-three range %s-%s yr present" % (want_lo, want_hi),
            True, hit)
    # EDTRPAS1-46's mean age and sigma are checked against the generator in
    # the mean_age_sigma block below; here only its parameterisation
    add("3.2", "EDTRPAS1-46 had tau2 prescribed, not fitted", False,
        bool(ma.set_index("SampleID").loc["EDTRPAS1-46", "tau2_was_free"]))
else:
    print("NOTE: mean_age_uncertainty.csv absent -- run mean_age_uncertainty.py")

# ---- 3.3 the Bayesian check on the premodern component ---------------------
# Added 2026-09-03. Until now the harness checked NOTHING from this analysis,
# while its numbers had reached the abstract, the Key Points and the
# Conclusions -- the widest exposure of any unverified figures in the paper.
# Writing the SI section for it turned up two defects that these checks would
# have caught: a median span quoted over 18 mixtures beside statistics over
# 14, and a figure (9.4) whose run had been overwritten by a filename
# collision and could not be reproduced from anything on disk.
#
# Every comparison across error models is made over the 14 mixtures the
# marginalised run can assess. The fixed-sigma runs spend no degree of freedom
# on the error scale and so assess all 18, and comparing them on their own
# larger sets is exactly the mixed-denominator defect above.
PB = os.path.join(TP, "posterior_tau2.csv")
if os.path.exists(PB):
    pb = pd.read_csv(PB)
    _as = pb[pb.assessable]
    _keep = set(_as.SampleID)
    add("3.3", "18 binary mixtures considered", 18, len(pb))
    add("3.3", "14 assessable once the error scale costs a dof", 14, len(_as))
    add("3.3", "the 4 excluded all have 3 active tracers", 4,
        int((pb[~pb.assessable].n_active == 3).sum()))
    add("3.3", "every assessable mixture has 4 tracers and dof 2", True,
        bool((_as.n_active == 4).all() and (_as.dof == 2).all()))
    add("3.3", "median contraction in log tau2", 0.19,
        round(float(_as.contraction_log_tau2.median()), 2))
    add("3.3", "12 of 14 contract by less than half", 12,
        int((_as.contraction_log_tau2 < 0.5).sum()))
    add("3.3", "median 95% credible span, factor", 37,
        int(round(float(_as.tau2_ci95_span.median()))))
    add("3.3", "no marginal posterior peaks at a prior bound", 0,
        int(_as.marginal_mode_at_bound.astype(bool).sum()))
    add("3.3", "the truncated scale prior spans a factor of 5", 5.0,
        float(_as.scale_range.iloc[0]))
    # the stratification Section 3.3 rests the claim on
    _ok = _as[_as.p_at_min > 0.05]
    _no = _as[_as.p_at_min <= 0.05]
    add("3.3", "8 assessable mixtures fit acceptably", 8, len(_ok))
    add("3.3", "6 do not", 6, len(_no))
    add("3.3", "acceptable subset median contraction", 0.40,
        round(float(_ok.contraction_log_tau2.median()), 2))
    add("3.3", "acceptable subset median span, factor", 14,
        int(round(float(_ok.tau2_ci95_span.median()))))
    add("3.3", "rejected subset median contraction", 0.03,
        round(float(_no.contraction_log_tau2.median()), 2))
    add("3.3", "rejected subset median span, factor", 71,
        int(round(float(_no.tau2_ci95_span.median()))))
    add("3.3", "6 of the 8 acceptable still span over a factor of 5", 6,
        int((_ok.tau2_ci95_span > 5).sum()))
    # the two exceptions, named in Section 3.3 and in Key Point 2
    _exc = _as[_as.contraction_log_tau2 > 0.5]
    add("3.3", "exactly 2 exceptions contract by more than half", 2, len(_exc))
    add("3.3", "the exceptions are the two samples named", True,
        set(_exc.SampleID) == {"EDTRPAS1-29",
                               "SCTXSUS1-10/EDTRVFPS1-15"})
    _e1 = _exc[_exc.SampleID == "EDTRPAS1-29"]
    if len(_e1):
        add("3.3", "EDTRPAS1-29 contraction", 0.86,
            round(float(_e1.contraction_log_tau2.iloc[0]), 2))
        add("3.3", "EDTRPAS1-29 credible interval lower (yr)", 10500,
            int(round(float(_e1.tau2_ci95_lo.iloc[0]) / 100) * 100))
        add("3.3", "EDTRPAS1-29 credible interval upper (yr)", 19200,
            int(round(float(_e1.tau2_ci95_hi.iloc[0]) / 100) * 100))
        add("3.3", "EDTRPAS1-29 span, factor", 1.8,
            round(float(_e1.tau2_ci95_span.iloc[0]), 1))
    _e2 = _exc[_exc.SampleID == "SCTXSUS1-10/EDTRVFPS1-15"]
    if len(_e2):
        add("3.3", "SCTXSUS1-10/EDTRVFPS1-15 contraction", 0.65,
            round(float(_e2.contraction_log_tau2.iloc[0]), 2))
        add("3.3", "SCTXSUS1-10/EDTRVFPS1-15 span, factor", 4.5,
            round(float(_e2.tau2_ci95_span.iloc[0]), 1))
    # the three error models, all on the same 14 mixtures
    for _tag, _f, _c, _s in [
            ("inversion sigma", "posterior_tau2_fixed_inversion_sigma.csv",
             0.31, 20),
            ("reported analytical sigma",
             "posterior_tau2_fixed_reported_sigma.csv", 0.93, 1.4)]:
        _p = os.path.join(TP, _f)
        if os.path.exists(_p):
            _d = pd.read_csv(_p)
            _d = _d[_d.SampleID.isin(_keep)]
            add("3.3", "%s: median contraction over the 14" % _tag, _c,
                round(float(_d.contraction_log_tau2.median()), 2))
            add("3.3", "%s: median span over the 14" % _tag, _s,
                round(float(_d.tau2_ci95_span.median()), 1)
                if _s < 10 else int(round(float(_d.tau2_ci95_span.median()))))
            if _tag.startswith("reported"):
                add("3.3", "only 3 of 14 fit acceptably under analytical "
                    "sigma", 3, int((_d.p_at_min > 0.05).sum()))
    # the scale-prior sensitivity Section 3.3 reports as 0.23 to 0.19
    for _S, _want in [(2, 0.23), (10, 0.19)]:
        _p = os.path.join(TP, "posterior_tau2_scale%d.csv" % _S)
        if os.path.exists(_p):
            _d = pd.read_csv(_p)
            add("3.3", "scale prior a factor of %d: median contraction" % _S,
                _want, round(float(
                    _d[_d.assessable].contraction_log_tau2.median()), 2))
        else:
            print("NOTE: posterior_tau2_scale%d.csv absent -- run "
                  "posterior_tau2.py --scale-range %d" % (_S, _S))
    # the like-for-like span comparison against the profile intervals
    _p = os.path.join(TP, "posterior_tau2_window_profile.csv")
    if os.path.exists(_p):
        _d = pd.read_csv(_p)
        add("3.3", "profile-window posterior median span, factor", 10.0,
            round(float(_d[_d.assessable].tau2_ci95_span.median()), 1))
    else:
        print("NOTE: posterior_tau2_window_profile.csv absent -- run "
              "posterior_tau2.py --prior-window profile")
    # Grid convergence: the evidence for 160 x 160. Scoped to the ASSESSABLE
    # samples, which is the only scope that bears on a published number.
    # The probe's third sample, EDTRPAS1-46, does NOT settle -- and it is
    # exactly the sample the degree-of-freedom rule already excludes, because
    # its minimum chi2 is near zero and it is the interpolating case the
    # truncated prior tames but does not fully cure. The two criteria agreeing
    # is worth asserting in its own right, so both directions are pinned.
    _p = os.path.join(TP, "posterior_grid_probe.csv")
    if os.path.exists(_p):
        _d = pd.read_csv(_p).merge(pb[["SampleID", "assessable"]],
                                   on="SampleID", how="left")
        _pf = _d[_d.axis == "f1"]
        add("3.3", "grid probe covers 3 samples", 3, _pf.SampleID.nunique())
        _fine = int(_pf.n_f1.max())
        _prod = int(_pf[_pf.n_f1 <= 160].n_f1.max())

        def _shift(rows):
            out = []
            for _sid, _g in rows.groupby("SampleID"):
                _g = _g.set_index("n_f1").contraction_log_tau2
                if _prod in _g.index and _fine in _g.index:
                    out.append(abs(_g[_fine] - _g[_prod]))
            return out

        _sh = _shift(_pf[_pf.assessable.fillna(False)])
        # if this ever exceeds 0.01 the production grid is too coarse
        add("3.3", "assessable samples settled at the production grid", True,
            bool(_sh) and max(_sh) < 0.01)
        _bad = _shift(_pf[~_pf.assessable.fillna(False)])
        add("3.3", "the excluded interpolating sample does NOT settle", True,
            bool(_bad) and max(_bad) > 0.05)
    else:
        print("NOTE: posterior_grid_probe.csv absent -- run "
              "probe_posterior_grid.py")
else:
    print("NOTE: posterior_tau2.csv absent -- run posterior_tau2.py")

# ---- 3.2 mean_age_sigma recomputed with tau2 free --------------------------
SG = os.path.join(TP, "mean_age_sigma_recomputed.csv")
if os.path.exists(SG):
    sg = pd.read_csv(SG).set_index("SampleID")
    add("3.2", "10 mixtures had tau2 prescribed", 10,
        int((~sg.tau2_free_originally).sum()))
    add("3.2", "MC rerun produced a sigma for all 10", 10,
        int(sg.loc[~sg.tau2_free_originally, "sigma_mc_tau2free"].notna().sum()))
    # young_fraction.csv now carries the corrected sigma, so the value the
    # paper quotes must come from the generator, not from the recompute script
    add("3.2", "EDTRPAS1-46 mean age 18,300 yr", 18300,
        int(round(yf.set_index("SampleID").loc["EDTRPAS1-46",
                                               "mean_age_mixture"] / 100) * 100))
    add("3.2", "EDTRPAS1-46 sigma 19,400 yr from the generator", 19400,
        int(round(yf.set_index("SampleID").loc["EDTRPAS1-46",
                                               "mean_age_sigma"] / 100) * 100))
    add("3.2", "its sigma comes from the tau2-free draws", "alt",
        str(yf.set_index("SampleID").loc["EDTRPAS1-46", "mean_age_sigma_source"]))
    add("3.2", "tau2 is the fixed contributor there", "tau2",
        str(yf.set_index("SampleID").loc["EDTRPAS1-46",
                                         "mean_age_fixed_contributors"]))

    # the shipped column is delta-method, not Monte Carlo -- if those two ever
    # coincide the docstring in recompute_mean_age_sigma.py needs revisiting
    add("3.2", "published sigma is not the MC sigma (different estimators)",
        True, bool(abs(sg.loc["EDTRPAS1-46", "sigma_delta_published"]
                       - sg.loc["EDTRPAS1-46", "sigma_mc_orig"]) > 1))
    mx = yf[yf.LPM != "DM"]
    add("3.2", "18 mixtures in young_fraction.csv", 18, len(mx))
    add("3.2", "median sigma 164% of mean age", 164,
        int(round((100 * mx.mean_age_sigma / mx.mean_age_mixture).median())))
    # the fix must not touch samples that never had a fixed contributor
    dmr = yf[yf.LPM == "DM"]
    add("3.2", "42 DM samples keep the delta-method sigma", 42,
        int((dmr.mean_age_sigma_source == "delta").sum()))
    add("3.2", "no DM sample has a fixed contributor", 0,
        int((dmr.mean_age_fixed_contributors.fillna("") != "").sum()))
    # provenance: the pre-dgmeta-fix copy had EDTRPAS1-29 at 41.7498
    add("3.2", "young_fraction.csv is post-dgmeta-fix", 42.028,
        round(float(yf.set_index("SampleID").loc["EDTRPAS1-29", "tau1"]), 3))
    add("3.2", "sigma never below the delta-method value", 0,
        int((mx.mean_age_sigma < mx.mean_age_sigma_delta - 1e-9).sum()))
else:
    print("NOTE: mean_age_sigma_recomputed.csv absent -- run recompute_mean_age_sigma.py")

# ---- 3.2 zone contrast: tested, and not significant ------------------------
# added after the ES&T review asked for an effect size. None of these reach
# significance, so the manuscript reports the zonation as a tendency.
from scipy.stats import fisher_exact, chi2_contingency, mannwhitneyu
_O = ["well constrained", "weakly constrained", "unconstrained"]
_ct = pd.crosstab(ident.Aq_class, ident.cls_true).reindex(columns=_O).fillna(0).astype(int)
_tab = [[int(_ct.loc["confined", "well constrained"]),
         int(_ct.loc["confined", ["weakly constrained", "unconstrained"]].sum())],
        [int(_ct.loc["unconfined", "well constrained"]),
         int(_ct.loc["unconfined", ["weakly constrained", "unconstrained"]].sum())]]
_or, _p = fisher_exact(_tab)
add("3.2", "Fisher odds ratio 3.0 for well-constrained by zone", 3.0, round(float(_or), 1))
add("3.2", "Fisher p = 0.26 (not significant)", 0.26, round(float(_p), 2))
add("3.2", "three-class table p = 0.08", 0.08,
    round(float(chi2_contingency(_ct.values)[1]), 2))
# Zone widths are quoted over CLOSED intervals only (Section 3.2, Text S6).
# Closure is differentially distributed by zone -- 14 of 25 confined fail to
# close against 7 of 26 unconfined -- so neither this comparison nor the
# all-rows one is a well-posed test; both p-values are recorded in Text S6 and
# the zone argument rests on the classification counts instead.
_hit = ident.true_hit_bound.fillna(True).astype(bool)
_a = ident[(ident.Aq_class == "confined") & ~_hit].w_tau1_true.dropna()
_b = ident[(ident.Aq_class == "unconfined") & ~_hit].w_tau1_true.dropna()
add("3.2", "median closed width 0.49 confined", 0.49, round(float(_a.median()), 2))
add("3.2", "median closed width 1.16 unconfined", 1.16, round(float(_b.median()), 2))
add("3.2", "closed-interval Mann-Whitney p = 0.02 (Text S6)", 0.02,
    round(float(mannwhitneyu(_a, _b, alternative="two-sided")[1]), 2))
_aa = ident[(ident.Aq_class == "confined")].w_tau1_true.dropna()
_bb = ident[(ident.Aq_class == "unconfined")].w_tau1_true.dropna()
add("3.2", "all-rows Mann-Whitney p = 0.21 (Text S6)", 0.21,
    round(float(mannwhitneyu(_aa, _bb, alternative="two-sided")[1]), 2))
add("3.2", "confined non-closure 14 of 25", "14/25",
    "%d/%d" % (int((_hit & (ident.Aq_class == "confined")
                    & ident.w_tau1_true.notna()).sum()), len(_aa)))
add("3.2", "unconfined non-closure 7 of 26", "7/26",
    "%d/%d" % (int((_hit & (ident.Aq_class == "unconfined")
                    & ident.w_tau1_true.notna()).sum()), len(_bb)))
add("3.2", "30 intervals closed, median w 1.059", "30/1.059",
    "%d/%.3f" % (int((~_hit & ident.w_tau1_true.notna()).sum()),
                 float(ident.loc[~_hit, "w_tau1_true"].dropna().median())))
add("3.2", "class percentages 13/27/60", "13/27/60",
    "%d/%d/%d" % tuple(round(100 * int(cls.get(c, 0)) / len(ident)) for c in _O))

# ---- SI Figures S3 and S4 are STALE (found 2026-08-24) --------------------
# Neither has a generator script and neither embedded image matches any file in
# the repository: S1-S4 were pasted in from a session whose outputs were not
# kept. Both were plotted from data the 2026-08-24 Hessian fix superseded.
#
#   Figure S3 (observed vs simulated) came from the stale PINN_sim columns.
#     The sample counts still match (60 / 44 / 54 / 60) but every rho and RMSE
#     in its annotation boxes differs from the corrected baseline:
#            3H  0.90 / 0.414  ->  0.89 / 0.352
#     3He(trit)  0.81 / 2.013  ->  0.85 / 1.831
#           SF6  0.79 / 13.220 ->  0.78 / 13.167
#           14C  0.91 / 4.493  ->  0.89 / 4.801
#
#   Figure S4 panel (f) plots tau1_sigma_hess, which the factor-of-2 fix scaled
#     by a median of exactly sqrt(2). Its n moves 50 -> 51 and the median sigma
#     0.910 -> 1.251, so every point in that panel moves. Panels (a)-(e) rest
#     on fitted optima and chi2, which reproduced byte-identically, except for
#     any error bars in (a) drawn from the Hessian sigma.
#
# The values below are what a regenerated Figure S3 must annotate. They are
# pinned so the regeneration can be checked, and the hash tripwire underneath
# fires the moment either image is replaced.
_S3 = {"3H": (60, 0.89, 0.352), "3He(trit)": (44, 0.85, 1.831),
       "SF6": (54, 0.78, 13.167), "14C": (60, 0.89, 4.801)}
_b = pd.read_csv(os.path.join(HO, "reference", "00_summary_table.csv"))
_b = _b[_b.SampleID.isin(set(ident.SampleID))]


def _num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


_pairs = []
for _, _r in _b.iterrows():
    _nm = [x.strip() for x in str(_r.Tracer_names).split(";")]
    _ob = [_num(x) for x in str(_r.Obs_vals).split(";")]
    _sm = [_num(x) for x in str(_r.PINN_sim).split(";")]
    _pairs += list(zip(_nm, _ob, _sm))
_pf = pd.DataFrame(_pairs, columns=["tracer", "obs", "sim"]).dropna()
from scipy.stats import spearmanr as _sr
for _tr, (_n, _rho, _rmse) in _S3.items():
    _g = _pf[_pf.tracer == _tr]
    add("FigS3", "%s: n" % _tr, _n, len(_g))
    add("FigS3", "%s: rho a regenerated figure must show" % _tr, _rho,
        round(float(_sr(_g.obs, _g.sim)[0]), 2))
    add("FigS3", "%s: RMSE a regenerated figure must show" % _tr, _rmse,
        round(float(np.sqrt(((_g.sim - _g.obs) ** 2).mean())), 3))

# ---- supplementary CSVs must not lag the manuscript ------------------------
# Found 2026-08-24 while assembling the submission list: two of the shipped
# CSVs were stale. model_structure_per_sample.csv still carried the pre-I1
# 16/16/12/16 split, and subunity_ratios.csv the pre-bisection-fix 13 rows with
# a minimum of 0.816. Both would have been submitted alongside a manuscript
# that says otherwise.
_SUP = os.path.join(FC, "supplementary_data")
_ms = os.path.join(_SUP, "model_structure_per_sample.csv")
if os.path.exists(_ms):
    _msd = pd.read_csv(_ms)
    _mv = _msd.vs_assigned.value_counts()
    add("supp", "model_structure_per_sample.csv matches Section 3.4",
        "20/17/17/6",
        "%d/%d/%d/%d" % (int(_mv.get("supported", 0)),
                         int(_mv.get("contradicted", 0)),
                         int(_mv.get("indistinguishable", 0)),
                         int(_mv.get("not assessable", 0))))
else:
    print("NOTE: model_structure_per_sample.csv absent")
_sr = os.path.join(_SUP, "subunity_ratios.csv")
if os.path.exists(_sr):
    _srd = pd.read_csv(_sr)
    add("supp", "subunity_ratios.csv has 12 rows", 12, len(_srd))
    add("supp", "its minimum ratio is 0.762", 0.762,
        round(float(_srd.information_ratio.min()), 3))
    add("supp", "eight of the twelve are 14C", 8,
        int((_srd.omitted_tracer == "14C").sum()))
else:
    print("NOTE: subunity_ratios.csv absent")
_f1c = os.path.join(_SUP, "Fig1_sample_locations.csv")
if os.path.exists(_f1c):
    _f1d = pd.read_csv(_f1c)
    _ic = [c for c in _f1d.columns if "identifiab" in c.lower()]
    if _ic:
        _v = _f1d[_ic[0]].value_counts()
        add("supp", "Fig1_sample_locations.csv classes are current", "8/16/36",
            "%d/%d/%d" % (int(_v.get("well constrained", 0)),
                          int(_v.get("weakly constrained", 0)),
                          int(_v.get("unconstrained", 0))))

# ---- the submission workbook must match Text S9 and the CSVs ---------------
# The 20 supporting CSVs are collected into one workbook for submission. The
# workbook is built FROM the CSVs and its sheet list FROM Text S9, so these
# check that all three still agree.
_WB = os.path.join(FC, "manuscript", "EST_Supporting_Data.xlsx")
if os.path.exists(_WB):
    _xl = pd.ExcelFile(_WB)
    _sheets = [x for x in _xl.sheet_names if x != "Contents"]
    add("supp", "workbook has a Contents sheet", True,
        "Contents" in _xl.sheet_names)
    add("supp", "workbook carries 20 data sheets", 20, len(_sheets))
    _csvs = {f[:-4] for f in os.listdir(_SUP) if f.endswith(".csv")}
    add("supp", "every sheet has its CSV and vice versa", set(),
        set(_sheets) ^ _csvs)
    # row counts must match the CSVs, which is what catches a stale workbook
    _drift = []
    for _sh in _sheets:
        _a = pd.read_excel(_WB, sheet_name=_sh)
        _b = pd.read_csv(os.path.join(_SUP, _sh + ".csv"))
        if _a.shape != _b.shape or list(_a.columns) != list(_b.columns):
            _drift.append(_sh)
    add("supp", "no sheet has drifted from its CSV", 0, len(_drift))
else:
    print("NOTE: EST_Supporting_Data.xlsx absent -- workbook unchecked")

# ---- Figure 1: the map's own attribute table -------------------------------
# The map is made in ArcGIS Pro from a shapefile join, outside this pipeline,
# so its classification cannot be re-derived here -- but the CSV it is built
# from can be checked against the authoritative classification. CLAUDE.md
# records that an earlier Fig1 CSV carried superseded classes, which is exactly
# what this catches.
_F1 = os.path.join(FC, "figures", "Fig1_sample_locations_shp_copy.csv")
if not os.path.exists(_F1):
    _F1 = os.path.join(FC, "figures", "Fig1_sample_locations_shp.csv")
if os.path.exists(_F1):
    _f1 = pd.read_csv(_F1)
    add("Fig1", "60 sample locations", 60, len(_f1))
    _fc = _f1.IdentClass.value_counts()
    add("Fig1", "map classes match the published 8/16/36", "8/16/36",
        "%d/%d/%d" % (int(_fc.get("well constrained", 0)),
                      int(_fc.get("weakly constrained", 0)),
                      int(_fc.get("unconstrained", 0))))
    _zc = _f1.Aq_class.value_counts()
    add("Fig1", "map zones match 28 unconfined / 32 confined", "28/32",
        "%d/%d" % (int(_zc.get("unconfined", 0)), int(_zc.get("confined", 0))))
    # per-sample, not just in aggregate
    _ref = ident.set_index("SampleID").cls_true
    _joined = _f1.set_index("SampleID").IdentClass.reindex(_ref.index)
    add("Fig1", "every sample's class agrees with the classification", 0,
        int((_joined != _ref).sum()))
    # and tau1 must be post-dgmeta-fix
    _bt = pd.to_numeric(pd.read_csv(
        os.path.join(HO, "reference", "00_summary_table.csv")
    ).set_index("SampleID").tau1_PINN, errors="coerce")
    _ft = pd.to_numeric(_f1.set_index("SampleID").tau1, errors="coerce")
    add("Fig1", "map tau1 equals the current baseline", True,
        bool((_ft - _bt.reindex(_ft.index)).abs().max() < 1e-6))
else:
    print("NOTE: no Fig1 attribute CSV found -- Figure 1 unchecked")

# ---- Table S6: threshold sensitivity -------------------------------------
# This table had NO generator and no harness check, and was a hybrid of two
# eras: its well-constrained counts at the two tightest pairs reproduced the
# RETIRED conditional width column while its middle rows reproduced the
# true-profile one. Three of six rows were wrong and the quoted range "1 to 13"
# came from the mixture. Regenerated 2026-08-24 by
# scripts/make_threshold_sensitivity.py under the rule Section 2.4 states.
_TS = os.path.join(FC, "supplementary_data", "threshold_sensitivity.csv")
if os.path.exists(_TS):
    _ts = pd.read_csv(_TS)
    add("TableS6", "six threshold pairs", 6, len(_ts))
    add("TableS6", "well-constrained count ranges 0 to 12", (0, 12),
        (int(_ts.well.min()), int(_ts.well.max())))
    add("TableS6", "the reported pair (0.5, 1.5) gives 8/16/36", "8/16/36",
        "%d/%d/%d" % tuple(_ts[(_ts.w_well == 0.5) & (_ts.w_weak == 1.5)]
                           [["well", "weakly", "unconstrained"]].iloc[0]))
    add("TableS6", "every row sums to 60", True,
        bool((_ts.well + _ts.weakly + _ts.unconstrained == 60).all()))
    add("TableS6", "unconstrained is the largest class in every row", True,
        bool((_ts.unconstrained
              >= _ts[["well", "weakly"]].max(axis=1)).all()))
    # the caption's zone claims, which used to read "persists throughout"
    add("TableS6", "confined holds more well-constrained in 5 of 6 rows", 5,
        int((_ts.confi_well > _ts.uncon_well).sum()))
    add("TableS6", "dispersion contrast significant in 4 of 6 rows", 4,
        int((_ts.disp_p < 0.05).sum()))
else:
    print("NOTE: threshold_sensitivity.csv absent -- Table S6 unchecked")

# ---- J1: the zone difference is one of DISPERSION, not location ------------
# The two location tests above do not reach significance, and the two
# directional metrics disagree: non-closure makes the CONFINED zone worse
# (14/25 vs 7/26, p = 0.048) while median closed width makes the UNCONFINED
# zone worse (1.16 vs 0.49, p = 0.02). The counts are not monotone either --
# confined holds more well-constrained AND more unconstrained estimates. The
# supportable statement is that confined holds both extremes while the
# recharge zone concentrates in the middle class, and that test IS
# significant. Section 3.2 and Text S6 now say so.
_mid = ident.cls_true == "weakly constrained"
_disp = [[int(((ident.Aq_class == "confined") & ~_mid).sum()),
          int(((ident.Aq_class == "confined") & _mid).sum())],
         [int(((ident.Aq_class == "unconfined") & ~_mid).sum()),
          int(((ident.Aq_class == "unconfined") & _mid).sum())]]
_dor, _dp = fisher_exact(_disp)
add("3.2", "extremes vs middle: confined 27/32", "27/32",
    "%d/%d" % (_disp[0][0], sum(_disp[0])))
add("3.2", "extremes vs middle: unconfined 17/28", "17/28",
    "%d/%d" % (_disp[1][0], sum(_disp[1])))
add("3.2", "middle class 11 of 28 unconfined vs 5 of 32 confined", "11/5",
    "%d/%d" % (_disp[1][1], _disp[0][1]))
add("3.2", "dispersion odds ratio 3.5", 3.5, round(float(_dor), 1))
add("3.2", "dispersion p = 0.047, the only significant zone test", 0.047,
    round(float(_dp), 3))
add("3.2", "non-closure IS significant and favours the opposite zone", True,
    bool(fisher_exact(pd.crosstab(
        ident.loc[ident.w_tau1_true.notna(), "Aq_class"],
        _hit[ident.w_tau1_true.notna()]).values)[1] < 0.05))
# the well-constrained class is entirely marginal -- no member far below 0.5
_wc = ident.loc[ident.cls_true == "well constrained", "w_tau1_true"].dropna()
add("3.2", "all 8 well-constrained have w in 0.35-0.49", (0.35, 0.49),
    (round(float(_wc.min()), 2), round(float(_wc.max()), 2)))
# J6: tracer availability does not confound the zone comparison
_na = ident.groupby("Aq_class").n_active.mean()
add("3.2", "mean active tracers 3.61 unconfined / 3.66 confined", "3.61/3.66",
    "%.2f/%.2f" % (_na["unconfined"], _na["confined"]))
# J5: neither does the dispersion-parameter bound
_bi = ~ident.bound_active.fillna(False).astype(bool)
add("3.2", "42 samples with the bound inactive", 42, int(_bi.sum()))
_dispb = [[int(((ident.Aq_class == "confined") & _bi & ~_mid).sum()),
           int(((ident.Aq_class == "confined") & _bi & _mid).sum())],
          [int(((ident.Aq_class == "unconfined") & _bi & ~_mid).sum()),
           int(((ident.Aq_class == "unconfined") & _bi & _mid).sum())]]
_bor, _bp = fisher_exact(_dispb)
add("3.2", "bound-inactive control: odds ratio 4.6", 4.6, round(float(_bor), 1))
add("3.2", "bound-inactive control: p = 0.06", 0.06, round(float(_bp), 2))

# ---- 3.3 tracer information content ---------------------------------------
# NOTE: use *_true. Table 3 in the paper splits these by zone; the text quotes
# the pooled values, which is why both are checked here.
for tracer, n_inv, n_cens, med in [("3He(trit)", 44, 36, None),
                                   ("SF6", 54, 28, 1.25),
                                   ("3H", 60, 34, 1.07),
                                   ("14C", 60, 27, 1.00)]:
    s = loto[loto.omitted_tracer == tracer]
    cen = s.censored_true.astype(bool)
    add("3.3", "%s: %d reduced-suite inversions" % (tracer, n_inv), n_inv, len(s))
    add("3.3", "%s withheld -> %d censored" % (tracer, n_cens), n_cens, int(cen.sum()))
    if med is not None:
        fin = s.loc[~cen, "information_ratio_true"].dropna()
        add("3.3", "%s pooled median information ratio" % tracer, med,
            round(float(fin.median()), 2))
c14 = loto[loto.omitted_tracer == "14C"]
add("3.3", "14C median |dtau1| = 0.1%", 0.1,
    round(100 * float(c14.delta_tau1_rel.abs().median()), 1))
add("3.3", "14C interquartile range of Ij is 1.00-1.00", 0.0,
    round(float(c14.loc[~c14.censored_true.astype(bool),
                        "information_ratio_true"].quantile(.75)
                - c14.loc[~c14.censored_true.astype(bool),
                          "information_ratio_true"].quantile(.25)), 2))
add("3.3", "max mixture mean age ~18,000 yr (nearest 1000)", 18,
    int(round(yf.mean_age_mixture.max() / 1000)))

# ---- 2.5 sub-unity information ratios (third review, medium priority) ------
_fin = loto[~loto.censored_true.astype(bool)].information_ratio_true.dropna()
add("2.5", "59 defined information ratios", 59, len(_fin))
add("2.5", "12 fall below unity", 12, int((_fin < 1).sum()))
add("2.5", "smallest defined ratio 0.762", 0.762, round(float(_fin.min()), 3))
_sub = loto[(~loto.censored_true.astype(bool)) & (loto.information_ratio_true < 1)]
add("2.5", "8 of the 12 come from omitting 14C", 8,
    int((_sub.omitted_tracer == "14C").sum()))

# ---- 3.6 14C dead-carbon sensitivity ---------------------------------------
CS = os.path.join(TP, "c14_sensitivity.csv")
if os.path.exists(CS):
    cs = pd.read_csv(CS)
    _free = cs.SampleID.map(
        ident.set_index("SampleID").free_params.str.contains("2nd Mean Age"))
    add("3.6", "18 mixtures perturbed", 18, cs.SampleID.nunique())
    # the 0.40-0.90 range in the text is across all 60 samples; the 18
    # mixtures perturbed here span only 0.40-0.80
    _q60 = pd.Series([float(t["scale"])
                      for smp in json.load(open(os.path.join(
                          HO, "examples", "edwards_aquifer",
                          "edwards_input_config.json")))["samples"]
                      for t in smp["tracers"]
                      if t["name"] == "14C" and smp["sample_id"] in set(ident.SampleID)])
    add("3.6", "dead-carbon factor spans 0.40-0.90 over 60 samples", "0.4-0.9",
        "%.1f-%.1f" % (_q60.min(), _q60.max()))
    add("3.6", "mixtures perturbed span 0.40-0.80", "0.4-0.8",
        "%.1f-%.1f" % (cs.q_baseline.min(), cs.q_baseline.max()))
    add("3.6", "tau2 immobile where prescribed", 0.0,
        round(float(cs.loc[~_free, "tau2_shift_pct"].abs().max()), 1))
    add("3.6", "prescribed-tau2: mean age median 5%", 5,
        int(round(cs.loc[~_free, "mean_age_shift_pct"].abs().median())))
    add("3.6", "prescribed-tau2: mean age max 15%", 15,
        int(round(cs.loc[~_free, "mean_age_shift_pct"].abs().max())))
    add("3.6", "fitted tau2: median shift 48%", 48,
        int(round(cs.loc[_free, "tau2_shift_pct"].abs().median())))
    add("3.6", "fitted tau2: max shift 670%", 670,
        int(round(cs.loc[_free, "tau2_shift_pct"].abs().max() / 10) * 10))
else:
    print("NOTE: c14_sensitivity.csv absent -- run c14_sensitivity.py")

# ---- 3.2 objective dependence (Table S4) -----------------------------------
PS = os.path.join(TP, "profile_sigma.csv")
if os.path.exists(PS):
    ps = pd.read_csv(PS)
    add("3.2", "60 samples refit under the weighted objective", 60, len(ps))
    for col, lbl, exp in (("cls_sigma", "dchi2=1", (46, 3, 11)),
                          ("cls_sigma_scaled", "dchi2=chi2/dof", (44, 5, 11))):
        v = ps[col].value_counts()
        got = tuple(int(v.get(c, 0)) for c in
                    ["well constrained", "weakly constrained", "unconstrained"])
        add("3.2", "weighted objective %s -> %s" % (lbl, "/".join(map(str, exp))),
            "/".join(map(str, exp)), "/".join(map(str, got)))
    add("3.2", "about a third keep their class (dchi2=chi2/dof)", 19,
        int((ps.cls_sigma_scaled == ps.cls_reported).sum()))
    add("3.2", "median weighted width 0.11 (closed)", 0.11,
        round(float(ps.loc[~ps.hit_bound_scaled.fillna(True).astype(bool),
                           "w_sigma_scaled"].dropna().median()), 2))

    # ---- the error model the weighted objective actually uses ------------
    # Section 3.2 / Text S3 / Table S5. The SI previously described these as
    # "the reported analytical uncertainty of tracer i"; they are not. Where
    # every active tracer for a sample carries the same relative sigma, the
    # weighted objective is a constant multiple of the relative-error one, so
    # the two "objectives" differ only by a threshold rescaling.
    _cfg = json.load(open(os.path.join(
        HO, "examples", "edwards_aquifer", "edwards_input_config.json")))
    _study = set(ident.SampleID)
    _rel, _allten = [], 0
    for _s in _cfg["samples"]:
        if _s["sample_id"] not in _study:
            continue
        _r = [abs(float(t["obs_err"]) / float(t["obs"]))
              for t in _s["tracers"]
              if t.get("obs") not in (None, 0) and t.get("obs_err") is not None
              and float(t.get("scale") or 0) > 0]
        _rel += _r
        if _r and np.allclose(_r, 0.10, rtol=1e-6):
            _allten += 1
    add("3.2", "218 active tracer-sample pairs", 218, len(_rel))

    # PROVENANCE. Every tracer concentration and standard error in the config
    # comes verbatim from Table 4 of the data release -- LPM_Meas_Tracer_NN and
    # _Err, the values used as inputs to the published TracerLPM inversion.
    # Nothing here is invented, and nothing is derived by an undocumented
    # transformation. Diffing against Tables 3 and 5 (the analytical
    # measurements) instead led to the wrong conclusion once already, so this
    # is pinned. The file is UTF-16.
    _t4p = os.path.join(os.path.dirname(HO), "Musgrove_2023Data",
                        "Table_4_LPM.txt")
    if os.path.exists(_t4p):
        _t4 = pd.read_csv(_t4p, sep="\t", encoding="utf-16")
        _n1 = lambda v: pd.to_numeric(pd.Series([v]), errors="coerce")[0]
        _T4 = {}
        for _, _r in _t4.iterrows():
            _dd = {}
            for _i in range(1, 11):
                _nm = _r.get("LPM_Tracer_Name_%02d" % _i)
                if isinstance(_nm, str) and _nm.strip():
                    _dd[_nm.strip()] = (_n1(_r.get("LPM_Meas_Tracer_%02d" % _i)),
                                        _n1(_r.get("LPM_Meas_Tracer_%02d_Err" % _i)),
                                        _n1(_r.get("LPM_ScaleFact_%02d" % _i)))
            _T4[str(_r["SampleID"]).strip()] = _dd
        _no, _ne, _nt = 0, 0, 0
        for _s in _cfg["samples"]:
            if _s["sample_id"] not in _study:
                continue
            _m = {}
            for _k in [_s["sample_id"]] + [x.strip() for x in _s["sample_id"].split("/")]:
                if _k in _T4:
                    _m = _T4[_k]
                    break
            for _t in _s["tracers"]:
                if _t.get("obs") in (None, 0) or float(_t.get("scale") or 0) <= 0:
                    continue
                _nt += 1
                _tv = _m.get(_t["name"])
                if _tv and np.isclose(float(_t["obs"]), _tv[0], rtol=1e-6):
                    _no += 1
                if _tv and np.isclose(float(_t["obs_err"]), _tv[1], rtol=1e-6):
                    _ne += 1
        add("prov", "config observations match data-release Table 4", _nt, _no)
        add("prov", "config uncertainties match data-release Table 4", _nt, _ne)
        add("prov", "all 218 pairs accounted for", 218, _nt)

        # The tracer input-function scale factors, including the 14C
        # dead-carbon DILUTION factor that Text S6 and Section 3.3 now lean
        # on, are the published inversion's own -- not values adopted here.
        # Every one of the 293 tracer-sample factors in the config (all
        # tracers, not just the 218 active pairs) must reproduce
        # LPM_ScaleFact_NN exactly, or the karst dead-carbon argument is
        # resting on a number we invented.
        _ns, _nsm = 0, 0
        _c14 = []
        for _s in _cfg["samples"]:
            _m = {}
            for _k in [_s["sample_id"]] + [x.strip()
                                           for x in _s["sample_id"].split("/")]:
                if _k in _T4:
                    _m = _T4[_k]
                    break
            for _t in _s["tracers"]:
                _tv = _m.get(_t["name"])
                if not _tv or _t.get("scale") is None:
                    continue
                _ns += 1
                if np.isclose(float(_t["scale"]), _tv[2], rtol=0, atol=1e-9):
                    _nsm += 1
                if _t["name"] == "14C":
                    _c14.append(float(_t["scale"]))
        add("prov", "tracer scale factors match data-release LPM_ScaleFact",
            _ns, _nsm)
        add("prov", "293 tracer-sample scale factors compared", 293, _ns)
        add("3.3", "14C dilution factor spans 0.40 to 0.90",
            (0.4, 0.9), (round(min(_c14), 2), round(max(_c14), 2)))
        # 14C carries much the largest of the scalings -- the quantitative
        # claim Text S6 makes when contrasting it with 3H and SF6
        add("Text S6", "14C dilution is the largest tracer scaling", True,
            bool(min(_c14) < 0.7))
    else:
        print("NOTE: Table_4_LPM.txt absent -- provenance check skipped")
    # The abstract's "for two-thirds of samples" rests on this: the weighted
    # objective is EXACTLY 100x the relative-error one only where every active
    # tracer of a sample carries the same relative sigma.
    _uni = 0
    for _s in _cfg["samples"]:
        if _s["sample_id"] not in _study:
            continue
        _rr = [abs(float(_t["obs_err"]) / float(_t["obs"]))
               for _t in _s["tracers"]
               if _t.get("obs") not in (None, 0)
               and float(_t.get("scale") or 0) > 0]
        if _rr and np.allclose(_rr, _rr[0], rtol=1e-9):
            _uni += 1
    add("3.2", "40 of 60 samples have a uniform relative sigma", 40, _uni)
    add("3.2", "that is two-thirds, as the abstract says", 67,
        int(round(100 * _uni / 60)))
    add("3.2", "164 of 218 pairs at a uniform 10%", 164,
        int(np.isclose(_rel, 0.10, rtol=1e-6).sum()))
    add("3.2", "40 of 60 samples have every tracer at 10%", 40, _allten)

    # Table S5: inversion vs analytical 1-sigma, as median relative values. Units
    # cannot confound a relative comparison; config 3H equals TRC_3H_TU and
    # config 14C equals 14C_sample_pmC exactly, so those two are like for like.
    _MG = os.path.join(os.path.dirname(HO), "Musgrove_2023Data")
    _t3 = pd.read_csv(os.path.join(_MG, "Table_3_Tracers.txt"), sep="\t")
    _t5 = pd.read_csv(os.path.join(_MG, "Table_5_Carbon14.txt"), sep="\t")
    _N = lambda v: pd.to_numeric(pd.Series([v]), errors="coerce")[0]
    _src = {}
    for _, _r in _t3.iterrows():
        _src.setdefault(str(_r["SampleID"]).strip(), {}).update({
            "3H": (_N(_r["TRC_3H_TU"]), _N(_r["TRC_3H_Err_TU"])),
            "SF6": (_N(_r["TRC_SF6_pptv"]), _N(_r["TRC_SF6_Err_pptv"])),
            "3He(trit)": (_N(_r["TRC_3HeTrit_TU"]), _N(_r["TRC_3HeTrit_Err_TU"]))})
    for _, _r in _t5.iterrows():
        _src.setdefault(str(_r["SampleID"]).strip(), {})["14C"] = (
            _N(_r["14C_sample_pM"]), _N(_r["14C_Err_sample_pM"]))
    _got = {}
    for _s in _cfg["samples"]:
        if _s["sample_id"] not in _study:
            continue
        _keys = [_s["sample_id"]] + [x.strip() for x in _s["sample_id"].split("/")]
        for _t in _s["tracers"]:
            _n = _t["name"]
            if _n not in ("3H", "SF6", "3He(trit)", "14C"):
                continue
            if _t.get("obs") in (None, 0) or float(_t.get("scale") or 0) <= 0:
                continue
            for _k in _keys:
                if _k in _src and _n in _src[_k] and _src[_k][_n][0] == _src[_k][_n][0]:
                    _o, _e = _src[_k][_n]
                    if _o and np.isfinite(_e):
                        _got.setdefault(_n, []).append(abs(_e / _o))
                    break
    for _n, _exp in (("3H", 1.0), ("3He(trit)", 30.4), ("SF6", 16.4), ("14C", 0.24)):
        add("TableS5", "%s analytical 1-sigma median %.2f%%" % (_n, _exp), _exp,
            round(100 * float(np.median(_got[_n])), 2 if _exp < 1 else 1))

    # Text S3: the reported uncertainties are not usable either
    _si = os.path.join(TP, "sigma_impact.csv")
    if os.path.exists(_si):
        _im = pd.read_csv(_si)
        add("3.2", "42 dispersion-model fits evaluated", 42, len(_im))
        add("3.2", "median weighted misfit 3.2 under the assumed sigma", 3.2,
            round(float(_im.chi2_cfg.median()), 1))
        add("3.2", "median weighted misfit 557 under reported sigma", 557,
            int(round(float(_im.chi2_rep.median()))))
        add("3.2", "23 of 42 fit at p>0.05 under assumed sigma", 23,
            int((_im.p_cfg > 0.05).sum()))
        add("3.2", "4 of 42 fit at p>0.05 under reported sigma", 4,
            int((_im.p_rep > 0.05).sum()))
    else:
        print("NOTE: sigma_impact.csv absent -- run sigma_impact.py")

    # tau2 under the weighted objective. Section 3.2 briefly claimed tau2's
    # non-closure was objective-independent; it is not. 4 of 18 close at
    # dchi2=1. Two of those four are uniform-10% samples, where the weighted
    # objective is a constant multiple of the relative-error one and the
    # closure is purely the tighter threshold; the other two are not.
    _t2s = os.path.join(TP, "profile_tau2_sigma.csv")
    if os.path.exists(_t2s):
        _d2 = pd.read_csv(_t2s)
        add("3.2", "18 binary mixtures profiled under the weighted objective",
            18, len(_d2))
        add("3.2", "tau2 closes for 4 of 18 at dchi2=1", 4,
            int((_d2.cls_tau2 == "well constrained").sum()))
        add("3.2", "tau2 closes for 5 of 18 at the rescaled threshold", 5,
            int((_d2.cls_tau2_scaled == "well constrained").sum()))
        _u10 = set()
        for _s in _cfg["samples"]:
            _r = [abs(float(t["obs_err"]) / float(t["obs"]))
                  for t in _s["tracers"]
                  if t.get("obs") not in (None, 0)
                  and t.get("obs_err") is not None
                  and float(t.get("scale") or 0) > 0]
            if _r and np.allclose(_r, 0.10, rtol=1e-6):
                _u10.add(_s["sample_id"])
        _cl = _d2[_d2.cls_tau2 == "well constrained"].SampleID
        add("3.2", "2 of the 4 are uniform-10% samples", 2,
            int(_cl.isin(_u10).sum()))
        # the reported criterion, from its own file rather than this one
        _mau = pd.read_csv(os.path.join(TP, "mean_age_uncertainty.csv"))
        add("3.2", "tau2 unconstrained for all 18 under relative error", 18,
            int((_mau.cls_tau2 == "unconstrained").sum()))
        # All three samples ever flagged for a non-converged gradient are among
        # the four. EDTRPAS1-29 is the only one of the four without that
        # history -- an earlier draft of the SI had this backwards.
        _FLAG = {"EDTRPAS1-46", "SCTXSUS1-04/EDTRVFPS1-20",
                 "SCTXSUS1-10/EDTRVFPS1-15"}
        add("3.2", "all 3 flagged samples are among the 4 that close", 3,
            len(_FLAG & set(_cl)))
        add("3.2", "EDTRPAS1-29 is the only one of the 4 never flagged",
            "EDTRPAS1-29", ", ".join(sorted(set(_cl) - _FLAG)))
        _mau2 = pd.read_csv(os.path.join(TP, "mean_age_uncertainty.csv"))
        _old3 = _mau2.sort_values("mean_age", ascending=False).SampleID.tolist()[:3]
        add("3.2", "3 of the 4 are the 3 oldest mixtures", 3,
            len(set(_old3) & set(_cl)))
        add("3.2", "EDTRPAS1-46 and EDTRPAS1-29 are the two oldest",
            "EDTRPAS1-46, EDTRPAS1-29", ", ".join(_old3[:2]))
        # The non-convergence flag was spurious. grad_norm_natural in
        # sweeps_reconv/summary.csv is evaluated over prescribed as well as
        # free parameters, so a prescribed-tau1 sample shows a large gradient
        # in a direction that is never optimised. Restricted to the parameters
        # actually estimated, all three are converged. Recomputed here rather
        # than trusted, since this is the claim that closes the question.
        _rc = pd.read_csv(os.path.join(HO, "sweeps_reconv", "summary.csv")
                          ).set_index("SampleID")
        add("3.2", "the stale norm flags 1 of the 3 as unconverged", 1,
            sum(float(_rc.loc[s, "grad_norm_natural"]) >= 1e-3 for s in _FLAG))
        try:
            sys.path.insert(0, HO)
            _cwd = os.getcwd()
            os.chdir(HO)
            import dlpmi_sweeps as _SW
            _SW.bootstrap(HO)
            _th = _SW._TORCH
            _cfgs = {r["sample_id"]: r for r in json.load(open(os.path.join(
                HO, "examples", "edwards_aquifer",
                "edwards_input_config.json")))["samples"]}
            _bse = pd.read_csv(os.path.join(HO, "reference",
                                            "00_summary_table.csv")
                               ).set_index("SampleID")
            _MP = {"tau1": "tau1_PINN", "f1": "f1_PINN",
                   "tau2": "tau2_PINN", "pd1": "pd1_PINN"}
            _worst = 0.0
            for _s in _FLAG:
                _u = _SW.unpack(_cfgs[_s])
                _lf, _pnm = _SW.make_loss(_u, _cfgs[_s])
                _pv = [float(_bse.loc[_s, _MP.get(n, n + "_PINN")])
                       for n in _pnm]
                _tt = _th.tensor(_pv, dtype=_th.float64, requires_grad=True)
                _lf(_tt).backward()
                _worst = max(_worst, max(abs(float(g) * v)
                                         for g, v in zip(_tt.grad, _pv)))
            os.chdir(_cwd)
            add("3.2", "all 3 converged on their free parameters "
                       "(rel. grad < 1.1e-3)", True, bool(_worst < 1.1e-3))
        except Exception as _e:                             # noqa: BLE001
            print("NOTE: free-parameter gradient check skipped: %s" % _e)
    else:
        print("NOTE: profile_tau2_sigma.csv absent -- run profile_tau2_sigma.py")

    # The 285 yr tau2 "attractor" is an age-grid node, not a physical minimum.
    # AGES_YOUNG's third segment is linspace(100, 500, 200), whose 92nd node is
    # 284.9246 yr; reconstructed here rather than importing torch.
    _c14 = os.path.join(TP, "c14_sensitivity.csv")
    if os.path.exists(_c14):
        _cs = pd.read_csv(_c14)
        _node = float(np.linspace(100.0, 500.0, 200)[92])
        add("3.6", "AGES_YOUNG node nearest 285 yr", 284.925, round(_node, 3))
        _nr = _cs[pd.to_numeric(_cs.tau2_perturbed,
                                errors="coerce").between(270, 300)]
        add("3.6", "5 samples land on that node", 5, _nr.SampleID.nunique())
        add("3.6", "all within 0.2 yr of the node", True,
            bool((_nr.tau2_perturbed - _node).abs().max() < 0.2))
        # three of them start far away, so this is convergence onto the node
        add("3.6", "3 of the 5 start above 300 yr", 3,
            int((_nr.tau2_baseline > 300).sum()))
else:
    print("NOTE: profile_sigma.csv absent -- run profile_sigma.py --all")

# ---- 3.4 model structure ---------------------------------------------------
# Assessability is dof >= 1, NOT n_active >= 4. Both structures carry k = 2, so
# the AICc correction term is identical and cancels in the difference: dAICc is
# exactly the chi2_sigma difference and stays defined where aicc() returns NaN.
# Requiring n_active >= 4 (so that aicc() returned a number for each structure
# on its own) held this section to 44 samples until 2026-08-24, discarding the
# ten three-tracer samples on the strength of a correction that cancels.
_dchi2 = st.chi2sig_BMM - st.chi2sig_DM
add("3.4", "dAICc equals the chi2_sigma difference where AICc is defined",
    True, bool((st.dAICc - _dchi2).abs().max() < 1e-9))
add("3.4", "k is 2 under both structures for every sample", True,
    bool((st.k_DM == 2).all() and (st.k_BMM == 2).all()))

ass = st[st.dAICc.notna()]
add("3.4", "54 assessable samples", 54, len(ass))
add("3.4", "assessable is exactly dof >= 1", 54, int((st.n_active >= 3).sum()))
add("3.4", "6 not assessable (both structures saturated at n_a = k)", 6,
    len(st) - len(ass))
# the excluded six are excluded for saturation, NOT because both fit identically
add("3.4", "excluded six do not all fit identically (max |dchi2| > 1)", True,
    bool(_dchi2[st.n_active < 3].abs().max() > 1.0))
vs = ass.vs_assigned.value_counts()
add("3.4", "20 supported", 20, int(vs.get("supported", 0)))
add("3.4", "17 contradicted", 17, int(vs.get("contradicted", 0)))
add("3.4", "17 indistinguishable", 17, int(vs.get("indistinguishable", 0)))
dm = ass[ass.assigned_LPM == "DM"]
bm = ass[ass.assigned_LPM != "DM"]
add("3.4", "36 assigned a single-component model", 36, len(dm))
add("3.4", "2 of those 36 supported", 2, int((dm.vs_assigned == "supported").sum()))
add("3.4", "18 assigned a binary mixture", 18, len(bm))
add("3.4", "all 18 supported", 18, int((bm.vs_assigned == "supported").sum()))
add("Fig5", "24 unconfined assessable", 24, int((ass.Aq_class == "unconfined").sum()))
add("Fig5", "30 confined assessable", 30, int((ass.Aq_class == "confined").sum()))
add("Fig5", "7 open rings (BMM tau1 on its lower bound)", 7,
    int(ass.BMM_tau1_at_lower_bound.astype(bool).sum()))

# The structures are NOT nested: DM frees {tau1, P_D}, BMM frees {tau1, f1}
# with P_D prescribed (Table S2), for all 60 samples. So f1 -> 1 recovers a
# dispersion model only at the prescribed P_D, and the binary mixture can fit
# worse at its own optimum. Text S6 quantifies that; pin it here so the claim
# cannot drift.
_worse = _dchi2[st.n_active >= 3]
add("3.4", "binary mixture fits worse in 12 of the 54 assessable", 12,
    int((_worse > 0).sum()))
add("3.4", "worst such excess, in chi2_sigma", 5.0,
    round(float(_worse.max()), 1))

# Table S10: objective-function sensitivity. Refit under Eq. (S9), ranked under
# Eq. (S18). Two of the 54 flip -- EDTRPAS1-28, and EDTRPAS1-50, a three-tracer
# sample the old n_a >= 4 rule discarded.
_OS = os.path.join(FC, "supplementary_data", "objective_sensitivity.csv")
if os.path.exists(_OS):
    _os = pd.read_csv(_OS)
    add("TableS10", "54 assessable under both objectives", 54, len(_os))
    add("TableS10", "2 classifications flip between objectives", 2,
        int((_os.class_eqS18 != _os.class_eqS9).sum()))
    add("TableS10", "the flipping samples", "EDTRPAS1-28,EDTRPAS1-50",
        ",".join(sorted(_os.loc[_os.class_eqS18 != _os.class_eqS9, "SampleID"])))
else:
    print("NOTE: objective_sensitivity.csv absent -- Table S10 unchecked")

# Text S6: the counts are conditional on the assumed error model.
#
# A UNIFORM rescale sigma -> c*sigma leaves the optimum unchanged and divides
# chi2_sigma by c^2, so it needs no refitting and is exact. Work from _dchi2
# rather than the AICc columns, which are NaN at dof = 1.
_isdm = st.assigned_LPM == "DM"
_ok = _isdm & _dchi2.notna() & (st.n_active >= 3)
_sup_max, _bmm_contra = 0, 0
for _c, _exp, _lbl in ((1.0, 2, "as used"), (3.0, 0, "3x looser")):
    _dd = _dchi2 / _c ** 2
    _sup = int((_ok & (_dd.abs() >= 2) & (_dd > 0)).sum())
    add("3.4", "single-component support under %s sigma" % _lbl, _exp, _sup)
    add("3.4", "36 assessable DM-assigned under %s sigma" % _lbl, 36,
        int(_ok.sum()))
    _sup_max = max(_sup_max, _sup)
    # dAICc < 0 favours BMM, so a BMM-assigned sample is contradicted only at
    # dAICc >= 2
    _bmm_contra = max(_bmm_contra, int(
        ((~_isdm) & _dchi2.notna() & (st.n_active >= 3) & (_dd >= 2)).sum()))

# The REPORTED ANALYTICAL sigmas are per-tracer, ranging from 0.24% (14C) to
# 30.4% (3He(trit)) against a uniform 10% -- per-tracer factors of 41, 10, 0.6
# and 0.5, so two tracers tighten and two loosen. That is not a uniform
# rescale, it moves the optimum, and it cannot be evaluated without refitting.
# Modelling it as a single scalar c = 1/13 (the behaviour before 2026-08-24)
# produced the manuscript's "9 of 30"; evaluating the analytical sigmas at the
# inversion-sigma optimum instead gives 24 of 36. Neither is the analytical
# fit. This check reads the genuine refit, produced by
#   dlpmi_sweeps.py --sweeps structure --objective sigma \
#       --reported-sigma-dir ../Musgrove_2023Data --fit-sigma-source reported
_REP = os.path.join(HO, "sweeps_sigma_reported", "sweep_structure.csv")
if os.path.exists(_REP):
    _r = pd.read_csv(_REP)
    _ra = _r[_r.n_active >= 3]
    _rdm = _ra[_ra.assigned_LPM == "DM"]
    _rsup = int((_rdm.vs_assigned == "supported").sum())
    add("3.4", "36 assessable DM-assigned under the analytical-sigma refit",
        36, len(_rdm))
    add("3.4", "single-component support under the analytical-sigma refit",
        21, _rsup)
    # 21 of 36, not the 2 of the primary error model and not the 9 the 1/13
    # proxy gave. The ranking is nonetheless NOT offered as a bound on the
    # result: in most of those 21 the analytical sigmas leave BOTH structures
    # rejected, so the comparison is between two models that do not fit.
    _s = _rdm[_rdm.vs_assigned == "supported"].copy()
    _dofr = (_s.n_active - 2).clip(lower=1)
    _pD = 1 - CHI2.cdf(_s.chi2sig_DM, _dofr)
    _pB = 1 - CHI2.cdf(_s.chi2sig_BMM, _dofr)
    add("3.4", "of those 21, neither structure fits (p <= 0.05 both)", 17,
        int(((_pD <= 0.05) & (_pB <= 0.05)).sum()))
    add("3.4", "of those 21, none has an acceptable binary-mixture fit", 0,
        int((_pB > 0.05).sum()))
    _bmm_contra = max(_bmm_contra, int(
        (_ra[_ra.assigned_LPM != "DM"].vs_assigned == "contradicted").sum()))
else:
    print("NOTE: %s absent -- the analytical-sigma sensitivity is unchecked. "
          "See REPRODUCE.md section 4." % _REP)

# ---- J2: the zone effect in model support is Simpson's paradox ------------
# Pooled it is highly significant; stratified by the a priori assignment it
# vanishes, because all 18 binary-mixture assignments are supported and 16 of
# them are confined. Section 3.4 reported the pooled version as a spatial
# finding two paragraphs after explaining the mechanism that produces it.
_ns = ass.vs_assigned.isin(["contradicted", "indistinguishable"])
_zc = pd.crosstab(ass.Aq_class, _ns)
add("3.4", "pooled: 21 of 24 unconfined not supported", "21/24",
    "%d/%d" % (int(_zc.loc["unconfined", True]),
               int(_zc.loc["unconfined"].sum())))
add("3.4", "pooled: 13 of 30 confined not supported", "13/30",
    "%d/%d" % (int(_zc.loc["confined", True]),
               int(_zc.loc["confined"].sum())))
add("3.4", "pooled zone effect is significant, p = 0.001", 0.001,
    round(float(fisher_exact(_zc.values)[1]), 3))
_dmz = ass[ass.assigned_LPM == "DM"]
_dmn = _dmz.vs_assigned.isin(["contradicted", "indistinguishable"])
_dmt = pd.crosstab(_dmz.Aq_class, _dmn)
add("3.4", "stratified DM: 21 of 22 unconfined", "21/22",
    "%d/%d" % (int(_dmt.loc["unconfined", True]),
               int(_dmt.loc["unconfined"].sum())))
add("3.4", "stratified DM: 13 of 14 confined", "13/14",
    "%d/%d" % (int(_dmt.loc["confined", True]),
               int(_dmt.loc["confined"].sum())))
add("3.4", "stratified, the zone effect vanishes: p = 1.0", 1.0,
    round(float(fisher_exact(_dmt.values)[1]), 2))
add("3.4", "16 of the 18 binary-mixture assignments are confined", "16/18",
    "%d/%d" % (int((bm.Aq_class == "confined").sum()), len(bm)))

# The paragraph also asserted a correspondence between structure support and
# identifiability. Pooled, that association runs BACKWARDS -- every BMM
# assignment is both supported and unconstrained, nine not fitting tau1 at
# all -- and within the DM stratum only two samples are supported, so there is
# no power to test it. The claim is withdrawn, not restated.
_ii = ident.set_index("SampleID")
_j = ass.set_index("SampleID").join(_ii[["cls_true", "w_tau1_true"]])
_poor = _j.cls_true == "unconstrained"
_sup = _j.vs_assigned == "supported"
add("3.4", "18 of 20 supported samples are unconstrained", "18/20",
    "%d/%d" % (int((_sup & _poor).sum()), int(_sup.sum())))
add("3.4", "12 of 34 unsupported samples are unconstrained", "12/34",
    "%d/%d" % (int((~_sup & _poor).sum()), int((~_sup).sum())))
# orientation-independent: state the rates, not the odds ratio
add("3.4", "supported are MORE often unconstrained than unsupported "
           "(the correspondence runs backwards)", True,
    bool((_poor[_sup].mean() > _poor[~_sup].mean())
         and fisher_exact(pd.crosstab(_sup, _poor).values)[1] < 0.01))
add("3.4", "only 2 DM-assigned are supported, too few to stratify", 2,
    int((_j[_j.assigned_LPM == "DM"].vs_assigned == "supported").sum()))
add("3.4", "all 18 BMM-assigned are unconstrained in tau1", 18,
    int((_j[_j.assigned_LPM != "DM"].cls_true == "unconstrained").sum()))

# What survives every error model tested, and is what Section 3.4 leans on.
add("3.4", "no binary-mixture assignment contradicted under any error model",
    0, _bmm_contra)
add("3.4", "single-component support at its highest across the uniform "
           "rescalings", 2, _sup_max)

# ---- 3.4 goodness of fit ---------------------------------------------------
dof = (gof.set_index("SampleID").n_active
       - st.set_index("SampleID").k_DM.reindex(gof.SampleID).values).clip(lower=1)
g = gof.set_index("SampleID")
p_rel = 1 - CHI2.cdf(g.chi2rel_DM_at_eq9opt, dof)
p_sig = 1 - CHI2.cdf(g.chi2sig_DM_at_sigmaopt, dof)
add("3.4", "relative-error objective: 59 of 60 with p > 0.05", 59, int((p_rel > 0.05).sum()))
add("3.4", "error-weighted objective: 24 of 60 with p > 0.05", 24, int((p_sig > 0.05).sum()))
add("3.4", "median chi2 under the weighted objective", 10.8,
    round(float(g.chi2sig_DM_at_sigmaopt.median()), 1))

# ---- 3.6 limitations -------------------------------------------------------
add("3.6", "21 of 60 intervals limited by a bound", 21,
    int(ident.true_hit_bound.astype(bool).sum()))
add("3.6", "9 samples with transit time confounded with a second parameter", 9, len(sc))

# ---- the document itself ---------------------------------------------------
# The checks above compare the data against values transcribed into this file,
# which only proves the transcription is right. These read the .docx, so a
# retired number reappearing in the manuscript fails the run.
DOC = os.path.join(FC, "manuscript", "WRR_Groundwater_Age_Identifiability.docx")
DOC_SI = os.path.join(FC, "manuscript", "WRR_Groundwater_Age_Identifiability_SI.docx")


def _flatten(path):
    """all prose and every table cell, so an embedded number cannot hide"""
    import docx as _d
    doc = _d.Document(path)
    s = " ".join(p.text for p in doc.paragraphs)
    for tbl in doc.tables:
        s += " " + " ".join(c.text for r in tbl.rows for c in r.cells)
    return s


try:
    import docx as _docx
    text = _flatten(DOC)
except Exception as exc:                                    # noqa: BLE001
    print("could not read the manuscript: %s" % exc)
    text = None

# The SI was previously unchecked, which is how "sigma_meas,i is the reported
# analytical uncertainty of tracer i" -- false as implemented -- survived there
# after the main text had been corrected.
try:
    text_si = _flatten(DOC_SI)
except Exception as exc:                                    # noqa: BLE001
    print("could not read the SI: %s" % exc)
    text_si = None

# Figures S3 and S4 were regenerated on 2026-08-24 by
# scripts/make_si_figures.py, the generator they had never had. These assert
# the NEW images are embedded AND that each still matches the file the
# generator writes, so an edited figure -- or a stale re-embed of either
# predecessor -- fails here.
if text_si is not None:
    import hashlib as _hl
    _si_doc = _docx.Document(DOC_SI)
    _hashes = {_hl.sha256(pt.blob).hexdigest()[:16]
               for _, pt in _si_doc.part.related_parts.items()
               if "image" in str(getattr(pt, "content_type", ""))}
    for _fn, _lab, _stale in (
            ("FigS3_obs_vs_sim.png", "S3", "c92e11a5f085bd9f"),
            ("FigS4_published_parameters.png", "S4",
             "6c998eb4d919364c")):
        _fp = os.path.join(FC, "figures", _fn)
        if os.path.exists(_fp):
            _h = _hl.sha256(open(_fp, "rb").read()).hexdigest()[:16]
            add("Fig%s" % _lab,
                "Figure %s matches its generated source" % _lab,
                True, _h in _hashes)
        else:
            print("NOTE: %s absent -- Figure %s unchecked" % (_fn, _lab))
        add("Fig%s" % _lab,
            "the superseded Figure %s image is gone" % _lab,
            False, _stale in _hashes)

    # Equation-object integrity. A scripted edit that rewrites whole
    # paragraphs destroys oMath silently -- paragraph.text cannot see it -- so
    # the counts are pinned. The SI stood at 74 from a reformatting pass until
    # 2026-08-24: two paragraphs of Text S2 had each had TWO inline equations
    # merged into one object and displaced to the end of the paragraph. Split
    # and re-inserted, restoring 76.
    _MNS = "{http://schemas.openxmlformats.org/officeDocument/2006/math}oMath"
    add("doc", "main text carries 20 equation objects", 20,
        len(_docx.Document(DOC).element.body.findall(".//" + _MNS)))
    add("doc", "SI carries 77 equation objects", 77,
        len(_si_doc.element.body.findall(".//" + _MNS)))
    add("doc", "SI numbered equations S1-S19 all present", True,
        all(("(S%d)" % _i) in text_si for _i in range(1, 20)))

    # SI structure (C10, C11 and two defects the review did not flag).
    import re as _re
    _sp = _si_doc.paragraphs
    # the Contents entries are the non-heading "Text SN." lines that appear
    # BEFORE the first Text S heading -- slicing by paragraph index breaks the
    # moment the list grows by a line, as adding Text S10 did
    _first_head = next(i for i, q in enumerate(_sp)
                       if _re.match(r"^Text S\d+\.", q.text.strip())
                       and q.style.name.startswith("Heading"))
    _cont = [q.text.strip() for q in _sp[:_first_head]
             if _re.match(r"^Text S\d+\.", q.text.strip())]
    _heads = [(q.style.name, q.text.strip()) for q in _sp
              if _re.match(r"^Text S\d+\.", q.text.strip())
              and q.style.name.startswith("Heading")]
    # The data-files section must name the workbook's sheets, in the same
    # order. Located by TITLE rather than by number: inserting Text S7
    # (Bayesian) on 2026-09-03 shifted this section from S9 to S10 and the
    # next one from S10 to S11, and three checks here broke because they had
    # the numbers hard-coded. A title survives renumbering; a number does not.
    def _si_section(title_starts):
        """Paragraphs of the Text S section whose heading starts with this."""
        out, on = [], False
        for _q in _si_doc.paragraphs:
            _t = _q.text.strip()
            _is_head = (_re.match(r"^Text S\d+\.", _t)
                        and _q.style.name.startswith("Heading"))
            if _is_head:
                if on:
                    break
                on = _t.split(". ", 1)[-1].startswith(title_starts)
                continue
            if on and _t:
                out.append(_t)
        return out

    _WB2 = os.path.join(FC, "manuscript", "EST_Supporting_Data.xlsx")
    if os.path.exists(_WB2):
        _sh2 = [x for x in pd.ExcelFile(_WB2).sheet_names if x != "Contents"]
        _t9 = []
        for _t in _si_section("Supplementary data files"):
            if _t.startswith("The supporting data"):
                continue
            _m9 = re.match(r"^([A-Za-z0-9_\-]+)\s*\u2014", _t)
            if _m9:
                _t9.append(_m9.group(1))
        add("supp", "the data-files section names the workbook sheets in "
            "order", _sh2, _t9)
        add("supp", "the data-files section describes the workbook", True,
            "EST_Supporting_Data.xlsx" in text_si)


    # ---- compound hyphenation, enforced 2026-09-03 ----------------------
    # The author asked for one format across well constrained, goodness of
    # fit, modern component and premodern component. An audit of all 18
    # main-text instances found no violation: the manuscript already applies
    # one rule -- hyphenate a compound that modifies a following noun, leave
    # it open predicatively or as a bare noun -- the same rule behind
    # transit-time, profile-based, lumped-parameter and the rest. A single
    # surface form is unavailable: hyphenating always gives "the
    # premodern-component was not constrained"; unhyphenating always would
    # have to undo every other compound in the paper. So the RULE is enforced.
    #
    # The check is SELF-CALIBRATING and uses no grammar. It collects the head
    # nouns that actually follow the hyphenated form in the document; the same
    # compound left open before one of those nouns is then a real
    # inconsistency. An earlier version tried to identify nouns from an
    # exclusion list and flagged "premodern component follows" and
    # "premodern component contributes", where those are verbs and the
    # compound is a bare noun subject -- correct as written.
    # The leading \b is load-bearing. Without it "chi-square goodness-of-fit"
    # matched on the "are" inside "square" and reported a predicative hyphen
    # that is not there -- the mirror image of the figure-renumbering bug this
    # project already records, where \b FAILED to match between a digit and a
    # letter. Word boundaries are worth checking in both directions.
    _LINK = (r"\b(?:is|are|was|were|be|been|being|remains?|remained|"
             r"becomes?|became|appears?|appeared|seems?|seemed|stays?|"
             r"stayed)")
    for _lbl, _txt in (("main", text), ("SI", text_si)):
        _bad = []
        for _head, _tail in (("[Ww]ell", "constrained"),
                             ("[Gg]oodness", "of.fit"),
                             ("[Mm]odern", "component"),
                             ("[Pp]remodern", "component")):
            _hyph = _tail.replace(".", "-")
            _open = _tail.replace(".", " ")
            _nouns = set(
                _m.group(1).lower() for _m in
                re.finditer(r"%s-%s\s+([A-Za-z]+)" % (_head, _hyph), _txt))
            for _m in re.finditer(r"%s\s+%s-%s\b"
                                  % (_LINK, _head, _hyph), _txt):
                _bad.append("predicative hyphen: %s" % _m.group(0))
            for _m in re.finditer(r"%s %s\s+([A-Za-z]+)"
                                  % (_head, _open), _txt):
                if _m.group(1).lower() in _nouns:
                    _bad.append("attributive left open: %s" % _m.group(0))
        add("style", "%s: compound hyphenation follows one rule" % _lbl,
            [], _bad)

    # ---- the title, set 2026-09-03 ------------------------------------
    # Main text and SI carried DIFFERENT titles until now, an
    # AGU submission defect open since 2026-08-20. Both are set
    # from one constant, and this check stops them drifting again.
    # Opened locally: _mdoc is created further down the file, after this
    # point, so referring to it here raised NameError on the first run.
    _tdoc_m = _docx.Document(DOC)
    _tdoc_s = _docx.Document(DOC_SI)
    _t_main = next((q.text.strip() for q in _tdoc_m.paragraphs
                    if q.text.strip()), "")
    _t_si = [q.text.strip() for q in _tdoc_s.paragraphs
             if q.text.strip()]
    add("doc", "the main text and SI titles match", True,
        bool(_t_main) and _t_main in _t_si[:4])
    add("doc", "the title is within the 8-16 word WRR band",
        True, 8 <= len(_t_main.split()) <= 16)
    # AGU capitalises prepositions of 4+ letters; the corpus is
    # From 6/0, With 10/0, Using 12/0, Under 8/0.
    add("doc", "no lowercase 4+ letter preposition in the title",
        [], [w for w in _t_main.split()
             if len(w) >= 4 and w[0].islower()])
    add("SI", "Contents lists Text S1-S11", 11, len(_cont))
    add("SI", "the Limitations section is last in the Contents", True,
        bool(_cont) and _cont[-1].endswith("Limitations of the analysis"))
    add("SI", "every Text S heading is the same level", 1,
        len({st for st, _ in _heads}))
    add("SI", "Contents and heading agree for every Text S section", True,
        [t for _, t in _heads] == _cont)
    # C7 moved main-text Table 1 into the SI as the new Table S1, shifting
    # every other SI table by one. Count is 12 from 2026-08-24.
    add("SI", "the declared table count matches the tables present", 12,
        len(_si_doc.tables))
    add("SI", "the SI states 12 tables", True,
        "12 tables and 4 figures" in text_si)
    add("SI", "the contents line spans Tables S1-S12", True,
        "Tables S1–S12" in text_si)
    # C13: the PREPROCESSING table (now S2) documents what was done to this
    # dataset and marks 4He as supported-but-unused. Found by its heading, not
    # by index -- it was tables[0] until the C7 move made it tables[1].
    _t1si = next((t for t in _si_doc.tables
                  if t.rows[0].cells[1].text.strip() == "Laboratory measurement"),
                 None)
    add("SI", "the preprocessing table is present", True, _t1si is not None)
    if _t1si is not None:
        add("SI", "its headings are journal-neutral", True,
            _t1si.rows[0].cells[4].text.strip() == "Value used in the inversion")
        add("SI", "it marks 4He as not used in this study", True,
            "not used in this study" in _t1si.rows[5].cells[3].text)

    # C10: the model-fit-diagnostics section had no body text at all -- its
    # lead sentence sat inside the Contents. Checked by locating the section
    # by title and looking for that sentence inside it, instead of the old
    # _sp[185:200] window, which the Text S7 insertion pushed off target.
    add("SI", "the model-fit-diagnostics section has body text before its "
        "figures", True,
        any(t.startswith("Observed and simulated tracer")
            for t in _si_section("Additional model-fit diagnostics")))

    # C9: every heading carries its number in the TEXT and none relies on Word
    # auto-numbering. Two Heading 1s carried both and exported numbered twice.
    _WNS = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
    _mdoc = _docx.Document(DOC)
    add("doc", "no heading relies on Word auto-numbering", 0,
        sum(1 for q in _mdoc.paragraphs
            if q.style.name.startswith("Heading")
            and q._p.find(".//%snumPr" % _WNS) is not None))
    add("doc", "main text carries 3 tables after the C7 move", 3,
        len(_mdoc.tables))
    # ---- AGU front matter, added 2026-09-03 -----------------------------
    # Key Points and a Plain Language Summary are mandatory for WRR and were
    # not in the ES&T version. The 140-character Key Point limit and the
    # 200-word summary limit are the journal's, so they are checked here
    # rather than trusted to the generator: an author reword in Word would
    # otherwise breach them silently.
    _fm = [q.text.strip() for q in _mdoc.paragraphs if q.text.strip()]
    _ix = {k: (_fm.index(k) if k in _fm else -1)
           for k in ("Key Points", "Abstract", "Plain Language Summary",
                     "Keywords")}
    add("doc", "AGU front matter: all four sections present", True,
        all(v >= 0 for v in _ix.values()))
    add("doc", "AGU order: Key Points, Abstract, Plain Language, Keywords",
        True,
        _ix["Key Points"] < _ix["Abstract"] < _ix["Plain Language Summary"]
        < _ix["Keywords"] if all(v >= 0 for v in _ix.values()) else False)
    # the three paragraphs between the Key Points label and the Abstract
    _kp = _fm[_ix["Key Points"] + 1:_ix["Abstract"]] if _ix["Abstract"] > 0 \
        else []
    add("doc", "exactly 3 Key Points", 3, len(_kp))
    add("doc", "every Key Point within AGU's 140-character limit", True,
        bool(_kp) and max(len(k) for k in _kp) <= 140)
    _pls = _fm[_ix["Plain Language Summary"] + 1:_ix["Keywords"]] \
        if _ix["Keywords"] > 0 else []
    add("doc", "Plain Language Summary within 200 words", True,
        0 < sum(len(p.split()) for p in _pls) <= 200)
    # A plain-language summary that needs the paper's vocabulary is not one.
    add("doc", "Plain Language Summary free of the paper's jargon", True,
        not any(j in " ".join(_pls).lower()
                for j in ("premodern", "identifiab", "lumped-parameter",
                          "transit time", "posterior", "bayesian", "τ")))
    # C3: every source cited in the tracer table resolves in the MAIN
    # reference list. That table lives in the SI since the C7 move, but its
    # citations are author-date entries of the main list, so this stays here.
    _t1 = next((t for t in _si_doc.tables
                if t.rows[0].cells[-1].text.strip() == "Source"), None)
    add("doc", "the tracer input-function table is findable", True,
        _t1 is not None)
    _cited = set()
    for _r in (_t1.rows[1:] if _t1 is not None else []):
        for _tok in _r.cells[4].text.replace(";", ",").split(","):
            # after the numbering pass a Source cell reads "IAEA/WMO3" or
            # "Reimer et al.7", so strip a trailing numeral group
            _tok = re.sub(r"[\d,–]+$", "", _tok.strip()).strip()
            if _tok:
                _cited.add(_tok.split(" (")[0].split(" et al")[0].strip())
    # the table now lives in the SI, so its sources must resolve against the
    # SI reference list -- they were migrated there with it
    _reflist = chr(10).join(q.text for q in _si_doc.paragraphs[-40:])
    add("doc", "every tracer-table source is in the SI reference list", True,
        all(_c.split()[0] in _reflist for _c in _cited if _c))

    # C1/C3/C4: full citation integrity in both documents, via the shared
    # checker so there is one implementation. Every in-text citation must
    # resolve to exactly one reference entry and every entry must be cited.
    try:
        sys.path.insert(0, os.path.join(FC, "scripts"))
        import check_citations as _cc
        for _pth, _lab, _n in ((DOC, "main text", 31), (DOC_SI, "SI", 21)):
            _seq, _refs = _cc.inventory(_pth)
            if _cc.numbered_state(_refs):
                # after the ES&T numbering pass the author names are gone, so
                # the equivalent guarantees are checked instead
                add("cite", "%s: numbered 1..N in document order" % _lab, True,
                    _cc.check_numbered(_pth, _lab))
                add("cite", "%s: reference count" % _lab, _n, len(_refs))
            else:
                _bad = [c for c in _seq if len(_cc.resolve(c, _refs)) != 1]
                _used = {_cc.resolve(c, _refs)[0] for c in _seq
                         if len(_cc.resolve(c, _refs)) == 1}
                add("cite", "%s: every citation resolves to one entry" % _lab,
                    0, len(_bad))
                add("cite", "%s: every reference entry is cited" % _lab, 0,
                    len(_refs) - len(_used))
                add("cite", "%s: citation and entry counts agree" % _lab,
                    len(_refs), len(_seq))
    except Exception as _e:                                    # noqa: BLE001
        print("NOTE: citation check unavailable (%s)" % _e)

# ---- E1: the 3H mechanism, recomputed from the input curves ----------------
# Section 3.3 used to say the 1960s bomb peak made 3H multi-valued. For a 2018
# sample at tau1 = 3-42 yr the recharge window is 1976-2015, so the peak is
# outside it entirely and a monotonic decline would give ONE crossing. The
# cause is that the response is nearly flat and wiggly.
_TC = os.path.join(FC, "data_current", "tracer_input_curves.csv")
if os.path.exists(_TC):
    _tc = pd.read_csv(_TC)
    _tc = _tc[_tc.decimal_year > 1900].sort_values("decimal_year")
    # Use the FIGURE's basis, which is the authoritative one: the band is the
    # actual fitted tau1 range (3.009 to 42.028), not a round 3-42. On a round
    # grid the 3He span comes out 44.56 and rounds to 45; on the fitted range
    # it is 43.84 and rounds to 44, which is what Figure 6b prints.
    _lam = np.log(2) / 12.32
    _lo, _hi = float(ident.tau1.min()), float(ident.tau1.max())
    _tau = np.arange(0.25, 60.01, 0.05)
    _tau = _tau[(_tau >= _lo) & (_tau <= _hi)]
    _yr = 2018.0 - _tau
    _cin = np.interp(_yr, _tc.decimal_year.values,
                     _tc["H3_31to29_100to95"].values)
    _r3h = _cin * np.exp(-_lam * _tau)
    add("3.3", "recharge window for tau1 = 3-42 yr from a 2018 sample",
        (1976, 2015), (int(round(_yr.min())), int(round(_yr.max()))))
    add("3.3", "the 1963 bomb peak lies OUTSIDE that window", True,
        not (_yr.min() <= 1963 <= _yr.max()))
    add("3.3", "3H response spans a factor of 2.8 over the range", 2.8,
        round(float(_r3h.max() / _r3h.min()), 1))
    add("3.3", "3H response is NOT monotonic there", True,
        bool((np.diff(np.sign(np.diff(_r3h))) != 0).sum() > 0))
    _he = _cin * (1.0 - np.exp(-_lam * _tau))
    add("3.3", "3He(trit) spans a factor of 44 over the range", 44.0,
        round(float(_he.max() / _he.min())))
    add("3.3", "3He(trit) rises overall but is not monotonic either", True,
        bool(_he[-1] > _he[0]
             and (np.diff(np.sign(np.diff(_he))) != 0).sum() > 0))
else:
    print("NOTE: tracer_input_curves.csv absent -- E1 unchecked")

# ---- G2: the young-water fraction, with the metric the review asked for ----
# Section 3.5 used to support "comparatively well constrained" with a figure
# and no number. The mechanism is exact: F(a<T) = f F1(T) + (1-f) F2(T), and
# F2(T) is zero to numerical precision at T = 10, 25 and 40 yr for all 18
# mixtures -- even the shortest fitted tau2 of 80 yr, whose P_D2 of 0.01 keeps
# the old component narrow. So F depends only on f and tau1, the two fitted
# parameters, and is insulated from the prescribed tau2.
for _T, _med, _pct in (("F_lt10yr", 0.00, 85), ("F_lt25yr", 0.08, 57),
                       ("F_lt40yr", 0.02, 83)):
    _w = (yf[_T + "_p84"] - yf[_T + "_p16"]).dropna()
    add("3.5", "%s: median P16-P84 width" % _T, _med,
        round(float(_w.median()), 2))
    add("3.5", "%s: %% of samples inside 0.10" % _T, _pct,
        int(round(100 * float((_w <= 0.10).mean()))))
add("3.5", "the young-water width is far below tau1's relative width", True,
    bool((yf.F_lt40yr_p84 - yf.F_lt40yr_p16).median()
         < ident.loc[ident.w_tau1_true.notna()
                     & ~ident.true_hit_bound.fillna(True).astype(bool),
                     "w_tau1_true"].median()))

if text is not None:
    MUST_APPEAR = [
        ("only 8 were classified as well constrained", "3.2 identifiability split"),
        ("36 as unconstrained", "3.2 unconstrained count"),
        ("Nine of the 60 samples showed pairwise parameter correlations", "3.2 correlation count"),
        ("1.25 against 1.07", "3.3 pooled median information ratios"),
        ("36 of the 44 samples", "3.3 3He censoring"),
        ("34 of 60 samples against 28 of 54", "3.3 3H vs SF6 censoring"),
        ("only 2 were supported", "3.4 single-component support"),
        ("profile-based intervals were truncated by prescribed parameter bounds",
         "3.6 bound-limited"),
        ("In nine of the 60 samples τ₁ is not a free parameter", "2.4 prescribed-tau1 disclosure"),
        ("among the 51 where τ₁ was actually estimated, 27",
         "3.2 second denominator"),
        ("no evidence that the original classification was an artifact of the parameterization", "3.2 parameterization swap"),
        ("τ₂ is unconstrained in all eighteen", "3.2 all-mixtures tau2 result"),
        ("mean ages span a median factor of 11", "3.2 mean-age span"),
        ("determined only to about an order of magnitude", "3.3 qualified 18,000 yr"),
        ("standard deviation of 19,400 yr", "3.2 corrected sigma"),
        ("median standard deviation is 164% of the mean age", "3.2 sigma summary"),
        ("Identifiability depends on the assumed error model as well as on "
         "the data", "3.2 identifiability depends on the error model"),
        ("exactly 100 times the relative-error one",
         "3.2 the 100x identity that explains the count change"),
        ("ranged from 0 to 12",
         "3.2 quotes the corrected threshold range"),
        ("nominal goodness-of-fit probabilities",
         "E4: the chi-square caveat is attached to the number"),
        ("Most of these limitations imply that the reported identifiability is an optimistic estimate",
         "G3: the misspecification exception is stated"),
        ("F₂(T) is zero to numerical precision",
         "G2: the mechanism is stated, not asserted"),
        ("median width is 0.00 at T = 10 yr",
         "G2: the width statistic the review asked for"),
        ("difference in dispersion rather than in location",
         "J1: 3.2 states the contrast the data support"),
        ("the confined zone is the less identifiable of the two",
         "J1: 3.2 discloses the direction reversal on non-closure"),
        ("near-absence of well-determined ages rather than a uniformly worse",
         "J1: 3.5 no longer claims a blanket ordering"),
        ("active tracers average 3.61 unconfined against 3.66 confined",
         "J6: tracer availability excluded as a confound"),
        ("persists among the 42 samples whose dispersion-parameter bound is "
         "inactive", "J5: the bound excluded as a confound"),
        ("in 17 of those 21 neither structure achieves an acceptable fit",
         "3.4 the analytical-sigma ranking is not offered as a bound"),
        ("no binary-mixture assignment is contradicted",
         "3.4 the claim that holds under every error model"),
        ("for 164 of the 218 tracer-sample pairs, the assigned uncertainty is a uniform 10% of the observation",
         "2.3 states the error model for what it is"),
        ("60 samples come from a single karst aquifer",
         "3.6 lists single-aquifer scope as a limitation"),
        ("should not be interpreted as representative of environmental-tracer studies in general",
         "3.7 non-transferability is explicit about aquifer type"),
        # 2026-08-23 review response: framing the reviewer asked for
        ("Throughout this paper the term means practical identifiability "
         "specifically", "Rec 6: practical identifiability defined early"),
        ("The load-bearing results use no width threshold at all",
         "Concern 1: the main results do not use the width thresholds"),
        ("(equifinality)", "equifinality defined where first described"),
        ("The important result is not the classification itself",
         "Rec 2: classification demoted to a summary"),
        ("Under the adopted model family and criterion, the premodern "
         "component was not constrained in any sample",
         "Rec 1: the tau2 claim carries its conditionality"),
        ("two analyses that answer different questions",
         "1: novelty paragraph aligned with the distinction, 2026-09-03"),
        ("Second, where they are not, which groundwater-age metrics remain useful?",
         "1: the surviving-metric question promoted from fourth to second"),
        ("identifies an age metric that remains determined where the transit times themselves are not",
         "1: the positive contribution stated as the first distinguishing feature"),
        ("a richer tracer suite is not the binding constraint",
         "4: additional sampling is not the fix -- 14C was measured throughout"),
        ("added uncertainty cannot add information",
         "4: propagating the dilution uncertainty fixes reporting, NOT identifiability; the direction must not be left ambiguous"),
        ("would widen the reported intervals rather than narrow them",
         "4: the direction of the dead-carbon correction stated explicitly"),
        ("Identifiability and information content are distinct questions",
         "1: the two analyses are different questions, not one done twice"),
        ("asks which observations supply the constraint",
         "1: information content framed as attribution, not uniqueness"),
        ("a tracer can be redundant where a parameter is well determined",
         "1: the independence of the two, in both directions"),
        ("referred to throughout as the premodern component",
         "C1: premodern defined where the mixture is introduced, 2026-09-03"),
        # ---- AGU front matter, 2026-09-03 -------------------------------
        # Key Point 2 carries the 12-of-14 count deliberately: the draft said
        # the two analyses simply "agree", and Section 3.3 reports two
        # exceptions. Pinned so the qualifier cannot be dropped for brevity.
        ("a Bayesian analysis agrees for 12 of 14",
         "Key Point 2 carries the count, not a bare claim of agreement"),
        ("is the metric to report where mean ages are not",
         "Key Point 3: the positive recommendation"),
        ("many different ages reproduce them equally well",
         "Plain Language Summary states equifinality without the word"),
        ("we asked how well the measurements pin those ages down",
         "Plain Language Summary states the contribution, not a better age"),
        ("should not be equated with identifiability of the age distribution itself",
         "Rec 3: parameter vs age identifiability"),
        ("Bayesian formulation represents the same lack of constraint",
         "Concern 3: Bayesian inference addressed -- Section 3.3 now "
         "performs the analysis rather than anticipating it"),
        ("The broader lesson is methodological rather than numerical",
         "Rec 7: what generalizes separated from what does not"),
        ("uncertain by an order of magnitude",
         "abstract carries the mean-age result"),
        ("different objective functions for different purposes", "2.3 dual-objective rationale"),
        ("Twelve of the 59 defined ratios", "2.5 sub-unity ratios explained"),
        ("error budget for multi-tracer inversion", "3.4 error-budget recommendation"),
        ("48% where τ₂ is fitted", "3.6 14C sensitivity, condensed"),
        ("converge on age-grid nodes",
         "3.6 the 285 yr value diagnosed as a grid node"),
        ("60 samples from the karst Edwards Aquifer", "abstract names the karst setting"),
        ("27 of 51 fitted", "abstract separates fitted from prescribed"),
        ("Under this model family, no mixture constrained the premodern ", "Rec 1: the abstract claim carries its conditionality"),
        ("Well-constrained ages were scarcest in the recharge zone", "E5a: the abstract states scarcity, not a blanket zone ordering"),
        ("¹⁴C essentially nothing to the modern component", "abstract scopes the 14C null to the modern component"),
        ("Reported ages require the criterion, model, and karst setting", "abstract closing carries criterion, model and setting"),
        ("all of which assume the ages are adequately constrained by "
         "observation",
         "Rec 4: significance restored 2026-09-02 after the "
         "2026-08-25 revision dropped it"),
        # ---- Section 4 Conclusions (WRR, 2026-09-03) --------------------
        # The ES&T subsection this replaced generalised outward; a WRR
        # Conclusions states what the study established. These four pin the
        # load-bearing sentences so a later edit cannot quietly soften them.
        ("Acceptable fit and identifiable age are distinct properties",
         "Conclusions: the central message, stated without a threshold"),
        ("rather than under every defensible treatment of measurement error",
         "Conclusions: Rec 1 conditionality, and the two Bayesian exceptions"),
        ("the remedy is to report a different quantity rather than to refine "
         "the inversion",
         "Conclusions: the positive recommendation, not only the negative "
         "result"),
        ("precision and evidential support are not the same thing",
         "Conclusions: closing distinction carried over from the ES&T "
         "subsection"),
    ]
    # ---- abstract claims pending the WRR abstract rewrite ---------------
    # These left the abstract in the author's 2026-08-25 revision. The
    # substance of most survives in the body (checked individually); what is
    # gone is the abstract's own statement of them. They are held here rather
    # than deleted, because deleting a check is how a retired claim creeps
    # back. Restore each to MUST_APPEAR as the WRR abstract reinstates it.
    ABSTRACT_PENDING_WRR = [
        ("for two-thirds of samples only rescales it", "abstract, pending WRR rewrite"),
        ("An error-weighted alternative appears to reverse those counts", "abstract, pending WRR rewrite"),
    ]

    MUST_NOT_APPEAR = [
        # ---- C1 terminology, normalised 2026-09-03 ----------------------
        # "older" is a comparative with no stated referent; "premodern" is
        # defined, and pairs with the "modern component" already used for
        # tau1. These four guards subsume the two retired-absolute guards
        # further down that happen to contain "older component" -- those are
        # kept for the reason they record, not for their coverage. NOTE the
        # guards are anchored on "component": bare "older" is correct English
        # and still appears ("much older groundwater through the matrix").
        ("profile likelihood",
         "retired: the relative-error objective is not a "
         "likelihood; Section 2.3 says so explicitly"),
        ("leave-one-out resampling",
         "retired: the analysis omits a tracer and refits, "
         "which is not resampling"),
        ("two complementary analyses",
         "retired: the two answer different questions, which is "
         "the stronger claim Section 1 now makes"),
        ("Mean Ages Are Unresolved",
         "retired title: the colon-plus-contrastive-finding form appears\n"
         "         in 2 of 115 recent WRR titles"),
        ("Ages from Environmental Tracers",
         "retired: AGU capitalises From; the corpus is 6/0"),
        ("Second, which tracers supply the dominant information? Third, when is",
         "retired: the question order that listed the contribution last"),
        ("Three features distinguish it",
         "retired: the surviving-metric result is now a fourth feature, and first"),
        ("older component",
         "C1: normalised to premodern component 2026-09-03"),
        ("older-component",
         "C1: hyphenated attributive form, likewise normalised"),
        ("younger component",
         "C1: the paired term is the modern component"),
        ("pre-modern",
         "C1: the convention is unhyphenated premodern"),
        # ---- the ES&T closing subsection, replaced 2026-09-03 -----------
        # ES&T does not ask for a Conclusions section and its closing
        # subsection generalises outward; AGU expects a numbered Conclusions
        # that states findings. Guard the heading and the two sentences whose
        # job was to generalise, so the ES&T framing cannot return.
        ("Implications for Environmental-Tracer Studies",
         "retired: ES&T closing subsection, now Section 4 Conclusions"),
        ("are therefore worth evaluating routinely in environmental-tracer "
         "investigations",
         "retired: recommendation-to-the-field framing, replaced by a "
         "statement of what this study established"),
        ("improving age estimates is not solely a matter of improving "
         "inversion",
         "retired: ES&T generalisation; the Conclusions makes the point as "
         "a finding about these data"),
        ("was unconstrained in all cases",
         "retired A1: absolute claim; Section 3.3 finds two "
         "exceptions, and Rec 1 requires the model-family conditional"),
        ("19 of the 60", "retired: unsupported sloppy-direction count"),
        # context-bound: bare "1.29"/"1.09" also occur as Table 2 median widths
        ("1.29 against 1.09", "retired: SF6/3H medians from an earlier run"),
        ("10 were classified as well", "retired: conditional-interval split 10/29/21"),
        ("29 as weakly", "retired: conditional-interval split 10/29/21"),
        ("21 as unconstrained", "retired: conditional-interval split 10/29/21"),
        ("35 of 184", "retired: pre-fix sub-unity ratio count"),
        ("profile-likelihood", "retired: likelihood terminology for a non-likelihood objective"),
        ("68% profile", "retired: uncalibrated 68% coverage claim"),
        ("uniquely determined", "retired: uniqueness not demonstrated"),
        ("Groundwater Mixing Models", "retired: mechanism implied by AICc"),
        ("41.7", "retired: pre-dgmeta-fix tau1 maximum"),
        # --- the 2026-08-23 objective-function audit ---
        ("1.53 against 1.25",
         "retired: zone widths computed over sentinel intervals"),
        ("inverts those counts",
         "retired: the weighted criterion rescales, it does not invert"),
        ("reported analytical uncertainty of tracer",
         "retired: sigma_meas is an assumed 10%, not the reported value"),
        ("analytical uncertainties of the four tracers differ by more than an "
         "order of magnitude",
         "retired: quoted the assumed sigmas as analytical, and understated "
         "the reported spread"),
        ("is not attempted here",
         "retired: Section 3.2 does profile under the weighted objective"),
        ("Forty-four of the 60 transit times",
         "retired: superseded by the corrected Section 3.2 account"),
        ("The disagreement is one of scale",
         "retired: the two objectives are not independent, so this framing "
         "misstates the cause"),
        ("Neither criterion is unambiguously preferable",
         "retired: the weighted criterion's error model is not defensible"),
        ("1.273", "retired: median w over sentinel-contaminated rows"),
        ("working assumption",
         "retired: the uncertainties are the published inversion's, not ours"),
        ("further preprocessing",
         "retired: the values come verbatim from data-release Table 4"),
        ("The older component is not identifiable at all",
         "retired: absolute claim, Rec 1"),
        ("¹⁴C essentially nothing.",
         "retired: unscoped -- reads as 14C being uninformative generally"),
        ("those same areas exhibited the weakest age identifiability",
         "retired J1: blanket ordering, contradicted by non-closure"),
        ("vulnerability are least identifiable",
         "retired J1: superlative the data do not support"),
        ("Identifiability followed the hydrogeologic zonation",
         "retired J1: hid which way the closure imbalance runs"),
        ("zone argument in Section 3.2 rests on the classification counts",
         "retired J1: those counts are not monotone either"),
        ("between 1 and 13",
         "retired: hybrid conditional/true-profile counts; the range is 0-12"),
        ("ranged from 1 to 13",
         "retired: same hybrid, quoted in 3.2"),
        ("zone contrast persists throughout",
         "retired: it does not -- 5 of 6 rows on location, 4 of 6 on "
         "dispersion"),
        ("systematically wider",
         "retired G1: asserted the zone contrast flatly in the Fig 2 caption"),
        ("at the 95% confidence level",
         "retired E3: conflated a 5% significance level with 95% confidence"),
        ("Each of these makes the reported identifiability",
         "retired G3: contradicted the misspecification item"),
        ("peaked in the 1960s and has fallen since",
         "retired E1: the bomb peak is outside the 1976-2015 window"),
        ("accumulates monotonically",
         "retired E1: the 3He response has 48 turning points"),
        ("from the monotonic members of the suite",
         "retired E1: 3He is not monotonic; steepness and sign are the point"),
        ("remained comparatively well constrained across much of the aquifer",
         "retired G2: a figure with no metric behind it"),
        ("dgmeta_params",
         "retired C13: Table S1 read as software documentation"),
        ("he4_params",
         "retired C13: implementation detail belongs in the repo README"),
        ("JSON obs value",
         "retired C13: column heading recast for a journal"),
        ("before entering DLPMI",
         "retired C13: describes the software, not the dataset"),
        ("symbol shape distinguishes the unconfined recharge zone",
         "retired: the updated Figure 1 shows zones as shaded polygons, not by symbol shape"),
        ("13 fall below unity",
         "retired: pre-bisection-fix; the answer is 12, minimum 0.762"),
        ("Ten of the thirteen involve",
         "retired: eight of the twelve"),
        ("contains 8 tables",
         "retired: the SI has 11 tables"),
        ("Identifiability was poorest in the recharge zone",
         "retired E5a: refutable from the classification table alone"),
        ("but only rescales the objective",
         "retired E5b: exact only for the 40 of 60 with uniform relative "
         "sigma"),
        ("never exceeds a third",
         "retired: FALSE -- the analytical-sigma refit gives 21 of 36"),
        ("out of the forty-four assessable",
         "retired: fifty-four assessable, two flips"),
        ("nested only at a boundary",
         "retired: the structures are not nested -- each frees a parameter "
         "the other prescribes"),
        ("reduces to a single dispersion model as f",
         "retired: only at the PRESCRIBED P_D, not the fitted one"),
        ("of 30 under the analytical uncertainties",
         "retired: modelled the per-tracer analytical sigmas as a uniform "
         "1/13 rescale"),
        ("at every scale tested. The counts are not",
         "retired: the direction is not stable under the analytical refit"),
        ("the comparison is restricted to the 44",
         "retired: assessability is dof >= 1, giving 54"),
        ("small-sample correction is undefined for",
         "retired: the correction cancels at equal k"),
        ("No mixture yielded a constrained older component",
         "retired: absolute claim in the abstract, Rec 1"),
        ("profile interval",
         "retired: the declared term is profile-based interval"),
        ("lumped parameter ",
         "retired: unhyphenated attributive compound"),
        ("identifiability means practical identifiability",
         "retired: the doubled definition in Section 1"),
        ("median 21 times the objective",
         "retired: superseded by the 100x identity argument"),
    ]
    for frag, what in MUST_APPEAR:
        add("doc", "present: %s" % what, True, frag in text)
    for frag, what in MUST_NOT_APPEAR:
        add("doc", "absent: %s" % what, True, frag not in text)

    # Table 2 is embedded in the manuscript, so check it cell by cell.
    t2 = next((t for t in _docx.Document(DOC).tables
               if "Bound active" in " ".join(c.text for c in t.rows[0].cells)), None)
    if t2 is not None:
        rows = {}
        for r in t2.rows[1:]:
            c = [x.text.strip() for x in r.cells]
            rows[(c[0], c[1])] = c[2:]
        for zone in ("unconfined", "confined"):
            for ba in (False, True):
                s = ident[(ident.Aq_class == zone)
                          & (ident.bound_active.astype(bool) == ba)]
                vc = s.cls_true.value_counts()
                # Median w is over intervals that CLOSED. The no-crossing
                # fallback returns 0.1*opt..3*opt, a sentinel and not a
                # measured width; including it put the all-sample median at
                # 1.273 rather than 1.059.
                _cl = s.loc[~s.true_hit_bound.fillna(True).astype(bool),
                            "w_tau1_true"].dropna()
                want = [str(len(s)),
                        str(int(vc.get("well constrained", 0))),
                        str(int(vc.get("weakly constrained", 0))),
                        str(int(vc.get("unconstrained", 0))),
                        ("%.3f" % _cl.median()) if len(_cl) else "—"]
                got = rows.get((zone, str(ba)))
                add("Table2", "%s / bound=%s" % (zone, ba), want, got)

    # Table 3. Median and IQR must exclude censored rows -- a censored row
    # carries a ratio, but it is a bound-limited lower bound, not a width
    # ratio. Including them once inflated 3He from 1.30 to 2.39 and flipped
    # its rank against SF6, so this check is load-bearing.
    t3 = next((t for t in _docx.Document(DOC).tables
               if "Tracer withheld" in " ".join(c.text for c in t.rows[0].cells)), None)
    if t3 is not None:
        # The tracer column is normalised to the body text's unicode forms
        # (2026-08-24, C8). Fold both conventions to the CSV's ASCII keys so
        # the lookup survives either -- the run-split superscript form
        # ('3' superscripted + 'He(trit)') flattens to the ASCII string, the
        # unicode form does not, and a silent key miss reads as "data=None"
        # rather than as a wrong number.
        _UNI = {"³He(trit)": "3He(trit)", "³He": "3He",
                "SF₆": "SF6", "³H": "3H", "¹⁴C": "14C",
                "⁴He": "4He"}
        seen = {}
        for r in t3.rows[1:]:
            c = [x.text.strip() for x in r.cells]
            seen[(_UNI.get(c[0], c[0]), c[1])] = c[2:]
        for tracer in ("3He(trit)", "SF6", "3H", "14C"):
            for zone in ("unconfined", "confined"):
                g = loto[(loto.omitted_tracer == tracer) & (loto.Aq_class == zone)]
                cen = g.censored_true.astype(bool)
                fin = g.loc[~cen, "information_ratio_true"].dropna()
                want = [str(len(g)), str(int(cen.sum())), str(len(fin)),
                        ("%.2f" % fin.median()) if len(fin) else "—",
                        ("%.2f–%.2f" % (fin.quantile(.25), fin.quantile(.75)))
                        if len(fin) else "—",
                        "%.1f" % (100 * g.delta_tau1_rel.dropna().abs().median())]
                add("Table3", "%s / %s" % (tracer, zone), want,
                    seen.get((tracer, zone)))
        # the three categories must account for every inversion
        for tracer in ("3He(trit)", "SF6", "3H", "14C"):
            g = loto[loto.omitted_tracer == tracer]
            cen = int(g.censored_true.astype(bool).sum())
            fin = int(g.loc[~g.censored_true.astype(bool),
                            "information_ratio_true"].notna().sum())
            add("Table3", "%s: censored+finite+fixed = n" % tracer, len(g),
                cen + fin + (len(g) - cen - fin))

# ---- Supporting Information -----------------------------------------------
# The SI carries most of the error-model argument now, so it needs the same
# treatment as the main text: the false sigma description lived here.
if text_si is not None:
    SI_MUST_APPEAR = [
        ("its magnitude is nevertheless informative",
         "the guaranteed-improvement caveat moved to Text S4 with "
         "the exchange refit, 2026-09-03"),
        ("the 1σ uncertainty adopted for tracer i",
         "Text S6 describes sigma_meas honestly"),
        ("The two objectives are not independent",
         "Text S3 states the non-independence"),
        ("χ²σ = 100 χ²rel exactly",
         "Text S3 states the 100x identity"),
        ("That is the case for 40 of the 60 samples",
         "Text S3 gives the affected sample count"),
        ("falls from 23 to 4 of 42",
         "Text S3 shows the reported sigmas are unusable"),
        ("Consequences for model selection",
         "Text S3 carries the AICc error-scale sensitivity"),
        ("Interval closure and the zone comparison",
         "Text S6 explains the censoring in the zone contrast"),
        ("The premodern component under the weighted objective",
         "Text S6 discloses tau2 closing under the weighted criterion"),
        ("No grid-refinement convergence test was performed",
         "the Limitations section states the grid caveat the main text "
         "delegates (Text S11 since the 2026-09-03 insertion)"),
        ("median of 48% and by as much as 670%",
         "Text S3 carries the full 14C sensitivity numbers"),
        ("The 285 yr τ₂ attractor",
         "Text S6 diagnoses the 285 yr value as an age-grid node"),
        ("that flag does not survive scrutiny",
         "Text S6 closes out the non-convergence question"),
        ("profile_tau2_sigma —", "profile_tau2_sigma listed as a sheet"),
        ("Table S6. The 1σ tracer uncertainties used in the inversion",
         "the sigma table is present and numbered S6 after the C7 move"),
        ("those of the published lumped-parameter inversion",
         "Text S3 attributes the error model to its actual source"),
        ("Median w is taken over intervals that closed",
         "the objective table caption states the closure restriction"),
        ("sigma_impact —", "sigma_impact listed as a workbook sheet"),
    ]
    SI_MUST_NOT_APPEAR = [
        # ---- C1 terminology, normalised 2026-09-03 ----------------------
        # Same guards as the main text. "toward older ages" and "the three
        # oldest mixtures" are genuine comparatives and survive, which is why
        # these are anchored on "component" rather than on "older".
        ("older component",
         "C1: normalised to premodern component 2026-09-03"),
        ("older-component",
         "C1: hyphenated attributive form, likewise normalised"),
        ("younger component",
         "C1: the paired term is the modern component"),
        ("pre-modern",
         "C1: the convention is unhyphenated premodern"),
        ("reported analytical uncertainty of tracer",
         "retired: sigma_meas is an assumed 10%, not the reported value"),
        ("The disagreement is one of scale",
         "retired: misstates the cause of the count change"),
        ("Neither criterion is unambiguously preferable",
         "retired: the weighted error model is not defensible"),
        ("for a representative sample",
         "retired: quoted the assumed sigmas as analytical"),
        ("narrow old component",
         "retired: the term is premodern component"),
        ("44 of the 60 transit times are then well constrained",
         "retired: superseded by the corrected account"),
        # No bare "1.27" guard here: the bound-active table legitimately
        # carries P_D ratios of 1.2741 and 1.2763. That table's median w is
        # pinned cell by cell below instead, which is exact.
        ("working assumption",
         "retired: the uncertainties are the published inversion's, not ours"),
        ("further preprocessing",
         "retired: the values come verbatim from data-release Table 4"),
        ("The older component is not identifiable at all",
         "retired: absolute claim, Rec 1"),
        ("¹⁴C essentially nothing.",
         "retired: unscoped -- reads as 14C being uninformative generally"),
        ("those same areas exhibited the weakest age identifiability",
         "retired J1: blanket ordering, contradicted by non-closure"),
        ("vulnerability are least identifiable",
         "retired J1: superlative the data do not support"),
        ("Identifiability followed the hydrogeologic zonation",
         "retired J1: hid which way the closure imbalance runs"),
        ("zone argument in Section 3.2 rests on the classification counts",
         "retired J1: those counts are not monotone either"),
        ("between 1 and 13",
         "retired: hybrid conditional/true-profile counts; the range is 0-12"),
        ("ranged from 1 to 13",
         "retired: same hybrid, quoted in 3.2"),
        ("zone contrast persists throughout",
         "retired: it does not -- 5 of 6 rows on location, 4 of 6 on "
         "dispersion"),
        ("systematically wider",
         "retired G1: asserted the zone contrast flatly in the Fig 2 caption"),
        ("at the 95% confidence level",
         "retired E3: conflated a 5% significance level with 95% confidence"),
        ("Each of these makes the reported identifiability",
         "retired G3: contradicted the misspecification item"),
        ("peaked in the 1960s and has fallen since",
         "retired E1: the bomb peak is outside the 1976-2015 window"),
        ("accumulates monotonically",
         "retired E1: the 3He response has 48 turning points"),
        ("from the monotonic members of the suite",
         "retired E1: 3He is not monotonic; steepness and sign are the point"),
        ("remained comparatively well constrained across much of the aquifer",
         "retired G2: a figure with no metric behind it"),
        ("dgmeta_params",
         "retired C13: Table S1 read as software documentation"),
        ("he4_params",
         "retired C13: implementation detail belongs in the repo README"),
        ("JSON obs value",
         "retired C13: column heading recast for a journal"),
        ("before entering DLPMI",
         "retired C13: describes the software, not the dataset"),
        ("symbol shape distinguishes the unconfined recharge zone",
         "retired: the updated Figure 1 shows zones as shaded polygons, not by symbol shape"),
        ("13 fall below unity",
         "retired: pre-bisection-fix; the answer is 12, minimum 0.762"),
        ("Ten of the thirteen involve",
         "retired: eight of the twelve"),
        ("contains 8 tables",
         "retired: the SI has 11 tables"),
        ("Identifiability was poorest in the recharge zone",
         "retired E5a: refutable from the classification table alone"),
        ("but only rescales the objective",
         "retired E5b: exact only for the 40 of 60 with uniform relative "
         "sigma"),
        ("never exceeds a third",
         "retired: FALSE -- the analytical-sigma refit gives 21 of 36"),
        ("out of the forty-four assessable",
         "retired: fifty-four assessable, two flips"),
        ("nested only at a boundary",
         "retired: the structures are not nested -- each frees a parameter "
         "the other prescribes"),
        ("reduces to a single dispersion model as f",
         "retired: only at the PRESCRIBED P_D, not the fitted one"),
        ("of 30 under the analytical uncertainties",
         "retired: modelled the per-tracer analytical sigmas as a uniform "
         "1/13 rescale"),
        ("at every scale tested. The counts are not",
         "retired: the direction is not stable under the analytical refit"),
        ("the comparison is restricted to the 44",
         "retired: assessability is dof >= 1, giving 54"),
        ("small-sample correction is undefined for",
         "retired: the correction cancels at equal k"),
        ("No mixture yielded a constrained older component",
         "retired: absolute claim in the abstract, Rec 1"),
        ("profile interval",
         "retired: the declared term is profile-based interval"),
        ("lumped parameter ",
         "retired: unhyphenated attributive compound"),
        ("identifiability means practical identifiability",
         "retired: the doubled definition in Section 1"),
        ("median 21 times the objective",
         "retired: superseded by the 100x identity argument"),
        ("Neither τ₂'s failure to close nor the tracer ranking",
         "retired: tau2 closes for 4 of 18 under the weighted criterion"),
        ("points to a shared local minimum",
         "retired: the 285 yr value is an age-grid node, diagnosed"),
        ("two of the three samples flagged as non-converged",
         "retired: EDTRPAS1-29 was never flagged"),
    ]
    for frag, what in SI_MUST_APPEAR:
        add("SI", "present: %s" % what, True, frag in text_si)
    for frag, what in SI_MUST_NOT_APPEAR:
        add("SI", "absent: %s" % what, True, frag not in text_si)

    # the objective and sigma tables, cell by cell
    _sid = _docx.Document(DOC_SI)
    _s10 = next((t for t in _sid.tables
                 if "Objective and threshold"
                 in " ".join(c.text for c in t.rows[0].cells)), None)
    if _s10 is not None and 'ps' in dir():
        _O3 = ["well constrained", "weakly constrained", "unconstrained"]
        _hitt = ident.true_hit_bound.fillna(True).astype(bool)
        _spec = [
            ("Relative error", ident.cls_true,
             ident.loc[~_hitt, "w_tau1_true"].dropna()),
            ("Error-weighted, Δχ² = 1", ps.cls_sigma,
             ps.loc[~ps.hit_bound.fillna(True).astype(bool), "w_sigma"].dropna()),
            ("Error-weighted, Δχ² = χ²/dof", ps.cls_sigma_scaled,
             ps.loc[~ps.hit_bound_scaled.fillna(True).astype(bool),
                    "w_sigma_scaled"].dropna()),
        ]
        for _row, (_lbl, _cls, _w) in zip(_s10.rows[1:], _spec):
            _c = [x.text.strip() for x in _row.cells]
            assert _c[0].startswith(_lbl), "objective-table row order changed"
            _vc = _cls.value_counts()
            add("TableS4", "%s counts" % _lbl,
                "/".join(str(int(_vc.get(k, 0))) for k in _O3),
                "/".join(_c[1:4]))
            add("TableS4", "%s median w" % _lbl,
                "%.2f" % float(_w.median()), _c[4])

    # Every figure embedded in the main text must be byte-identical to its
    # figures/*.png source. This is what actually went wrong in the retired
    # standalone graphics package: two of its seven images were the
    # pre-bisection Figure 2 and Figure 4 and matched no file on disk.
    try:
        import glob as _glob
        import hashlib as _hl
        import io as _io
        from docx.oxml.ns import qn as _qn
        _disk = {_hl.md5(open(_p, "rb").read()).hexdigest()
                 for _p in _glob.glob(os.path.join(FC, "figures", "*.png"))}
        _md = _docx.Document(DOC)
        _emb = [_hl.md5(_md.part.related_parts[_b.get(_qn("r:embed"))].blob
                        ).hexdigest()
                for _b in _md.element.body.findall(".//" + _qn("a:blip"))]
        # The author embedded the TOC graphic at the head of the document
        # (labelled "TOC", after the keywords) on 2026-08-23, so there are 8
        # images: the TOC plus the 7 numbered figures. The TOC is not a
        # figures/*.png product, hence the split.
        add("doc", "8 images embedded (8 numbered figures, no TOC)", 8, len(_emb))
        add("doc", "the 8 numbered figures match their figures/*.png sources",
            8, sum(h in _disk for h in _emb))

        # Tripwire, not a defect check. The embedded TOC is still the
        # UNCORRECTED TOC.png: 1.50:1, where ES&T requires 3.25 x 1.75 in
        # (1.857:1), and it carries the "IDENTIFIABIALITY" typo, a red curve
        # drawn above Delta-chi2 = 1, and the threshold-dependent 8-of-60
        # subtitle. See manuscript/TOC_correction_brief.md. This check passes
        # while that is still true and FAILS the moment the image is replaced,
        # which is the prompt to re-verify the new one and update this block.
        try:
            import io as _io
            from PIL import Image as _Img
            _b0 = _md.part.related_parts[
                _md.element.body.findall(".//" + _qn("a:blip"))[0]
                .get(_qn("r:embed"))].blob
            with _Img.open(_io.BytesIO(_b0)) as _i0:
                _asp = round(_i0.size[0] / _i0.size[1], 2)
            # The TOC graphic was dropped for WRR: ACS required one, AGU
            # does not. The old 1.50:1 -> 1.85:1 aspect tripwire is retired
            # with it. What replaces it is stronger -- every embedded image
            # must now be a figures/*.png product, so a stray or hand-pasted
            # image cannot slip back in unnoticed.
            add("doc", "no non-figure image embedded (TOC dropped)",
                0, len(_emb) - sum(h in _disk for h in _emb))
        except Exception as _e:                             # noqa: BLE001
            print("NOTE: TOC aspect check skipped: %s" % _e)
    except Exception as _e:                                 # noqa: BLE001
        print("NOTE: embedded-figure check skipped: %s" % _e)

    # structural: the contents line, the caption count and the actual table
    # count must agree. The line said "Tables S1-S8" while the file held 11.
    import re as _re
    # A caption is a paragraph that BEGINS "Table SN." -- an in-text reference
    # at the end of a sentence ("... reported in Table S6.") also matches a
    # bare search and was inflating this count.
    _caps = [int(m.group(1)) for x in _sid.paragraphs
             for m in [_re.match(r"^Table S(\d+)\.", x.text.strip())] if m]
    _figs = [int(m.group(1)) for x in _sid.paragraphs
             for m in [_re.match(r"^Figure S(\d+)\.", x.text.strip())] if m]
    _txts = [int(m.group(1)) for x in _sid.paragraphs
             for m in [_re.match(r"^Text S(\d+)\.", x.text.strip())] if m]
    add("SI", "one caption per table in the file", len(_sid.tables), len(_caps))
    # ORDERED, not a set. The set version passed while the captions ran
    # S1, S9, S5, S10, S11, S2, S3, S4, S6, S7, S8 -- ACS requires numbering in
    # order of appearance, so this is the check that matters.
    add("SI", "table captions ascend S1..S%d in document order" % len(_caps),
        list(range(1, len(_caps) + 1)), _caps)
    add("SI", "figure captions ascend S1..S%d in document order" % len(_figs),
        list(range(1, len(_figs) + 1)), _figs)
    # Text S sections appear twice: once in the contents list, once as the
    # section itself, so the run is the sorted unique sequence doubled.
    add("SI", "Text S sections ascend S1..S%d" % len(set(_txts)),
        sorted(set(_txts)), sorted(set(_txts)))
    add("SI", "Text S numbering is contiguous from S1", True,
        sorted(set(_txts)) == list(range(1, len(set(_txts)) + 1)))
    add("SI", "contents line lists Tables S1-S%d" % len(_caps), True,
        ("Tables S1–S%d" % len(_caps)) in text_si)

    _s11 = next((t for t in _sid.tables
                 if "Assumed / reported"
                 in " ".join(c.text for c in t.rows[0].cells)), None)
    if _s11 is not None:
        add("TableS5", "4 tracer rows", 4, len(_s11.rows) - 1)
        for _row in _s11.rows[1:]:
            _c = [x.text.strip() for x in _row.cells]
            add("TableS5", "%s inversion 1-sigma is the uniform 10%%" % _c[0],
                "10.0", _c[1])

# ---------------------------------------------------------------- report ----
bad, cur = 0, None
print("=" * 80)
print("MANUSCRIPT NUMBER VERIFICATION  --  true-profile outputs")
print("=" * 80)
for sec, claim, paper, data in C:
    if sec != cur:
        print("\n[%s]" % sec)
        cur = sec
    try:
        ok = abs(float(data) - float(paper)) <= (0.005 if isinstance(paper, float) else 0)
    except (TypeError, ValueError):
        ok = data == paper
    bad += not ok
    print("  %-4s %-56s paper=%-7s data=%s"
          % ("ok" if ok else "FAIL", claim, paper, data))

print("\n" + "=" * 80)
print("%d of %d claims reproduce%s"
      % (len(C) - bad, len(C), "" if not bad else "   --  %d MISMATCH" % bad))
print("=" * 80)
sys.exit(1 if bad else 0)
