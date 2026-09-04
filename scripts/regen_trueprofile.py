#!/usr/bin/env python3
"""
regen_trueprofile.py -- rebuild the figures and tables that depend on the
tau1 profile interval, using the TRUE profile-likelihood results.

Affected artifacts (nothing else depends on the interval):
    Fig 2  tau1 with its 68% profile interval
    Fig 3  identifiability vs depth + classification by zone
    Fig 4  information ratios by tracer and zone
    Fig 8  panel (b) buckets samples by identifiability class
    Table 2  identifiability classification
    Table 3  information ratios by tracer

Figures 5, 6 and 7 are unaffected: Fig 5 is AICc-based, Fig 6 is Monte Carlo,
Fig 7 is input-function geometry.

The existing plotting functions are reused unchanged; only their inputs are
swapped, with the true-profile columns renamed onto the names those functions
already expect. The "not assessable" legend entry is dropped, since the class
has had no members since the dgmeta regeneration.
"""
from __future__ import annotations
import os
import sys

import numpy as np
import pandas as pd

FC = r"D:\projects\carbon isotopic\DLMPI_reanalysis\From_Claude_chat"
HO = r"D:\projects\carbon isotopic\DLMPI_reanalysis\DLPMI_edwards_handoff"
TP = os.path.join(HO, "sweeps_trueprofile")
OUT = os.path.join(FC, "figures")

sys.path.insert(0, os.path.join(FC, "scripts"))
import make_figures as MF
import make_figures_new as MN

# The strip overlay in fig4() jitters points with np.random.uniform. Without a
# seed this script emitted a visibly different Figure 4 on every run -- same
# data, different scatter -- so the embedded figure could never be reproduced.
# make_figures.py seeds in main(), which this script bypasses by calling the
# plotting functions directly.
np.random.seed(0)

# the class has had no members since the dgmeta regeneration
MF.CLS_ORDER = ["well constrained", "weakly constrained", "unconstrained"]
MN.CLS_ORDER = list(MF.CLS_ORDER)

# ── inputs ──────────────────────────────────────────────────────────────
ident = pd.read_csv(os.path.join(TP, "sweep_identifiability_trueprofile.csv"))
loto = pd.read_csv(os.path.join(TP, "profile_true_loto.csv"))
base = pd.read_csv(os.path.join(HO, "reference", "00_summary_table.csv"))
depth = pd.read_csv(os.path.join(FC, "figures",
                                 "Fig1_sample_locations_SUPERSEDED.csv"))

# ── rename true-profile columns onto the names the plotters expect ──────
idt = ident.copy()
idt["w_tau1"] = idt["w_tau1_true"]
idt["identifiability_final"] = idt["cls_true"]
idt["profile_hit_bound"] = idt["true_hit_bound"].fillna(True).astype(bool)

# fig2 reads the interval from `base`, not from `idt`
b = base.merge(ident[["SampleID", "tau1_lo_true", "tau1_hi_true"]],
               on="SampleID", how="left")
b["tau1_profile_lo"] = b["tau1_lo_true"]
b["tau1_profile_hi"] = b["tau1_hi_true"]

lo = loto.copy()
lo["information_ratio"] = lo["information_ratio_true"]
lo["censored"] = lo["censored_true"].astype(bool)
lo["w_full"] = lo["w_full_true"]
lo["w_reduced"] = lo["w_reduced_true"]

dep = depth[["SampleID", "Depth_m"]].drop_duplicates("SampleID")
dep["Depth_m"] = pd.to_numeric(dep["Depth_m"], errors="coerce")

print("inputs:", len(idt), "samples,", len(lo), "loto rows")
print("classes:", idt.identifiability_final.value_counts().to_dict())

# ── figures ─────────────────────────────────────────────────────────────
print(MF.fig2(idt, b, OUT))
print(MF.fig3(idt, dep, OUT))


def fig4_trueprofile(lo, out):
    """Replacement for make_figures.fig4.

    The original plots every row with a numeric ratio inside the boxes and
    labels that count "samples with a finite ratio" -- but censored rows carry
    a numeric ratio too. Under true profiling 125 of 218 inversions are
    censored, so that design shows mostly boundary-limited values as though
    they were likelihood-derived. It also still drops three samples under the
    not-assessable rule retired in the dgmeta regeneration.

    Panel (a) therefore leads with the robust statistic -- the fraction of
    inversions in which withholding the tracer leaves tau1 unconstrained --
    and panel (b) shows only the genuinely finite ratios, with their counts.
    """
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    order = ["3He(trit)", "SF6", "3H", "14C"]
    label = {"3He(trit)": "$^{3}$He$_{trit}$", "SF6": "SF$_6$",
             "3H": "$^{3}$H", "14C": "$^{14}$C"}
    zc = {"unconfined": "#56B4E9", "confined": "#009E73"}
    CM = MF.CM

    fig, axes = plt.subplots(1, 2, figsize=(17.5 * CM, 7.6 * CM))

    ax = axes[0]
    for i, tr in enumerate(order):
        for j, (zone, col) in enumerate(zc.items()):
            g = lo[(lo.omitted_tracer == tr) & (lo.Aq_class == zone)]
            if not len(g):
                continue
            frac = g.censored.mean()
            x = i + (j - 0.5) * 0.34
            ax.bar(x, frac, 0.30, color=col, edgecolor="black", linewidth=0.4)
            ax.text(x, frac + 0.02, f"{int(g.censored.sum())}/{len(g)}",
                    ha="center", fontsize=6, color="0.25")
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label[t] for t in order])
    ax.set_xlabel("Tracer withheld")
    ax.set_ylabel(r"Fraction leaving $\tau_1$ unconstrained")
    ax.set_ylim(0, 1.12)
    ax.set_title("(a) how often omission destroys the constraint", loc="left")
    ax.spines[["top", "right"]].set_visible(False)
    h = [plt.Rectangle((0, 0), 1, 1, fc=c, ec="black", lw=0.4, label=z)
         for z, c in zc.items()]
    ax.legend(handles=h, frameon=False, loc="upper right",
              title="aquifer zone", title_fontsize=7)

    ax = axes[1]
    for i, tr in enumerate(order):
        for j, (zone, col) in enumerate(zc.items()):
            g = lo[(lo.omitted_tracer == tr) & (lo.Aq_class == zone)]
            v = g[~g.censored].information_ratio.dropna()
            x = i + (j - 0.5) * 0.34
            ax.text(x, 0.63, f"n={len(v)}", ha="center", va="bottom",
                    fontsize=6, color="0.3")
            if len(v) < 2:
                if len(v) == 1:
                    ax.scatter([x], v.values, s=14, c=col, edgecolors="black",
                               linewidths=0.4, zorder=4)
                continue
            bp = ax.boxplot([v.values], positions=[x], widths=0.28,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=1.0),
                            whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                            boxprops=dict(lw=0.5))
            bp["boxes"][0].set_facecolor(col)
            bp["boxes"][0].set_alpha(0.85)
            ax.scatter(np.full(len(v), x) + np.random.uniform(-.05, .05, len(v)),
                       v.values, s=4, c="black", alpha=0.35, zorder=5)
    ax.axhline(1.0, color="0.3", ls=":", lw=0.8)
    ax.set_xticks(range(len(order)))
    ax.set_xticklabels([label[t] for t in order])
    ax.set_xlabel("Tracer withheld")
    ax.set_ylabel(r"Information ratio $I_j$ for $\tau_1$ (finite only)")
    ax.set_ylim(0.6, None)
    ax.set_title("(b) interval ratio where it is defined", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    fig.text(0.012, 0.015,
             "Panel (b) excludes censored inversions, in which the interval "
             "terminates at a prescribed bound; n is the number remaining. "
             r"$^{3}$He$_{trit}$ retains only one.", fontsize=5.8, color="0.3")
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    p = os.path.join(out, "Fig4_information_ratio.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


print(fig4_trueprofile(lo, OUT))

yf = pd.read_csv(os.path.join(FC, "data_current", "young_fraction.csv"))
p8, rho = MN.fig8_published(yf, idt, OUT)
print(p8, f"(Spearman rho={rho:.4f})")

# ── Table 2: identifiability classification ─────────────────────────────
ORD = ["well constrained", "weakly constrained", "unconstrained"]

# median_w is taken over intervals that actually CLOSED. Where the outward
# walk finds no Delta-chi2 crossing inside the search window, true_profile()
# returns hi = min(3*opt, hi_b) and lo = max(lo_b, 0.1*opt) and sets
# true_hit_bound -- a sentinel (w = 2.90 unclipped), not a measured width.
# Including those rows mixes measured widths with manufactured ones; it put
# the all-sample median at 1.273 instead of 1.059 before 2026-08-23. The
# classification is unaffected: no crossing inside the window means
# unconstrained, which is what those samples are called either way.


def _med_closed(g):
    """median relative width over closed intervals only, and their count."""
    w = g.loc[~g.profile_hit_bound, "w_tau1"].dropna()
    return (round(float(w.median()), 3) if len(w) else "\u2014"), len(w)


rows = []
for zone in ["unconfined", "confined"]:
    for ba in [False, True]:
        g = idt[(idt.Aq_class == zone) & (idt.bound_active.astype(bool) == ba)]
        if not len(g):
            continue
        m, nc = _med_closed(g)
        rows.append({"Zone": zone, "Bound_active": ba, "n": len(g),
                     **{c: int((g.identifiability_final == c).sum()) for c in ORD},
                     "n_closed": nc, "median_w": m})
m, nc = _med_closed(idt)
rows.append({"Zone": "all", "Bound_active": "\u2014", "n": len(idt),
             **{c: int((idt.identifiability_final == c).sum()) for c in ORD},
             "n_closed": nc, "median_w": m})
t2 = pd.DataFrame(rows)
p2 = os.path.join(OUT, "Table2_identifiability.csv")
t2.to_csv(p2, index=False)
print("\n" + p2); print(t2.to_string(index=False))

# ── Table 3: information ratios (censoring is now the primary column) ───
ORDER = ["3He(trit)", "SF6", "3H", "14C"]
rows = []
for tr in ORDER:
    for z in ["unconfined", "confined"]:
        g = lo[(lo.omitted_tracer == tr) & (lo.Aq_class == z)]
        # Censored rows carry a ratio value, but it is a lower bound produced
        # by an interval that ran to a limit -- not a measured width ratio.
        # Median and IQR must come from `fin`, matching panel (b) of Fig. 4
        # and the caption's "inversions with a defined ratio". Using every row
        # here inflated 3He from 1.30 to 2.39 and reversed its rank against
        # SF6. The displacement column below is different: censoring affects
        # the interval, not the point estimate, so it keeps all rows.
        fin = g[~g.censored].information_ratio.dropna()
        rows.append({
            "Tracer withheld": tr, "Zone": z,
            "n inversions": len(g),
            "n censored": int(g.censored.sum()),
            "n finite": len(fin),
            "Median I_j": round(float(fin.median()), 2) if len(fin) else np.nan,
            "IQR I_j": (f"{fin.quantile(.25):.2f}-{fin.quantile(.75):.2f}"
                        if len(fin) else "-"),
            "Median |dtau1| (%)": round(
                float((g.delta_tau1_rel.dropna().abs() * 100).median()), 1),
        })
t3 = pd.DataFrame(rows)
p3 = os.path.join(OUT, "Table3_information_ratios.csv")
t3.to_csv(p3, index=False)
print("\n" + p3); print(t3.to_string(index=False))
