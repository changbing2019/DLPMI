#!/usr/bin/env python3
"""
make_figures.py — manuscript figures and tables for the DLPMI Edwards paper.

*** PARTLY SUPERSEDED — read before running. ***

This script reads sweeps_final/ and sweeps/, which hold the CONDITIONAL chi2
intervals (identifiability 10 well / 29 weakly / 21 unconstrained). The paper
now reports TRUE PROFILE likelihood (8 / 16 / 36). Figures 2, 3 and 4 in the
manuscript come from regen_trueprofile.py, which reads sweeps_trueprofile/.

Running this script unguarded overwrites Fig2, Fig3 and Fig4 in the output
directory with the retired numbers, and the difference is not visually
obvious. Regenerate those three with regen_trueprofile.py instead.

Fig7_young_water_fraction is still current here: the young-water fraction is
computed from Monte Carlo draws, not from the profile intervals.

By default this now writes Fig7 only; the superseded outputs are skipped
unless --allow-superseded is passed. Fig. 1 (location map) cannot be generated
here: coordinates are present for only 27 of the 60 samples in the site
spreadsheet.

Usage:  python make_figures.py --results <dir> --sites <xlsx> --out figures/
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

# Okabe-Ito, colourblind-safe and distinguishable in greyscale
C_WELL, C_WEAK, C_UNC, C_NA = "#0072B2", "#E69F00", "#D55E00", "#999999"
CLS_COLOR = {"well constrained": C_WELL, "weakly constrained": C_WEAK,
             "unconstrained": C_UNC, "not assessable": C_NA}
CLS_ORDER = ["well constrained", "weakly constrained", "unconstrained",
             "not assessable"]
ZONE_MARK = {"unconfined": "o", "confined": "^"}
CM = 1 / 2.54

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.dpi": 400, "figure.dpi": 120, "axes.grid": False,
})


def load(results, sites):
    idt = pd.read_csv(os.path.join(results, "sweeps_final",
                                   "sweep_identifiability.csv"))
    lo = pd.read_csv(os.path.join(results, "sweeps", "sweep_loto.csv"))
    yf = pd.read_csv(os.path.join(results, "sweeps", "young_fraction.csv"))
    st = pd.read_csv(os.path.join(results, "sweeps_sigma",
                                 "sweep_structure.csv"))
    base = pd.read_csv(os.path.join(results, "reference",
                                    "00_summary_table.csv"))
    xl = pd.read_excel(sites)
    dep = xl[["SampleID", "Depth_m"]].drop_duplicates("SampleID")
    dep["Depth_m"] = pd.to_numeric(dep["Depth_m"], errors="coerce")
    return idt, lo, yf, st, base, dep


# ── Fig. 2 ────────────────────────────────────────────────────────────────
def fig2(idt, base, out):
    EX = {"EDTRPAS1-38", "EDTRPAS1-39", "EDTRPAS1-40",
          "EDTRPAS1-47", "EDTRPAS1-48"}
    b = base[~base.SampleID.isin(EX)].merge(
        idt[["SampleID", "Aq_class", "identifiability_final"]],
        on="SampleID", how="left")
    b = b.dropna(subset=["tau1_PINN"])

    fig, axes = plt.subplots(1, 2, figsize=(17.5 * CM, 8.0 * CM),
                             sharex=True)
    for ax, zone in zip(axes, ["unconfined", "confined"]):
        z = b[b.Aq_class == zone].sort_values("tau1_PINN").reset_index(drop=True)
        y = np.arange(len(z))
        lo = z.tau1_profile_lo.values
        hi = z.tau1_profile_hi.values
        t = z.tau1_PINN.values
        ok = np.isfinite(lo) & np.isfinite(hi)
        ax.hlines(y[ok], lo[ok], hi[ok], color="0.55", lw=0.9, zorder=1)
        for cls in CLS_ORDER:
            m = (z.identifiability_final == cls).values
            if m.sum():
                ax.scatter(t[m], y[m], s=17, c=CLS_COLOR[cls],
                           marker=ZONE_MARK[zone], edgecolors="black",
                           linewidths=0.35, zorder=3, label=cls)
        ax.set_xscale("log")
        ax.set_xlabel("Modern-component transit time $\\tau_1$ (yr)")
        ax.set_yticks([])
        ax.set_title(f"({'ab'[zone == 'confined']}) {zone} ($n$ = {len(z)})",
                     loc="left")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_ylim(-1, len(z))
    for ax in axes:
        ax.set_xticks([1, 3, 10, 30])
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    handles = [Line2D([], [], marker="s", ls="", mfc=CLS_COLOR[c],
                      mec="black", mew=0.35, ms=4.5, label=c)
               for c in CLS_ORDER]
    handles.append(Line2D([], [], color="0.55", lw=0.9,
                          label="68% profile interval"))
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.03), columnspacing=1.1,
               handlelength=1.2)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    p = os.path.join(out, "Fig2_transit_time_intervals.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ── Fig. 3 ────────────────────────────────────────────────────────────────
def fig3(idt, dep, out):
    d = idt.merge(dep, on="SampleID", how="left")
    d = d.dropna(subset=["Depth_m", "tau1"])

    fig, axes = plt.subplots(1, 2, figsize=(17.5 * CM, 7.5 * CM))

    ax = axes[0]
    for cls in CLS_ORDER:
        for zone, mk in ZONE_MARK.items():
            m = (d.identifiability_final == cls) & (d.Aq_class == zone)
            if m.sum():
                ax.scatter(d.tau1[m], d.Depth_m[m], s=22, marker=mk,
                           c=CLS_COLOR[cls], edgecolors="black",
                           linewidths=0.35)
    ax.set_xscale("log"); ax.invert_yaxis()
    ax.set_xticks([3, 5, 10, 20, 40])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Modern-component transit time $\\tau_1$ (yr)")
    ax.set_ylabel("Well depth (m)")
    ax.set_title("(a) identifiability with depth", loc="left")
    ax.spines[["top", "right"]].set_visible(False)

    ax = axes[1]
    order = ["unconfined", "confined"]
    width = 0.19
    for i, cls in enumerate(CLS_ORDER):
        vals = [((idt.Aq_class == z) & (idt.identifiability_final == cls)).sum()
                for z in order]
        x = np.arange(len(order)) + (i - 1.5) * width
        ax.bar(x, vals, width, color=CLS_COLOR[cls], edgecolor="black",
               linewidth=0.4, label=cls)
        for xi, v in zip(x, vals):
            if v:
                ax.text(xi, v + 0.25, str(v), ha="center", fontsize=6.5)
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([f"{z}\n($n$ = {(idt.Aq_class == z).sum()})"
                        for z in order])
    ax.set_ylabel("Number of samples")
    # headroom for the count labels, which sit above each bar
    _vmax = max(((idt.Aq_class == z) & (idt.identifiability_final == c)).sum()
                for z in order for c in CLS_ORDER)
    ax.set_ylim(0, _vmax * 1.22 + 1)
    ax.set_title("(b) classification by zone", loc="left")
    ax.legend(frameon=False, loc="upper center", ncol=2,
              columnspacing=0.9, handlelength=1.3)
    ax.spines[["top", "right"]].set_visible(False)

    mk = [Line2D([], [], marker=m, ls="", mfc="white", mec="black",
                 ms=4.5, label=z) for z, m in ZONE_MARK.items()]
    axes[0].legend(handles=mk, frameon=False, loc="lower left")
    fig.tight_layout()
    p = os.path.join(out, "Fig3_identifiability.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ── Fig. 4 ────────────────────────────────────────────────────────────────
def fig4(lo, out):
    NA = {"EDTRPAS1-46", "SCTXSUS1-10/EDTRVFPS1-15",
          "SCTXSUS1-04/EDTRVFPS1-20"}
    d = lo[~lo.SampleID.isin(NA)].copy()
    order = ["3He(trit)", "SF6", "3H", "14C"]
    label = {"3He(trit)": "$^{3}$He$_{trit}$", "SF6": "SF$_6$",
             "3H": "$^{3}$H", "14C": "$^{14}$C"}
    zc = {"unconfined": "#56B4E9", "confined": "#009E73"}

    fig, ax = plt.subplots(figsize=(12.5 * CM, 7.5 * CM))
    pos, ticks = [], []
    for i, tr in enumerate(order):
        for j, (zone, col) in enumerate(zc.items()):
            v = d[(d.omitted_tracer == tr) &
                  (d.Aq_class == zone)].information_ratio.dropna()
            if not len(v):
                continue
            x = i + (j - 0.5) * 0.32
            bp = ax.boxplot([v.values], positions=[x], widths=0.26,
                            patch_artist=True, showfliers=False,
                            medianprops=dict(color="black", lw=1.0),
                            whiskerprops=dict(lw=0.6),
                            capprops=dict(lw=0.6),
                            boxprops=dict(lw=0.5))
            bp["boxes"][0].set_facecolor(col)
            bp["boxes"][0].set_alpha(0.85)
            ax.scatter(np.full(len(v), x) + np.random.uniform(-.05, .05, len(v)),
                       v.values, s=4, c="black", alpha=0.35, zorder=4)
            nc = int(d[(d.omitted_tracer == tr) &
                       (d.Aq_class == zone)].censored.sum())
            ax.text(x, 0.435, f"n={len(v)}\nc={nc}", ha="center",
                    va="bottom", fontsize=5.8, color="0.3")
        ticks.append(i)
    ax.axhline(1.0, color="0.3", ls=":", lw=0.8)
    ax.text(3.42, 1.02, "$I_j$ = 1", fontsize=6, color="0.3", va="bottom",
            ha="right")
    ax.set_yscale("log")
    ax.set_xticks(ticks); ax.set_xticklabels([label[t] for t in order])
    ax.set_xlabel("Tracer withheld")
    ax.set_ylabel("Information ratio $I_j$ for $\\tau_1$")
    ax.set_ylim(0.40, None)
    ax.spines[["top", "right"]].set_visible(False)
    h = [plt.Rectangle((0, 0), 1, 1, fc=c, alpha=0.85, ec="black", lw=0.5,
                       label=z) for z, c in zc.items()]
    ax.legend(handles=h, frameon=False, loc="upper right", title="aquifer zone",
              title_fontsize=7)
    fig.text(0.012, 0.02, "n = samples with a finite ratio; "
                          "c = censored (omission left $\\tau_1$ unconstrained)",
             fontsize=5.8, color="0.3")
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    p = os.path.join(out, "Fig4_information_ratio.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ── Fig. 7 ────────────────────────────────────────────────────────────────
def fig7_young_water(yf, idt, out):
    # named for the manuscript slot it fills, not the slot it once filled --
    # the mismatch between fig5() and Fig7_... is what let this drift twice
    d = yf.merge(idt[["SampleID", "Aq_class"]], on="SampleID", how="left",
                 suffixes=("", "_i"))
    zc = "Aq_class" if "Aq_class" in yf.columns else "Aq_class_i"
    Ts = [10, 25, 40]
    fig, axes = plt.subplots(1, 3, figsize=(17.5 * CM, 6.5 * CM), sharey=True)
    for ax, T in zip(axes, Ts):
        for zone, col, mk in [("unconfined", "#56B4E9", "o"),
                              ("confined", "#009E73", "^")]:
            z = d[d[zc] == zone].sort_values(f"F_lt{T}yr").reset_index(drop=True)
            y = np.arange(len(z))
            p16 = z[f"F_lt{T}yr_p16"].values
            p84 = z[f"F_lt{T}yr_p84"].values
            v = z[f"F_lt{T}yr"].values
            off = 0 if zone == "unconfined" else len(d[d[zc] == "unconfined"])
            ok = np.isfinite(p16) & np.isfinite(p84)
            ax.hlines(y[ok] + off, p16[ok], p84[ok], color="0.6", lw=0.8)
            ax.scatter(v, y + off, s=13, c=col, marker=mk,
                       edgecolors="black", linewidths=0.3, label=zone)
        ax.set_xlim(-0.03, 1.03)
        ax.set_xlabel(f"$F(a < {T}$ yr$)$")
        ax.set_yticks([])
        ax.set_title(f"({'abc'[Ts.index(T)]}) $T$ = {T} yr", loc="left")
        ax.spines[["top", "right", "left"]].set_visible(False)
    axes[0].set_ylabel("samples, ordered within zone")
    h = [Line2D([], [], marker=m, ls="", mfc=c, mec="black", mew=0.3, ms=4.5,
                label=z)
         for z, c, m in [("unconfined", "#56B4E9", "o"),
                         ("confined", "#009E73", "^")]]
    h.append(Line2D([], [], color="0.6", lw=0.8, label="P16-P84 (Monte Carlo)"))
    fig.legend(handles=h, frameon=False, loc="lower center", ncol=3,
               bbox_to_anchor=(0.5, -0.02))
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    # Figure 7 in the EST manuscript.  This slot has moved twice: 5 -> 6 when
    # make_figures_new.py added the model-structure figure at 5, then 6 -> 7
    # when the tracer-response figure took 6.  The 5 was never updated here,
    # so re-running this script used to overwrite the wrong figure.
    p = os.path.join(out, "Fig7_young_water_fraction.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ── Tables ────────────────────────────────────────────────────────────────
def tables(idt, st, out):
    rows = []
    for zone in ["unconfined", "confined"]:
        for ba in [False, True]:
            s = idt[(idt.Aq_class == zone) & (idt.bound_active == ba)]
            rows.append(dict(
                Zone=zone, Bound_active=ba, n=len(s),
                **{c: int((s.identifiability_final == c).sum())
                   for c in CLS_ORDER},
                median_w=round(float(s.w_tau1.median()), 3)
                if s.w_tau1.notna().any() else np.nan))
    t3 = pd.DataFrame(rows)
    t3.loc[len(t3)] = dict(Zone="all", Bound_active="—", n=len(idt),
                           **{c: int((idt.identifiability_final == c).sum())
                              for c in CLS_ORDER},
                           median_w=round(float(idt.w_tau1.median()), 3))
    p3 = os.path.join(out, "Table3_identifiability.csv")
    t3.to_csv(p3, index=False)

    keep = [c for c in ["SampleID", "Aq_class", "assigned_LPM", "n_active",
                        "chi2sig_DM", "chi2sig_BMM", "AICc_DM", "AICc_BMM",
                        "dAICc", "preferred", "support", "vs_assigned"]
            if c in st.columns]
    t4 = st[keep].sort_values("dAICc")
    p4 = os.path.join(out, "Table4_model_structure.csv")
    t4.to_csv(p4, index=False)

    # convenience dump only; the CSV above is the artifact. to_markdown needs
    # the optional 'tabulate' package, whose absence used to abort this
    # function after both CSVs were already on disk.
    try:
        md = t3.to_markdown(index=False)
    except ImportError:
        print("   (skipping Table3_identifiability.md: pip install tabulate)")
    else:
        with open(os.path.join(out, "Table3_identifiability.md"), "w") as f:
            f.write(md)
    return p3, p4


# Outputs built from the conditional chi2 sweeps. The manuscript takes all of
# these from regen_trueprofile.py instead; regenerating them here overwrites
# current files with retired numbers that look almost identical.
SUPERSEDED = """\
  Fig2_transit_time_intervals.png   -> regen_trueprofile.py
  Fig3_identifiability.png          -> regen_trueprofile.py
  Fig4_information_ratio.png        -> regen_trueprofile.py
  Table3_identifiability.csv/.md    -> regen_trueprofile.py (now Table 2)
  Table4_model_structure.csv        -> regen_trueprofile.py (now Table 4)"""


def main():
    ap = argparse.ArgumentParser(
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Superseded outputs, skipped unless --allow-superseded:\n"
               + SUPERSEDED)
    ap.add_argument("--results", required=True)
    ap.add_argument("--sites", required=True)
    ap.add_argument("--out", default="figures")
    ap.add_argument("--allow-superseded", action="store_true",
                    help="also write the conditional-interval figures and "
                         "tables, overwriting the true-profile versions")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    np.random.seed(0)
    idt, lo, yf, st, base, dep = load(a.results, a.sites)

    # current: the young-water fraction comes from Monte Carlo draws, not from
    # the profile intervals, so it is unaffected by the conditional/true split
    outputs = [fig7_young_water(yf, idt, a.out)]

    if a.allow_superseded:
        print("WARNING: writing conditional-chi2 outputs. The manuscript "
              "reports true profile likelihood (8/16/36, not 10/29/21).\n")
        outputs += [fig2(idt, base, a.out), fig3(idt, dep, a.out),
                    fig4(lo, a.out), *tables(idt, st, a.out)]

    for p in outputs:
        print("->", p)

    if not a.allow_superseded:
        print("\nskipped (superseded by regen_trueprofile.py; pass "
              "--allow-superseded to write anyway):\n" + SUPERSEDED)


if __name__ == "__main__":
    main()
