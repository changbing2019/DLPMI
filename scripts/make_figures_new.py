#!/usr/bin/env python3
"""
make_figures_new.py — additional manuscript figures for the DLPMI Edwards paper.

Companion to make_figures.py, which produces Figs 2-4 and the young-water
figure. This script adds the three figures needed to bring the main text to an
eight-figure package:

    Fig. 5  model-structure support (dAICc by sample and zone)
    Fig. 7  why young water is least identifiable (tracer response vs tau1)
    Fig. 8  agreement with previously published estimates

Reads the current CSVs in data_current/. Style matches make_figures.py.

Usage:  python make_figures_new.py --data data_current --out figures/
"""
from __future__ import annotations
import argparse, os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.ticker import ScalarFormatter
from scipy.stats import spearmanr

# Okabe-Ito, colourblind-safe and distinguishable in greyscale
C_WELL, C_WEAK, C_UNC, C_NA = "#0072B2", "#E69F00", "#D55E00", "#999999"
CLS_COLOR = {"well constrained": C_WELL, "weakly constrained": C_WEAK,
             "unconstrained": C_UNC}
CLS_ORDER = ["well constrained", "weakly constrained", "unconstrained"]
SUP_COLOR = {"supported": C_WELL, "indistinguishable": C_NA,
             "contradicted": C_UNC}
SUP_ORDER = ["supported", "indistinguishable", "contradicted"]
ZONE_MARK = {"unconfined": "o", "confined": "^"}
ZONE_COLOR = {"unconfined": "#56B4E9", "confined": "#009E73"}
CM = 1 / 2.54

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.dpi": 400, "figure.dpi": 120, "axes.grid": False,
})

H3COL = "H3_31to29_100to95"      # the band used for the Edwards samples
LAM_H3 = np.log(2) / 12.32       # tritium decay constant, yr^-1
T_SAMPLE = 2018.0                # nominal collection year


# ── Fig. 5 — model-structure support ──────────────────────────────────────
def fig5_structure(st, out):
    """dAICc per sample, by zone. dAICc = AICc(BMM) - AICc(DM); negative
    favours the binary mixture. Restricted to the 54 assessable samples --
    those retaining at least one degree of freedom (n_active >= 3), not the
    n_active >= 4 used before 2026-08-24. The AICc correction is identical
    under both structures (k = 2 each) and cancels in the difference, so a
    three-tracer sample is comparable even though AICc itself is undefined
    there; the sweep supplies dAICc from the chi2_sigma difference."""
    a = st[st.n_active >= 3].copy()
    fig, axes = plt.subplots(1, 2, figsize=(17.5 * CM, 8.5 * CM), sharex=True)

    for ax, zone in zip(axes, ["unconfined", "confined"]):
        z = a[a.Aq_class == zone].sort_values("dAICc").reset_index(drop=True)
        y = np.arange(len(z))
        ax.axvspan(-2, 2, color="0.90", zorder=0, lw=0)
        for x in (-10, 10):
            ax.axvline(x, color="0.55", ls=":", lw=0.7, zorder=1)
        ax.axvline(0, color="0.25", lw=0.7, zorder=1)
        ax.hlines(y, 0, z.dAICc.values, color="0.6", lw=0.8, zorder=2)
        for cls in SUP_ORDER:
            m = (z.vs_assigned == cls).values
            if m.sum():
                ax.scatter(z.dAICc.values[m], y[m], s=18, c=SUP_COLOR[cls],
                           marker=ZONE_MARK[zone], edgecolors="black",
                           linewidths=0.35, zorder=4, label=cls)
        # boundary solutions: BMM tau1 pinned at its lower bound
        b = z.BMM_tau1_at_lower_bound.fillna(False).astype(bool).values
        if b.sum():
            ax.scatter(z.dAICc.values[b], y[b], s=62, facecolors="none",
                       edgecolors="black", linewidths=0.7, zorder=3)
        ax.set_xscale("symlog", linthresh=2)
        ax.set_xlabel(r"$\Delta$AICc  =  AICc(BMM) $-$ AICc(DM)")
        ax.set_yticks([])
        ax.set_title(f"({'ab'[zone == 'confined']}) {zone} "
                     f"($n$ = {len(z)} assessable)", loc="left")
        ax.spines[["top", "right", "left"]].set_visible(False)
        ax.set_ylim(-1, len(z))
        ax.set_xlim(-2000, 60)
        ax.annotate(r"$\longleftarrow$ binary mixture favoured",
                    xy=(0.03, 0.955), xycoords="axes fraction", fontsize=6,
                    color="0.35")

    handles = [Line2D([], [], marker="s", ls="", mfc=SUP_COLOR[c], mec="black",
                      mew=0.35, ms=4.5, label=c) for c in SUP_ORDER]
    handles += [
        Line2D([], [], marker="o", ls="", mfc="none", mec="black", mew=0.7,
               ms=6.5, label=r"BMM $\tau_1$ at lower bound"),
        plt.Rectangle((0, 0), 1, 1, fc="0.90", ec="none",
                      label=r"$|\Delta$AICc$| < 2$ (indistinguishable)"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False,
               bbox_to_anchor=(0.5, -0.04), columnspacing=1.1, handlelength=1.3)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    p = os.path.join(out, "Fig5_model_structure.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ── Fig. 7 — why young water is least identifiable ────────────────────────
def fig7_response(tc, idt, out):
    """(a) atmospheric input functions; (b, c) piston-flow response of a
    sample collected in 2018 as a function of tau1. Illustrative: the PFM is
    the simplest case, chosen to isolate the shape of the input function from
    the effect of dispersion."""
    c = tc[(tc.decimal_year > 1935) & (tc.decimal_year <= 2020)] \
        .sort_values("decimal_year")
    yr, h3_in = c.decimal_year.values, c[H3COL].values
    sf6_in, c14_in = c.SF6_NH_pptv.values, c.C14_NHZone2_pmC.values

    tau = np.arange(0.25, 60.01, 0.05)
    ry = T_SAMPLE - tau
    h3_rec = np.interp(ry, yr, h3_in)
    h3 = h3_rec * np.exp(-LAM_H3 * tau)              # decayed tritium
    he3 = h3_rec * (1.0 - np.exp(-LAM_H3 * tau))     # tritiogenic helium-3
    sf6 = np.interp(ry, yr, sf6_in)

    t_lo, t_hi = idt.tau1.min(), idt.tau1.max()

    fig = plt.figure(figsize=(17.5 * CM, 12.5 * CM))
    gs = fig.add_gridspec(2, 2, height_ratios=[1, 1], hspace=0.42, wspace=0.28)
    ax0 = fig.add_subplot(gs[0, :])
    ax1 = fig.add_subplot(gs[1, 0])
    ax2 = fig.add_subplot(gs[1, 1])

    # (a) input functions ---------------------------------------------------
    ax0.plot(yr, h3_in, color=C_UNC, lw=1.1, label=r"$^{3}$H (TU, left)")
    ax0.set_yscale("log")
    ax0.set_ylabel(r"$^{3}$H in recharge (TU)")
    ax0.set_xlabel("Calendar year")
    ax0.axvspan(2017, 2018.6, color="0.80", lw=0, zorder=0)
    ax0.annotate("sampled 2017-2018", xy=(2017.4, 2.6), xytext=(1996, 2.1),
                 ha="center", fontsize=6, color="0.30",
                 arrowprops=dict(arrowstyle="-", lw=0.5, color="0.45"))
    ax0.annotate(f"bomb peak {h3_in.max():.0f} TU (1963)",
                 xy=(1963, h3_in.max()), xytext=(1972, 620), fontsize=6,
                 color=C_UNC,
                 arrowprops=dict(arrowstyle="-", lw=0.5, color=C_UNC))
    axb = ax0.twinx()
    axb.plot(yr, sf6_in, color=ZONE_COLOR["confined"], lw=1.1)
    axb.set_ylabel(r"SF$_6$ in recharge (pptv)", color=ZONE_COLOR["confined"])
    axb.tick_params(axis="y", colors=ZONE_COLOR["confined"])
    axb.spines["right"].set_color(ZONE_COLOR["confined"])
    ax0.set_title("(a) atmospheric input functions: a decaying spike and a "
                  "monotonic ramp", loc="left")
    ax0.set_xlim(1935, 2021)
    ax0.spines[["top"]].set_visible(False)
    axb.spines[["top"]].set_visible(False)

    # (b) 3H / 3He_trit response -------------------------------------------
    ax1.axvspan(t_lo, t_hi, color="0.90", lw=0, zorder=0)
    ax1.plot(tau, h3, color=C_UNC, lw=1.2, label=r"$^{3}$H")
    ax1.plot(tau, he3, color=C_WEAK, lw=1.2, label=r"$^{3}$He$_{trit}$")
    ax1.set_yscale("log")
    # non-uniqueness: a single 3H concentration consistent with many tau1
    lvl = np.median(h3[(tau >= t_lo) & (tau <= t_hi)])
    ax1.axhline(lvl, color="0.35", ls="--", lw=0.7)
    sgn = np.sign(h3 - lvl)
    xing = tau[:-1][np.diff(sgn) != 0]
    xing = xing[(xing >= t_lo) & (xing <= t_hi)]
    if len(xing):
        ax1.scatter(xing, np.full(len(xing), lvl), s=16, c="0.2", zorder=5,
                    marker="v")
        ax1.annotate(f"one $^{{3}}$H value, {len(xing)} admissible "
                     r"$\tau_1$",
                     xy=(xing.mean(), lvl), xytext=(0.30, 0.14),
                     textcoords="axes fraction", fontsize=6, color="0.2",
                     arrowprops=dict(arrowstyle="->", lw=0.5, color="0.35"))
    # dynamic range over the fitted band explains the information ratios
    bnd = (tau >= t_lo) & (tau <= t_hi)
    fr_h3 = h3[bnd].max() / h3[bnd].min()
    fr_he = he3[bnd].max() / he3[bnd].min()
    ax1.annotate(f"over the shaded range $^{{3}}$H spans only "
                 f"{fr_h3:.1f}$\\times$,\n$^{{3}}$He$_{{trit}}$ spans "
                 f"{fr_he:.0f}$\\times$",
                 xy=(0.35, 0.80), xycoords="axes fraction", fontsize=6,
                 color="0.25")
    ax1.set_xlabel(r"Transit time $\tau_1$ (yr)")
    ax1.set_ylabel("Concentration in 2018 sample (TU)")
    ax1.set_title(r"(b) $^{3}$H, $^{3}$He$_{trit}$: multi-valued in $\tau_1$",
                  loc="left")
    ax1.legend(frameon=False, loc="upper left")
    ax1.spines[["top", "right"]].set_visible(False)
    ax1.set_xlim(0, 60)

    # (c) SF6 response ------------------------------------------------------
    ax2.axvspan(t_lo, t_hi, color="0.90", lw=0, zorder=0)
    ax2.plot(tau, sf6, color=ZONE_COLOR["confined"], lw=1.2)
    lvl2 = np.median(sf6[(tau >= t_lo) & (tau <= t_hi)])
    ax2.axhline(lvl2, color="0.35", ls="--", lw=0.7)
    sgn2 = np.sign(sf6 - lvl2)
    x2 = tau[:-1][np.diff(sgn2) != 0]
    ax2.scatter(x2, np.full(len(x2), lvl2), s=16, c="0.2", zorder=5, marker="v")
    ax2.annotate(r"one SF$_6$ value, one $\tau_1$",
                 xy=(x2[0], lvl2), xytext=(0.36, 0.16),
                 textcoords="axes fraction", fontsize=6, color="0.2",
                 arrowprops=dict(arrowstyle="->", lw=0.5, color="0.35"))
    ax2.set_xlabel(r"Transit time $\tau_1$ (yr)")
    ax2.set_ylabel("Concentration in 2018 sample (pptv)")
    ax2.set_title(r"(c) SF$_6$: single-valued in $\tau_1$", loc="left")
    ax2.spines[["top", "right"]].set_visible(False)
    ax2.set_xlim(0, 60)

    fig.text(0.012, 0.005,
             "Shaded band in (b) and (c): range of $\\tau_1$ fitted in this "
             "study (%.1f-%.1f yr). Piston-flow response, shown to isolate the "
             "shape of the input function from the effect of dispersion."
             % (t_lo, t_hi), fontsize=5.8, color="0.3")
    # Figure 6 in the EST manuscript.  It was Figure 7 until the forward
    # references to results objects were removed from the Methods section:
    # that made 3.5 para 2 (this figure) the first citation in the text,
    # ahead of the young-water-fraction figure in para 4, so the two swapped.
    p = os.path.join(out, "Fig6_tracer_response.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p


# ── Fig. 8 — agreement with published estimates ───────────────────────────
def fig8_published(yf, idt, out):
    d = yf.merge(idt[["SampleID", "identifiability_final", "Aq_class"]],
                 on="SampleID", how="left", suffixes=("", "_i"))
    zc = "Aq_class" if "Aq_class" in yf.columns else "Aq_class_i"
    d = d.dropna(subset=["mean_age_mixture", "LPM_MeanAgeFinal_yrs"])
    rho, _ = spearmanr(d.mean_age_mixture, d.LPM_MeanAgeFinal_yrs)
    d = d.assign(absdev=np.abs(np.log10(d.mean_age_mixture /
                                        d.LPM_MeanAgeFinal_yrs)))

    fig, axes = plt.subplots(1, 2, figsize=(17.5 * CM, 8.0 * CM))

    # (a) present vs published --------------------------------------------
    ax = axes[0]
    lim = [2, 3e4]
    ax.plot(lim, lim, color="0.4", ls="--", lw=0.7, zorder=1)
    for zone in ["unconfined", "confined"]:
        z = d[d[zc] == zone]
        ax.scatter(z.LPM_MeanAgeFinal_yrs, z.mean_age_mixture, s=18,
                   marker=ZONE_MARK[zone], c=ZONE_COLOR[zone],
                   edgecolors="black", linewidths=0.35, zorder=3, label=zone)
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlim(lim); ax.set_ylim(lim)
    ax.set_xlabel("Published mixture-weighted mean age (yr)")
    ax.set_ylabel("This study, mixture-weighted mean age (yr)")
    ax.set_title("(a) agreement across four orders of magnitude", loc="left")
    ax.annotate(f"Spearman $\\rho$ = {rho:.3f}\n$n$ = {len(d)}",
                xy=(0.05, 0.86), xycoords="axes fraction", fontsize=7)
    ax.legend(frameon=False, loc="lower right", title="aquifer zone",
              title_fontsize=7)
    ax.spines[["top", "right"]].set_visible(False)

    # (b) divergence concentrates in poorly identifiable samples ----------
    ax = axes[1]
    data, labels, ns = [], [], []
    for cls in CLS_ORDER:
        v = d[d.identifiability_final == cls].absdev.dropna().values
        if len(v):
            data.append(v); labels.append(cls); ns.append(len(v))
    bp = ax.boxplot(data, positions=np.arange(len(data)), widths=0.5,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color="black", lw=1.0),
                    whiskerprops=dict(lw=0.6), capprops=dict(lw=0.6),
                    boxprops=dict(lw=0.5))
    for b, cls in zip(bp["boxes"], labels):
        b.set_facecolor(CLS_COLOR[cls]); b.set_alpha(0.85)
    rng = np.random.default_rng(0)
    for i, v in enumerate(data):
        ax.scatter(np.full(len(v), i) + rng.uniform(-.09, .09, len(v)), v,
                   s=5, c="black", alpha=0.40, zorder=4)
    top = max(v.max() for v in data)
    for i, (v, n) in enumerate(zip(data, ns)):
        ax.text(i, top * 1.03, f"$n$ = {n}", ha="center", va="bottom",
                fontsize=6, color="0.3")
    ax.set_xticks(np.arange(len(labels)))
    ax.set_xticklabels([l.replace(" ", "\n") for l in labels])
    ax.set_ylabel(r"$|\log_{10}$(this study / published)$|$")
    ax.set_xlabel("Transit-time identifiability class")
    ax.set_title("(b) divergence tracks identifiability", loc="left")
    ax.set_ylim(-0.02, top * 1.14)
    ax.spines[["top", "right"]].set_visible(False)

    fig.tight_layout()
    p = os.path.join(out, "Fig8_published_comparison.png")
    fig.savefig(p, bbox_inches="tight"); plt.close(fig)
    return p, rho


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data_current")
    ap.add_argument("--out", default="figures")
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)

    idt = pd.read_csv(os.path.join(A.data, "sweep_identifiability.csv"))
    st = pd.read_csv(os.path.join(A.data, "sweep_structure_sigma.csv"))
    yf = pd.read_csv(os.path.join(A.data, "young_fraction.csv"))
    tc = pd.read_csv(os.path.join(A.data, "tracer_input_curves.csv"))

    print(fig5_structure(st, A.out))
    print(fig7_response(tc, idt, A.out))
    p8, rho = fig8_published(yf, idt, A.out)
    print(p8)
    print(f"(check) Spearman rho = {rho:.4f}")


if __name__ == "__main__":
    main()
