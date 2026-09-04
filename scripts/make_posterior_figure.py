#!/usr/bin/env python3
"""Figure for the Bayesian tau2 analysis (posterior_tau2.py).

Three panels, in the order the argument runs:

  (a) the marginal posteriors for tau2 themselves, against the prior. Plotted
      as density RELATIVE to the prior, so a flat line at 1.0 means the data
      added nothing. Normalising tau2 by each sample's own source value puts
      every sample on one axis, since the prior is 0.05x-5x that value.

  (b) how much the answer depends on the assumed error model. The same
      samples under three treatments, connected sample by sample so the
      collapse is visible rather than inferred from medians.

  (c) the caveat: marginalising an unknown error scale lets it inflate to
      absorb misfit, so a badly fitting sample cannot contract. Low
      contraction therefore means two different things depending on the fit,
      and the panel separates them.

Reads only files written by posterior_tau2.py plus the two fixed-sigma runs.
"""
from __future__ import annotations

import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt                                    # noqa: E402
import numpy as np                                                 # noqa: E402
import pandas as pd                                                # noqa: E402
from matplotlib.lines import Line2D                                # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
CHAT = os.path.dirname(HERE)
ROOT = os.path.dirname(CHAT)
ST = os.path.join(ROOT, "DLPMI_edwards_handoff", "sweeps_trueprofile")
OUTDIR = os.path.join(CHAT, "figures")

INK, MID, GRID = "#16212A", "#54646F", "#D6DCE0"
TEAL, OXIDE, PALE = "#0F6C8C", "#AC4E1C", "#9FB4BE"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Calibri", "DejaVu Sans"],
    "font.size": 9, "axes.edgecolor": MID, "axes.labelcolor": INK,
    "text.color": INK, "xtick.color": MID, "ytick.color": MID,
    "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 200,
})


def short(sid):
    return sid.split("/")[0]


def main():
    prim = pd.read_csv(os.path.join(ST, "posterior_tau2.csv"))
    marg = pd.read_csv(os.path.join(ST, "posterior_tau2_marginals.csv"))
    inv = pd.read_csv(os.path.join(ST,
                                   "posterior_tau2_fixed_inversion_sigma.csv"))
    rep = pd.read_csv(os.path.join(ST,
                                   "posterior_tau2_fixed_reported_sigma.csv"))

    keep = prim[prim.assessable].copy()
    ids = set(keep.SampleID)
    src = dict(zip(prim.SampleID, prim.tau2_src))

    fig = plt.figure(figsize=(7.2, 5.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1.0],
                          hspace=0.46, wspace=0.30)
    a1 = fig.add_subplot(gs[0, :])
    a2 = fig.add_subplot(gs[1, 0])
    a3 = fig.add_subplot(gs[1, 1])

    # ── (a) the marginal posteriors, as cumulative distributions ────────
    # Densities are plotted cumulatively rather than directly: one sample's
    # peak reaches 78x the prior density, which on a shared linear axis
    # flattens the other thirteen onto the baseline and hides exactly the
    # structure this panel exists to show. A CDF is bounded in [0, 1], and
    # because the prior is log-uniform its own CDF is the straight diagonal,
    # so "follows the diagonal" reads directly as "the data added nothing".
    hi_two = keep.nlargest(2, "contraction_log_tau2").SampleID.tolist()
    # both step up within a factor of ~1.2 of each other, so labelling each at
    # its median would overprint; anchor them at different heights instead
    label_y = {hi_two[0]: 0.72, hi_two[1]: 0.28}
    for sid, g in marg[marg.SampleID.isin(ids)].groupby("SampleID"):
        g = g.sort_values("tau2")
        x = g.tau2.values / src[sid]
        y = np.cumsum(g.posterior_density.values)
        y /= y[-1]
        if sid in hi_two:
            a1.plot(x, y, color=OXIDE, lw=2.0, zorder=4)
            yl = label_y[sid]
            j = int(np.argmin(np.abs(y - yl)))
            a1.annotate(short(sid), xy=(x[j], yl), xytext=(10, -2),
                        textcoords="offset points", fontsize=7.5,
                        color=OXIDE, fontweight="bold", va="center",
                        arrowprops=dict(arrowstyle="-", color=OXIDE, lw=.8,
                                        shrinkA=0, shrinkB=2))
        else:
            a1.plot(x, y, color=TEAL, lw=1.1, alpha=.6, zorder=2)
    a1.plot([0.05, 5.0], [0.0, 1.0], color=INK, ls=(0, (5, 3)), lw=1.3,
            zorder=3)
    a1.set_xscale("log")
    a1.set_xlim(0.05, 5.0)
    a1.set_ylim(0, 1)
    a1.set_xlabel("τ₂ ÷ the value the source study used")
    a1.set_ylabel("cumulative posterior\nprobability")
    a1.set_title("(a)  12 of 14 posteriors barely depart from the prior",
                 fontsize=9.2, loc="left", pad=7)
    a1.xaxis.grid(True, color=GRID, lw=.7)
    a1.set_axisbelow(True)
    a1.legend(handles=[
        Line2D([], [], color=TEAL, lw=1.4, alpha=.7,
               label="12 of 14 samples"),
        Line2D([], [], color=OXIDE, lw=2.0,
               label="the 2 that do contract"),
        Line2D([], [], color=INK, lw=1.3, ls=(0, (5, 3)),
               label="prior"),
    ], loc="upper left", frameon=False, fontsize=7.8, handlelength=2.0)

    # ── (b) dependence on the assumed error model ───────────────────────
    runs = [("reported\nanalytical σ", rep), ("inversion σ\n(10% convention)",
                                              inv),
            ("scale\nmarginalised", prim)]
    xs = np.arange(len(runs))
    tab = {}
    for k, (_lab, df) in enumerate(runs):
        d = df[df.SampleID.isin(ids)].set_index("SampleID")
        tab[k] = d.contraction_log_tau2
    for sid in ids:
        a2.plot(xs, [tab[k][sid] for k in range(len(runs))],
                color=PALE, lw=.9, marker="o", ms=3, alpha=.85, zorder=2)
    for k in range(len(runs)):
        a2.plot([k], [tab[k].median()], marker="_", ms=26, mew=2.6,
                color=OXIDE, zorder=4)
    a2.set_xticks(xs, [r[0] for r in runs], fontsize=7.6)
    a2.set_xlim(-0.45, len(runs) - 0.55)
    a2.set_ylim(-0.08, 1.04)
    a2.set_ylabel("contraction in log τ₂\n(0 = data added nothing)")
    a2.set_title("(b)  apparent precision follows the\n      assumed error "
                 "model", fontsize=9.2, loc="left", pad=7)
    a2.yaxis.grid(True, color=GRID, lw=.7)
    a2.set_axisbelow(True)
    for k, (_lab, df) in enumerate(runs):
        d = df[df.SampleID.isin(ids)]
        a2.annotate("%d/%d fit" % (int((d.p_at_min > 0.05).sum()), len(d)),
                    xy=(k, -0.045), ha="center", fontsize=7.2, color=MID)
    a2.annotate("median", xy=(2.32, tab[2].median()), fontsize=7.2,
                color=OXIDE, va="center")

    # ── (c) the caveat: misfit is absorbed by the error scale ───────────
    ok = keep[keep.p_at_min > 0.05]
    no = keep[keep.p_at_min <= 0.05]
    for d, c, lab in ((no, MID, "fit rejected (p ≤ 0.05)"),
                      (ok, TEAL, "fit acceptable (p > 0.05)")):
        a3.scatter(d.p_at_min.clip(lower=3e-4), d.contraction_log_tau2,
                   s=34, color=c, edgecolor="white", linewidth=.6,
                   label=lab, zorder=3)
    a3.axvline(0.05, color=OXIDE, ls=(0, (4, 3)), lw=1.0, zorder=2)
    a3.set_xscale("log")
    a3.set_xlabel("goodness of fit at the posterior mode, p")
    a3.set_ylabel("contraction in log τ₂")
    a3.set_ylim(-0.08, 1.04)
    a3.set_title("(c)  a rejected fit cannot contract:\n      the error scale "
                 "absorbs the misfit", fontsize=9.2, loc="left", pad=7)
    a3.grid(True, color=GRID, lw=.7)
    a3.set_axisbelow(True)
    a3.legend(loc="upper left", frameon=False, fontsize=7.4,
              handletextpad=.3, borderpad=.2)

    os.makedirs(OUTDIR, exist_ok=True)
    p = os.path.join(OUTDIR, "Fig_posterior_tau2.png")
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print("wrote %s  (%.0f KB)" % (p, os.path.getsize(p) / 1024))
    print("panel (a): %d marginals (%d highlighted)" % (len(ids),
                                                        len(hi_two)))
    print("panel (b): medians %s"
          % ["%+.3f" % tab[k].median() for k in range(len(runs))])
    print("panel (c): %d acceptable, %d rejected" % (len(ok), len(no)))


if __name__ == "__main__":
    main()
