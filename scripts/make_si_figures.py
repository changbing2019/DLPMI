#!/usr/bin/env python3
"""make_si_figures.py — SI Figures S3 and S4, which had no generator.

Both figures were pasted into the SI from a session whose outputs were not
kept: neither embedded image matched any file in the repository, and both were
plotted from data the 2026-08-24 Hessian fix superseded.

  Figure S3  observed vs simulated, per tracer. Came from the stale PINN_sim
             columns. The sample counts were right (60 / 44 / 54 / 60) but
             every rho and RMSE in the annotation boxes was wrong. The tell was
             DLPMI's 3H correlation (0.90) coming out WORSE than the published
             LPM's (0.98) -- the reanalysis appearing to fit worse than the
             study it reproduces.

  Figure S4  parameters and goodness-of-fit against the published TracerLPM
             results. Panel (f) plots tau1_sigma_hess, which the factor-of-2
             fix scaled by a median of exactly sqrt(2); its n moves 50 -> 51.
             Panels (a)-(e) rest on fitted optima and chi2, which reproduced
             byte-identically, except error bars in (a) drawn from the Hessian
             sigma.

Design follows the originals so the SI stays visually consistent: the same
Okabe-Ito palette, serif 8 pt, 400 dpi and figure widths as make_figures.py
and make_figures_new.py.

Reads only DLPMI_edwards_handoff/reference/00_summary_table.csv and the data
release's Table_4_LPM.txt (UTF-16), which supplies the published dispersion
parameter -- the one quantity Figure S4b needs that the summary table lacks.

Usage:  python make_si_figures.py [--out ../figures]
"""
from __future__ import annotations

import argparse
import os

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt                              # noqa: E402
from matplotlib.lines import Line2D                          # noqa: E402
from scipy.stats import spearmanr                            # noqa: E402

R = r"D:\projects\carbon isotopic\DLMPI_reanalysis"
BASE = os.path.join(R, "DLPMI_edwards_handoff", "reference",
                    "00_summary_table.csv")
IDENT = os.path.join(R, "DLPMI_edwards_handoff", "sweeps_trueprofile",
                     "sweep_identifiability_trueprofile.csv")
T4 = os.path.join(R, "Musgrove_2023Data", "Table_4_LPM.txt")

C_WELL, C_WEAK, C_UNC, C_NA = "#0072B2", "#E69F00", "#D55E00", "#999999"
CM = 1 / 2.54

plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8.5, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "legend.fontsize": 7, "axes.linewidth": 0.6,
    "xtick.major.width": 0.6, "ytick.major.width": 0.6,
    "savefig.dpi": 400, "figure.dpi": 120, "axes.grid": False,
})

#: (column label, panel title, axis unit, log axes?)
TRACERS = [("3H", r"$^{3}$H", "TU", True),
           ("3He(trit)", r"$^{3}$He$_{trit}$", "TU", True),
           ("SF6", r"SF$_{6}$", "pptv", True),
           ("14C", r"$^{14}$C", "pmC", False)]


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return np.nan


def load():
    """The 60-sample study set, with the published P_D joined on."""
    d = pd.read_csv(BASE)
    keep = set(pd.read_csv(IDENT).SampleID)
    d = d[d.SampleID.isin(keep)].copy()
    t4 = pd.read_csv(T4, sep="\t", encoding="utf-16")
    t4["SampleID"] = t4.SampleID.astype(str).str.strip()
    # these release columns are object dtype and carry sentinels such as
    # "init. val." where a parameter was not optimised, so coerce rather than
    # compare strings to numbers
    pdl = t4.set_index("SampleID")[["LPM_ModParm1_C1",
                                    "LPM_ModParm1_C1_Err"]].apply(
        pd.to_numeric, errors="coerce")
    pdl.columns = ["pd1_LPM", "pd1_LPM_err"]
    d = d.merge(pdl, left_on="SampleID", right_index=True, how="left")
    for _c in ("pd1_PINN", "tau1_PINN", "tau1_LPM", "f1_PINN", "f1_LPM",
               "chi2_PINN", "chi2_LPM", "chi2_prob_PINN", "chi2_prob_LPM",
               "tau1_sigma_hess", "tau1_sigma_LPM", "f1_sigma_hess",
               "f1_sigma_LPM", "dof"):
        if _c in d:
            d[_c] = pd.to_numeric(d[_c], errors="coerce")
    d["is_bmm"] = d.LPM.astype(str).str.startswith("BMM")
    d["mixed"] = d.AgeCat.astype(str).str.strip().str.lower().eq("mixed")
    return d


def lpm_sims():
    """Published TracerLPM simulated concentrations, read from the data
    release and keyed by tracer NAME.

    The summary table's own LPM_sim column CANNOT be used: it lists the
    simulated value for every tracer the LPM carries, in the release's fixed
    order (3H, 3He(trit), SF6, 14C, 4He), while Tracer_names lists only the
    ACTIVE tracers. The two desynchronise for the 16 samples where 3He(trit)
    is not active, shifting every later value by one position. The shipped
    RelErr_LPM column is corrupted by exactly this: 9 of 60 samples carry
    impossible values, up to 2.4e12%, because a 14C simulation is divided by a
    4He observation of zero.

    LPM_Mod_Tracer_NN is indexed by the same NN as LPM_Tracer_Name_NN, so
    reading them together is unambiguous.
    """
    t4 = pd.read_csv(T4, sep="	", encoding="utf-16")
    out = {}
    for _, r in t4.iterrows():
        sid = str(r.SampleID).strip()
        for i in range(1, 11):
            nm = r.get("LPM_Tracer_Name_%02d" % i)
            if not isinstance(nm, str) or not nm.strip():
                continue
            out[(sid, nm.strip())] = num(r.get("LPM_Mod_Tracer_%02d" % i))
    return out


def pairs(d):
    """Long form: one row per active tracer-sample pair."""
    lpm = lpm_sims()
    rows = []
    for _, r in d.iterrows():
        nm = [x.strip() for x in str(r.Tracer_names).split(";")]
        ob = [num(x) for x in str(r.Obs_vals).split(";")]
        pn = [num(x) for x in str(r.PINN_sim).split(";")]
        for a, b, c in zip(nm, ob, pn):
            key = None
            for cand in [str(r.SampleID)] + [x.strip() for x
                                             in str(r.SampleID).split("/")]:
                if (cand, a) in lpm:
                    key = (cand, a)
                    break
            rows.append(dict(SampleID=r.SampleID, tracer=a, obs=b, pinn=c,
                             lpm=lpm.get(key, np.nan) if key else np.nan,
                             mixed=bool(r.mixed)))
    return pd.DataFrame(rows)


def envelope(ax, lo, hi, frac, log):
    """Shaded +/- frac band about the 1:1 line."""
    x = np.geomspace(lo, hi, 200) if log else np.linspace(lo, hi, 200)
    ax.fill_between(x, x * (1 - frac), x * (1 + frac), color="0.85",
                    lw=0, zorder=0)
    ax.plot(x, x, ls="--", color="0.25", lw=0.7, zorder=1)


# ════════════════════════════════ Figure S3 ═══════════════════════════════
def fig_s3(d, out):
    p = pairs(d)
    fig, axes = plt.subplots(2, 2, figsize=(17.5 * CM, 15.0 * CM))
    for ax, (col, title, unit, log) in zip(axes.ravel(), TRACERS):
        g = p[p.tracer == col].dropna(subset=["obs"])
        gp = g.dropna(subset=["pinn"])
        gl = g.dropna(subset=["lpm"])
        both = pd.concat([g.obs, gp.pinn, gl.lpm]).dropna()
        lo, hi = float(both.min()), float(both.max())
        if log:
            lo, hi = max(lo, 1e-4) * 0.6, hi * 1.6
        else:
            pad = 0.08 * (hi - lo)
            lo, hi = lo - pad, hi + pad
        envelope(ax, lo, hi, 0.10, log)

        for sub, mark, fc, ec, lab in (
                (gp[~gp.mixed], "o", C_WELL, "none", "DLPMI  Modern"),
                (gp[gp.mixed], "s", C_WEAK, "none", "DLPMI  Mixed"),
                (gl[~gl.mixed], "^", "none", C_WELL, "LPM \u2013 Modern"),
                (gl[gl.mixed], "v", "none", C_WEAK, "LPM \u2013 Mixed")):
            if not len(sub):
                continue
            y = sub.pinn if mark in ("o", "s") else sub.lpm
            ax.scatter(sub.obs, y, s=15, marker=mark,
                       facecolors=fc if fc != "none" else "none",
                       edgecolors=ec if ec != "none" else "black",
                       linewidths=0.5, zorder=4, label=lab)
        if log:
            ax.set_xscale("log")
            ax.set_yscale("log")
        ax.set_xlim(lo, hi)
        ax.set_ylim(lo, hi)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Observed (%s)" % unit)
        ax.set_ylabel("Simulated (%s)" % unit)
        ax.set_title("(%s) %s" % ("abcd"[TRACERS.index((col, title, unit,
                                                        log))], title),
                     loc="center")

        def stat(sub, ycol):
            if len(sub) < 3:
                return np.nan, np.nan
            return (spearmanr(sub.obs, sub[ycol])[0],
                    float(np.sqrt(((sub[ycol] - sub.obs) ** 2).mean())))
        rp, ep = stat(gp, "pinn")
        rl, el = stat(gl, "lpm")
        ax.text(0.035, 0.965,
                "DLPMI:  $\\rho$=%.2f  RMSE=%.3f\nLPM:      $\\rho$=%.2f  "
                "RMSE=%.3f\nn = %d" % (rp, ep, rl, el, len(gp)),
                transform=ax.transAxes, va="top", ha="left", fontsize=6.2,
                bbox=dict(fc="white", ec="0.6", lw=0.5, pad=2.2))
        ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D([], [], marker="o", ls="", mfc=C_WELL, mec="none", ms=4,
               label="DLPMI  Modern"),
        Line2D([], [], marker="s", ls="", mfc=C_WEAK, mec="none", ms=4,
               label="DLPMI  Mixed"),
        Line2D([], [], marker="^", ls="", mfc="none", mec=C_WELL, mew=0.6,
               ms=4.5, label="LPM \u2013 Modern"),
        Line2D([], [], marker="v", ls="", mfc="none", mec=C_WEAK, mew=0.6,
               ms=4.5, label="LPM \u2013 Mixed"),
        Line2D([], [], ls="--", color="0.25", lw=0.8, label="1:1 line"),
        plt.Rectangle((0, 0), 1, 1, fc="0.85", ec="none",
                      label=r"$\pm$10% envelope"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.015), columnspacing=1.2,
               handlelength=1.4)
    fig.tight_layout(rect=(0, 0.045, 1, 1))
    fp = os.path.join(out, "FigS3_obs_vs_sim.png")
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    return fp


# ════════════════════════════════ Figure S4 ═══════════════════════════════
def _scatter_by_class(ax, d, x, y, xerr=None, yerr=None):
    """Four classes: Modern/Mixed x DM/BMM, as in the original."""
    for mixed, bmm, mark, col, lab in (
            (False, False, "o", C_WELL, "Modern (DM)"),
            (True, False, "o", C_WEAK, "Mixed (DM)"),
            (False, True, "^", C_WELL, "Modern (BMM)"),
            (True, True, "^", C_WEAK, "Mixed (BMM)")):
        s = d[(d.mixed == mixed) & (d.is_bmm == bmm)]
        s = s.dropna(subset=[x, y])
        if not len(s):
            continue
        if xerr is not None or yerr is not None:
            ax.errorbar(s[x], s[y],
                        xerr=s[xerr].abs() if xerr else None,
                        yerr=s[yerr].abs() if yerr else None,
                        fmt="none", ecolor=col, elinewidth=0.5, alpha=0.45,
                        zorder=2)
        ax.scatter(s[x], s[y], s=16, marker=mark, c=col, edgecolors="none",
                   zorder=4, label=lab)


def fig_s4(d, out):
    fig, axes = plt.subplots(2, 3, figsize=(23.0 * CM, 14.5 * CM))
    (a, b, c), (e, f, g) = axes

    # (a) tau1 ------------------------------------------------------------
    s = d.dropna(subset=["tau1_PINN", "tau1_LPM"])
    lo = max(min(s.tau1_PINN.min(), s.tau1_LPM.min()) * 0.55, 0.5)
    hi = max(s.tau1_PINN.max(), s.tau1_LPM.max()) * 1.8
    envelope(a, lo, hi, 0.20, True)
    _scatter_by_class(a, d, "tau1_LPM", "tau1_PINN",
                      xerr="tau1_sigma_LPM", yerr="tau1_sigma_hess")
    lx, ly = np.log10(s.tau1_LPM), np.log10(s.tau1_PINN)
    a.text(0.035, 0.965, "$R^2$ = %.3f\nRMSD(log) = %.3f\nn = %d"
           % (np.corrcoef(lx, ly)[0, 1] ** 2,
              float(np.sqrt(((ly - lx) ** 2).mean())), len(s)),
           transform=a.transAxes, va="top", fontsize=6.2,
           bbox=dict(fc="white", ec="0.6", lw=0.5, pad=2.2))
    a.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
          xlabel=r"$\tau_1$ TracerLPM (yr)", ylabel=r"$\tau_1$ DLPMI (yr)")
    a.set_title(r"(a) Mean Age $\tau_1$")

    # (b) dispersion parameter -------------------------------------------
    s = d.dropna(subset=["pd1_PINN", "pd1_LPM"])
    s = s[(s.pd1_PINN > 0) & (s.pd1_LPM > 0)]
    lo = min(s.pd1_PINN.min(), s.pd1_LPM.min()) * 0.55
    hi = max(s.pd1_PINN.max(), s.pd1_LPM.max()) * 1.8
    envelope(b, lo, hi, 0.20, True)
    _scatter_by_class(b, d[~d.is_bmm], "pd1_LPM", "pd1_PINN")
    sd = s[~s.is_bmm]
    lx, ly = np.log10(sd.pd1_LPM), np.log10(sd.pd1_PINN)
    n_dm = int((~d.is_bmm).sum())
    b.text(0.035, 0.965,
           "$R^2$ = %.3f\nn = %d of %d DM\n(%d without an\noptimised $P_D$)"
           % (np.corrcoef(lx, ly)[0, 1] ** 2, len(sd), n_dm, n_dm - len(sd)),
           transform=b.transAxes, va="top", fontsize=6.2,
           bbox=dict(fc="white", ec="0.6", lw=0.5, pad=2.2))
    b.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
          xlabel=r"$P_D$ TracerLPM", ylabel=r"$P_D$ DLPMI")
    b.set_title(r"(b) Dispersion Parameter $P_D$  ($P_D$ fixed in BMM)")

    # (c) mixing fraction -------------------------------------------------
    s = d[d.is_bmm].dropna(subset=["f1_PINN", "f1_LPM"])
    envelope(c, 0.0, 1.15, 0.20, False)
    c.errorbar(s.f1_LPM, s.f1_PINN, xerr=s.f1_sigma_LPM.abs(),
               yerr=s.f1_sigma_hess.abs(), fmt="none", ecolor=C_NA,
               elinewidth=0.5, alpha=0.6, zorder=2)
    c.scatter(s.f1_LPM, s.f1_PINN, s=18, marker="^", c=C_WELL,
              edgecolors="none", zorder=4)
    c.set(xlim=(0, 1.15), ylim=(0, 1.15), xlabel=r"$f$ TracerLPM",
          ylabel=r"$f$ DLPMI")
    c.set_title(r"(c) Mixing Fraction $f$ (BMM, N = %d)" % len(s))

    # (e) chi-square ------------------------------------------------------
    s = d.dropna(subset=["chi2_PINN", "chi2_LPM"])
    s = s[(s.chi2_PINN > 0) & (s.chi2_LPM > 0)]
    lo = min(s.chi2_PINN.min(), s.chi2_LPM.min()) * 0.4
    hi = max(s.chi2_PINN.max(), s.chi2_LPM.max()) * 2.5
    x = np.geomspace(lo, hi, 200)
    e.fill_between(x, x, hi, color="#FDE9E4", lw=0, zorder=0)
    e.fill_between(x, lo, x, color="#E6F0F7", lw=0, zorder=0)
    e.plot(x, x, ls="-.", color="0.25", lw=0.7, zorder=1)
    _scatter_by_class(e, s, "chi2_LPM", "chi2_PINN")
    # Count over every sample with both misfits defined, not only those the
    # log axis can show: six have a chi2 of exactly zero. The original
    # figure annotated a 60-sample count on a 54-point plot without saying so.
    allc = d.dropna(subset=["chi2_PINN", "chi2_LPM"])
    n_win = int((allc.chi2_PINN <= allc.chi2_LPM).sum())
    e.text(0.035, 0.965,
           "DLPMI $\\leq$ LPM:  %d/%d\nLPM $<$ DLPMI:  %d/%d\n"
           "(%d shown; %d have $\\chi^2 = 0$)"
           % (n_win, len(allc), len(allc) - n_win, len(allc), len(s),
              len(allc) - len(s)),
           transform=e.transAxes, va="top", fontsize=6.2,
           bbox=dict(fc="white", ec="0.6", lw=0.5, pad=2.2))
    e.text(0.03, 0.86, "LPM wins", transform=e.transAxes, fontsize=6,
           color="#C0392B", style="italic")
    e.text(0.97, 0.05, "DLPMI wins", transform=e.transAxes, fontsize=6,
           color="#1F6FA8", style="italic", ha="right")
    e.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
          xlabel=r"$\chi^2$ TracerLPM", ylabel=r"$\chi^2$ DLPMI")
    e.set_title(r"(d) Chi-Square $\chi^2$")

    # (f) chi-square probability -----------------------------------------
    s = d.dropna(subset=["chi2_prob_PINN", "chi2_prob_LPM"])
    f.axvline(0.05, color="0.55", ls="--", lw=0.7, zorder=1)
    f.axhline(0.05, color="0.55", ls="--", lw=0.7, zorder=1)
    _scatter_by_class(f, s, "chi2_prob_LPM", "chi2_prob_PINN")
    ok_p, ok_l = s.chi2_prob_PINN > 0.05, s.chi2_prob_LPM > 0.05
    for xx, yy, lab, n, col in (
            (0.05, 0.42, "DLPMI only", int((ok_p & ~ok_l).sum()), "#1F6FA8"),
            (0.62, 0.40, "Both", int((ok_p & ok_l).sum()), "#1E8449"),
            (0.02, 0.02, "Neither", int((~ok_p & ~ok_l).sum()), "#C0392B"),
            (0.66, 0.02, "LPM only", int((~ok_p & ok_l).sum()), "#B9770E")):
        f.text(xx, yy, "%s\nn=%d" % (lab, n), transform=f.transAxes,
               fontsize=6, color=col)
    f.set(xlim=(-0.05, 1.08), ylim=(-0.05, 1.08),
          xlabel=r"$\chi^2$ prob. TracerLPM",
          ylabel=r"$\chi^2$ prob. DLPMI")
    f.set_title(r"(e) $\chi^2$ Probability ($p>0.05$ = acceptable)")

    # (g) tau1 uncertainty ------------------------------------------------
    s = d.dropna(subset=["tau1_sigma_hess", "tau1_sigma_LPM"])
    s = s[(s.tau1_sigma_hess > 0) & (s.tau1_sigma_LPM > 0)]
    lo = min(s.tau1_sigma_hess.min(), s.tau1_sigma_LPM.min()) * 0.4
    hi = max(s.tau1_sigma_hess.max(), s.tau1_sigma_LPM.max()) * 2.5
    envelope(g, lo, hi, 0.20, True)
    sc = g.scatter(s.tau1_sigma_LPM, s.tau1_sigma_hess, s=18,
                   c=s.dof.astype(float), cmap="viridis_r",
                   edgecolors="none", zorder=4)
    cb = fig.colorbar(sc, ax=g, pad=0.02, fraction=0.045)
    cb.set_label("Degrees of freedom", fontsize=7)
    cb.ax.tick_params(labelsize=6)
    lx, ly = np.log10(s.tau1_sigma_LPM), np.log10(s.tau1_sigma_hess)
    g.text(0.035, 0.965, "$R^2$ = %.3f\nn = %d"
           % (np.corrcoef(lx, ly)[0, 1] ** 2, len(s)),
           transform=g.transAxes, va="top", fontsize=6.2,
           bbox=dict(fc="white", ec="0.6", lw=0.5, pad=2.2))
    g.set(xscale="log", yscale="log", xlim=(lo, hi), ylim=(lo, hi),
          xlabel=r"$\sigma(\tau_1)$ TracerLPM Solver (yr)",
          ylabel=r"$\sigma(\tau_1)$ DLPMI Hessian (yr)")
    g.set_title(r"(f) $\tau_1$ Uncertainty Comparison")

    for ax in axes.ravel():
        ax.spines[["top", "right"]].set_visible(False)

    handles = [
        Line2D([], [], marker="o", ls="", mfc=C_WELL, mec="none", ms=4,
               label="Modern (DM)"),
        Line2D([], [], marker="o", ls="", mfc=C_WEAK, mec="none", ms=4,
               label="Mixed (DM)"),
        Line2D([], [], marker="^", ls="", mfc=C_WELL, mec="none", ms=4.5,
               label="Modern (BMM)"),
        Line2D([], [], marker="^", ls="", mfc=C_WEAK, mec="none", ms=4.5,
               label="Mixed (BMM)"),
        Line2D([], [], ls="--", color="0.25", lw=0.8, label="1:1 line"),
        plt.Rectangle((0, 0), 1, 1, fc="0.85", ec="none",
                      label=r"$\pm$20% envelope"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6, frameon=False,
               bbox_to_anchor=(0.5, -0.02), columnspacing=1.2,
               handlelength=1.4)
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fp = os.path.join(out, "FigS4_published_parameters.png")
    fig.savefig(fp, bbox_inches="tight")
    plt.close(fig)
    return fp


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(R, "From_Claude_chat",
                                                  "figures"))
    A = ap.parse_args()
    os.makedirs(A.out, exist_ok=True)
    np.random.seed(0)          # nothing jitters here, but keep it declared
    d = load()
    print("study set: %d samples (%d BMM, %d mixed)"
          % (len(d), int(d.is_bmm.sum()), int(d.mixed.sum())))
    print(fig_s3(d, A.out))
    print(fig_s4(d, A.out))

    p = pairs(d)
    print("\nFigure S3 annotation values:")
    for col, _, _, _ in TRACERS:
        g = p[p.tracer == col].dropna(subset=["obs", "pinn"])
        print("  %-10s n=%2d  rho=%.2f  RMSE=%8.3f"
              % (col, len(g), spearmanr(g.obs, g.pinn)[0],
                 float(np.sqrt(((g.pinn - g.obs) ** 2).mean()))))


if __name__ == "__main__":
    main()
