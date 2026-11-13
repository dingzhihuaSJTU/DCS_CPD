"""Figures for the CNSNS manuscript (regenerated from results/*.csv)."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dcs_cpd as D
from synthetic import synthetic_series

FIG = HERE.parent / "manuscript" / "figs_submission"
SUP = HERE.parent / "manuscript" / "supplementary" / "figs"
plt.rcParams.update({
    "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
    "axes.titlesize": 8, "legend.fontsize": 7, "xtick.labelsize": 7,
    "ytick.labelsize": 7, "axes.linewidth": 0.6, "lines.linewidth": 0.9,
    "text.usetex": False, "mathtext.fontset": "dejavuserif",
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def score_example():
    x, truth = synthetic_series(length=1200, period=32, snr=2.0,
                                family="polarity", seed=11)
    out = D.score_curve(x, tau=32, snr=2.0)
    pred = D.detect(x, tau=32, snr=2.0)
    fig, axes = plt.subplots(3, 1, figsize=(4.6, 4.2), sharex=True)
    axes[0].plot(x, color="0.25", lw=0.7)
    for c in truth:
        axes[0].axvline(c, color="tab:red", ls="--", lw=0.8)
    axes[0].set_ylabel(r"$x_t$")
    axes[0].set_title("(a) record and true breaks", loc="left")
    axes[1].plot(np.arange(len(out["cos"])) + 32, out["cos"], color="0.25", lw=0.7)
    axes[1].axhline(out["yt"], color="tab:blue", ls=":", lw=0.9)
    axes[1].set_ylabel(r"$y_i$")
    axes[1].set_title("(b) similarity series and threshold $y_T$", loc="left")
    axes[2].plot(out["score"], color="tab:green", lw=0.9)
    axes[2].axhline(0.5, color="0.5", ls=":", lw=0.8)
    for c in truth:
        axes[2].axvline(c, color="tab:red", ls="--", lw=0.8)
    for c in pred:
        axes[2].axvline(c, color="tab:orange", ls="-", lw=0.7, alpha=0.9)
    axes[2].set_ylabel(r"score $S_i$")
    axes[2].set_xlabel("sample")
    axes[2].set_title("(c) score, detections", loc="left")
    fig.tight_layout()
    fig.savefig(FIG / "Figure_4.pdf")
    plt.close(fig)


def period_sweep():
    d = pd.read_csv(HERE / "results" / "period.csv")
    p = d.pivot_table(index="ratio", columns="snr", values="f1")
    fig, ax = plt.subplots(figsize=(3.4, 2.3))
    for snr, style in ((0.5, "o-"), (2.0, "s-"), (5.0, "^-")):
        ax.plot(p.index, p[snr], style, ms=3, label=r"SNR$=%g$" % snr)
    ax.axvspan(1.0, 1.10, color="0.85", zorder=0)
    ax.set_xlabel("period dilation ratio")
    ax.set_ylabel(r"mean $F_1$")
    ax.set_ylim(-0.02, 0.85)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25, lw=0.4)
    fig.tight_layout()
    fig.savefig(FIG / "Figure_5.pdf")
    plt.close(fig)


def runtime():
    d = pd.read_csv(HERE / "results" / "runtime.csv")
    p = d.pivot_table(index="length", columns="method", values="seconds")
    labels = {"dcs_cpd": "DCS-CPD", "bottomup_l2": "BottomUp-$\\ell_2$",
              "window_l2": "Window-$\\ell_2$", "pelt_l2": "PELT-$\\ell_2$",
              "mosum": "MOSUM", "binseg_l2": "BinSeg-$\\ell_2$",
              "bocpd": "BOCPD"}
    fig, ax = plt.subplots(figsize=(3.4, 2.4))
    for col in ("dcs_cpd", "bottomup_l2", "window_l2", "pelt_l2", "mosum",
                "binseg_l2", "bocpd"):
        ax.plot(p.index, p[col], "o-", ms=2.5, lw=0.9, label=labels[col])
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("record length $L$")
    ax.set_ylabel("run time (s)")
    ax.legend(frameon=False, ncol=2, fontsize=6)
    ax.grid(alpha=0.25, lw=0.4, which="both")
    fig.tight_layout()
    fig.savefig(SUP / "Figure_S4.pdf")
    plt.close(fig)


if __name__ == "__main__":
    FIG.mkdir(parents=True, exist_ok=True)
    score_example()
    period_sweep()
    runtime()
    print("figures written to", FIG)
