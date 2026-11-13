"""Figure S.5: where the two thresholds sit (the ablation of Section S.17).

For each family and signal-to-noise ratio the record's own alpha quantile is
drawn as a box, the alpha quantile of a change-free record as a dashed line,
and the derived threshold of Eq. (13) as a marker.  The gap between the box and
the dashed line is the bias that the breaks introduce into the empirical rule.
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dcs_cpd as D
import synthetic as S

SUP = HERE.parent / "manuscript" / "supplementary" / "figs"
TAU, ALPHA, BETA, THETA, REPS = 32, 0.005, 0.6, 0.7, 15
plt.rcParams.update({
    "font.family": "serif", "font.size": 9, "axes.labelsize": 9,
    "axes.titlesize": 9, "legend.fontsize": 7.5, "xtick.labelsize": 8,
    "ytick.labelsize": 8, "axes.linewidth": 0.6, "lines.linewidth": 0.9,
    "text.usetex": False, "mathtext.fontset": "dejavuserif",
    "figure.dpi": 200, "savefig.bbox": "tight", "savefig.pad_inches": 0.02,
})


def main():
    snrs = (0.5, 2.0, 5.0)
    fig, axes = plt.subplots(1, 3, figsize=(6.3, 2.9), sharey=True)
    for ax, snr in zip(axes, snrs):
        clean = []
        for rep in range(10):
            x, _ = S.synthetic_series(length=2000, period=TAU, snr=snr,
                                      family="none", seed=5000 + rep)
            clean.append(float(np.quantile(D.cosine_series(x, TAU), ALPHA)))
        clean_mean = float(np.mean(clean))

        data, derived = [], []
        for family in S.FAMILIES:
            own = []
            for rep in range(REPS):
                x, _ = S.synthetic_series(length=2000, period=TAU, snr=snr,
                                          family=family, seed=1000 + rep)
                y = D.cosine_series(x, TAU)
                own.append(float(np.quantile(y, ALPHA)))
                if rep == 0:
                    derived.append(D.threshold(ALPHA, snr, TAU))
            data.append(own)
        derived_mean = float(np.mean(derived))

        bp = ax.boxplot(data, widths=0.55, patch_artist=True, showfliers=False,
                        medianprops=dict(color="black", linewidth=0.9),
                        boxprops=dict(facecolor="0.85", edgecolor="black", linewidth=0.6),
                        whiskerprops=dict(color="black", linewidth=0.6),
                        capprops=dict(color="black", linewidth=0.6))
        ax.axhline(clean_mean, color="black", linestyle="--", linewidth=0.9)
        ax.plot(range(1, 5), [derived_mean] * 4, linestyle="none", marker="D",
                markersize=4.2, color="tab:red", zorder=5)
        ax.set_xticks(range(1, 5))
        ax.set_xticklabels(["level", "polarity", "phase", "period"], rotation=32, ha="right")
        ax.set_title(rf"$\mathrm{{SNR}}={snr:g}$", loc="left")
        ax.axhline(0.0, color="0.65", linewidth=0.5)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel(r"threshold $y_T$")
    handles = [
        plt.Line2D([], [], color="black", linestyle="--", linewidth=0.9,
                   label=r"$\alpha$-quantile, change-free record"),
        plt.Line2D([], [], color="tab:red", marker="D", linestyle="none",
                   markersize=4.2, label="derived threshold, Eq. (13)"),
        plt.Rectangle((0, 0), 1, 1, facecolor="0.85", edgecolor="black",
                      linewidth=0.6, label=r"$\alpha$-quantile, record with breaks"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, -0.30))
    out = SUP / "Figure_S5.pdf"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    print("wrote", out)


if __name__ == "__main__":
    main()
