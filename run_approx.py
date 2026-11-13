"""Accuracy of the null approximations: moments and tail level.

For a stationary record of length 200000 (tau = 32, several SNR) the empirical
mean, standard deviation and 0.5% quantile of the similarity series are
compared with the closed forms of Lemma 1 and Lemma 2.
"""
import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dcs_cpd as D


def main():
    rows = []
    for snr in (0.5, 1.0, 2.0, 5.0):
        for tau in (16, 32, 64):
            rng = np.random.default_rng(12345)
            length = 200000
            t = np.arange(length)
            b = 1.0
            period = tau
            var_xi = (0.5 + b * b) / snr
            x = b + np.sin(2 * np.pi * t / period) + rng.normal(0, np.sqrt(var_xi), length)
            y = D.cosine_series(x, tau)
            yt = D.threshold(0.005, snr, tau)
            mu_pred = snr / (snr + 1 + 1.0 / tau)
            spread = np.sqrt(1.0 / (tau * snr ** 2)
                             + 2.0 / ((tau - 1) * (snr + 1 + 1.0 / tau) ** 2))
            rows.append({
                "snr": snr, "tau": tau,
                "mean_pred": mu_pred, "mean_emp": float(y.mean()),
                "sd_pred": mu_pred * spread, "sd_emp": float(y.std()),
                "yt": yt, "alpha_nominal": 0.005,
                "alpha_emp": float((y <= yt).mean()),
                "q005_emp": float(np.quantile(y, 0.005)),
            })
    out = HERE / "results" / "approx.csv"
    with out.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print("wrote", out)


if __name__ == "__main__":
    main()
