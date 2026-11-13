"""Threshold rule and the integer rule: the ablation behind Table S.9.

A  derived threshold  y_T = Eq. (13) from (alpha, tau, snr); decision on the score at 1/2.
B  own quantile      y_T = empirical q of the record's own similarity series, with q
                     chosen per SNR so that the false-alarm rate matches A; same score.
C  integer rule      same y_T as A, decision on n_i >= ceil(n_0).

A and C select the same supra-threshold positions; the run reports how far their
localisation and F1 differ.  A and B differ only in the threshold.
"""
from __future__ import annotations

import csv
import math
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import dcs_cpd as D
import metrics as M
import synthetic as S

TAU, ALPHA, BETA, THETA, REPS = 32, 0.005, 0.6, 0.7, 15
MATCHED_Q = {0.5: 0.075, 2.0: 0.015, 5.0: 0.003}


def counts(x, y_t, tau):
    y = D.cosine_series(x, tau)
    m = 2 * tau
    c = np.concatenate(([0.0], np.cumsum((y <= y_t).astype(float))))
    n = np.full(len(x), np.nan)
    lo = m - 1
    n[lo:len(y)] = c[lo + 1:len(y) + 1] - c[lo + 1 - m:len(y) + 1 - m]
    return y, n


def main():
    fam_rows, null_rows = [], []
    stats = dict(records=0, same_n=0, same_pos=0, df1=[], dpos=[])
    for family in S.FAMILIES:
        for snr in (0.5, 2.0, 5.0):
            fa, fb, fc = [], [], []
            for rep in range(REPS):
                x, truth = S.synthetic_series(length=2000, period=TAU, snr=snr,
                                              family=family, seed=1000 + rep)
                oa = D.score_curve(x, tau=TAU, snr=snr, alpha=ALPHA, beta=BETA, theta=THETA)
                det_a = D.localize(oa["score"], TAU, p_t=0.5)
                fa.append(M.evaluate(truth, det_a, epsilon=8, n=len(x))["f1"])

                y, n = counts(x, oa["yt"], TAU)
                det_c = D.localize(np.nan_to_num(n, nan=-1e9), TAU,
                                   p_t=math.ceil(oa["n0"]) - 0.5)
                fc.append(M.evaluate(truth, det_c, epsilon=8, n=len(x))["f1"])

                ob = D.score_curve(x, tau=TAU, snr=snr, alpha=ALPHA, beta=BETA,
                                   theta=THETA, y_t=float(np.quantile(y, MATCHED_Q[snr])))
                det_b = D.localize(ob["score"], TAU, p_t=0.5)
                fb.append(M.evaluate(truth, det_b, epsilon=8, n=len(x))["f1"])

                st = stats
                st["records"] += 1
                st["same_n"] += int(len(det_a) == len(det_c))
                if len(det_a) == len(det_c):
                    st["same_pos"] += int(all(abs(p - q) <= 1 for p, q in zip(sorted(det_a), sorted(det_c))))
                    if det_a:
                        st["dpos"] += [abs(p - q) for p, q in zip(sorted(det_a), sorted(det_c))]
                st["df1"].append(M.evaluate(truth, det_a, epsilon=8, n=len(x))["f1"]
                                 - M.evaluate(truth, det_c, epsilon=8, n=len(x))["f1"])
            fam_rows.append(dict(family=family, snr=snr, f1_derived=float(np.mean(fa)),
                                 f1_quantile=float(np.mean(fb)), f1_integer=float(np.mean(fc))))
    for snr in (0.5, 2.0, 5.0):
        na, nb = [], []
        for rep in range(10):
            x, _ = S.synthetic_series(length=2000, period=TAU, snr=snr, family="none", seed=5000 + rep)
            oa = D.score_curve(x, tau=TAU, snr=snr, alpha=ALPHA, beta=BETA, theta=THETA)
            na.append(len(D.localize(oa["score"], TAU, p_t=0.5)))
            y = D.cosine_series(x, TAU)
            ob = D.score_curve(x, tau=TAU, snr=snr, alpha=ALPHA, beta=BETA, theta=THETA,
                               y_t=float(np.quantile(y, MATCHED_Q[snr])))
            nb.append(len(D.localize(ob["score"], TAU, p_t=0.5)))
        null_rows.append(dict(snr=snr, q=MATCHED_Q[snr], det_derived=float(np.mean(na)),
                              det_quantile=float(np.mean(nb))))
    out = HERE / "results"
    out.mkdir(parents=True, exist_ok=True)
    with open(out / "threshold.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(fam_rows[0].keys())); w.writeheader(); w.writerows(fam_rows)
    with open(out / "threshold_null.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(null_rows[0].keys())); w.writeheader(); w.writerows(null_rows)
    st = stats
    print(f"records {st['records']} | same count {st['same_n']} | same pos {st['same_pos']} "
          f"| mean |dF1| {np.mean(np.abs(st['df1'])):.4f} | mean |dpos| {np.mean(st['dpos']):.2f}")
    for r in fam_rows:
        print(f"{r['family']:9s} snr={r['snr']:<4} A={r['f1_derived']:.3f} B={r['f1_quantile']:.3f} C={r['f1_integer']:.3f}")
    for r in null_rows:
        print(f"null snr={r['snr']:<4} q={r['q']} A={r['det_derived']:.2f} B={r['det_quantile']:.2f}")


if __name__ == "__main__":
    main()
