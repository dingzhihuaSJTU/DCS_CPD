"""Supporting experiments: run time, parameter sensitivity, false alarms and
posterior calibration.

Sub-commands
------------
``runtime``     average run time against record length (Fig. 6)
``sensitivity`` F1 against tau, alpha, beta, theta (Table 3)
``null``        false alarms on records without change points (Table 5)
``calibration`` reliability and Brier score of the point-wise score
``period``      F1 against the period-dilation ratio (Fig. 5)
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import baselines as B
import dcs_cpd as D
import metrics as M
from synthetic import synthetic_series, period_sweep_series


def _row(**kw):
    return kw


def runtime(ns):
    rows = []
    rng = np.random.default_rng(7)
    for length in (500, 1000, 2000, 5000):
        for trial in range(ns.reps):
            x, _ = synthetic_series(length=length, period=32, snr=2.0,
                                    family="level", seed=1000 + trial)
            for method in ("dcs_cpd", "binseg_l2", "pelt_l2", "window_l2",
                           "bottomup_l2", "mosum", "bocpd"):
                t0 = time.perf_counter()
                if method == "dcs_cpd":
                    D.detect(x, tau=32, snr=2.0)
                elif method == "mosum":
                    B.mosum(x, bandwidth=32)
                elif method == "bocpd":
                    B.bocpd(x)
                else:
                    B.METHODS[method](x, min_size=128)
                rows.append({"length": length, "trial": trial, "method": method,
                             "seconds": time.perf_counter() - t0})
    return rows


def sensitivity(ns):
    rows = []
    grid = {"tau": (16, 32, 64), "alpha": (0.001, 0.005, 0.01),
            "beta": (0.4, 0.6, 0.8), "theta": (0.3, 0.5, 0.7, 0.9)}
    base = dict(tau=32, alpha=0.005, beta=0.6, theta=0.7)
    for family in ("level", "polarity", "period"):
        for snr in (2.0,):
            for rep in range(ns.reps):
                x, truth = synthetic_series(length=2000, period=32, snr=snr,
                                            family=family, seed=5000 + rep)
                for name, values in grid.items():
                    for value in values:
                        cfg = dict(base)
                        cfg[name] = value
                        pred = D.detect(x, tau=cfg["tau"], snr=snr,
                                        alpha=cfg["alpha"], beta=cfg["beta"],
                                        theta=cfg["theta"])
                        rows.append({"family": family, "snr": snr, "rep": rep,
                                     "parameter": name, "value": value,
                                     "f1": M.evaluate(truth, pred, 8, len(x))["f1"]})
    return rows


def null(ns):
    """False alarms per record and per 1000 samples on change-free series."""
    rows = []
    for snr in (0.5, 1.0, 2.0, 5.0):
        for rep in range(ns.reps):
            x, _ = synthetic_series(length=2000, period=32, snr=snr,
                                    family="none", seed=8000 + rep)
            for method in ("dcs_cpd", "pelt_l2", "pelt_rbf", "mosum", "bocpd"):
                if method == "dcs_cpd":
                    pred = D.detect(x, tau=32, snr=snr)
                elif method == "mosum":
                    pred = B.mosum(x, bandwidth=32)
                elif method == "bocpd":
                    pred = B.bocpd(x)
                else:
                    pred = B.METHODS[method](x, min_size=128)
                rows.append({"snr": snr, "rep": rep, "method": method,
                             "n_false": len(pred),
                             "per_1000": 1000.0 * len(pred) / len(x)})
    return rows


def calibration(ns):
    """Reliability of the point-wise score under H0 and under a mixed record."""
    rows = []
    for snr in (0.5, 1.0, 2.0, 5.0):
        for rep in range(ns.reps):
            x, truth = synthetic_series(length=5000, period=32, snr=snr,
                                        family="level", seed=9000 + rep)
            out = D.score_curve(x, tau=32, snr=snr)
            score = out["score"]
            labels = np.zeros(len(x))
            for c in truth:
                labels[max(0, c - 8):c + 9] = 1.0
            valid = np.arange(2 * 32, len(x))
            for bin_lo in np.arange(0.0, 1.0, 0.1):
                sel = valid[(score[valid] >= bin_lo) & (score[valid] < bin_lo + 0.1)]
                if sel.size:
                    rows.append({"snr": snr, "rep": rep, "bin": round(float(bin_lo), 2),
                                 "predicted": float(score[sel].mean()),
                                 "observed": float(labels[sel].mean()),
                                 "n": int(sel.size)})
    return rows


def period(ns):
    rows = []
    for ratio in (1.00, 1.02, 1.05, 1.08, 1.10, 1.15, 1.20, 1.50):
        for snr in (0.5, 2.0, 5.0):
            for rep in range(ns.reps):
                x, truth = period_sweep_series(ratio=ratio, snr=snr,
                                               seed=6000 + rep)
                pred = D.detect(x, tau=32, snr=snr)
                rows.append({"ratio": ratio, "snr": snr, "rep": rep,
                             "f1": M.evaluate(truth, pred, 8, len(x))["f1"],
                             "n": len(pred)})
    return rows


COMMANDS = {"runtime": runtime, "sensitivity": sensitivity, "null": null,
            "calibration": calibration, "period": period}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("command", choices=sorted(COMMANDS))
    ap.add_argument("--reps", type=int, default=10)
    ap.add_argument("--out", default=None)
    ns = ap.parse_args()
    rows = COMMANDS[ns.command](ns)
    out = Path(ns.out or (HERE / "results" / (ns.command + ".csv")))
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote %s (%d rows)" % (out, len(rows)))


if __name__ == "__main__":
    main()
