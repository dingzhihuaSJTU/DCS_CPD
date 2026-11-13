"""Synthetic benchmark: DCS-CPD against distribution- and segmentation-based
detectors on four change families and three signal-to-noise ratios.

Usage
-----
    python run_synthetic.py --reps 20 --out results/synthetic.csv

Every detector is scored with the one-to-one matching F1 at a tolerance of
``--epsilon`` samples and with the same minimum segment length ``4*tau``.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

import baselines as B
import dcs_cpd as D
import metrics as M
from synthetic import FAMILIES, synthetic_series

DCS_METHODS = ("dcs_cpd", "dcs_cpd_est", "cc_moving")
RUPTURES = ("binseg_l2", "pelt_l2", "bottomup_l2", "window_l2",
            "binseg_rbf", "pelt_rbf", "bottomup_rbf", "window_rbf",
            "binseg_normal", "pelt_normal", "bottomup_normal", "window_normal")
OTHERS = ("mosum", "cusum", "bocpd")
ALL_METHODS = DCS_METHODS + RUPTURES + OTHERS


def cc_moving(x, tau, alpha=0.005):
    """Moving-correlation jump detector (Wu et al., 2019).

    The Pearson correlation of adjacent blocks is computed exactly as in the
    reference method; a jump is declared where the correlation series falls
    below its ``alpha`` quantile and the negative excursion reaches its local
    minimum.  The minimum of block pair ``j`` sits at sample ``j + tau``, which
    is where the two blocks meet.
    """
    y = D.cosine_series(x, tau, centered=True)
    if y.size == 0:
        return []
    thr = float(np.quantile(y, alpha))
    below = y <= thr
    if not below.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], below.astype(int), [0]))))
    runs = list(zip(edges[::2], edges[1::2]))
    cands = sorted(((float(y[a:b].min()), a + int(np.argmin(y[a:b])) + tau)
                    for a, b in runs), key=lambda z: z[0])
    kept = []
    for _, pos in cands:
        if all(abs(pos - q) >= 4 * tau for q in kept):
            kept.append(pos)
    return [p for p in sorted(kept) if 0 < p < len(x)]


def run_one(job):
    family, snr, rep, method, args = job
    x, truth = synthetic_series(length=args["length"], period=args["period"],
                                snr=snr, family=family, seed=args["seed0"] + rep)
    tau = args["period"]
    t0 = time.perf_counter()
    status, error = "ok", ""
    try:
        if method == "dcs_cpd":
            pred = D.detect(x, tau=tau, snr=snr, alpha=args["alpha"],
                            beta=args["beta"], theta=args["theta"])
        elif method == "dcs_cpd_est":
            pred = D.detect(x, tau=tau, snr=None, alpha=args["alpha"],
                            beta=args["beta"], theta=args["theta"])
        elif method == "cc_moving":
            pred = cc_moving(x, tau=tau, alpha=args["alpha"])
        elif method == "mosum":
            pred = B.mosum(x, bandwidth=tau)
        elif method == "cusum":
            pred = B.cusum(x)
        elif method == "bocpd":
            pred = B.bocpd(x)
        else:
            pred = B.METHODS[method](x, min_size=4 * tau)
    except Exception as exc:                      # pragma: no cover
        pred, status, error = [], "error", "%s: %s" % (type(exc).__name__, exc)
    elapsed = time.perf_counter() - t0
    row = {"family": family, "snr": snr, "rep": rep, "method": method,
           "seed": args["seed0"] + rep, "runtime_seconds": elapsed,
           "status": status, "error": error}
    row.update(M.evaluate(truth, pred, epsilon=args["epsilon"], n=len(x)))
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=20)
    ap.add_argument("--length", type=int, default=2000)
    ap.add_argument("--period", type=int, default=32)
    ap.add_argument("--families", default=",".join(FAMILIES))
    ap.add_argument("--snrs", default="0.5,2,5")
    ap.add_argument("--epsilon", type=int, default=8)
    ap.add_argument("--alpha", type=float, default=0.005)
    ap.add_argument("--beta", type=float, default=0.6)
    ap.add_argument("--theta", type=float, default=0.7)
    ap.add_argument("--seed0", type=int, default=20260101)
    ap.add_argument("--methods", default=",".join(ALL_METHODS))
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default=str(HERE / "results" / "synthetic.csv"))
    ns = ap.parse_args()

    families = [f for f in ns.families.split(",") if f]
    snrs = [float(s) for s in ns.snrs.split(",")]
    methods = [m for m in ns.methods.split(",") if m]
    unknown = set(methods) - set(ALL_METHODS)
    if unknown:
        ap.error("unknown methods: %s" % sorted(unknown))

    args = {"length": ns.length, "period": ns.period, "epsilon": ns.epsilon,
            "alpha": ns.alpha, "beta": ns.beta, "theta": ns.theta,
            "seed0": ns.seed0}
    jobs = list(itertools.product(families, snrs, range(ns.reps), methods,
                                  [args]))
    started = time.time()
    rows = []
    with ProcessPoolExecutor(max_workers=ns.workers) as pool:
        for i, row in enumerate(pool.map(run_one, jobs, chunksize=4), 1):
            rows.append(row)
            if i % 200 == 0:
                print("  %d/%d  %.0fs" % (i, len(jobs), time.time() - started),
                      flush=True)

    out = Path(ns.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    import csv
    with out.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    with out.with_suffix(".config.json").open("w") as fh:
        json.dump({"argv": sys.argv, "python": platform.python_version(),
                   "numpy": np.__version__, "args": vars(ns),
                   "seconds": time.time() - started}, fh, indent=2)
    print("wrote %s (%d rows, %.0fs)" % (out, len(rows), time.time() - started))


if __name__ == "__main__":
    main()
