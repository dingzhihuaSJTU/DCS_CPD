"""Change point metrics with one-to-one matching.

A prediction is credited to at most one ground truth point and a ground truth
point to at most one prediction, greedily by increasing distance.  The public
reference implementation of the earlier submission increments the true-positive
count for every ground truth point that has *any* prediction inside the
tolerance, which allows a single prediction to be counted several times and can
push precision above one.  The matching below cannot exceed ``min(n_true,
n_pred)`` matches by construction.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np


def clean_points(points: Iterable[float], n: int | None = None) -> List[int]:
    out = sorted({int(round(float(p))) for p in points})
    if n is not None:
        out = [p for p in out if 0 < p < n]
    return out


def match_points(true_points, pred_points, epsilon: int, n: int | None = None):
    true = clean_points(true_points, n)
    pred = clean_points(pred_points, n)
    candidates = sorted((abs(t - p), t, p) for t in true for p in pred
                        if abs(t - p) <= epsilon)
    used_t, used_p, matches = set(), set(), []
    for _, t, p in candidates:
        if t not in used_t and p not in used_p:
            used_t.add(t)
            used_p.add(p)
            matches.append((t, p))
    return (matches, [t for t in true if t not in used_t],
            [p for p in pred if p not in used_p])


def evaluate(true_points, pred_points, epsilon: int = 8,
             n: int | None = None) -> Dict[str, float]:
    true = clean_points(true_points, n)
    pred = clean_points(pred_points, n)
    matches, missing, false = match_points(true, pred, epsilon, n)
    tp, fn, fp = len(matches), len(missing), len(false)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    errors = np.asarray([abs(t - p) for t, p in matches], dtype=float)
    signed = np.asarray([p - t for t, p in matches], dtype=float)
    boundaries = [0] + true + ([n] if n is not None else [])
    segments = max(len(boundaries) - 1, 1)
    return {
        "tp": float(tp), "fp": float(fp), "fn": float(fn),
        "precision": precision, "recall": recall, "f1": f1,
        "mean_localization_error": float(errors.mean()) if errors.size else float("nan"),
        "median_localization_error": float(np.median(errors)) if errors.size else float("nan"),
        "mean_signed_error": float(signed.mean()) if signed.size else float("nan"),
        "false_alarms_per_segment": float(fp / segments),
        "n_predictions": float(len(pred)),
    }


def summarize(values) -> Dict[str, float]:
    v = np.asarray([x for x in values if np.isfinite(x)], dtype=float)
    if not v.size:
        return {"mean": float("nan"), "median": float("nan"),
                "ci95_low": float("nan"), "ci95_high": float("nan")}
    if v.size == 1:
        return {"mean": float(v[0]), "median": float(v[0]),
                "ci95_low": float(v[0]), "ci95_high": float(v[0])}
    return {"mean": float(v.mean()), "median": float(np.median(v)),
            "ci95_low": float(np.quantile(v, 0.025)),
            "ci95_high": float(np.quantile(v, 0.975))}
