"""Reproducible one-dimensional baselines for the P1 protocol."""
from __future__ import annotations

from typing import Iterable, List
import math
import numpy as np


def _points(values: Iterable[int], n: int) -> List[int]:
    return sorted({int(x) for x in values if 0 < int(x) < n})


def ruptures_detector(x, method: str, model: str = "l2", min_size: int = 32, penalty: float | None = None, jump: int = 8, width: int | None = None) -> List[int]:
    """Thin wrapper around the reference ``ruptures`` implementations.

    ``method='window'`` maps to the sliding-window detector, which needs an
    explicit ``width`` instead of ``min_size``. The penalty follows the same
    BIC-style convention (``log(n) * var``) for every detector so that the
    comparison uses one cost/significance rule.
    """
    import ruptures as rpt
    y = np.asarray(x, dtype=float).reshape(-1, 1)
    n = len(y)
    if model in ("rbf", "normal"):
        sd = float(np.std(y))
        if sd > 0:
            y = (y - float(np.mean(y))) / sd
    if method.lower() == "window":
        if width is None:
            width = max(min_size + 1, 2 * min_size)
        algo = rpt.Window(model=model, width=int(width), jump=jump).fit(y)
    else:
        cls = {"pelt": rpt.Pelt, "binseg": rpt.Binseg, "bottomup": rpt.BottomUp}[method.lower()]
        algo = cls(model=model, min_size=min_size, jump=jump).fit(y)
    if penalty is None:
        penalty = math.log(max(n, 2)) * max(float(np.var(y)), 1e-8)
    return _points(algo.predict(pen=float(penalty)), n)


def binseg_l2(x, **kwargs): return ruptures_detector(x, "binseg", "l2", **kwargs)
def pelt_l2(x, **kwargs): return ruptures_detector(x, "pelt", "l2", **kwargs)
def bottomup_l2(x, **kwargs): return ruptures_detector(x, "bottomup", "l2", **kwargs)
def window_l2(x, **kwargs): return ruptures_detector(x, "window", "l2", **kwargs)
def binseg_rbf(x, **kwargs): return ruptures_detector(x, "binseg", "rbf", **kwargs)
def pelt_rbf(x, **kwargs): return ruptures_detector(x, "pelt", "rbf", **kwargs)
def bottomup_rbf(x, **kwargs): return ruptures_detector(x, "bottomup", "rbf", **kwargs)
def window_rbf(x, **kwargs): return ruptures_detector(x, "window", "rbf", **kwargs)
def binseg_normal(x, **kwargs): return ruptures_detector(x, "binseg", "normal", **kwargs)
def pelt_normal(x, **kwargs): return ruptures_detector(x, "pelt", "normal", **kwargs)
def bottomup_normal(x, **kwargs): return ruptures_detector(x, "bottomup", "normal", **kwargs)
def window_normal(x, **kwargs): return ruptures_detector(x, "window", "normal", **kwargs)


def moving_correlation(x, window: int = 32, threshold: float = 0.85, min_distance: int | None = None) -> List[int]:
    """Moving correlation of adjacent windows, with deterministic peak picking."""
    x = np.asarray(x, dtype=float); n = len(x); w = int(window)
    if min_distance is None: min_distance = 4 * w
    scores = np.zeros(n)
    for i in range(0, n - 2 * w + 1):
        a, b = x[i:i+w], x[i+w:i+2*w]
        sa, sb = a.std(), b.std()
        scores[i+w] = 0.0 if sa == 0 or sb == 0 else np.corrcoef(a, b)[0, 1]
    candidates = [i for i in range(w, n - w + 1) if scores[i] < threshold]
    return _nonmax_candidates(candidates, -scores, min_distance, n)


def mosum(x, bandwidth: int = 32, threshold: float = 3.0, min_distance: int | None = None) -> List[int]:
    """Two-sided standardized MOSUM for mean changes."""
    x = np.asarray(x, dtype=float); n = len(x); g = int(bandwidth)
    if min_distance is None: min_distance = 4 * g
    scale = np.std(x, ddof=1) or 1.0
    scores = np.zeros(n)
    for t in range(g, n-g):
        scores[t] = abs(x[t:t+g].mean() - x[t-g:t].mean()) / (scale * math.sqrt(2.0 / g))
    candidates = [t for t in range(g, n-g) if scores[t] >= threshold]
    return _nonmax_candidates(candidates, -scores, min_distance, n)


def _nonmax_candidates(candidates, ranking, min_distance, n):
    selected = []
    for i in sorted(candidates, key=lambda z: (ranking[z], z)):
        if all(abs(i-j) >= min_distance for j in selected): selected.append(i)
    return sorted(selected)


def cusum(x, threshold: float = 5.0, drift: float = 0.5, baseline_window: int = 32, min_distance: int = 32) -> List[int]:
    """Two-sided standardized Page CUSUM with a frozen initial baseline."""
    x = np.asarray(x, dtype=float); n = len(x)
    if n <= baseline_window + 1: return []
    mu = float(np.mean(x[:baseline_window])); sigma = float(np.std(x[:baseline_window], ddof=1) or 1.0)
    up = down = 0.0; out = []; last = -min_distance
    for t, z in enumerate((x - mu) / sigma):
        up = max(0.0, up + z - drift); down = min(0.0, down + z + drift)
        if t >= baseline_window and (up >= threshold or -down >= threshold) and t-last >= min_distance:
            out.append(t); last = t; up = down = 0.0
    return out


def bocpd(x, hazard: float = 1/100.0, min_distance: int | None = None, prior_window: int = 32,
          max_run_length: int = 512, kappa0: float = 1.0, alpha0: float = 1.0) -> List[int]:
    """Bayesian online change point detection (Adams & MacKay, 2007).

    Run-length posteriors are propagated in log space with a Normal--Inverse--Gamma
    conjugate prior on the segment mean and variance, so the predictive density is
    Student-$t$ and becomes sharper as the current run grows. A change point is
    reported when the posterior *mode* of the run length resets, i.e. when
    ``argmax_r R_t(r) < argmax_r R_{t-1}(r)`` and the completed run was at least
    ``min_distance`` samples long.

    Notes
    -----
    An earlier version of this function thresholded ``R_t(0)``, the posterior mass
    of a zero-length run. That criterion can never fire: because the change and
    continuation hypotheses share the same predictive factor, the recursion gives
    ``R_t(0) = hazard`` identically at every step, for every input series. The
    detector therefore always returned an empty list. The mode-reset statistic
    above is the standard remedy.
    """
    from scipy.special import gammaln, logsumexp
    x = np.asarray(x, dtype=float); n = len(x)
    if n < 2: return []
    w = max(2, min(int(prior_window), n))
    if min_distance is None: min_distance = max(8, min(n // 20, 32))
    mu0 = float(np.mean(x[:w]))
    var0 = float(np.var(x[:w], ddof=1))
    if not np.isfinite(var0) or var0 <= 0: var0 = 1.0
    beta0 = 0.5 * var0
    log_run = np.array([0.0]); mu = np.array([mu0]); kappa = np.array([kappa0])
    alpha = np.array([alpha0]); beta = np.array([beta0])
    out: List[int] = []
    prev_mode = 0
    for t in range(w, n):
        value = x[t]
        k = min(len(log_run), max_run_length)
        log_run, mu, kappa, alpha, beta = log_run[:k], mu[:k], kappa[:k], alpha[:k], beta[:k]
        scale = np.sqrt(beta * (kappa + 1.0) / (alpha * kappa)); df = 2.0 * alpha
        z = (value - mu) / scale
        log_pred = (gammaln((df + 1) / 2) - gammaln(df / 2) - 0.5 * np.log(np.pi * df)
                    - np.log(scale) - ((df + 1) / 2) * np.log1p(z * z / df))
        growth = log_run + log_pred + np.log1p(-hazard)
        change = logsumexp(log_run + log_pred + np.log(hazard))
        size = min(len(log_run) + 1, max_run_length + 1)
        new_log_run = np.empty(size); new_log_run[0] = change; new_log_run[1:] = growth[:size - 1]
        new_log_run -= logsumexp(new_log_run)
        new_mu = np.empty(size); new_kappa = np.empty(size); new_alpha = np.empty(size); new_beta = np.empty(size)
        new_mu[0], new_kappa[0], new_alpha[0], new_beta[0] = mu0, kappa0, alpha0, beta0
        new_kappa[1:] = kappa + 1.0
        new_alpha[1:] = alpha + 0.5
        new_mu[1:] = (kappa * mu + value) / (kappa + 1.0)
        new_beta[1:] = beta + kappa * (value - mu) ** 2 / (2.0 * (kappa + 1.0))
        mode = int(np.argmax(new_log_run))
        if mode < prev_mode and prev_mode >= min_distance and (not out or t - out[-1] >= min_distance):
            out.append(t)
        prev_mode = mode
        log_run, mu, kappa, alpha, beta = new_log_run, new_mu, new_kappa, new_alpha, new_beta
    return out

METHODS = {
    "binseg_l2": binseg_l2, "pelt_l2": pelt_l2, "bottomup_l2": bottomup_l2, "window_l2": window_l2,
    "binseg_rbf": binseg_rbf, "pelt_rbf": pelt_rbf, "bottomup_rbf": bottomup_rbf, "window_rbf": window_rbf,
    "binseg_normal": binseg_normal, "pelt_normal": pelt_normal, "bottomup_normal": bottomup_normal,
    "window_normal": window_normal,
    "wu2019_moving_corr": moving_correlation, "mosum": mosum, "cusum": cusum, "bocpd": bocpd,
}
