"""DCS-CPD: windowed structural correlation change point detector.

Reference implementation of the procedure described in the manuscript.
All indices in this module are 0-based.  Equation numbers in the comments
refer to the main text of the CNSNS manuscript.

Pipeline
--------
1. ``cosine_series``  Eq. (4)   windowed cosine similarity between adjacent
                                non-overlapping blocks of length ``tau``.
2. ``threshold``      Eq. (12)  null quantile of the similarity statistic at
                                level ``alpha`` for a given SNR.
3. ``score_curve``    Eq. (33)  counting statistic over the last ``2*tau``
                                similarities, mapped to a decision score by
                                the sigmoid of Theorem 1.
4. ``localize``       Algorithm 2  centroid of the maximal plateau inside a
                                ``4*tau`` search window.
"""
from __future__ import annotations

import numpy as np
from scipy.special import gammaln
from scipy.stats import norm

DEFAULTS = dict(tau=32, alpha=0.005, beta=0.6, theta=0.7)


# --------------------------------------------------------------------------
# 1. windowed cosine similarity, Eq. (4)
# --------------------------------------------------------------------------
def cosine_series(x: np.ndarray, tau: int, centered: bool = False) -> np.ndarray:
    """Similarity of adjacent blocks of length ``tau``.

    ``centered=False`` is the uncentered cosine similarity of Eq. (4),
    ``centered=True`` the Pearson correlation coefficient used by the
    moving-correlation detector of Wu et al. (2019).  Both are computed on the
    same overlapping block pairs, so a comparison between them isolates the
    effect of the measure.
    """
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n < 2 * tau:
        return np.empty(0)
    windows = np.lib.stride_tricks.sliding_window_view(x, 2 * tau)
    a, b = windows[:, :tau], windows[:, tau:]
    if centered:
        a = a - a.mean(axis=1, keepdims=True)
        b = b - b.mean(axis=1, keepdims=True)
    num = np.einsum("ij,ij->i", a, b)
    den = np.sqrt(np.einsum("ij,ij->i", a, a) * np.einsum("ij,ij->i", b, b))
    with np.errstate(divide="ignore", invalid="ignore"):
        y = np.where(den > 0, num / np.where(den > 0, den, 1.0), 0.0)
    return y


# --------------------------------------------------------------------------
# 2. null threshold, Eq. (12)
# --------------------------------------------------------------------------
def threshold(alpha: float, snr: float, tau: int) -> float:
    """``y_T = mu * exp( z_alpha * s )`` with ``mu = s/(s+1+1/tau)`` (Eq. 12)."""
    mu = snr / (snr + 1.0 + 1.0 / tau)
    spread = np.sqrt(1.0 / (tau * snr ** 2)
                     + 2.0 / ((tau - 1) * (snr + 1.0 + 1.0 / tau) ** 2))
    return float(mu * np.exp(norm.ppf(alpha) * spread))


def snr_from_cosines(y: np.ndarray, q: float = 0.75, lo: float = 1e-2,
                     hi: float = 1e3) -> float:
    """Label-free SNR from the moment identity ``E[y] = s/(1+s)``.

    The upper quartile of the empirical similarity series is used as a robust
    stand-in for the stationary mean: at most ``1-q`` of the record may be
    disturbed by changes before the estimate is contaminated.  Inverting the
    identity gives ``s = m/(1-m)``.
    """
    y = np.asarray(y, dtype=float)
    y = y[np.isfinite(y)]
    m = float(np.quantile(y, q)) if y.size else 0.5
    m = min(max(m, 1e-3), 1.0 - 1e-3)
    return float(np.clip(m / (1.0 - m), lo, hi))


# --------------------------------------------------------------------------
# 3. decision score, Eqs. (24)-(33) and Proposition 2
# --------------------------------------------------------------------------
def score_constants(tau: int, alpha: float, beta: float) -> dict:
    """Slope ``k``, envelope offset ``R`` and critical count ``n_0``."""
    delta = tau - tau / (3.0 * beta)
    scale = (1.0 - beta) * (1.0 + 2.0 * delta)
    k = 1.0 / scale + np.log((1.0 - alpha) / alpha)
    R = (tau * beta / scale
         + 0.5 * np.log(1.0 + 2.0 * delta)
         + 0.5 * np.log(4.0 * np.pi * tau * beta * (1.0 - beta))
         + 2.0 * tau * np.log(1.0 - alpha))
    lbinom = gammaln(2 * tau + 1) - gammaln(tau + 1) - gammaln(tau + 1)
    return {"k": k, "R": R, "delta": delta, "lbinom": lbinom}


def critical_count(tau: int, alpha: float, beta: float, theta: float,
                   prior: float | None = None) -> float:
    """``n_0`` of Eq. (38); ``prior`` defaults to ``1/(8 tau)``."""
    c = score_constants(tau, alpha, beta)
    if prior is None:
        prior = 1.0 / (8.0 * tau)
    prior = float(np.clip(prior, 1e-6, 1 - 1e-6))
    num = (c["R"] + (1.0 - theta) * (tau / ((1 - beta) * (1 + 2 * c["delta"]))
                                     + c["lbinom"])
           - np.log(prior / (1.0 - prior)))
    return float(num / c["k"])


def score_curve(x: np.ndarray, tau: int = 32, snr: float | None = None,
                alpha: float = 0.005, beta: float = 0.6, theta: float = 0.7,
                prior: float | None = None, y_t: float | None = None,
                centered: bool = False) -> dict:
    """Point-wise change point score.

    Returns a dict with the score curve (same length as ``x``), the similarity
    series, the threshold and ``n_0``.  The score at position ``i`` counts
    similarities in ``y[i-2*tau+1 : i+1]``, i.e. the window that ends at
    ``i``.  Positions ``i < 2*tau-1`` carry no score and are set to zero.

    ``centered=True`` switches the similarity to the Pearson coefficient.  The
    closed-form threshold of Eq. (12) is derived for the uncentered cosine, so
    the centered variant replaces it by the empirical ``alpha`` quantile of its
    own similarity series, the calibration rule used by moving-correlation
    detectors.
    """
    x = np.asarray(x, dtype=float)
    y = cosine_series(x, tau, centered=centered)
    if y.size == 0:
        return {"score": np.zeros(len(x)), "cos": y, "yt": np.nan, "n0": np.nan,
                "snr": np.nan}
    if y_t is None:
        if centered:
            y_t = float(np.quantile(y, alpha))
        else:
            if snr is None:
                snr = snr_from_cosines(y)
            y_t = threshold(alpha, float(snr), tau)
    m = 2 * tau
    n0 = critical_count(tau, alpha, beta, theta, prior)
    k = score_constants(tau, alpha, beta)["k"]
    crossing = (y <= y_t).astype(float)
    csum = np.concatenate(([0.0], np.cumsum(crossing)))
    n = np.full(len(x), np.nan)
    # window [i-m+1, i] of the similarity series, for i = m-1 ... len(y)-1
    hi = len(y)
    lo = m - 1
    counts = csum[lo + 1:hi + 1] - csum[lo + 1 - m:hi + 1 - m]
    n[lo:hi] = counts
    score = np.zeros(len(x))
    valid = np.isfinite(n)
    score[valid] = 1.0 / (1.0 + np.exp(-k * (n[valid] - n0)))
    return {"score": score, "cos": y, "yt": float(y_t), "n0": float(n0),
            "snr": float(snr) if snr is not None else np.nan}


# --------------------------------------------------------------------------
# 4. localisation, Algorithm 2
# --------------------------------------------------------------------------
def localize(score: np.ndarray, tau: int, p_t: float = 0.5,
             min_sep: int | None = None) -> list:
    """Peak selection with a ``4*tau`` exclusion radius (Algorithm 2).

    Supra-threshold positions are grouped into connected runs.  Each run is
    represented by the centroid of its maximal plateau, which equals the peak
    when the maximum is unique and stays at the centre of the evidence when the
    posterior saturates.  Candidates are then accepted greedily in order of
    decreasing peak height, subject to a minimum separation of ``4*tau``
    samples, the resolution limit of the two-window comparison.
    """
    s = np.nan_to_num(np.asarray(score, dtype=float), nan=0.0)
    above = s >= p_t
    if not above.any():
        return []
    edges = np.flatnonzero(np.diff(np.concatenate(([0], above.astype(int), [0]))))
    candidates = []
    for start, stop in zip(edges[::2], edges[1::2]):
        seg = s[start:stop]
        peak = float(seg.max())
        at = np.flatnonzero(seg >= peak - 1e-12)
        candidates.append((peak, start + float(at.mean())))
    candidates.sort(key=lambda item: -item[0])
    accepted = []
    for peak, position in candidates:
        sep = 4 * tau if min_sep is None else int(min_sep)
        if all(abs(position - a) >= sep for a in accepted):
            accepted.append(position)
    return sorted(accepted)


def detect(x: np.ndarray, tau: int = 32, snr: float | None = None,
           alpha: float = 0.005, beta: float = 0.6, theta: float = 0.7,
           prior: float | None = None, p_t: float = 0.5,
           min_sep: int | None = None) -> list:
    """Convenience wrapper returning the detected change point positions."""
    out = score_curve(x, tau=tau, snr=snr, alpha=alpha, beta=beta, theta=theta,
                      prior=prior)
    if not out["cos"].size:
        return []
    return [int(round(p)) for p in localize(out["score"], tau, p_t, min_sep)
            if 0 < int(round(p)) < len(x)]
