"""Deterministic synthetic generators for the CNSNS evaluation.

Every record is ``x = b + f + xi`` with a sinusoidal core component ``f``,
a constant bias ``b`` and white Gaussian noise ``xi``.  The four change
families differ only in which parameter of the core component is switched
between the two alternating regimes:

===========  ==========================================================
``level``    bias sign inversion, ``b -> -b``
``polarity`` core component sign inversion, ``f -> -f``
``phase``    core component phase jump, ``phi -> phi + pi/2``
``period``   period dilation, ``T -> 1.05 T``
===========  ==========================================================

The noise variance is set to ``(0.5 + b^2)/snr``, so ``snr`` is the power
ratio between the noise-free record and the noise, as used throughout the
manuscript.  Breakpoints are spaced ``8*period`` apart and alternate between
the two regimes, so the change point prior is ``1/(8*period)`` and the
minimum segment length is ``8*period = 4*tau`` for the default ``tau=period``.
"""
from __future__ import annotations

import numpy as np

FAMILIES = ("level", "polarity", "phase", "period")
PERIOD_RATIO = 1.05
PHASE_JUMP = np.pi / 2


def change_points(length: int, period: int, spacing: int = 8) -> list:
    ends = np.arange(spacing * period, length, spacing * period)
    return [int(e) for e in ends]


def synthetic_series(length: int = 2000, period: int = 32, snr: float = 2.0,
                     family: str = "level", b: float = 1.0, seed: int = 0,
                     spacing: int = 8):
    """Return ``(x, truth)`` for one synthetic record."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    base = np.sin(2 * np.pi * t / period)
    truth = change_points(length, period, spacing)
    mask = np.zeros(length)
    for k, e in enumerate(truth):
        mask[e:] = 1.0 if k % 2 == 0 else 0.0
    noise_sd = np.sqrt((0.5 + b * b) / snr)

    if family == "none":
        x = b + base + rng.normal(0, noise_sd, length)
    elif family == "level":
        x = b * (1 - 2 * mask) + base + rng.normal(0, noise_sd, length)
    elif family == "polarity":
        x = b + base * (1 - 2 * mask) + rng.normal(0, noise_sd, length)
    elif family == "phase":
        shifted = np.sin(2 * np.pi * t / period + PHASE_JUMP)
        x = b + np.where(mask > 0, shifted, base) + rng.normal(0, noise_sd, length)
    elif family == "period":
        stretched = np.sin(2 * np.pi * t / (PERIOD_RATIO * period))
        x = b + np.where(mask > 0, stretched, base) + rng.normal(0, noise_sd, length)
    else:
        raise ValueError("unknown family: %r" % (family,))
    return x, truth


def period_sweep_series(length: int = 2000, period: int = 32, snr: float = 2.0,
                        ratio: float = 1.05, b: float = 1.0, seed: int = 0,
                        spacing: int = 8):
    """Period-dilation record with a free dilation ratio (boundary study)."""
    rng = np.random.default_rng(seed)
    t = np.arange(length)
    base = np.sin(2 * np.pi * t / period)
    truth = change_points(length, period, spacing)
    mask = np.zeros(length)
    for k, e in enumerate(truth):
        mask[e:] = 1.0 if k % 2 == 0 else 0.0
    stretched = np.sin(2 * np.pi * t / (ratio * period))
    noise_sd = np.sqrt((0.5 + b * b) / snr)
    x = b + np.where(mask > 0, stretched, base) + rng.normal(0, noise_sd, length)
    return x, truth
