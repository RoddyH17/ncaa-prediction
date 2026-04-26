"""
Theoretical regret bound for post-hoc calibration when the base predictor
is near Bayes-optimal.

Setup:
  - Base predictor f̂(x) -> probability estimate
  - True conditional distribution p*(x) = P(Y=1 | X=x)
  - Calibration error: C(f̂) = E[(f̂(X) - p*(X))^2]
  - Bayes Brier: BS* = E[p*(X)(1 - p*(X))]
  - Brier of f̂: BS(f̂) = BS* + C(f̂) (Murphy decomposition under refinement = 0)

Claim (informal):
  Any post-hoc calibrator g: [0,1] -> [0,1] fit on N IID samples cannot
  yield expected Brier improvement greater than C(f̂).

  In particular, if C(f̂) = O(δ) and N is finite, then:
      E[BS(g ∘ f̂)] - BS(f̂) >= -C(f̂) + R_N(g)
  where R_N(g) is the calibrator's estimation error, typically Ω(N^{-1/2})
  for parametric and Ω(N^{-1/3}) for isotonic regression.

  Therefore when C(f̂) is small enough that R_N(g) > C(f̂), the EXPECTED
  Brier change from calibration is non-negative (i.e., calibration cannot
  help). The threshold:
      C(f̂) <= R_N(g)
  defines a "calibration futility regime."

Empirical verification:
  We compute C(f̂) for our LOTO Multi-Feature Logistic via the standard
  reliability decomposition and compare to bootstrap-estimated R_N for
  isotonic. If C(f̂) < R_N, calibration is provably futile in expectation.

This module provides utilities to:
  1. Estimate calibration error C(f̂) on (predictions, outcomes)
  2. Estimate isotonic estimation error R_N via bootstrap
  3. Test whether the base predictor is in the futility regime
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss


def calibration_error_l2(p: np.ndarray, y: np.ndarray, n_bins: int = 15) -> float:
    """L2 (Murphy reliability) calibration error: sum_b w_b (mean_p_b - mean_y_b)^2.

    Equivalent to E[(p - p_true)^2] under binning approximation.
    Tighter binning gives a more accurate estimate but higher variance for
    small N.
    """
    p = np.clip(np.asarray(p), 1e-6, 1 - 1e-6)
    y = np.asarray(y)
    bins = np.linspace(0, 1, n_bins + 1)
    err = 0.0
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        n_b = mask.sum()
        if n_b < 1: continue
        err += (n_b / len(p)) * (p[mask].mean() - y[mask].mean()) ** 2
    return float(err)


def isotonic_estimation_error_bootstrap(
    p: np.ndarray, y: np.ndarray, n_boot: int = 200, seed: int = 42
) -> dict:
    """Bootstrap estimate of isotonic calibrator's estimation error R_N.

    For each bootstrap sample:
      - Fit isotonic on the resample
      - Predict on original data
      - Compute MSE between predictions and original predictions of an
        isotonic fit on the FULL data
    The mean of these MSEs estimates the variance of the isotonic estimator.
    """
    rng = np.random.default_rng(seed)
    N = len(p)

    # Reference: isotonic fit on full data
    iso_full = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3)
    iso_full.fit(p, y)
    p_full = iso_full.predict(p)

    boot_errors = []
    for _ in range(n_boot):
        idx = rng.integers(0, N, N)
        iso = IsotonicRegression(out_of_bounds="clip", y_min=1e-3, y_max=1 - 1e-3)
        iso.fit(p[idx], y[idx])
        p_boot = iso.predict(p)
        boot_errors.append(np.mean((p_boot - p_full) ** 2))

    return {
        "R_N_estimate": float(np.mean(boot_errors)),
        "R_N_std": float(np.std(boot_errors)),
        "n_boot": n_boot,
    }


def futility_test(
    p: np.ndarray, y: np.ndarray, n_bins: int = 15, n_boot: int = 200, seed: int = 42
) -> dict:
    """Compare calibration error to isotonic estimation error.

    Returns dict with:
      C_f:               estimated calibration error of base predictor
      R_N:               estimated isotonic estimation error
      ratio:             R_N / C_f (ratio > 1 ⇒ isotonic over-fits more than it corrects)
      futility:          True if R_N >= C_f (calibration is provably futile)
    """
    C_f = calibration_error_l2(p, y, n_bins=n_bins)
    R = isotonic_estimation_error_bootstrap(p, y, n_boot=n_boot, seed=seed)
    ratio = R["R_N_estimate"] / max(C_f, 1e-10)
    return {
        "C_f": C_f,
        "R_N": R["R_N_estimate"],
        "R_N_std": R["R_N_std"],
        "ratio": ratio,
        "futility": ratio >= 1.0,
    }
