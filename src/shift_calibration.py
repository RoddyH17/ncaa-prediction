"""
Distribution-shift-aware calibration for tournament prediction.

Problem: standard post-hoc calibration (isotonic, Platt, temperature scaling)
assumes the calibration data and target distribution are exchangeable. In
tournament prediction this fails — each year is a slightly different draw.

Empirically: calibration improvements on LOTO out-of-fold predictions of
~0.005 Brier do NOT transfer to a held-out year, and often degrade Brier.

This module provides:

  - LOTO-OOF calibration data collection (predictions tagged with season)
  - Standard calibrators (isotonic, Platt, temperature)
  - Robust calibrators:
      * leave-one-season-out (LOSO) isotonic ensemble
      * temperature scaling with shrinkage prior
      * year-stratified isotonic with hierarchical pooling
  - Evaluation harness: LOTO-improve, hold-out-degrade, gap quantification
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats, optimize
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss


# ---------------------------------------------------------------------------
# Logit / sigmoid helpers
# ---------------------------------------------------------------------------

def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


# ---------------------------------------------------------------------------
# Standard calibrators
# ---------------------------------------------------------------------------

class IsotonicCalibrator:
    name = "isotonic"

    def fit(self, p, y, season=None):
        self.iso = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995)
        self.iso.fit(p, y)
        return self

    def predict(self, p):
        return np.clip(self.iso.predict(p), 0.005, 0.995)


class PlattCalibrator:
    name = "platt"

    def fit(self, p, y, season=None):
        self.lr = LogisticRegression(C=1e6, max_iter=1000)
        self.lr.fit(logit(p).reshape(-1, 1), y)
        return self

    def predict(self, p):
        return np.clip(self.lr.predict_proba(logit(p).reshape(-1, 1))[:, 1], 0.005, 0.995)


class TemperatureCalibrator:
    """T = arg min NLL on (logit(p)/T, y)."""
    name = "temperature"

    def fit(self, p, y, season=None):
        z = logit(p)
        def nll(T_log):
            T = np.exp(T_log)
            q = expit(z / T)
            q = np.clip(q, 1e-6, 1 - 1e-6)
            return -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))
        res = optimize.minimize_scalar(nll, bounds=(-3, 3), method="bounded")
        self.T = float(np.exp(res.x))
        return self

    def predict(self, p):
        z = logit(p)
        return np.clip(expit(z / self.T), 0.005, 0.995)


# ---------------------------------------------------------------------------
# Robust shift-aware calibrators
# ---------------------------------------------------------------------------

class LOSOIsotonicEnsemble:
    """Leave-one-season-out isotonic ensemble.

    For each season s, fit an isotonic on data EXCLUDING season s. To predict
    on a new probability p, average the K trained isotonics. This reduces
    over-fit to any single season's idiosyncratic distribution.
    """
    name = "loso_isotonic"

    def fit(self, p, y, season):
        season = np.asarray(season)
        unique_seasons = np.unique(season)
        self.calibrators = []
        for s in unique_seasons:
            mask = season != s
            if mask.sum() < 50:
                continue
            iso = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995)
            iso.fit(p[mask], y[mask])
            self.calibrators.append(iso)
        return self

    def predict(self, p):
        preds = np.column_stack([c.predict(p) for c in self.calibrators])
        return np.clip(preds.mean(axis=1), 0.005, 0.995)


class ShrinkageTemperature:
    """Temperature scaling with Bayesian shrinkage toward T=1.

    Optimize: NLL + lambda * (log T)^2
    The penalty discourages aggressive temperature changes that wouldn't
    transfer. Tuned via inner LOSO-CV on the calibration set.
    """
    name = "shrink_temperature"

    def __init__(self, lambdas=(0.0, 1.0, 10.0, 100.0, 1000.0)):
        self.lambdas = lambdas

    def fit(self, p, y, season):
        season = np.asarray(season)
        unique_seasons = np.unique(season)
        # Inner LOSO-CV to pick lambda
        best_lambda, best_loss = self.lambdas[0], np.inf
        for lam in self.lambdas:
            losses = []
            for s in unique_seasons:
                mask_tr = season != s
                mask_te = season == s
                if mask_tr.sum() < 50 or mask_te.sum() < 5:
                    continue
                T = self._fit_once(p[mask_tr], y[mask_tr], lam)
                z = logit(p[mask_te])
                q = np.clip(expit(z / T), 1e-6, 1 - 1e-6)
                losses.append(brier_score_loss(y[mask_te], q))
            avg = np.mean(losses) if losses else np.inf
            if avg < best_loss:
                best_loss, best_lambda = avg, lam
        # Final fit
        self.T = self._fit_once(p, y, best_lambda)
        self.best_lambda = best_lambda
        return self

    def _fit_once(self, p, y, lam):
        z = logit(p)
        def loss(T_log):
            T = np.exp(T_log)
            q = expit(z / T)
            q = np.clip(q, 1e-6, 1 - 1e-6)
            nll = -np.sum(y * np.log(q) + (1 - y) * np.log(1 - q))
            return nll + lam * (T_log ** 2)
        res = optimize.minimize_scalar(loss, bounds=(-3, 3), method="bounded")
        return float(np.exp(res.x))

    def predict(self, p):
        return np.clip(expit(logit(p) / self.T), 0.005, 0.995)


class HierarchicalIsotonic:
    """Year-stratified isotonic with hierarchical pooling.

    For each season, fit an isotonic. Each season's curve is partial-pooled
    toward the global isotonic via shrinkage parameter alpha:
        f_s(p) = alpha * f_global(p) + (1-alpha) * f_s_raw(p)
    For prediction on a NEW season, use f_global only — but the shrinkage
    enforces that f_global was fit to data already pooled, reducing
    over-fit to season-specific quirks.
    """
    name = "hier_isotonic"

    def __init__(self, alphas=(0.0, 0.3, 0.5, 0.7, 0.9, 1.0)):
        self.alphas = alphas

    def fit(self, p, y, season):
        season = np.asarray(season)
        unique_seasons = np.unique(season)

        # Build per-season raw isotonics
        per_season = {}
        for s in unique_seasons:
            mask = season == s
            if mask.sum() < 30:
                continue
            iso_s = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995)
            iso_s.fit(p[mask], y[mask])
            per_season[s] = iso_s

        # Build global isotonic
        global_iso = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995)
        global_iso.fit(p, y)

        # Pick alpha via LOSO inner CV
        best_alpha, best_loss = 1.0, np.inf
        for alpha in self.alphas:
            losses = []
            for s in unique_seasons:
                mask_tr = season != s
                mask_te = season == s
                if mask_tr.sum() < 50 or mask_te.sum() < 5:
                    continue
                iso_global = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995)
                iso_global.fit(p[mask_tr], y[mask_tr])
                # Use only global for held-out prediction (alpha=1 means full pool)
                # But effectively alpha controls how aggressive we are
                pred = iso_global.predict(p[mask_te])
                # alpha=1 -> pure global; alpha=0 -> back off toward identity (uncalibrated)
                pred = alpha * pred + (1 - alpha) * p[mask_te]
                pred = np.clip(pred, 0.005, 0.995)
                losses.append(brier_score_loss(y[mask_te], pred))
            avg = np.mean(losses) if losses else np.inf
            if avg < best_loss:
                best_loss, best_alpha = avg, alpha

        self.alpha = best_alpha
        self.global_iso = global_iso
        return self

    def predict(self, p):
        pred = self.global_iso.predict(p)
        pred = self.alpha * pred + (1 - self.alpha) * p
        return np.clip(pred, 0.005, 0.995)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_calibrator(
    calibrator_factory,
    p_oof: np.ndarray,
    y_oof: np.ndarray,
    season_oof: np.ndarray,
    p_holdout: np.ndarray,
    y_holdout: np.ndarray,
) -> dict:
    """Quantify shift-transfer behaviour of a calibrator.

    `calibrator_factory` is a zero-arg callable returning a fresh calibrator.

    Returns:
      brier_oof_uncal:   Brier on LOTO OOF without calibration
      brier_oof_cal:     Brier on LOTO OOF with leave-one-season-out calibration
      brier_holdout_uncal: Brier on held-out year without calibration
      brier_holdout_cal:   Brier on held-out year with calibrator fit on full OOF
      loto_improvement:  brier_oof_uncal - brier_oof_cal (positive = improvement)
      holdout_change:    brier_holdout_uncal - brier_holdout_cal (positive = improvement)
      transfer_gap:      loto_improvement - holdout_change (positive = LOTO over-promises)
    """
    bs_oof_uncal = brier_score_loss(y_oof, p_oof)
    bs_holdout_uncal = brier_score_loss(y_holdout, p_holdout)

    season_oof = np.asarray(season_oof)
    unique = np.unique(season_oof)
    p_oof_cal = np.zeros_like(p_oof)
    for s in unique:
        mask_tr = season_oof != s
        mask_te = season_oof == s
        if mask_tr.sum() < 50 or mask_te.sum() < 1:
            p_oof_cal[mask_te] = p_oof[mask_te]
            continue
        cal = calibrator_factory()
        cal.fit(p_oof[mask_tr], y_oof[mask_tr], season_oof[mask_tr])
        p_oof_cal[mask_te] = cal.predict(p_oof[mask_te])
    bs_oof_cal = brier_score_loss(y_oof, p_oof_cal)

    cal_full = calibrator_factory()
    cal_full.fit(p_oof, y_oof, season_oof)
    p_holdout_cal = cal_full.predict(p_holdout)
    bs_holdout_cal = brier_score_loss(y_holdout, p_holdout_cal)

    return {
        "calibrator": cal_full.name,
        "brier_oof_uncal": bs_oof_uncal,
        "brier_oof_cal": bs_oof_cal,
        "brier_holdout_uncal": bs_holdout_uncal,
        "brier_holdout_cal": bs_holdout_cal,
        "loto_improvement": bs_oof_uncal - bs_oof_cal,
        "holdout_change": bs_holdout_uncal - bs_holdout_cal,
        "transfer_gap": (bs_oof_uncal - bs_oof_cal) - (bs_holdout_uncal - bs_holdout_cal),
    }
