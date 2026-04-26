"""
Calibration analysis: collect LOTO predictions for top models, build
reliability diagrams, decompose Brier score, test isotonic post-hoc calibration.

Usage:
    python scripts/run_calibration.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data
from src.pipeline import make_build_features_fn
from src.models import (
    SeedLogistic, KenPomLogistic, BarttovikLogistic,
    MultiFeatureLogistic, GradientBoostingModel,
)

plt.style.use("seaborn-v0_8-whitegrid")


def collect_loto_predictions(build_fn, model_factory, seasons):
    """Run LOTO and collect all (y_true, y_pred) pairs."""
    all_y_true = []
    all_y_pred = []
    for holdout in seasons:
        train = [s for s in seasons if s != holdout]
        X_train, y_train = build_fn(train)
        X_test, y_test = build_fn([holdout])
        if len(X_test) == 0:
            continue
        m = model_factory()
        m.fit(X_train, y_train)
        p = m.predict_proba(X_test)[:, 1]
        all_y_true.extend(y_test.tolist())
        all_y_pred.extend(p.tolist())
    return np.array(all_y_true), np.array(all_y_pred)


def reliability_data(y_true, y_pred, n_bins=10):
    """Compute reliability diagram bins."""
    bins = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    bin_pred = []
    bin_obs = []
    bin_count = []
    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_pred >= bins[i]) & (y_pred <= bins[i + 1])
        else:
            mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        if mask.sum() > 0:
            bin_pred.append(y_pred[mask].mean())
            bin_obs.append(y_true[mask].mean())
            bin_count.append(int(mask.sum()))
        else:
            bin_pred.append(np.nan)
            bin_obs.append(np.nan)
            bin_count.append(0)
    return bin_centers, np.array(bin_pred), np.array(bin_obs), np.array(bin_count)


def brier_decomposition(y_true, y_pred, n_bins=10):
    """Murphy decomposition: BS = Reliability - Resolution + Uncertainty."""
    base_rate = y_true.mean()
    uncertainty = base_rate * (1 - base_rate)

    bins = np.linspace(0, 1, n_bins + 1)
    reliability = 0.0
    resolution = 0.0
    n = len(y_true)

    for i in range(n_bins):
        if i == n_bins - 1:
            mask = (y_pred >= bins[i]) & (y_pred <= bins[i + 1])
        else:
            mask = (y_pred >= bins[i]) & (y_pred < bins[i + 1])
        nk = mask.sum()
        if nk == 0:
            continue
        f_k = y_pred[mask].mean()  # mean predicted in bin
        o_k = y_true[mask].mean()  # mean observed in bin
        reliability += (nk / n) * (f_k - o_k) ** 2
        resolution += (nk / n) * (o_k - base_rate) ** 2

    return reliability, resolution, uncertainty


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    seasons = [s for s in range(2014, 2026) if s != 2020]

    models = {
        "Seed Logistic": lambda: SeedLogistic(),
        "KenPom Logistic": lambda: KenPomLogistic(),
        "Barttorvik Logistic": lambda: BarttovikLogistic(),
        "Multi-Feature Logistic": lambda: MultiFeatureLogistic(C=0.5),
    }

    # Collect predictions
    preds = {}
    for name, factory in models.items():
        print(f"\n--- Collecting predictions: {name} ---")
        y_true, y_pred = collect_loto_predictions(build_fn, factory, seasons)
        preds[name] = (y_true, y_pred)

    # Reliability diagram
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for name, (yt, yp) in preds.items():
        bc, bp, bo, _ = reliability_data(yt, yp, n_bins=10)
        axes[0].plot(bp, bo, marker="o", label=name, linewidth=2)

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect calibration")
    axes[0].set_xlabel("Mean predicted probability")
    axes[0].set_ylabel("Observed frequency")
    axes[0].set_title("Reliability Diagram")
    axes[0].legend(loc="lower right", fontsize=10)
    axes[0].set_xlim(0, 1)
    axes[0].set_ylim(0, 1)

    # Histogram of predictions for best model
    yt_best, yp_best = preds["Multi-Feature Logistic"]
    axes[1].hist(yp_best, bins=20, color="steelblue", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Predicted probability (Multi-Feature Logistic)")
    axes[1].set_ylabel("Number of games")
    axes[1].set_title("Prediction Distribution")

    plt.tight_layout()
    plt.savefig("output/reliability_diagram.png", dpi=150, bbox_inches="tight")
    print("\nSaved reliability_diagram.png")

    # Brier decomposition + isotonic calibration
    rows = []
    for name, (yt, yp) in preds.items():
        rel, res, unc = brier_decomposition(yt, yp, n_bins=10)
        bs = brier_score_loss(yt, yp)

        # Cross-validated isotonic calibration: leave-one-fold-out within LOTO
        # Simpler: split data into 2 halves, calibrate on one, eval on other
        half = len(yt) // 2
        idx = np.random.RandomState(42).permutation(len(yt))
        cal_idx, eval_idx = idx[:half], idx[half:]

        iso = IsotonicRegression(out_of_bounds="clip")
        iso.fit(yp[cal_idx], yt[cal_idx])
        yp_iso = iso.predict(yp[eval_idx])
        bs_iso = brier_score_loss(yt[eval_idx], yp_iso)
        bs_uncal_eval = brier_score_loss(yt[eval_idx], yp[eval_idx])

        rows.append({
            "model": name,
            "brier": bs,
            "reliability": rel,
            "resolution": res,
            "uncertainty": unc,
            "brier_eval_uncal": bs_uncal_eval,
            "brier_eval_iso": bs_iso,
            "iso_improvement": bs_uncal_eval - bs_iso,
        })

    summary = pd.DataFrame(rows)
    print(f"\n{'='*70}\n  CALIBRATION SUMMARY\n{'='*70}")
    print(summary.to_string(index=False))
    summary.to_csv("output/calibration_summary.csv", index=False)
    print("\nSaved calibration_summary.csv")


if __name__ == "__main__":
    main()
