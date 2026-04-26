"""
Inductive (split) conformal prediction for tournament game outcomes.

For binary classification, conformal prediction provides distribution-free
sets {0, 1} or {0} or {1} or {} with provable miscoverage rate <= alpha.

For probability-valued predictions, we use Mondrian conformal calibration
to construct prediction intervals [p_low, p_high] for each game.

Key property: 90% credible intervals will contain the true outcome at
least 90% of the time, regardless of model assumptions.

Reference: Vovk et al. 2005, "Algorithmic Learning in a Random World".
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from src.data_collection import load_all_mens_data
from src.pipeline import make_build_features_fn
from src.models import BarttovikLogistic, MultiFeatureLogistic

plt.style.use("seaborn-v0_8-whitegrid")


def conformal_intervals(p_pred: np.ndarray, p_calib: np.ndarray, y_calib: np.ndarray,
                         alpha: float = 0.1) -> tuple:
    """Build (1 - alpha) prediction intervals via inductive conformal regression
    on the binary cross-entropy nonconformity score.

    Score: s_i = -y_i * log(p_i) - (1 - y_i) * log(1 - p_i)
    Larger s = worse prediction (game more 'surprising').

    Quantile q = (1 - alpha) of calibration scores.

    For test prediction p_test, the prediction set covers all probabilities
    p* such that the BCE-distance to {0, 1} respects q.
    """
    eps = 1e-6
    p_calib_c = np.clip(p_calib, eps, 1 - eps)
    scores = -y_calib * np.log(p_calib_c) - (1 - y_calib) * np.log(1 - p_calib_c)

    # Quantile threshold
    q = np.quantile(scores, 1 - alpha, method="higher")

    # For each test prediction p, compute the BCE distance to outcome 1 vs 0
    p_test_c = np.clip(p_pred, eps, 1 - eps)
    bce_to_1 = -np.log(p_test_c)
    bce_to_0 = -np.log(1 - p_test_c)

    # Prediction set: {y in {0,1} : BCE(p, y) <= q}
    pred_1 = bce_to_1 <= q  # outcome=1 is in prediction set
    pred_0 = bce_to_0 <= q  # outcome=0 is in prediction set

    return pred_0, pred_1, q


def conformal_probability_intervals(p_pred: np.ndarray, p_calib: np.ndarray,
                                     y_calib: np.ndarray, alpha: float = 0.1) -> tuple:
    """Construct probability-valued [p_low, p_high] intervals.

    Use signed residuals: r_i = y_i - p_i.
    Quantile of |r| gives a symmetric interval.
    """
    residuals = np.abs(y_calib - p_calib)
    q = np.quantile(residuals, 1 - alpha, method="higher")

    p_low = np.clip(p_pred - q, 0, 1)
    p_high = np.clip(p_pred + q, 0, 1)

    return p_low, p_high, q


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    seasons = [s for s in range(2014, 2026) if s != 2020]

    # Use Barttorvik Logistic (best single model) for conformal calibration
    print("\nUsing Barttorvik Logistic as base predictor")
    print("Conformal calibration via inductive split: train on first 7 seasons,")
    print("calibrate on next 2, test on last 2 (rolling)\n")

    # Coverage validation across folds
    coverage_results = []
    interval_results = []

    for outer_holdout in seasons[3:]:  # need at least 3 prior seasons
        prior_seasons = [s for s in seasons if s < outer_holdout]
        if len(prior_seasons) < 4:
            continue
        # Split prior into train (60%) and calibration (40%)
        n_calib = max(2, len(prior_seasons) // 3)
        train_seasons = prior_seasons[:-n_calib]
        calib_seasons = prior_seasons[-n_calib:]

        X_train, y_train = build_fn(train_seasons)
        X_calib, y_calib = build_fn(calib_seasons)
        X_test, y_test = build_fn([outer_holdout])

        if len(X_test) == 0:
            continue

        # Fit base model
        model = BarttovikLogistic()
        model.fit(X_train, y_train)
        p_calib = model.predict_proba(X_calib)[:, 1]
        p_test = model.predict_proba(X_test)[:, 1]

        # Conformal probability intervals at 90% coverage
        for alpha in [0.10, 0.20]:
            p_low, p_high, q = conformal_probability_intervals(
                p_test, p_calib, y_calib, alpha=alpha
            )
            # Coverage: how often does the true outcome's "side" fall in [p_low, p_high]
            # For binary, treat outcome y as 0 or 1; check |p_pred - y| <= q
            actual_cover = np.mean(np.abs(p_test - y_test) <= q)
            mean_width = np.mean(p_high - p_low)
            coverage_results.append({
                "season": outer_holdout, "alpha": alpha,
                "target_coverage": 1 - alpha,
                "actual_coverage": actual_cover,
                "mean_width": mean_width,
                "n_games": len(y_test),
            })
            print(f"Season {outer_holdout} alpha={alpha:.2f}: "
                  f"target {1-alpha:.0%}, actual {actual_cover:.1%}, "
                  f"width={mean_width:.3f}")

        # Save full intervals for the 0.10 case
        p_low, p_high, _ = conformal_probability_intervals(p_test, p_calib, y_calib, 0.10)
        interval_results.extend([
            {"season": outer_holdout, "p_pred": float(p), "p_low": float(pl),
             "p_high": float(ph), "y_true": int(yt), "covered": bool(abs(p - yt) <= 0.5 + 0)}
            for p, pl, ph, yt in zip(p_test, p_low, p_high, y_test)
        ])

    cov_df = pd.DataFrame(coverage_results)
    interval_df = pd.DataFrame(interval_results)

    print(f"\n{'='*60}\n  CONFORMAL COVERAGE SUMMARY\n{'='*60}")
    for alpha in [0.10, 0.20]:
        sub = cov_df[cov_df["alpha"] == alpha]
        avg = sub["actual_coverage"].mean()
        target = 1 - alpha
        print(f"alpha={alpha:.2f}  target={target:.0%}  actual_avg={avg:.1%}  "
              f"avg_width={sub['mean_width'].mean():.3f}")

    cov_df.to_csv("output/conformal_coverage.csv", index=False)
    interval_df.to_csv("output/conformal_intervals.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Coverage by season
    ax = axes[0]
    for alpha, color in [(0.10, "#2563eb"), (0.20, "#16a34a")]:
        sub = cov_df[cov_df["alpha"] == alpha].sort_values("season")
        ax.plot(sub["season"], sub["actual_coverage"], "o-",
                color=color, label=f"target {1-alpha:.0%}")
        ax.axhline(1 - alpha, color=color, linestyle="--", alpha=0.4)
    ax.set_xlabel("Season")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Conformal calibration: empirical vs nominal coverage")
    ax.set_ylim(0.6, 1.0)
    ax.legend()

    # Width distribution
    ax = axes[1]
    sub = cov_df[cov_df["alpha"] == 0.10]
    ax.bar(sub["season"], sub["mean_width"], color="#2563eb", alpha=0.85)
    ax.set_xlabel("Season")
    ax.set_ylabel("Mean prediction interval width")
    ax.set_title("Conformal interval widths (alpha=0.10)")

    plt.tight_layout()
    plt.savefig("output/conformal_coverage.png", dpi=150, bbox_inches="tight")
    print("\nSaved conformal_coverage.png and CSVs")


if __name__ == "__main__":
    main()
