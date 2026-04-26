"""
Time-varying coefficient analysis: how has the importance of each feature
changed over the 11-year span?

For each season, fit a logistic on training data ENDING at that season
(using prior tournament outcomes only). Track coefficients over time.

This is an empirical study of basketball evolution, not a prediction
optimization. Output: coefficient trajectories over 2014-2025.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data_collection import load_all_mens_data
from src.pipeline import make_build_features_fn

plt.style.use("seaborn-v0_8-whitegrid")

# Use a focused feature set with both Barttorvik and Four Factors
FEATURES = [
    "seed_diff",
    "rank_diff_POM",
    "bart_net_diff",
    "bart_adjoe_diff",
    "bart_adjde_diff",
    "efg_pct_diff",
    "to_pct_diff",
    "or_pct_diff",
    "ft_rate_diff",
    "momentum_margin_diff",
]


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    seasons = [s for s in range(2014, 2026) if s != 2020]

    # Build full matrix once
    X_all, y_all = build_fn(seasons)
    feat_cols = [c for c in FEATURES if c in X_all.columns]

    # Per-season fit (each season trained on data UP TO that season, exclusive)
    coefs_by_year = {}
    rolling_window = 5  # train on last 5 seasons

    for target_season in seasons:
        # Use prior K seasons (rolling window)
        train_seasons = [s for s in seasons if s < target_season][-rolling_window:]
        if len(train_seasons) < 3:
            continue

        mask = X_all["Season"].isin(train_seasons)
        X_train = X_all[mask][feat_cols].apply(pd.to_numeric, errors="coerce")
        y_train = y_all[mask.values]

        # Impute and scale
        imp = SimpleImputer(strategy="median")
        scl = StandardScaler()
        X_proc = scl.fit_transform(imp.fit_transform(X_train))

        lr = LogisticRegression(C=0.5, max_iter=2000, solver="lbfgs")
        lr.fit(X_proc, y_train)

        coefs_by_year[target_season] = dict(zip(feat_cols, lr.coef_[0]))
        print(f"Season {target_season} (trained on {len(train_seasons)} prior seasons): "
              f"intercept={lr.intercept_[0]:.3f}")

    # Build coefficient DataFrame
    coef_df = pd.DataFrame(coefs_by_year).T
    coef_df.index.name = "season"
    coef_df.to_csv("output/time_varying_coefs.csv")
    print(f"\nCoefficients over {len(coef_df)} seasons:")
    print(coef_df.round(3).to_string())

    # Plot top features
    fig, axes = plt.subplots(2, 1, figsize=(11, 9))

    # Top: Barttorvik features
    bart_feats = ["bart_net_diff", "bart_adjoe_diff", "bart_adjde_diff", "rank_diff_POM"]
    ax = axes[0]
    for feat in bart_feats:
        if feat in coef_df.columns:
            ax.plot(coef_df.index, coef_df[feat], "o-", label=feat, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Time-varying coefficients: rating-system features")
    ax.set_xlabel("Season")
    ax.set_ylabel("Standardized coefficient")
    ax.legend()

    # Bottom: tactical features (Four Factors + momentum)
    tactical = ["efg_pct_diff", "to_pct_diff", "or_pct_diff", "ft_rate_diff",
                "momentum_margin_diff", "seed_diff"]
    ax = axes[1]
    for feat in tactical:
        if feat in coef_df.columns:
            ax.plot(coef_df.index, coef_df[feat], "o-", label=feat, linewidth=2)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Time-varying coefficients: tactical & seed features")
    ax.set_xlabel("Season")
    ax.set_ylabel("Standardized coefficient")
    ax.legend(ncol=2)

    plt.tight_layout()
    plt.savefig("output/time_varying_coefs.png", dpi=150, bbox_inches="tight")
    print("\nSaved time_varying_coefs.png and time_varying_coefs.csv")

    # Trend analysis
    print("\n=== TREND ANALYSIS (linear slope of coefficient over seasons) ===")
    slopes = {}
    for col in coef_df.columns:
        # Slope of coefficient over seasons (years)
        years = np.array(coef_df.index.tolist())
        vals = coef_df[col].values
        slope, intercept = np.polyfit(years, vals, 1)
        slopes[col] = slope
        direction = "increasing" if slope > 0 else "decreasing"
        if abs(slope) > 0.005:
            print(f"  {col:<25s} slope = {slope:+.4f}/year ({direction} importance)")


if __name__ == "__main__":
    main()
