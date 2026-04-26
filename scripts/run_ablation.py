"""
Ablation study: systematically remove feature groups from MultiFeatureLogistic
and measure the LOTO Brier impact.

Usage:
    python scripts/run_ablation.py
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from src.data_collection import load_all_mens_data
from src.pipeline import make_build_features_fn
from src.evaluation import leave_one_tournament_out


# Feature groups for ablation (matches the column names in pipeline output)
FEATURE_GROUPS = {
    "Seed":           ["seed_diff"],
    "POM":            ["rank_diff_POM"],
    "Barttorvik":     ["bart_net_diff", "bart_adjoe_diff", "bart_adjde_diff", "bart_barthag_diff"],
    "Four Factors":   ["efg_pct_diff", "to_pct_diff", "or_pct_diff", "ft_rate_diff",
                       "opp_efg_pct_diff", "opp_to_pct_diff", "opp_or_pct_diff", "opp_ft_rate_diff"],
    "Efficiency":     ["net_eff_diff", "off_eff_diff", "def_eff_diff", "tempo_diff"],
    "Momentum":       ["momentum_margin_diff", "momentum_winpct_diff"],
    "Coach":          ["coach_tourney_apps_diff"],
}

ALL_FEATURES = [c for grp in FEATURE_GROUPS.values() for c in grp]


class AblationLogistic(BaseEstimator, ClassifierMixin):
    """Multi-feature logistic with custom column subset."""

    def __init__(self, exclude_cols=None, C=0.5):
        self.exclude_cols = exclude_cols or []
        self.C = C
        self.pipe = None

    def fit(self, X, y):
        cols = [c for c in ALL_FEATURES if c in X.columns and c not in self.exclude_cols]
        self.cols_ = cols
        self.pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs")),
        ])
        self.pipe.fit(X[cols], y)
        return self

    def predict_proba(self, X):
        return self.pipe.predict_proba(X[self.cols_])


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    seasons = [s for s in range(2014, 2026) if s != 2020]

    results = []

    # Full set
    print(f"\n{'='*60}\n  Full feature set (baseline)\n{'='*60}")
    df = leave_one_tournament_out(build_fn, lambda: AblationLogistic(exclude_cols=[]), seasons)
    full_brier = df["brier_score"].mean()
    results.append({"variant": "Full set", "n_features": len(ALL_FEATURES),
                    "brier": full_brier, "delta": 0.0})
    print(f"  Mean Brier: {full_brier:.4f}")

    # Drop each feature group
    for group_name, group_cols in FEATURE_GROUPS.items():
        print(f"\n{'='*60}\n  Drop: {group_name} ({len(group_cols)} features)\n{'='*60}")
        df = leave_one_tournament_out(
            build_fn,
            lambda gc=group_cols: AblationLogistic(exclude_cols=gc),
            seasons,
        )
        brier = df["brier_score"].mean()
        delta = brier - full_brier
        results.append({
            "variant": f"-- {group_name}",
            "n_features": len(ALL_FEATURES) - len(group_cols),
            "brier": brier,
            "delta": delta,
        })
        print(f"  Mean Brier: {brier:.4f} ({delta:+.4f} vs full)")

    # Single-feature baselines (only that group)
    for group_name, group_cols in FEATURE_GROUPS.items():
        if len(group_cols) > 4:  # skip large groups for single-feature mode
            continue
        exclude = [c for c in ALL_FEATURES if c not in group_cols]
        print(f"\n{'='*60}\n  Only: {group_name}\n{'='*60}")
        df = leave_one_tournament_out(
            build_fn,
            lambda ex=exclude: AblationLogistic(exclude_cols=ex),
            seasons,
        )
        brier = df["brier_score"].mean()
        delta = brier - full_brier
        results.append({
            "variant": f"Only {group_name}",
            "n_features": len(group_cols),
            "brier": brier,
            "delta": delta,
        })
        print(f"  Mean Brier: {brier:.4f} ({delta:+.4f} vs full)")

    # Print summary
    summary = pd.DataFrame(results).sort_values("brier")
    print(f"\n{'='*60}\n  ABLATION SUMMARY\n{'='*60}")
    print(summary.to_string(index=False))

    summary.to_csv("output/ablation_results.csv", index=False)
    print("\nSaved to output/ablation_results.csv")


if __name__ == "__main__":
    main()
