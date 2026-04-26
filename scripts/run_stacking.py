"""
Proper stacking ensemble: meta-learner trained on out-of-fold predictions
from base models. Compared to the existing weighted-blend ensemble that uses
linear convex combination, stacking allows the meta-learner to:
  - have an intercept (calibration shift)
  - apply logistic transformation
  - learn nonlinear interactions

Base models: Barttorvik Logistic, KenPom Logistic, Multi-Feature Logistic, XGBoost
Meta-learner: Logistic Regression (with optional interaction features)

Usage:
    python scripts/run_stacking.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data
from src.pipeline import make_build_features_fn
from src.models import (
    BarttovikLogistic, KenPomLogistic, MultiFeatureLogistic, GradientBoostingModel,
)


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    seasons = [s for s in range(2014, 2026) if s != 2020]

    base_factories = {
        "Barttorvik": lambda: BarttovikLogistic(),
        "KenPom": lambda: KenPomLogistic(),
        "MultiFeature": lambda: MultiFeatureLogistic(C=0.5),
        "XGBoost": lambda: GradientBoostingModel("xgboost"),
    }

    print(f"\nStacking ensemble with {len(base_factories)} base models")
    print("Meta-learner: Logistic Regression on base model OOF predictions\n")

    fold_results = []

    for outer_holdout in seasons:
        outer_train_seasons = [s for s in seasons if s != outer_holdout]
        X_train, y_train = build_fn(outer_train_seasons)
        X_test, y_test = build_fn([outer_holdout])

        n_train = len(y_train)
        n_models = len(base_factories)
        oof_preds = np.full((n_train, n_models), 0.5)
        names = list(base_factories.keys())

        # Inner LOTO: for each training season, hold it out, train on others, predict it
        for inner_holdout in outer_train_seasons:
            inner_train_seasons = [s for s in outer_train_seasons if s != inner_holdout]
            X_inner_train, y_inner_train = build_fn(inner_train_seasons)
            X_inner_val, y_inner_val = build_fn([inner_holdout])

            mask = (X_train["Season"] == inner_holdout).values
            if mask.sum() == 0:
                continue

            for j, (name, factory) in enumerate(base_factories.items()):
                m = factory()
                m.fit(X_inner_train, y_inner_train)
                preds = m.predict_proba(X_inner_val)[:, 1]
                oof_preds[mask, j] = preds

        # Train meta-learner on OOF predictions
        # Option 1: simple logistic on raw predictions
        # Option 2: include logit-transformed predictions for better calibration
        eps = 1e-6
        oof_logits = np.log(np.clip(oof_preds, eps, 1 - eps) /
                            (1 - np.clip(oof_preds, eps, 1 - eps)))

        meta = LogisticRegression(C=1.0, max_iter=2000, solver="lbfgs")
        meta.fit(oof_logits, y_train)

        # Train base models on full outer_train, predict holdout
        test_preds = np.zeros((len(y_test), n_models))
        for j, (name, factory) in enumerate(base_factories.items()):
            m = factory()
            m.fit(X_train, y_train)
            test_preds[:, j] = m.predict_proba(X_test)[:, 1]

        # Apply meta-learner
        test_logits = np.log(np.clip(test_preds, eps, 1 - eps) /
                             (1 - np.clip(test_preds, eps, 1 - eps)))
        stacked = meta.predict_proba(test_logits)[:, 1]
        bs = brier_score_loss(y_test, stacked)

        # Also compute simple-average ensemble for comparison
        avg_preds = test_preds.mean(axis=1)
        bs_avg = brier_score_loss(y_test, avg_preds)

        # And best single base model
        bs_singles = {name: brier_score_loss(y_test, test_preds[:, j])
                      for j, name in enumerate(names)}
        best_single_name = min(bs_singles, key=bs_singles.get)

        coef_str = ", ".join(f"{n}={c:.2f}" for n, c in zip(names, meta.coef_[0]))
        print(f"Season {outer_holdout}: stacked={bs:.4f}  avg={bs_avg:.4f}  "
              f"best={best_single_name}({bs_singles[best_single_name]:.4f})")
        print(f"  Meta logits coef: intercept={meta.intercept_[0]:.2f}, {coef_str}")

        fold_results.append({
            "season": outer_holdout,
            "stacked_brier": bs,
            "avg_brier": bs_avg,
            "best_single_name": best_single_name,
            "best_single_brier": bs_singles[best_single_name],
            **{f"{n}_brier": bs_singles[n] for n in names},
        })

    df = pd.DataFrame(fold_results)
    print(f"\n{'='*60}\n  STACKING ENSEMBLE SUMMARY\n{'='*60}")
    print(f"Stacked mean Brier:        {df['stacked_brier'].mean():.4f}")
    print(f"Simple average mean Brier: {df['avg_brier'].mean():.4f}")
    print(f"Per-base mean Brier:")
    for n in base_factories.keys():
        print(f"  {n:<14s} {df[f'{n}_brier'].mean():.4f}")

    df.to_csv("output/stacking_results.csv", index=False)
    print("\nSaved to output/stacking_results.csv")


if __name__ == "__main__":
    main()
