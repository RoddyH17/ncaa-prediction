"""
Evaluation framework: Brier score, calibration, backtest.

Target benchmarks:
  Seed baseline:     ~0.200 Brier
  KenPom logistic:   ~0.185 Brier
  Vegas line:        ~0.175 Brier
  Our target:        <0.170 Brier
"""

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss


def brier_score(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    """Brier score (MSE of probabilities). Lower is better."""
    return brier_score_loss(y_true, y_prob)


def calibration_table(y_true: np.ndarray, y_prob: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Reliability diagram data: predicted prob vs observed frequency."""
    bins = np.linspace(0, 1, n_bins + 1)
    records = []
    for i in range(n_bins):
        mask = (y_prob >= bins[i]) & (y_prob < bins[i + 1])
        if mask.sum() > 0:
            records.append({
                "bin_center": (bins[i] + bins[i + 1]) / 2,
                "predicted_avg": y_prob[mask].mean(),
                "observed_freq": y_true[mask].mean(),
                "count": mask.sum(),
            })
    return pd.DataFrame(records)


def leave_one_tournament_out(
    build_features_fn,
    train_model_fn,
    seasons: list[int],
) -> pd.DataFrame:
    """
    LOTO backtest: for each season, train on all other seasons, predict tournament.
    Returns DataFrame with columns: season, brier_score, n_games.
    """
    results = []
    for holdout in seasons:
        train_seasons = [s for s in seasons if s != holdout]
        X_train, y_train = build_features_fn(train_seasons)
        X_test, y_test = build_features_fn([holdout])

        model = train_model_fn()
        model.fit(X_train, y_train)
        y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else model.predict(X_test)

        bs = brier_score(y_test, y_prob)
        results.append({"season": holdout, "brier_score": bs, "n_games": len(y_test)})
        print(f"Season {holdout}: Brier = {bs:.4f} ({len(y_test)} games)")

    return pd.DataFrame(results)


def compare_models(model_results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Compare multiple models across backtest seasons."""
    summary = []
    for name, df in model_results.items():
        summary.append({
            "model": name,
            "mean_brier": df["brier_score"].mean(),
            "std_brier": df["brier_score"].std(),
            "best_season": df.loc[df["brier_score"].idxmin(), "season"],
            "worst_season": df.loc[df["brier_score"].idxmax(), "season"],
        })
    return pd.DataFrame(summary).sort_values("mean_brier")
