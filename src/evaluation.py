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


def loto_ensemble(
    build_features_fn,
    model_factories: dict[str, callable],
    seasons: list[int],
) -> pd.DataFrame:
    """
    LOTO backtest with weighted ensemble and per-fold weight optimization.

    For each holdout season:
      1. Train all base models on remaining seasons
      2. Use inner LOTO (leave-one-out within training) to get OOF predictions
      3. Optimize ensemble weights on OOF predictions
      4. Predict holdout with optimized ensemble
    """
    from src.models import WeightedEnsemble

    results = []
    for holdout in seasons:
        train_seasons = [s for s in seasons if s != holdout]

        # Build train/test data
        X_train, y_train = build_features_fn(train_seasons)
        X_test, y_test = build_features_fn([holdout])

        # Train all base models on full training set
        trained_models = {}
        for name, factory in model_factories.items():
            model = factory()
            model.fit(X_train, y_train)
            trained_models[name] = model

        # Collect OOF predictions for weight optimization via inner LOTO
        n_train = len(y_train)
        oof_preds = {name: np.full(n_train, 0.5) for name in model_factories}

        # Inner LOTO: for each inner holdout season, train on rest and predict
        for inner_holdout in train_seasons:
            inner_train = [s for s in train_seasons if s != inner_holdout]
            X_inner_train, y_inner_train = build_features_fn(inner_train)
            X_inner_val, y_inner_val = build_features_fn([inner_holdout])

            # Find indices of inner_holdout games in X_train
            mask = X_train["Season"] == inner_holdout
            if mask.sum() == 0:
                continue

            for name, factory in model_factories.items():
                inner_model = factory()
                inner_model.fit(X_inner_train, y_inner_train)
                preds = inner_model.predict_proba(X_inner_val)[:, 1]
                oof_preds[name][mask.values] = preds

        # Optimize ensemble weights on OOF predictions
        model_list = [trained_models[name] for name in model_factories]
        ensemble = WeightedEnsemble(model_list)

        # Use scipy to optimize on OOF predictions directly
        from scipy.optimize import minimize

        names = list(model_factories.keys())
        oof_matrix = np.column_stack([oof_preds[n] for n in names])

        def objective(w):
            w = np.abs(w)
            w = w / w.sum()
            blended = oof_matrix @ w
            return brier_score_loss(y_train, blended)

        n_models = len(names)
        w0 = np.ones(n_models) / n_models
        opt = minimize(
            objective, w0, method="SLSQP",
            bounds=[(0, 1)] * n_models,
            constraints={"type": "eq", "fun": lambda w: np.sum(w) - 1},
        )
        opt_weights = opt.x / opt.x.sum()
        ensemble.weights = list(opt_weights)

        # Predict holdout
        y_prob = ensemble.predict_proba(X_test)[:, 1]
        bs = brier_score(y_test, y_prob)

        weight_str = ", ".join(f"{n}={w:.2f}" for n, w in zip(names, opt_weights))
        print(f"Season {holdout}: Brier = {bs:.4f} | weights: {weight_str}")
        results.append({"season": holdout, "brier_score": bs, "n_games": len(y_test)})

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
