"""
Bayesian logistic regression with PyMC for NCAA tournament prediction.

Provides full posterior distributions over:
  - Feature coefficients (uncertainty about feature importance)
  - Per-game win probability (credible intervals)

Uses NUTS sampler. Compares to point-estimate frequentist logistic.

Trading application: only place trades when 95% credible interval excludes
the market price. This is more robust than thresholding mean predictions.
"""

import sys
sys.path.insert(0, ".")

import os
os.environ.setdefault("PYTENSOR_FLAGS", "cxx=")  # disable g++ warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
import pymc as pm

from src.data_collection import load_all_mens_data
from src.pipeline import make_build_features_fn

plt.style.use("seaborn-v0_8-whitegrid")

# Same focused feature set as time-varying analysis
FEATURES = [
    "seed_diff", "rank_diff_POM", "bart_net_diff", "bart_adjoe_diff",
    "bart_adjde_diff", "efg_pct_diff", "to_pct_diff", "or_pct_diff",
    "ft_rate_diff", "momentum_margin_diff",
]


def fit_bayesian_logistic(X_train, y_train, X_test, n_samples=2000):
    """Fit Bayesian logistic regression via ADVI (variational inference) for speed.

    Falls back from NUTS because PyTensor C-compilation is unavailable.
    ADVI gives mean-field Gaussian posterior — fast and adequate for this scale.
    """
    n_features = X_train.shape[1]

    with pm.Model() as model:
        beta = pm.Normal("beta", mu=0, sigma=1.0, shape=n_features)
        alpha = pm.Normal("alpha", mu=0, sigma=2.0)
        logits_train = alpha + pm.math.dot(X_train, beta)
        pm.Bernoulli("y_obs", logit_p=logits_train, observed=y_train)

        # ADVI: variational inference, fast even without g++
        approx = pm.fit(n=20000, method="advi", progressbar=False)
        trace = approx.sample(n_samples)

    posterior = trace.posterior
    beta_samples = posterior["beta"].values.reshape(-1, n_features)
    alpha_samples = posterior["alpha"].values.flatten()

    # X_test: (n_test, d), beta_samples: (n_samples, d) -> X_test @ beta.T: (n_test, n_samples)
    # alpha_samples: (n_samples,) -> add via broadcasting on last axis
    logits_test = alpha_samples[None, :] + X_test @ beta_samples.T
    p_test = 1 / (1 + np.exp(-logits_test))

    return p_test, beta_samples, alpha_samples


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    seasons = [s for s in range(2014, 2026) if s != 2020]

    # Build full matrix
    X_all, y_all = build_fn(seasons)
    feat_cols = [c for c in FEATURES if c in X_all.columns]

    # Train on 2014-2024, test on 2025
    train_seasons = [s for s in seasons if s != 2025]
    X_train_df = X_all[X_all["Season"].isin(train_seasons)][feat_cols]
    y_train = y_all[X_all["Season"].isin(train_seasons).values]
    X_test_df = X_all[X_all["Season"] == 2025][feat_cols]
    y_test = y_all[(X_all["Season"] == 2025).values]

    # Impute & scale
    imp = SimpleImputer(strategy="median")
    scl = StandardScaler()
    X_train_proc = scl.fit_transform(imp.fit_transform(X_train_df.apply(pd.to_numeric, errors="coerce")))
    X_test_proc = scl.transform(imp.transform(X_test_df.apply(pd.to_numeric, errors="coerce")))

    print(f"Train: {X_train_proc.shape}, Test: {X_test_proc.shape}")
    print("\nFitting Bayesian logistic regression with NUTS sampler...")

    p_samples, beta_samples, alpha_samples = fit_bayesian_logistic(
        X_train_proc, y_train.astype(float), X_test_proc,
        n_samples=2000,
    )
    print(f"Posterior samples shape: {p_samples.shape}")

    # Posterior summary
    p_mean = p_samples.mean(axis=1)
    p_low = np.quantile(p_samples, 0.025, axis=1)
    p_high = np.quantile(p_samples, 0.975, axis=1)
    p_std = p_samples.std(axis=1)

    bs_bayesian = brier_score_loss(y_test, p_mean)

    # Compare to frequentist
    lr = LogisticRegression(C=0.5, max_iter=2000)
    lr.fit(X_train_proc, y_train)
    p_freq = lr.predict_proba(X_test_proc)[:, 1]
    bs_freq = brier_score_loss(y_test, p_freq)

    print(f"\n=== TEST SET (2025) PERFORMANCE ===")
    print(f"Bayesian Brier:    {bs_bayesian:.4f}")
    print(f"Frequentist Brier: {bs_freq:.4f}")

    # Interval coverage
    cover_95 = np.mean((y_test >= p_low) & (y_test <= p_high))
    cover_50 = np.mean((y_test >= np.quantile(p_samples, 0.25, axis=1)) &
                        (y_test <= np.quantile(p_samples, 0.75, axis=1)))
    print(f"95% credible interval coverage: {cover_95:.1%}")
    print(f"50% credible interval coverage: {cover_50:.1%}")
    print(f"Mean 95% interval width: {np.mean(p_high - p_low):.3f}")

    # Coefficient posteriors
    print("\n=== POSTERIOR COEFFICIENTS (mean ± 95% CI) ===")
    coef_summary = []
    for i, name in enumerate(feat_cols):
        m = beta_samples[:, i].mean()
        lo = np.quantile(beta_samples[:, i], 0.025)
        hi = np.quantile(beta_samples[:, i], 0.975)
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"  {name:<22s} {m:+.3f}  [{lo:+.3f}, {hi:+.3f}] {sig}")
        coef_summary.append({"feature": name, "mean": m, "ci_low": lo, "ci_high": hi})

    pd.DataFrame(coef_summary).to_csv("output/bayesian_coefs.csv", index=False)

    # Save predictions with intervals
    pd.DataFrame({
        "p_mean": p_mean, "p_std": p_std,
        "p_low_95": p_low, "p_high_95": p_high,
        "y_true": y_test,
    }).to_csv("output/bayesian_predictions.csv", index=False)

    # Plot 1: posterior coefficients
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax = axes[0]
    means = [r["mean"] for r in coef_summary]
    los = [r["ci_low"] for r in coef_summary]
    his = [r["ci_high"] for r in coef_summary]
    y_pos = np.arange(len(feat_cols))
    ax.errorbar(means, y_pos, xerr=[np.array(means) - np.array(los),
                                     np.array(his) - np.array(means)],
                fmt="o", color="#2563eb", capsize=4)
    ax.axvline(0, color="black", linewidth=0.5)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(feat_cols)
    ax.set_xlabel("Posterior coefficient (mean ± 95% CI)")
    ax.set_title("Bayesian Logistic: Coefficient Posteriors")

    # Plot 2: Prediction intervals on test set
    ax = axes[1]
    sort_idx = np.argsort(p_mean)
    x = np.arange(len(p_mean))
    ax.fill_between(x, p_low[sort_idx], p_high[sort_idx], alpha=0.3,
                     color="#2563eb", label="95% CI")
    ax.plot(x, p_mean[sort_idx], color="#2563eb", linewidth=1.5, label="Posterior mean")
    actual_x = np.where(y_test[sort_idx] == 1)[0]
    ax.scatter(actual_x, np.ones(len(actual_x)) * 1.02, color="#16a34a", s=10,
               label="Actual=1", zorder=5)
    actual_x_0 = np.where(y_test[sort_idx] == 0)[0]
    ax.scatter(actual_x_0, np.ones(len(actual_x_0)) * -0.02, color="#dc2626", s=10,
               label="Actual=0", zorder=5)
    ax.set_xlabel("Game (sorted by predicted P(TeamA wins))")
    ax.set_ylabel("Predicted probability")
    ax.set_title(f"2025 Posterior Predictions (95% CI coverage: {cover_95:.0%})")
    ax.legend(loc="lower right")
    ax.set_ylim(-0.05, 1.07)

    plt.tight_layout()
    plt.savefig("output/bayesian_analysis.png", dpi=150, bbox_inches="tight")
    print("\nSaved bayesian_analysis.png and CSVs")


if __name__ == "__main__":
    main()
