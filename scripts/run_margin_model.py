"""
Margin Distribution Modeling for NCAA Tournament Prediction.

Instead of binary outcomes y ∈ {0, 1}, predict the score margin distribution
m = score_a - score_b ~ N(mu, sigma^2). Then P(team A wins) = P(m > 0) = Φ(mu/sigma).

This carries more information than binary modeling because the magnitude of
victory is informative (a 30-point win is stronger evidence than a 1-point win).

Predicted quantities:
  - Mean margin (mu): regression on features
  - Margin std (sigma): empirical residual std + per-game adjustments
  - Implied win probability: Φ(mu / sigma)
  - Margin RMSE: novel metric for sports prediction quality

Model: Gaussian regression on margin (Skellam-like with continuous approximation).
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import Ridge
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss, mean_squared_error

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import make_build_features_fn, _parse_seed_num

plt.style.use("seaborn-v0_8-whitegrid")

# Same feature set as Multi-Feature Logistic
FEATURES = [
    "seed_diff", "rank_diff_POM",
    "bart_net_diff", "bart_adjoe_diff", "bart_adjde_diff", "bart_barthag_diff",
    "off_eff_diff", "def_eff_diff", "net_eff_diff", "tempo_diff",
    "efg_pct_diff", "to_pct_diff", "or_pct_diff", "ft_rate_diff",
    "opp_efg_pct_diff", "opp_to_pct_diff", "opp_or_pct_diff", "opp_ft_rate_diff",
    "momentum_margin_diff", "momentum_winpct_diff",
]


def get_actual_margins(data: dict, X: pd.DataFrame) -> np.ndarray:
    """For each row in X (with TeamA, TeamB, Season), look up actual margin
    from tourney_compact (= score_A - score_B in canonical ordering).
    """
    tourney = data["tourney_compact"]
    actual_2026 = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")

    # Combine
    combined = pd.concat([
        tourney[["Season", "WTeamID", "WScore", "LTeamID", "LScore"]],
        actual_2026[["Season", "WTeamID", "WScore", "LTeamID", "LScore"]],
    ], ignore_index=True)

    # Build lookup: (Season, min(team), max(team)) -> margin (in canonical order)
    margins = {}
    for _, g in combined.iterrows():
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        s = int(g["Season"])
        wsc, lsc = int(g["WScore"]), int(g["LScore"])
        if w < l:
            margins[(s, w, l)] = wsc - lsc  # team A (winner) won
        else:
            margins[(s, l, w)] = -(wsc - lsc)  # team A (loser) lost, negative margin

    actual_margins = []
    for _, row in X.iterrows():
        s, ta, tb = int(row["Season"]), int(row["TeamA"]), int(row["TeamB"])
        m = margins.get((s, ta, tb), np.nan)
        actual_margins.append(m)
    return np.array(actual_margins)


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    seasons = [s for s in range(2014, 2026) if s != 2020]

    X_all, y_all = build_fn(seasons)
    feat_cols = [c for c in FEATURES if c in X_all.columns]

    # Get actual margins
    margins = get_actual_margins(data, X_all)
    print(f"Loaded {len(margins)} games, {(~np.isnan(margins)).sum()} with margins")

    # === LOTO evaluation: margin model vs binary logistic ===
    print(f"\n{'='*70}\n  LOTO: MARGIN GAUSSIAN MODEL vs BINARY LOGISTIC\n{'='*70}")

    margin_results = []
    for holdout in seasons:
        train_mask = (X_all["Season"] != holdout).values
        test_mask = (X_all["Season"] == holdout).values
        if test_mask.sum() == 0: continue

        # Skip games with NaN margin (e.g., if 2026 not in compact)
        valid_train = ~np.isnan(margins) & train_mask
        valid_test = ~np.isnan(margins) & test_mask

        X_train_n = X_all.loc[valid_train, feat_cols].apply(pd.to_numeric, errors="coerce").values
        m_train = margins[valid_train]
        y_train = y_all[valid_train]
        X_test_n = X_all.loc[valid_test, feat_cols].apply(pd.to_numeric, errors="coerce").values
        m_test = margins[valid_test]
        y_test = y_all[valid_test]

        # Pipeline
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("reg", Ridge(alpha=1.0)),
        ])
        pipe.fit(X_train_n, m_train)
        mu_test = pipe.predict(X_test_n)

        # Estimate sigma from training residuals
        mu_train = pipe.predict(X_train_n)
        residuals = m_train - mu_train
        sigma = float(residuals.std())

        # Convert to win probability via normal CDF
        p_win = stats.norm.cdf(mu_test / sigma)
        p_win = np.clip(p_win, 0.01, 0.99)

        # Metrics
        bs = brier_score_loss(y_test, p_win)
        rmse_margin = np.sqrt(mean_squared_error(m_test, mu_test))
        margin_results.append({
            "season": holdout,
            "n_games": int(valid_test.sum()),
            "brier_margin_model": bs,
            "rmse_margin": rmse_margin,
            "sigma_residual": sigma,
        })
        print(f"  {holdout}: Brier={bs:.4f}  margin_RMSE={rmse_margin:.2f}  sigma={sigma:.2f}")

    df = pd.DataFrame(margin_results)
    print(f"\n  Mean Brier (margin Gaussian): {df['brier_margin_model'].mean():.4f}")
    print(f"  Mean margin RMSE:             {df['rmse_margin'].mean():.2f} points")
    print(f"  Mean residual sigma:          {df['sigma_residual'].mean():.2f}")
    print(f"  vs LOTO Multi-Feature Brier:  ~0.189")
    df.to_csv("output/margin_model_loto.csv", index=False)

    # === Calibration check on 2025 actual (most recent year in pipeline) ===
    print(f"\n{'='*70}\n  2025 ACTUAL: MARGIN MODEL\n{'='*70}")

    train_seasons = [s for s in seasons if s != 2025]
    valid_train = ~np.isnan(margins) & X_all["Season"].isin(train_seasons).values
    valid_test = ~np.isnan(margins) & (X_all["Season"] == 2025).values

    X_train_n = X_all.loc[valid_train, feat_cols].apply(pd.to_numeric, errors="coerce").values
    m_train = margins[valid_train]
    X_test_n = X_all.loc[valid_test, feat_cols].apply(pd.to_numeric, errors="coerce").values
    m_test = margins[valid_test]
    y_test = y_all[valid_test]

    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])
    pipe.fit(X_train_n, m_train)
    mu_test = pipe.predict(X_test_n)
    sigma = float((m_train - pipe.predict(X_train_n)).std())
    p_win = np.clip(stats.norm.cdf(mu_test / sigma), 0.01, 0.99)
    bs = brier_score_loss(y_test, p_win)
    rmse_margin = np.sqrt(mean_squared_error(m_test, mu_test))
    print(f"  2025 Brier (margin model):  {bs:.4f}")
    print(f"  2025 margin RMSE:           {rmse_margin:.2f}")

    # Plot: predicted vs actual margin
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ax = axes[0]
    ax.scatter(mu_test, m_test, alpha=0.5)
    lim = max(abs(m_test).max(), abs(mu_test).max())
    ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.4)
    ax.axvline(0, color="gray", alpha=0.3)
    ax.axhline(0, color="gray", alpha=0.3)
    ax.set_xlabel("Predicted margin")
    ax.set_ylabel("Actual margin")
    ax.set_title(f"2026 actual: margin RMSE = {rmse_margin:.2f}")

    # Reliability: predicted P(win) vs observed
    ax = axes[1]
    bins = np.linspace(0, 1, 11)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    obs_freq = []
    for i in range(len(bins) - 1):
        mask = (p_win >= bins[i]) & (p_win < bins[i + 1])
        if mask.sum() > 0:
            obs_freq.append(y_test[mask].mean())
        else:
            obs_freq.append(np.nan)
    ax.plot(bin_centers, obs_freq, "o-", color="#2563eb")
    ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
    ax.set_xlabel("Predicted P(team A wins)")
    ax.set_ylabel("Observed frequency")
    ax.set_title("2026 calibration (margin → P(win))")

    plt.tight_layout()
    plt.savefig("output/margin_model_2026.png", dpi=150, bbox_inches="tight")
    print("\nSaved margin_model_2026.png")


if __name__ == "__main__":
    main()
