"""
Information-theoretic ceiling experiment for NCAA tournament prediction.

For our feature set X (Barttorvik + Four Factors + ratings + momentum + seed),
estimate the irreducible Bayes risk BS* via three independent methods:

  (1) MINE: I(X; Y) → H(Y|X) → upper bound on BS*
  (2) KMeans-discrete: empirical I(X; Y) and Σ p_k p_k(1-p_k) directly
  (3) Flexible-model CV: RF, GBM, kNN cross-validated Brier

Convergence of these three estimates around the empirical LOTO Brier of our
linear models (~0.189) supports the ceiling claim: NO model on this feature
set can do meaningfully better.

Outputs:
  output/info_ceiling_estimates.csv  - all three estimators + LOTO baselines
  output/info_ceiling_models.csv     - flexible-model per-method Brier
"""

import sys
sys.path.insert(0, ".")

import math
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data
from src.pipeline import make_build_features_fn
from src.models import MultiFeatureLogistic
from src.info_ceiling import (
    estimate_mi_mine, estimate_mi_kmeans, estimate_bs_flexible,
    binary_entropy, brier_lower_bound_from_cond_entropy,
)


FEATURES = [
    "seed_diff", "rank_diff_POM",
    "bart_net_diff", "bart_adjoe_diff", "bart_adjde_diff", "bart_barthag_diff",
    "off_eff_diff", "def_eff_diff", "net_eff_diff", "tempo_diff",
    "efg_pct_diff", "to_pct_diff", "or_pct_diff", "ft_rate_diff",
    "opp_efg_pct_diff", "opp_to_pct_diff", "opp_or_pct_diff", "opp_ft_rate_diff",
    "momentum_margin_diff", "momentum_winpct_diff",
]


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    seasons = [s for s in range(2014, 2026) if s != 2020]
    X_all, y_all = build_fn(seasons)
    feat_cols = [c for c in FEATURES if c in X_all.columns]

    X = X_all[feat_cols].apply(pd.to_numeric, errors="coerce").values
    y = y_all.astype(int)
    season_arr = X_all["Season"].values
    print(f"  N games: {len(X)}, features: {X.shape[1]}, P(Y=1)={y.mean():.3f}")

    H_Y = float(binary_entropy(y.mean()))
    print(f"  H(Y) = {H_Y:.4f} nats ({H_Y/math.log(2):.4f} bits)")

    rows = []

    # ============================================================
    # (1) MINE neural estimator
    # ============================================================
    print(f"\n{'='*70}\n  (1) MINE neural estimator\n{'='*70}")
    mine_runs = []
    for seed in [0, 1, 2, 3, 4]:
        out = estimate_mi_mine(
            X, y, n_epochs=2000, batch_size=256, lr=5e-4,
            hidden=64, ema_decay=0.99, seed=seed, verbose=False,
        )
        print(f"  seed={seed}: I(X;Y) = {out['mi_nats']:.4f} nats  "
              f"H(Y|X) = {out['H_Y_given_X']:.4f}")
        mine_runs.append(out)
    mi_mean = np.mean([r["mi_nats"] for r in mine_runs])
    mi_std = np.std([r["mi_nats"] for r in mine_runs])
    H_yx_mine = max(0.0, H_Y - mi_mean)
    bs_upper_mine, bs_lower_mine = brier_lower_bound_from_cond_entropy(H_yx_mine)
    print(f"\n  MINE:   I(X;Y) = {mi_mean:.4f} +/- {mi_std:.4f} nats")
    print(f"          H(Y|X) = {H_yx_mine:.4f} nats")
    print(f"          Brier upper bound (loose):    {bs_upper_mine:.4f}")
    print(f"          Brier lower bound (Fano-like): {bs_lower_mine:.4f}")
    rows.append({
        "method": "MINE",
        "mi_nats": mi_mean,
        "mi_std": mi_std,
        "H_Y_given_X": H_yx_mine,
        "BS_upper_bound": bs_upper_mine,
        "BS_lower_bound": bs_lower_mine,
        "BS_Bayes_estimate": np.nan,
    })

    # ============================================================
    # (2) KMeans discrete estimator
    # ============================================================
    print(f"\n{'='*70}\n  (2) KMeans discrete estimator\n{'='*70}")
    for K in [20, 50, 80, 120]:
        out = estimate_mi_kmeans(X, y, n_clusters=K, seed=42)
        bs_up, bs_lo = brier_lower_bound_from_cond_entropy(out["H_Y_given_X"])
        print(f"  K={K:3d}: I={out['mi_nats']:.4f}  H(Y|X)={out['H_Y_given_X']:.4f}  "
              f"BS*~={out['BS_Bayes_est']:.4f}")
        rows.append({
            "method": f"KMeans_K={K}",
            "mi_nats": out["mi_nats"],
            "mi_std": np.nan,
            "H_Y_given_X": out["H_Y_given_X"],
            "BS_upper_bound": bs_up,
            "BS_lower_bound": bs_lo,
            "BS_Bayes_estimate": out["BS_Bayes_est"],
        })

    # ============================================================
    # (3) Flexible-model estimator
    # ============================================================
    print(f"\n{'='*70}\n  (3) Flexible-model CV estimator\n{'='*70}")

    print("  -- 5-fold random CV (IID assumption) --")
    out_iid = estimate_bs_flexible(X, y, seasons=None, n_splits=5, seed=42)
    for name, res in out_iid["models"].items():
        print(f"    {name:<22s} Brier = {res['brier_mean']:.4f} +/- {res['brier_std']:.4f}")
        rows.append({
            "method": f"flexible_{name}_5foldIID",
            "mi_nats": np.nan, "mi_std": np.nan,
            "H_Y_given_X": np.nan,
            "BS_upper_bound": np.nan, "BS_lower_bound": np.nan,
            "BS_Bayes_estimate": res["brier_mean"],
        })
    print(f"    Floor estimate (IID):  {out_iid['bs_floor_estimate']:.4f}")

    print("\n  -- LOTO (year-shift) CV --")
    out_loto = estimate_bs_flexible(X, y, seasons=season_arr, seed=42)
    for name, res in out_loto["models"].items():
        print(f"    {name:<22s} Brier = {res['brier_mean']:.4f} +/- {res['brier_std']:.4f}")
        rows.append({
            "method": f"flexible_{name}_LOTO",
            "mi_nats": np.nan, "mi_std": np.nan,
            "H_Y_given_X": np.nan,
            "BS_upper_bound": np.nan, "BS_lower_bound": np.nan,
            "BS_Bayes_estimate": res["brier_mean"],
        })
    print(f"    Floor estimate (LOTO): {out_loto['bs_floor_estimate']:.4f}")

    # ============================================================
    # (4) Linear baseline LOTO Brier (Multi-Feature Logistic)
    # ============================================================
    print(f"\n{'='*70}\n  (4) Multi-Feature Logistic LOTO baseline\n{'='*70}")
    fold_brier = []
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        m = MultiFeatureLogistic(C=0.5)
        m.fit(X_all.loc[tr], y[tr])
        p = m.predict_proba(X_all.loc[te])[:, 1]
        fold_brier.append(brier_score_loss(y[te], p))
    bs_logit = float(np.mean(fold_brier))
    print(f"  Multi-Feature Logistic LOTO Brier = {bs_logit:.4f}")
    rows.append({
        "method": "MultiFeatureLogistic_LOTO",
        "mi_nats": np.nan, "mi_std": np.nan,
        "H_Y_given_X": np.nan,
        "BS_upper_bound": np.nan, "BS_lower_bound": np.nan,
        "BS_Bayes_estimate": bs_logit,
    })

    # ============================================================
    # Convergence summary
    # ============================================================
    print(f"\n{'='*70}\n  CONVERGENCE SUMMARY\n{'='*70}")
    df = pd.DataFrame(rows)
    df.to_csv("output/info_ceiling_estimates.csv", index=False)

    # Pull together Bayes-Brier estimates from each method
    converge = {
        "MINE upper bound (H(Y|X)/2)":       bs_upper_mine,
        "KMeans K=80 BS_Bayes_est":           df[df["method"] == "KMeans_K=80"]["BS_Bayes_estimate"].values[0],
        "Flexible RF (IID 5-fold)":           df[df["method"] == "flexible_RandomForest_500_5foldIID"]["BS_Bayes_estimate"].values[0],
        "Flexible GBM (IID 5-fold)":          df[df["method"] == "flexible_GBM_300_5foldIID"]["BS_Bayes_estimate"].values[0],
        "Flexible kNN (IID 5-fold)":          df[df["method"] == "flexible_kNN_30_5foldIID"]["BS_Bayes_estimate"].values[0],
        "Flexible RF (LOTO)":                 df[df["method"] == "flexible_RandomForest_500_LOTO"]["BS_Bayes_estimate"].values[0],
        "Flexible GBM (LOTO)":                df[df["method"] == "flexible_GBM_300_LOTO"]["BS_Bayes_estimate"].values[0],
        "Flexible kNN (LOTO)":                df[df["method"] == "flexible_kNN_30_LOTO"]["BS_Bayes_estimate"].values[0],
        "Multi-Feature Logistic (LOTO)":     bs_logit,
    }
    print()
    for name, val in converge.items():
        print(f"  {name:<45s} {val:.4f}")

    bs_estimates = [v for k, v in converge.items() if "MINE" not in k]
    print(f"\n  Range across non-MINE estimates: "
          f"[{min(bs_estimates):.4f}, {max(bs_estimates):.4f}]")
    print(f"  Mean: {np.mean(bs_estimates):.4f}, std: {np.std(bs_estimates):.4f}")

    print(f"\nSaved output/info_ceiling_estimates.csv")


if __name__ == "__main__":
    main()
