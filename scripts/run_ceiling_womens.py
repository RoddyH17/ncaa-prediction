"""
Validate the information ceiling + calibration artifact on women's tournament.

Replicates the men's analysis (run_info_ceiling.py + run_calibration_shift.py)
on women's data. Two questions:

  Q1. Does the same Bayes-risk ceiling pattern hold? (3-way convergence of
      MINE, KMeans, flexible-model estimators around the linear-logistic
      LOTO Brier)
  Q2. Does the in-sample-vs-LOSO calibration artifact replicate?
      (in-sample isotonic appears to improve, honest LOSO does not)

If both replicate on women's data, the findings are not artifacts of the
men's-specific feature set or season composition.

Outputs:
  output/ceiling_womens_estimates.csv
  output/ceiling_womens_calibration.csv
"""

import sys
sys.path.insert(0, ".")

import math
import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss
from sklearn.isotonic import IsotonicRegression

from scripts.build_womens_model import (
    load_womens_data, build_womens_features, WomensLogistic, WOMENS_FEATURES,
)
from src.info_ceiling import (
    estimate_mi_mine, estimate_mi_kmeans, estimate_bs_flexible,
    binary_entropy, brier_lower_bound_from_cond_entropy,
)


def main():
    print("Loading women's data + features (this may use cached Barttorvik)...")
    data = load_womens_data()
    seasons = [s for s in range(2014, 2026) if s != 2020]

    X_all, y_all = build_womens_features(data, seasons)
    feat_cols = [c for c in WOMENS_FEATURES if c in X_all.columns]
    X = X_all[feat_cols].apply(pd.to_numeric, errors="coerce").values
    y = y_all.astype(int)
    season_arr = X_all["Season"].values
    print(f"  N games: {len(X)}, features: {X.shape[1]}, P(Y=1)={y.mean():.3f}")

    H_Y = float(binary_entropy(y.mean()))
    print(f"  H(Y) = {H_Y:.4f} nats")

    rows = []

    # ============================================================
    # (1) Information ceiling estimators
    # ============================================================
    print(f"\n{'='*70}\n  Women's: Information ceiling estimators\n{'='*70}")

    # MINE
    print("  MINE (5 seeds)...")
    mine_runs = []
    for seed in [0, 1, 2, 3, 4]:
        out = estimate_mi_mine(X, y, n_epochs=2000, batch_size=256, lr=5e-4,
                               hidden=64, seed=seed)
        print(f"    seed={seed}: I={out['mi_nats']:.4f} nats")
        mine_runs.append(out)
    mi_mean = np.mean([r["mi_nats"] for r in mine_runs])
    mi_std = np.std([r["mi_nats"] for r in mine_runs])
    H_yx_mine = max(0.0, H_Y - mi_mean)
    bs_up_mine, bs_lo_mine = brier_lower_bound_from_cond_entropy(H_yx_mine)
    print(f"  MINE: I={mi_mean:.4f}+/-{mi_std:.4f}, H(Y|X)={H_yx_mine:.4f}, "
          f"BS_lower={bs_lo_mine:.4f}")
    rows.append({"method": "MINE", "value": H_yx_mine, "kind": "H(Y|X)"})
    rows.append({"method": "MINE_BS_lower", "value": bs_lo_mine, "kind": "BS"})

    # KMeans discrete
    for K in [20, 50, 80, 120]:
        out = estimate_mi_kmeans(X, y, n_clusters=K)
        print(f"  KMeans K={K}: I={out['mi_nats']:.4f}, BS_est={out['BS_Bayes_est']:.4f}")
        rows.append({"method": f"KMeans_K={K}", "value": out["BS_Bayes_est"], "kind": "BS"})

    # Flexible models (LOTO + IID)
    out_loto = estimate_bs_flexible(X, y, seasons=season_arr, seed=42)
    for name, res in out_loto["models"].items():
        print(f"  {name} LOTO Brier: {res['brier_mean']:.4f}")
        rows.append({"method": f"{name}_LOTO", "value": res["brier_mean"], "kind": "BS"})

    # ============================================================
    # (2) Linear logistic LOTO baseline
    # ============================================================
    print(f"\n{'='*70}\n  Women's: WomensLogistic LOTO baseline\n{'='*70}")
    p_oof = np.zeros(len(X_all))
    fold_briers = []
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        m = WomensLogistic(C=0.5)
        m.fit(X_all.loc[tr], y[tr])
        p_oof[te] = m.predict_proba(X_all.loc[te])[:, 1]
        fold_briers.append(brier_score_loss(y[te], p_oof[te]))
    bs_logit = brier_score_loss(y, p_oof)
    print(f"  WomensLogistic LOTO Brier = {bs_logit:.4f}")
    rows.append({"method": "WomensLogistic_LOTO", "value": bs_logit, "kind": "BS"})

    df = pd.DataFrame(rows)
    df.to_csv("output/ceiling_womens_estimates.csv", index=False)

    # ============================================================
    # (3) Calibration artifact: in-sample vs honest LOSO
    # ============================================================
    print(f"\n{'='*70}\n  Women's: Calibration artifact (in-sample vs LOSO)\n{'='*70}")

    iso_in = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995)
    iso_in.fit(p_oof, y)
    bs_in = brier_score_loss(y, iso_in.predict(p_oof))

    p_iso_loso = np.zeros_like(p_oof)
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.005, y_max=0.995)
        iso.fit(p_oof[tr], y[tr])
        p_iso_loso[te] = iso.predict(p_oof[te])
    bs_loso = brier_score_loss(y, p_iso_loso)

    print(f"  Uncalibrated LOTO Brier:        {bs_logit:.4f}")
    print(f"  In-sample isotonic (overfit):   {bs_in:.4f}  (delta {bs_in - bs_logit:+.4f})")
    print(f"  Honest LOSO isotonic:           {bs_loso:.4f}  (delta {bs_loso - bs_logit:+.4f})")
    print(f"  Artifact gap = in_sample - LOSO = {bs_in - bs_loso:+.4f}")

    cal_rows = pd.DataFrame([
        {"setting": "uncalibrated", "brier": bs_logit},
        {"setting": "in_sample_isotonic", "brier": bs_in},
        {"setting": "loso_isotonic", "brier": bs_loso},
        {"setting": "artifact_gap", "brier": bs_in - bs_loso},
    ])
    cal_rows.to_csv("output/ceiling_womens_calibration.csv", index=False)

    print(f"\n  Saved output/ceiling_womens_estimates.csv")
    print(f"  Saved output/ceiling_womens_calibration.csv")


if __name__ == "__main__":
    main()
