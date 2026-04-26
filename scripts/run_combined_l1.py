"""
Combined M+W training with L1 regularization for automatic feature selection.

Strategy:
  1. Pool men's and women's tournament games (1445 total games)
  2. Add `is_womens` binary flag as a feature
  3. Set Massey features to 0 for women's rows (no women's Massey data)
  4. Use L1-regularized LR (Lasso-style); smaller C => more features dropped
  5. Sweep C on LOSO; find best Brier
  6. Compare to L2 LR (current baseline) and per-gender LR (LR-only baseline)

L1 advantage: automatic feature selection that doesn't need a manual threshold.
Stable selection: features only kept if their effect is consistent across folds.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import _parse_seed_num
from scripts.build_womens_model import load_womens_data
from scripts.run_top3 import (
    build_combined_features, build_combined_features_2026,
    FEATURE_COLS as TOP3_FEATURES,
)
from src.seed_base_rate import compute_base_rate_table, lookup_p_a_wins


def add_seed_pair(X, is_w, seed_lookup_m, seed_lookup_w,
                  base_per_season_m, base_per_season_w,
                  base_full_m, base_full_w):
    vals = []
    for _, r in X.iterrows():
        season = int(r["Season"])
        ta, tb = int(r["TeamA"]), int(r["TeamB"])
        if int(r["is_womens"]) == 0:
            seed_a = seed_lookup_m.get((season, ta), 17)
            seed_b = seed_lookup_m.get((season, tb), 17)
            tbl = base_per_season_m.get(season, base_full_m)
        else:
            seed_a = seed_lookup_w.get((season, ta), 17)
            seed_b = seed_lookup_w.get((season, tb), 17)
            tbl = base_per_season_w.get(season, base_full_w)
        vals.append(lookup_p_a_wins(tbl, seed_a, seed_b))
    X = X.copy()
    X["seed_pair_winrate"] = vals
    return X


def build_lr_l1(C=0.1):
    """L1-regularized LR. Note: liblinear solver for L1."""
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("lr", LogisticRegression(C=C, penalty="l1", solver="liblinear",
                                   max_iter=3000, random_state=42)),
    ])


def build_lr_l2(C=0.1):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("lr", LogisticRegression(C=C, penalty="l2", solver="lbfgs",
                                   max_iter=2000)),
    ])


def loso_combined(X, y, is_w, season_arr, feats, pipeline_factory):
    """LOSO with single combined model (one model trained on M+W)."""
    p_oof = np.zeros(len(X))
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        if te.sum() == 0: continue
        pipe = pipeline_factory()
        Xtr = X.loc[tr, feats].apply(pd.to_numeric, errors="coerce")
        Xte = X.loc[te, feats].apply(pd.to_numeric, errors="coerce")
        pipe.fit(Xtr, y[tr])
        p_oof[te] = pipe.predict_proba(Xte)[:, 1]
    n_m = (is_w == 0).sum(); n_w_ = (is_w == 1).sum()
    bs_m = brier_score_loss(y[is_w == 0], p_oof[is_w == 0])
    bs_w = brier_score_loss(y[is_w == 1], p_oof[is_w == 1])
    bs_c = (bs_m * n_m + bs_w * n_w_) / (n_m + n_w_)
    return p_oof, bs_m, bs_w, bs_c


def get_l1_active_features(X_train, y_train, feats, C):
    """Fit L1 LR on training data, return list of features with non-zero coefs."""
    pipe = build_lr_l1(C=C)
    Xtr = X_train[feats].apply(pd.to_numeric, errors="coerce")
    pipe.fit(Xtr, y_train)
    coefs = pipe.named_steps["lr"].coef_[0]
    return [f for f, c in zip(feats, coefs) if abs(c) > 1e-8]


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]
    print("Loading data...")
    data_m = load_all_mens_data()
    data_w = load_womens_data()

    print("\nBuilding feature matrix (combined M+W)...")
    X, y, is_w = build_combined_features(data_m, data_w, seasons)
    season_arr = X["Season"].values

    # For combined model, set Massey to 0 for women's rows (women's has no Massey data)
    massey_cols = [c for c in TOP3_FEATURES if c.startswith("massey_")]
    X.loc[is_w == 1, massey_cols] = 0.0

    # Seed-pair feature
    seeds_m = data_m["seeds"]; seeds_w = data_w["seeds"]
    seed_lookup_m = {(int(r["Season"]), int(r["TeamID"])): _parse_seed_num(r["Seed"])
                     for _, r in seeds_m.iterrows()}
    seed_lookup_w = {(int(r["Season"]), int(r["TeamID"])): _parse_seed_num(r["Seed"])
                     for _, r in seeds_w.iterrows()}
    base_full_m = compute_base_rate_table(data_m["tourney_compact"], seeds_m)
    base_full_w = compute_base_rate_table(data_w["tourney_compact"], seeds_w)
    base_per_season_m = {s: compute_base_rate_table(data_m["tourney_compact"], seeds_m, exclude_season=s)
                         for s in seasons}
    base_per_season_w = {s: compute_base_rate_table(data_w["tourney_compact"], seeds_w, exclude_season=s)
                         for s in seasons}
    X = add_seed_pair(X, is_w, seed_lookup_m, seed_lookup_w,
                      base_per_season_m, base_per_season_w, base_full_m, base_full_w)

    base_feats = [c for c in TOP3_FEATURES if c in X.columns]
    feats_combined = base_feats + ["seed_pair_winrate", "is_womens"]
    print(f"  Features: {len(feats_combined)}  (incl. seed_pair_winrate + is_womens)")

    # ===========================================================
    # Step 1: L2 baseline (combined training, all features)
    # ===========================================================
    print(f"\n{'='*70}\n  Step 1: L2 LR baseline (combined M+W training)\n{'='*70}")
    rows = []
    best_l2 = (None, np.inf)
    for C in [0.05, 0.1, 0.3, 0.5, 1.0, 3.0]:
        _, bs_m, bs_w, bs_c = loso_combined(X, y, is_w, season_arr, feats_combined,
                                              lambda C=C: build_lr_l2(C))
        rows.append({"variant": "L2_combined_27", "C": C,
                     "men": bs_m, "women": bs_w, "combined": bs_c, "n_features": len(feats_combined)})
        print(f"  C={C}: men={bs_m:.4f}  women={bs_w:.4f}  combined={bs_c:.4f}")
        if bs_c < best_l2[1]:
            best_l2 = (C, bs_c)

    # ===========================================================
    # Step 2: L1 regularization sweep (combined training)
    # ===========================================================
    print(f"\n{'='*70}\n  Step 2: L1 LR (auto feature selection)\n{'='*70}")
    best_l1 = (None, np.inf)
    for C in [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]:
        _, bs_m, bs_w, bs_c = loso_combined(X, y, is_w, season_arr, feats_combined,
                                              lambda C=C: build_lr_l1(C))
        # Count active features at C on full data
        active = get_l1_active_features(X, y, feats_combined, C)
        rows.append({"variant": "L1_combined", "C": C,
                     "men": bs_m, "women": bs_w, "combined": bs_c, "n_features": len(active)})
        print(f"  C={C:>5.3f}: men={bs_m:.4f}  women={bs_w:.4f}  combined={bs_c:.4f}  "
              f"active features = {len(active)}")
        if bs_c < best_l1[1]:
            best_l1 = (C, bs_c, active)

    print(f"\n  Best L2: C={best_l2[0]}, Brier={best_l2[1]:.4f}")
    print(f"  Best L1: C={best_l1[0]}, Brier={best_l1[1]:.4f}, "
          f"features={len(best_l1[2])}/{len(feats_combined)}")
    print(f"\n  L1 active features at best C:")
    for f in best_l1[2]:
        print(f"    - {f}")
    print(f"\n  L1 dropped:")
    dropped = [f for f in feats_combined if f not in best_l1[2]]
    for f in dropped:
        print(f"    - {f}")

    df = pd.DataFrame(rows).sort_values("combined")
    df.to_csv("output/combined_l1_loso.csv", index=False)
    print(f"\nTop 10:")
    print(df.head(10).to_string(index=False))

    # ===========================================================
    # Step 3: Apply best L2 + best L1 to 2026
    # ===========================================================
    print(f"\n{'='*70}\n  Step 3: Apply to 2026\n{'='*70}")
    X_2026, _, is_w_2026 = build_combined_features_2026(data_m, data_w)
    X_2026.loc[is_w_2026 == 1, massey_cols] = 0.0
    X_2026 = add_seed_pair(X_2026, is_w_2026.astype(int),
                            seed_lookup_m, seed_lookup_w,
                            {2026: base_full_m}, {2026: base_full_w},
                            base_full_m, base_full_w)

    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")

    def br(pair_lk, actual):
        yt, yp = [], []
        for _, g in actual.iterrows():
            w_, l = int(g["WTeamID"]), int(g["LTeamID"])
            key = (w_, l) if w_ < l else (l, w_)
            yt.append(1 if w_ < l else 0)
            yp.append(pair_lk.get(key, 0.5))
        return brier_score_loss(yt, yp), len(yt)

    def evaluate(label, pipe, feats):
        pipe.fit(X[feats].apply(pd.to_numeric, errors="coerce"), y)
        p_2026 = pipe.predict_proba(X_2026[feats].apply(pd.to_numeric, errors="coerce"))[:, 1]
        p_2026 = np.clip(p_2026, 0.005, 0.995)
        pair_lk = {(int(r["TeamA"]), int(r["TeamB"])): float(p_2026[i])
                   for i, r in X_2026.reset_index(drop=True).iterrows()}
        bs_m_, n_m_ = br(pair_lk, actual_m)
        bs_w_, n_w_ = br(pair_lk, actual_w)
        bs_c_ = (bs_m_ * n_m_ + bs_w_ * n_w_) / (n_m_ + n_w_)
        print(f"  {label}: men={bs_m_:.4f}  women={bs_w_:.4f}  combined={bs_c_:.4f}  ({len(feats)} features)")
        return pair_lk, bs_m_, bs_w_, bs_c_, p_2026

    # Best L2 (all features)
    pipe_l2 = build_lr_l2(C=best_l2[0])
    lk_l2, _, _, bs_c_l2, p_l2 = evaluate(
        f"L2 C={best_l2[0]} all 27 features", pipe_l2, feats_combined
    )

    # Best L1 (active features only)
    pipe_l1 = build_lr_l1(C=best_l1[0])
    lk_l1, _, _, bs_c_l1, p_l1 = evaluate(
        f"L1 C={best_l1[0]} ({len(best_l1[2])} features)", pipe_l1, feats_combined
    )

    # Best L2 retrained on L1's selected features (compactness check)
    pipe_l2_sparse = build_lr_l2(C=0.1)
    lk_l2_sparse, _, _, bs_c_l2_sparse, _ = evaluate(
        f"L2 C=0.1 on L1's {len(best_l1[2])} features", pipe_l2_sparse, best_l1[2]
    )

    # Save submission with the best (best by 2026? No—by LOSO!)
    # Pick by LOSO: best_l2[1] vs best_l1[1]
    if best_l2[1] <= best_l1[1]:
        winner_pair_lk = lk_l2
        winner_label = f"L2_C={best_l2[0]}_all_27"
        winner_bs = bs_c_l2
    else:
        winner_pair_lk = lk_l1
        winner_label = f"L1_C={best_l1[0]}_{len(best_l1[2])}"
        winner_bs = bs_c_l1

    print(f"\n{'='*70}")
    print(f"  FINAL CHOICE (by LOSO): {winner_label}")
    print(f"  2026 Combined Brier: {winner_bs:.4f}")
    print(f"{'='*70}")

    # Save submission
    sub = pd.read_csv("output/submission_stage2.csv")
    sub[["s_str", "ta_str", "tb_str"]] = sub["ID"].str.split("_", expand=True)
    sub["TeamA"] = sub["ta_str"].astype(int); sub["TeamB"] = sub["tb_str"].astype(int)
    sub["Pred"] = sub.apply(
        lambda r: winner_pair_lk.get((r["TeamA"], r["TeamB"]), float(r["Pred"])),
        axis=1
    ).clip(0.005, 0.995)
    sub[["ID", "Pred"]].to_csv("output/submission_stage2_COMBINED_L1.csv", index=False)
    print(f"  Saved output/submission_stage2_COMBINED_L1.csv")


if __name__ == "__main__":
    main()
