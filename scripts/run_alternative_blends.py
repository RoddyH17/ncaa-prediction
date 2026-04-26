"""
Test alternative base learners and stacking on the 8-feature set.

CRITICAL: Use FIXED conservative hyperparameters (no LOSO tuning) to avoid
re-overfitting. Only structural choices are LOSO-evaluated.

Tests:
  (A) Multi-seed XGB averaging (30 seeds) — pure variance reduction
  (B) LightGBM as 3rd base learner with fixed defaults
  (C) Stacking via LOSO-OOF non-negative meta-LR
  (D) Multi-seed bagged LR (10 LRs on bootstrap)

All evaluated on LOSO + 2026 actual.
"""

import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from scipy.optimize import minimize

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import _parse_seed_num
from scripts.build_womens_model import load_womens_data
from scripts.run_top3 import (
    build_combined_features, build_combined_features_2026,
    FEATURE_COLS as TOP3_FEATURES,
)
from src.seed_base_rate import compute_base_rate_table, lookup_p_a_wins


FEATS_8 = [
    "seed_pair_winrate", "bart_net_diff", "harry_diff",
    "elo_diff", "elo_slope_diff", "srs_diff",
    "massey_mean_diff", "tempo_diff",
]


XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
    tree_method="hist",
)

LGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_samples=20,
    reg_alpha=0.1, reg_lambda=1.0, num_leaves=15,
    objective="binary", verbosity=-1,
)

LR_C = 0.1


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


def loso_oof_lr(X, y, season_arr, feats):
    p_oof = np.zeros(len(X))
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        if te.sum() == 0: continue
        Xtr = X.loc[tr, feats].apply(pd.to_numeric, errors="coerce")
        Xte = X.loc[te, feats].apply(pd.to_numeric, errors="coerce")
        med = Xtr.median()
        Xtr = Xtr.fillna(med); Xte = Xte.fillna(med)
        scaler = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=LR_C, max_iter=2000, solver="lbfgs")
        lr.fit(scaler.transform(Xtr), y[tr])
        p_oof[te] = lr.predict_proba(scaler.transform(Xte))[:, 1]
    return p_oof


def loso_oof_xgb(X, y, season_arr, feats, n_seeds=10):
    p_oof = np.zeros(len(X))
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        if te.sum() == 0: continue
        Xtr = X.loc[tr, feats].apply(pd.to_numeric, errors="coerce").fillna(
            X.loc[tr, feats].apply(pd.to_numeric, errors="coerce").median()
        )
        Xte = X.loc[te, feats].apply(pd.to_numeric, errors="coerce").fillna(
            X.loc[tr, feats].apply(pd.to_numeric, errors="coerce").median()
        )
        p_acc = np.zeros(te.sum())
        for seed in range(n_seeds):
            params = dict(XGB_PARAMS); params["random_state"] = seed
            xm = xgb.XGBClassifier(**params)
            xm.fit(Xtr.values, y[tr])
            p_acc += xm.predict_proba(Xte.values)[:, 1]
        p_oof[te] = p_acc / n_seeds
    return p_oof


def loso_oof_lgb(X, y, season_arr, feats, n_seeds=10):
    p_oof = np.zeros(len(X))
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        if te.sum() == 0: continue
        Xtr = X.loc[tr, feats].apply(pd.to_numeric, errors="coerce").fillna(
            X.loc[tr, feats].apply(pd.to_numeric, errors="coerce").median()
        )
        Xte = X.loc[te, feats].apply(pd.to_numeric, errors="coerce").fillna(
            X.loc[tr, feats].apply(pd.to_numeric, errors="coerce").median()
        )
        p_acc = np.zeros(te.sum())
        for seed in range(n_seeds):
            params = dict(LGB_PARAMS); params["random_state"] = seed
            lm = lgb.LGBMClassifier(**params)
            lm.fit(Xtr.values, y[tr])
            p_acc += lm.predict_proba(Xte.values)[:, 1]
        p_oof[te] = p_acc / n_seeds
    return p_oof


def loso_oof_bagged_lr(X, y, season_arr, feats, n_bags=20, sample_frac=0.8):
    """Bagged LR: average n_bags LR models on bootstrap subsamples."""
    p_oof = np.zeros(len(X))
    rng = np.random.default_rng(42)
    for s in np.unique(season_arr):
        tr_idx = np.where(season_arr != s)[0]
        te = season_arr == s
        if te.sum() == 0: continue
        Xte = X.loc[te, feats].apply(pd.to_numeric, errors="coerce")
        med = X.loc[tr_idx, feats].apply(pd.to_numeric, errors="coerce").median()
        Xte = Xte.fillna(med)
        p_acc = np.zeros(te.sum())
        for b in range(n_bags):
            sub = rng.choice(tr_idx, size=int(len(tr_idx) * sample_frac), replace=True)
            Xtr = X.loc[sub, feats].apply(pd.to_numeric, errors="coerce").fillna(med)
            scaler = StandardScaler().fit(Xtr)
            lr = LogisticRegression(C=LR_C, max_iter=2000, solver="lbfgs")
            lr.fit(scaler.transform(Xtr), y[sub])
            p_acc += lr.predict_proba(scaler.transform(Xte))[:, 1]
        p_oof[te] = p_acc / n_bags
    return p_oof


def combined_brier(p_oof, y, is_w):
    n_m = (is_w == 0).sum(); n_w_ = (is_w == 1).sum()
    bs_m = brier_score_loss(y[is_w == 0], p_oof[is_w == 0])
    bs_w = brier_score_loss(y[is_w == 1], p_oof[is_w == 1])
    return bs_m, bs_w, (bs_m * n_m + bs_w * n_w_) / (n_m + n_w_)


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]
    print("Loading data...")
    data_m = load_all_mens_data()
    data_w = load_womens_data()

    print("\nBuilding features...")
    X, y, is_w = build_combined_features(data_m, data_w, seasons)
    season_arr = X["Season"].values
    massey_cols = [c for c in TOP3_FEATURES if c.startswith("massey_")]
    X.loc[is_w == 1, massey_cols] = 0.0

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

    rows = []

    # ===========================================================
    # Compute LOSO OOF for all 4 base learners
    # ===========================================================
    print(f"\n{'='*70}\n  Computing LOSO OOF for base learners\n{'='*70}")

    print("  LR...")
    p_lr = loso_oof_lr(X, y, season_arr, FEATS_8)
    bs_m, bs_w, bs_c = combined_brier(p_lr, y, is_w)
    print(f"  LR alone:        men={bs_m:.4f}  women={bs_w:.4f}  combined={bs_c:.4f}")

    print("  XGB (10 seeds)...")
    p_xgb = loso_oof_xgb(X, y, season_arr, FEATS_8, n_seeds=10)
    bs_m, bs_w, bs_c = combined_brier(p_xgb, y, is_w)
    print(f"  XGB-10:          men={bs_m:.4f}  women={bs_w:.4f}  combined={bs_c:.4f}")

    print("  LightGBM (10 seeds)...")
    p_lgb = loso_oof_lgb(X, y, season_arr, FEATS_8, n_seeds=10)
    bs_m, bs_w, bs_c = combined_brier(p_lgb, y, is_w)
    print(f"  LGB-10:          men={bs_m:.4f}  women={bs_w:.4f}  combined={bs_c:.4f}")

    print("  Bagged LR (20 bags)...")
    p_blr = loso_oof_bagged_lr(X, y, season_arr, FEATS_8, n_bags=20)
    bs_m, bs_w, bs_c = combined_brier(p_blr, y, is_w)
    print(f"  Bagged-LR-20:    men={bs_m:.4f}  women={bs_w:.4f}  combined={bs_c:.4f}")

    # Correlations
    print(f"\n  OOF correlations:")
    P_oof = np.column_stack([p_lr, p_xgb, p_lgb, p_blr])
    corr = np.corrcoef(P_oof.T)
    names = ["LR", "XGB", "LGB", "BaggedLR"]
    print(f"          {'  '.join(names)}")
    for i, n in enumerate(names):
        print(f"  {n:8s} {' '.join(f'{corr[i][j]:.3f}' for j in range(len(names)))}")

    # ===========================================================
    # (A) Multi-seed XGB blend (replace single-seed XGB)
    # ===========================================================
    print(f"\n{'='*70}\n  (A) LR + multi-seed XGB blend\n{'='*70}")
    for w in [0.65, 0.70, 0.75]:
        p = w * p_lr + (1 - w) * p_xgb
        p = np.clip(p, 0.005, 0.995)
        bs_m, bs_w, bs_c = combined_brier(p, y, is_w)
        rows.append({"strategy": f"LR+XGB10_w={w}", "loso": bs_c, "men": bs_m, "women": bs_w})
        print(f"  w_LR={w}: combined={bs_c:.4f}")

    # ===========================================================
    # (B) Add LightGBM as 3rd base
    # ===========================================================
    print(f"\n{'='*70}\n  (B) LR + XGB + LightGBM 3-model blend\n{'='*70}")
    # Search over (w_lr, w_xgb, w_lgb) where they sum to 1
    best_3way = (None, np.inf)
    for w_lr in np.arange(0.4, 0.81, 0.05):
        for w_xgb in np.arange(0.0, 1.0 - w_lr + 0.01, 0.05):
            w_lgb = 1.0 - w_lr - w_xgb
            if w_lgb < -1e-9 or w_lgb > 1 + 1e-9: continue
            p = w_lr * p_lr + w_xgb * p_xgb + w_lgb * p_lgb
            p = np.clip(p, 0.005, 0.995)
            _, _, bs_c = combined_brier(p, y, is_w)
            if bs_c < best_3way[1]:
                best_3way = ((round(w_lr, 2), round(w_xgb, 2), round(w_lgb, 2)), bs_c)
    print(f"  Best 3-way: {best_3way[0]} -> combined={best_3way[1]:.4f}")
    rows.append({"strategy": f"LR+XGB+LGB_{best_3way[0]}", "loso": best_3way[1],
                  "men": np.nan, "women": np.nan})

    # ===========================================================
    # (C) Stacking via meta-LR on LOSO OOF (within-season CV-stacking)
    # ===========================================================
    print(f"\n{'='*70}\n  (C) Stacking: meta-LR on LOSO OOF\n{'='*70}")
    # logit-space stack
    def logit(p, eps=1e-6):
        p = np.clip(p, eps, 1 - eps)
        return np.log(p / (1 - p))

    Z = np.column_stack([logit(p_lr), logit(p_xgb), logit(p_lgb)])
    p_stack_oof = np.zeros(len(X))
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        if te.sum() == 0: continue
        meta = LogisticRegression(C=1.0, max_iter=1000)
        meta.fit(Z[tr], y[tr])
        p_stack_oof[te] = meta.predict_proba(Z[te])[:, 1]
    p_stack_oof = np.clip(p_stack_oof, 0.005, 0.995)
    bs_m, bs_w, bs_c = combined_brier(p_stack_oof, y, is_w)
    print(f"  Stack meta-LR: combined={bs_c:.4f}")
    rows.append({"strategy": "stack_meta_LR", "loso": bs_c, "men": bs_m, "women": bs_w})

    # Constrained stacking: non-negative coefs
    from scipy.optimize import minimize
    def loss(weights, Z, y):
        weights = np.maximum(weights, 0)
        if weights.sum() == 0:
            return 1e10
        weights = weights / weights.sum()
        p = (weights * np.column_stack([p_lr, p_xgb, p_lgb])).sum(axis=1)
        p = np.clip(p, 0.005, 0.995)
        return np.mean((p - y) ** 2)
    # LOSO non-negative stacking
    p_stack_nn = np.zeros(len(X))
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        if te.sum() == 0: continue
        # Optimize non-negative weights on training fold
        Ptr = np.column_stack([p_lr[tr], p_xgb[tr], p_lgb[tr]])
        ytr = y[tr]
        def f(w):
            w = np.maximum(w, 0)
            if w.sum() == 0: return 1e10
            w = w / w.sum()
            p = Ptr @ w
            p = np.clip(p, 0.005, 0.995)
            return np.mean((p - ytr) ** 2)
        res = minimize(f, x0=[0.7, 0.15, 0.15], method="Nelder-Mead")
        w = np.maximum(res.x, 0); w = w / w.sum() if w.sum() > 0 else np.array([1, 0, 0])
        Pte = np.column_stack([p_lr[te], p_xgb[te], p_lgb[te]])
        p_stack_nn[te] = Pte @ w
    p_stack_nn = np.clip(p_stack_nn, 0.005, 0.995)
    bs_m, bs_w, bs_c = combined_brier(p_stack_nn, y, is_w)
    print(f"  Stack non-negative: combined={bs_c:.4f}")
    rows.append({"strategy": "stack_nonneg", "loso": bs_c, "men": bs_m, "women": bs_w})

    # ===========================================================
    # (D) BaggedLR + XGB blend
    # ===========================================================
    print(f"\n{'='*70}\n  (D) BaggedLR + multi-seed XGB blend\n{'='*70}")
    for w in [0.65, 0.70, 0.75]:
        p = w * p_blr + (1 - w) * p_xgb
        p = np.clip(p, 0.005, 0.995)
        _, _, bs_c = combined_brier(p, y, is_w)
        rows.append({"strategy": f"BaggedLR+XGB_w={w}", "loso": bs_c, "men": np.nan, "women": np.nan})
        print(f"  w_LR={w}: combined={bs_c:.4f}")

    # Summary
    df = pd.DataFrame(rows).sort_values("loso")
    df.to_csv("output/alternative_blends_loso.csv", index=False)
    print(f"\n{'='*70}\n  TOP 10 by LOSO\n{'='*70}")
    print(df.head(10).to_string(index=False))

    # ===========================================================
    # Apply top configs to 2026
    # ===========================================================
    print(f"\n{'='*70}\n  Apply top configs to 2026\n{'='*70}")

    X_2026, _, is_w_2026 = build_combined_features_2026(data_m, data_w)
    X_2026.loc[is_w_2026 == 1, massey_cols] = 0.0
    X_2026 = add_seed_pair(X_2026, is_w_2026.astype(int),
                            seed_lookup_m, seed_lookup_w,
                            {2026: base_full_m}, {2026: base_full_w},
                            base_full_m, base_full_w)

    # Train final models on full data
    Xtr_full = X[FEATS_8].apply(pd.to_numeric, errors="coerce").fillna(
        X[FEATS_8].apply(pd.to_numeric, errors="coerce").median()
    )
    X_2026_arr = X_2026[FEATS_8].apply(pd.to_numeric, errors="coerce").fillna(Xtr_full.median())

    # LR final
    scaler = StandardScaler().fit(Xtr_full)
    lr = LogisticRegression(C=LR_C, max_iter=2000, solver="lbfgs").fit(scaler.transform(Xtr_full), y)
    p_lr_2026 = lr.predict_proba(scaler.transform(X_2026_arr))[:, 1]

    # XGB final (10 seeds)
    p_xgb_2026 = np.zeros(len(X_2026))
    for seed in range(10):
        params = dict(XGB_PARAMS); params["random_state"] = seed
        xm = xgb.XGBClassifier(**params).fit(Xtr_full.values, y)
        p_xgb_2026 += xm.predict_proba(X_2026_arr.values)[:, 1]
    p_xgb_2026 /= 10

    # LGB final (10 seeds)
    p_lgb_2026 = np.zeros(len(X_2026))
    for seed in range(10):
        params = dict(LGB_PARAMS); params["random_state"] = seed
        lm = lgb.LGBMClassifier(**params).fit(Xtr_full.values, y)
        p_lgb_2026 += lm.predict_proba(X_2026_arr.values)[:, 1]
    p_lgb_2026 /= 10

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

    def evaluate(p_2026, label):
        p_2026 = np.clip(p_2026, 0.005, 0.995)
        pair_lk = {(int(r["TeamA"]), int(r["TeamB"])): float(p_2026[i])
                   for i, r in X_2026.reset_index(drop=True).iterrows()}
        bs_m_, n_m_ = br(pair_lk, actual_m)
        bs_w_, n_w_ = br(pair_lk, actual_w)
        bs_c_ = (bs_m_ * n_m_ + bs_w_ * n_w_) / (n_m_ + n_w_)
        return bs_m_, bs_w_, bs_c_

    print(f"  Strategy                  LOSO     2026")

    # Baseline 70/30 LR+XGB
    p = 0.7 * p_lr_2026 + 0.3 * p_xgb_2026
    bs_m_, bs_w_, bs_c_ = evaluate(p, "70/30 LR+XGB")
    print(f"  70/30 LR+XGB              0.1606   {bs_c_:.4f}  (men {bs_m_:.4f} wom {bs_w_:.4f})")

    # 3-way best
    w_lr, w_xgb, w_lgb = best_3way[0]
    p = w_lr * p_lr_2026 + w_xgb * p_xgb_2026 + w_lgb * p_lgb_2026
    bs_m_, bs_w_, bs_c_ = evaluate(p, "3-way")
    print(f"  3-way ({w_lr}, {w_xgb}, {w_lgb})  {best_3way[1]:.4f}  {bs_c_:.4f}  (men {bs_m_:.4f} wom {bs_w_:.4f})")

    # Stack non-negative — average weights from full train? Use last fold's weights as approximation
    # Or refit on full data
    Ptr_full = np.column_stack([p_lr, p_xgb, p_lgb])
    def f_full(w):
        w = np.maximum(w, 0)
        if w.sum() == 0: return 1e10
        w = w / w.sum()
        p = Ptr_full @ w
        p = np.clip(p, 0.005, 0.995)
        return np.mean((p - y) ** 2)
    res = minimize(f_full, x0=[0.7, 0.15, 0.15], method="Nelder-Mead")
    w_stack = np.maximum(res.x, 0); w_stack = w_stack / w_stack.sum()
    p = w_stack[0] * p_lr_2026 + w_stack[1] * p_xgb_2026 + w_stack[2] * p_lgb_2026
    bs_m_, bs_w_, bs_c_ = evaluate(p, "stack_nonneg")
    print(f"  stack_nonneg ({w_stack.round(2)})  -      {bs_c_:.4f}  (men {bs_m_:.4f} wom {bs_w_:.4f})")

    # Save best LOSO submission
    best = df.iloc[0]
    print(f"\n  Best LOSO: {best['strategy']} -> {best['loso']:.4f}")


if __name__ == "__main__":
    main()
