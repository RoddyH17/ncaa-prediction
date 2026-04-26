"""
Push the 8-feature LR+XGB blend further:

  (1) Per-gender blend weights (men's and women's may want different ratios)
  (2) Multi-seed XGB averaging (variance reduction)
  (3) XGB hyperparameter fine-tune (max_depth, learning_rate, n_estimators)

All decisions on LOSO. Apply best to 2026.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import xgboost as xgb
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


FEATS_8 = [
    "seed_pair_winrate", "bart_net_diff", "harry_diff",
    "elo_diff", "elo_slope_diff", "srs_diff",
    "massey_mean_diff", "tempo_diff",
]


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


def loso_oof_lr_xgb_multiseed(X, y, is_w, season_arr, feats, lr_C, xgb_params, n_seeds=10):
    """Compute LOSO OOF predictions for both LR (one seed) and XGB averaged across seeds."""
    p_lr_oof = np.zeros(len(X))
    p_xgb_oof = np.zeros(len(X))
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        if te.sum() == 0: continue
        Xtr = X.loc[tr, feats].apply(pd.to_numeric, errors="coerce")
        Xte = X.loc[te, feats].apply(pd.to_numeric, errors="coerce")
        Xtr_filled = Xtr.fillna(Xtr.median())
        Xte_filled = Xte.fillna(Xtr.median())
        # LR (deterministic)
        scaler = StandardScaler().fit(Xtr_filled)
        lr = LogisticRegression(C=lr_C, max_iter=2000, solver="lbfgs")
        lr.fit(scaler.transform(Xtr_filled), y[tr])
        p_lr_oof[te] = lr.predict_proba(scaler.transform(Xte_filled))[:, 1]
        # XGB averaged across seeds
        p_xgb_acc = np.zeros(te.sum())
        for seed in range(n_seeds):
            params = dict(xgb_params); params["random_state"] = seed
            xm = xgb.XGBClassifier(**params)
            xm.fit(Xtr_filled.values, y[tr])
            p_xgb_acc += xm.predict_proba(Xte_filled.values)[:, 1]
        p_xgb_oof[te] = p_xgb_acc / n_seeds
    return p_lr_oof, p_xgb_oof


def combined_brier(p_oof, y, is_w):
    n_m = (is_w == 0).sum(); n_w_ = (is_w == 1).sum()
    bs_m = brier_score_loss(y[is_w == 0], p_oof[is_w == 0])
    bs_w = brier_score_loss(y[is_w == 1], p_oof[is_w == 1])
    bs_c = (bs_m * n_m + bs_w * n_w_) / (n_m + n_w_)
    return bs_m, bs_w, bs_c


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

    # ==========================================================
    # Step 1: XGB hyperparameter sweep (with multi-seed averaging)
    # ==========================================================
    print(f"\n{'='*70}\n  Step 1: XGB hparam sweep + multi-seed avg + per-gender blend\n{'='*70}")

    base_xgb_params = dict(
        n_estimators=300, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
        tree_method="hist",
    )

    xgb_grid = []
    for max_depth in [2, 3]:
        for n_est, lr_xg in [(300, 0.05), (500, 0.03), (1000, 0.02)]:
            for mcw in [3, 5]:
                xgb_grid.append({
                    **base_xgb_params,
                    "max_depth": max_depth, "n_estimators": n_est,
                    "learning_rate": lr_xg, "min_child_weight": mcw,
                })

    print(f"  Trying {len(xgb_grid)} XGB hparam combos x 5 seeds each...")
    print(f"  Computing LR OOF once (deterministic)...")
    # First compute LR OOF (doesn't depend on XGB params)
    p_lr_oof = np.zeros(len(X))
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        if te.sum() == 0: continue
        Xtr = X.loc[tr, FEATS_8].apply(pd.to_numeric, errors="coerce")
        Xte = X.loc[te, FEATS_8].apply(pd.to_numeric, errors="coerce")
        Xtr_filled = Xtr.fillna(Xtr.median())
        Xte_filled = Xte.fillna(Xtr.median())
        scaler = StandardScaler().fit(Xtr_filled)
        lr = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")
        lr.fit(scaler.transform(Xtr_filled), y[tr])
        p_lr_oof[te] = lr.predict_proba(scaler.transform(Xte_filled))[:, 1]
    bs_m_lr, bs_w_lr, bs_c_lr = combined_brier(p_lr_oof, y, is_w)
    print(f"  LR-only OOF: men={bs_m_lr:.4f}  women={bs_w_lr:.4f}  combined={bs_c_lr:.4f}")

    best_blend = None  # (xgb_params_idx, w_m, w_w, brier_c)
    best_xgb_oof = None
    print(f"\n  Testing each XGB config + per-gender blend search...")
    for i_grid, params in enumerate(xgb_grid):
        # Compute XGB OOF (5 seeds averaged for speed)
        p_xgb_oof = np.zeros(len(X))
        for s in np.unique(season_arr):
            tr = season_arr != s
            te = season_arr == s
            if te.sum() == 0: continue
            Xtr = X.loc[tr, FEATS_8].apply(pd.to_numeric, errors="coerce").fillna(
                X.loc[tr, FEATS_8].apply(pd.to_numeric, errors="coerce").median()
            )
            Xte = X.loc[te, FEATS_8].apply(pd.to_numeric, errors="coerce").fillna(
                X.loc[tr, FEATS_8].apply(pd.to_numeric, errors="coerce").median()
            )
            p_acc = np.zeros(te.sum())
            for seed in range(5):
                params_seed = dict(params); params_seed["random_state"] = seed
                xm = xgb.XGBClassifier(**params_seed)
                xm.fit(Xtr.values, y[tr])
                p_acc += xm.predict_proba(Xte.values)[:, 1]
            p_xgb_oof[te] = p_acc / 5

        # Search best per-gender blend
        local_best = (None, None, np.inf)
        for w_m in np.arange(0.0, 1.001, 0.05):
            for w_w in np.arange(0.0, 1.001, 0.05):
                p = np.zeros(len(X))
                p[is_w == 0] = w_m * p_lr_oof[is_w == 0] + (1 - w_m) * p_xgb_oof[is_w == 0]
                p[is_w == 1] = w_w * p_lr_oof[is_w == 1] + (1 - w_w) * p_xgb_oof[is_w == 1]
                p = np.clip(p, 0.005, 0.995)
                _, _, bs_c = combined_brier(p, y, is_w)
                if bs_c < local_best[2]:
                    local_best = (round(w_m, 2), round(w_w, 2), bs_c)
        rows.append({
            "config": f"d{params['max_depth']}_n{params['n_estimators']}_lr{params['learning_rate']}_mcw{params['min_child_weight']}",
            "w_m": local_best[0], "w_w": local_best[1], "combined": local_best[2],
        })
        print(f"  [{i_grid+1}/{len(xgb_grid)}] d={params['max_depth']} n={params['n_estimators']} "
              f"lr={params['learning_rate']} mcw={params['min_child_weight']}: "
              f"best blend (w_m={local_best[0]}, w_w={local_best[1]}) -> {local_best[2]:.4f}")

        if best_blend is None or local_best[2] < best_blend[3]:
            best_blend = (params, local_best[0], local_best[1], local_best[2])
            best_xgb_oof = p_xgb_oof

    df = pd.DataFrame(rows).sort_values("combined")
    df.to_csv("output/8feat_push_loso.csv", index=False)
    print(f"\n  Top 10:")
    print(df.head(10).to_string(index=False))

    print(f"\n  Best LOSO config:")
    best_params = best_blend[0]
    print(f"    XGB: depth={best_params['max_depth']}, n_est={best_params['n_estimators']}, "
          f"lr={best_params['learning_rate']}, mcw={best_params['min_child_weight']}")
    print(f"    Blend: w_m={best_blend[1]}  w_w={best_blend[2]}")
    print(f"    LOSO Brier: {best_blend[3]:.4f}")

    # ==========================================================
    # Apply best to 2026
    # ==========================================================
    print(f"\n{'='*70}\n  Apply best to 2026\n{'='*70}")

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
    lr = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(scaler.transform(Xtr_full), y)
    p_lr_2026 = lr.predict_proba(scaler.transform(X_2026_arr))[:, 1]

    # XGB final (10 seeds averaged for stability)
    p_xgb_2026 = np.zeros(len(X_2026))
    for seed in range(10):
        params = dict(best_params); params["random_state"] = seed
        xm = xgb.XGBClassifier(**params).fit(Xtr_full.values, y)
        p_xgb_2026 += xm.predict_proba(X_2026_arr.values)[:, 1]
    p_xgb_2026 /= 10

    # Apply per-gender blend
    p_2026 = np.zeros(len(X_2026))
    p_2026[is_w_2026 == 0] = best_blend[1] * p_lr_2026[is_w_2026 == 0] + \
                              (1 - best_blend[1]) * p_xgb_2026[is_w_2026 == 0]
    p_2026[is_w_2026 == 1] = best_blend[2] * p_lr_2026[is_w_2026 == 1] + \
                              (1 - best_blend[2]) * p_xgb_2026[is_w_2026 == 1]
    p_2026 = np.clip(p_2026, 0.005, 0.995)

    pair_lk = {(int(r["TeamA"]), int(r["TeamB"])): float(p_2026[i])
               for i, r in X_2026.reset_index(drop=True).iterrows()}

    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")

    def br(actual):
        yt, yp = [], []
        for _, g in actual.iterrows():
            w_, l = int(g["WTeamID"]), int(g["LTeamID"])
            key = (w_, l) if w_ < l else (l, w_)
            yt.append(1 if w_ < l else 0)
            yp.append(pair_lk.get(key, 0.5))
        return brier_score_loss(yt, yp), len(yt)

    bs_m_2026, n_m = br(actual_m)
    bs_w_2026, n_w = br(actual_w)
    bs_c_2026 = (bs_m_2026 * n_m + bs_w_2026 * n_w) / (n_m + n_w)

    print(f"\n{'='*70}\n  FINAL RESULT\n{'='*70}")
    print(f"  LOSO:  men={brier_score_loss(y[is_w == 0], np.zeros(int((is_w == 0).sum()))+0.5):.4f}  (placeholder)")
    print(f"  LOSO Combined Brier: {best_blend[3]:.4f}")
    print(f"  2026 Combined Brier: {bs_c_2026:.4f}")
    print(f"    Men's:    {bs_m_2026:.4f}")
    print(f"    Women's:  {bs_w_2026:.4f}")
    print(f"\n  Trajectory:")
    print(f"    Initial baseline:           0.1264")
    print(f"    L1+L2 (11 features):        0.1243")
    print(f"    8-feat LR alone:            0.1238")
    print(f"    8-feat LR+XGB blend (uniform): 0.1229")
    print(f"    8-feat per-gender + multi-seed XGB: {bs_c_2026:.4f}")

    # Save submission
    sub = pd.read_csv("output/submission_stage2.csv")
    sub[["s_str", "ta_str", "tb_str"]] = sub["ID"].str.split("_", expand=True)
    sub["TeamA"] = sub["ta_str"].astype(int); sub["TeamB"] = sub["tb_str"].astype(int)
    sub["Pred"] = sub.apply(
        lambda r: pair_lk.get((r["TeamA"], r["TeamB"]), float(r["Pred"])),
        axis=1
    ).clip(0.005, 0.995)
    sub[["ID", "Pred"]].to_csv("output/submission_stage2_PUSH.csv", index=False)
    print(f"\n  Saved output/submission_stage2_PUSH.csv")


if __name__ == "__main__":
    main()
