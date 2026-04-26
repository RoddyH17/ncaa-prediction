"""
Cross-year validation of over-tuning patterns.

We replicate the five over-tuning experiments of Section~5 holding out 2024
and 2025 (instead of 2026), to verify that the LOSO-improves but test-degrades
pattern is not a 2026-specific artifact.

For each held-out year h:
  1. Train on {2014..2025} \ {h, 2020}; LOSO over those years
  2. Run baseline 70/30 LR+XGB on 8 features → LOSO + B_h Brier
  3. Run each over-tuning method → measure LOSO improvement & B_h change
  4. Report (delta LOSO, delta B_h)

Outputs:
  output/cross_year_overtuning.csv
"""

import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import _parse_seed_num
from scripts.build_womens_model import load_womens_data
from scripts.run_top3 import build_combined_features, FEATURE_COLS as TOP3_FEATURES
from src.seed_base_rate import compute_base_rate_table, lookup_p_a_wins


FEATS_8 = [
    "seed_pair_winrate", "bart_net_diff", "harry_diff",
    "elo_diff", "elo_slope_diff", "srs_diff",
    "massey_mean_diff", "tempo_diff",
]
LR_C = 0.1
XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
    tree_method="hist", random_state=42,
)


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


def fit_blend(X_train, y_train, lr_C, w_lr, xgb_params, X_pred):
    """Fit LR+XGB blend, return predictions on X_pred."""
    Xtr = X_train[FEATS_8].apply(pd.to_numeric, errors="coerce").fillna(
        X_train[FEATS_8].apply(pd.to_numeric, errors="coerce").median()
    )
    med = Xtr.median()
    Xpred = X_pred[FEATS_8].apply(pd.to_numeric, errors="coerce").fillna(med)
    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=lr_C, max_iter=2000, solver="lbfgs")
    lr.fit(scaler.transform(Xtr), y_train)
    p_lr = lr.predict_proba(scaler.transform(Xpred))[:, 1]
    xm = xgb.XGBClassifier(**xgb_params)
    xm.fit(Xtr.values, y_train)
    p_xgb = xm.predict_proba(Xpred.values)[:, 1]
    return np.clip(w_lr * p_lr + (1 - w_lr) * p_xgb, 0.005, 0.995)


def loso_brier(X, y, season_arr, is_w, lr_C, w_lr, xgb_params, exclude_seasons=()):
    """Compute LOSO Brier excluding given seasons (held-out years)."""
    p_oof = np.zeros(len(X))
    counted = np.zeros(len(X), dtype=bool)
    seasons = [s for s in np.unique(season_arr) if s not in exclude_seasons]
    for s in seasons:
        tr_idx = np.where((season_arr != s) & (~np.isin(season_arr, exclude_seasons)))[0]
        te_idx = np.where(season_arr == s)[0]
        if len(te_idx) == 0:
            continue
        p = fit_blend(X.iloc[tr_idx], y[tr_idx], lr_C, w_lr, xgb_params, X.iloc[te_idx])
        p_oof[te_idx] = p
        counted[te_idx] = True
    n_m = ((is_w == 0) & counted).sum()
    n_w_ = ((is_w == 1) & counted).sum()
    bs_m = brier_score_loss(y[(is_w == 0) & counted], p_oof[(is_w == 0) & counted])
    bs_w = brier_score_loss(y[(is_w == 1) & counted], p_oof[(is_w == 1) & counted])
    return (bs_m * n_m + bs_w * n_w_) / (n_m + n_w_)


def heldout_brier(X, y, season_arr, is_w, holdout_year, lr_C, w_lr, xgb_params):
    """Train on all seasons except holdout, evaluate on holdout."""
    tr_idx = np.where((season_arr != holdout_year) & (season_arr != 2020))[0]
    te_idx = np.where(season_arr == holdout_year)[0]
    if len(te_idx) == 0:
        return None
    p = fit_blend(X.iloc[tr_idx], y[tr_idx], lr_C, w_lr, xgb_params, X.iloc[te_idx])
    bs_m = brier_score_loss(y[te_idx][is_w[te_idx] == 0], p[is_w[te_idx] == 0])
    bs_w = brier_score_loss(y[te_idx][is_w[te_idx] == 1], p[is_w[te_idx] == 1])
    n_m = (is_w[te_idx] == 0).sum()
    n_w_ = (is_w[te_idx] == 1).sum()
    return (bs_m * n_m + bs_w * n_w_) / (n_m + n_w_)


def main():
    seasons_all = [s for s in range(2014, 2026) if s != 2020]
    print("Loading...")
    data_m = load_all_mens_data()
    data_w = load_womens_data()
    X, y, is_w = build_combined_features(data_m, data_w, seasons_all)
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
                         for s in seasons_all}
    base_per_season_w = {s: compute_base_rate_table(data_w["tourney_compact"], seeds_w, exclude_season=s)
                         for s in seasons_all}
    X = add_seed_pair(X, is_w, seed_lookup_m, seed_lookup_w,
                      base_per_season_m, base_per_season_w, base_full_m, base_full_w)

    # Configurations: baseline + 4 over-tuning variants
    configs = {
        "baseline_70/30": dict(lr_C=0.1, w_lr=0.7, xgb_params=XGB_PARAMS),
        "deep_xgb_d4_n1000": dict(lr_C=0.1, w_lr=0.7,
                                   xgb_params={**XGB_PARAMS, "max_depth": 4, "n_estimators": 1000, "learning_rate": 0.02}),
        "high_C_w_75": dict(lr_C=0.24, w_lr=0.78,
                             xgb_params={**XGB_PARAMS, "max_depth": 4, "n_estimators": 850, "learning_rate": 0.054,
                                         "min_child_weight": 2, "subsample": 1.0, "colsample_bytree": 0.6,
                                         "reg_alpha": 0.27, "reg_lambda": 1.23}),
        "low_w_60": dict(lr_C=0.1, w_lr=0.6, xgb_params=XGB_PARAMS),
        "high_w_85": dict(lr_C=0.1, w_lr=0.85, xgb_params=XGB_PARAMS),
    }

    results = []
    for holdout in [2024, 2025, 2026]:
        if holdout == 2026:
            print(f"\n=== Hold out {holdout} (no 2026 in our data — use 2026 actual files) ===")
            # 2026 not in season_arr; we cant evaluate inside this framework cleanly
            # Skip for this script (we have 2026 results elsewhere)
            continue
        print(f"\n=== Hold out {holdout} ===")
        baseline_loso = loso_brier(X, y, season_arr, is_w, exclude_seasons=(holdout, 2020),
                                     **configs["baseline_70/30"])
        baseline_h = heldout_brier(X, y, season_arr, is_w, holdout, **configs["baseline_70/30"])
        print(f"  Baseline: LOSO={baseline_loso:.4f}  hold-out {holdout}={baseline_h:.4f}")

        for name, cfg in configs.items():
            if name == "baseline_70/30":
                continue
            loso = loso_brier(X, y, season_arr, is_w, exclude_seasons=(holdout, 2020), **cfg)
            heldout = heldout_brier(X, y, season_arr, is_w, holdout, **cfg)
            d_loso = loso - baseline_loso
            d_heldout = heldout - baseline_h
            ratio = d_heldout / d_loso if abs(d_loso) > 1e-9 else float('nan')
            results.append({
                "holdout_year": holdout,
                "method": name,
                "baseline_loso": baseline_loso,
                "method_loso": loso,
                "delta_loso": d_loso,
                "baseline_heldout": baseline_h,
                "method_heldout": heldout,
                "delta_heldout": d_heldout,
                "ratio": ratio,
            })
            print(f"  {name:30s} delta_LOSO={d_loso:+.4f}  delta_{holdout}={d_heldout:+.4f}  ratio={ratio:+.2f}")

    df = pd.DataFrame(results)
    df.to_csv("output/cross_year_overtuning.csv", index=False)
    print(f"\nSaved output/cross_year_overtuning.csv")
    print()
    print("Summary by method (mean across years):")
    print(df.groupby("method")[["delta_loso", "delta_heldout", "ratio"]].mean().to_string())


if __name__ == "__main__":
    main()
