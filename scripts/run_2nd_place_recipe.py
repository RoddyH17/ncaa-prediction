"""
2nd place 2026 Kaggle exact recipe: LR(40%) + XGB(60%) blend on the SAME
feature set with separate scaling.

Recipe (from 2nd place writeup):
  - Same 35 features as differentials (we use our 26-feature set + seed_pair)
  - LR with StandardScaler (C=1.0)
  - XGB with raw features (max_depth=4, n_estimators=300, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3, reg_alpha=0.1,
    reg_lambda=1.0, eval_metric='logloss')
  - blended = 0.4 * lr_preds + 0.6 * xgb_preds
  - Clip [0.02, 0.98]
  - Trained on combined men's + women's

If this LOSO-Brier-beats our current best (+seed_pair LR alone at 0.1637), use it.
Else stick with simpler.
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


XGB_HPARAMS = dict(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
    random_state=42, tree_method="hist",
)

LR_C = 1.0
LR_WEIGHT = 0.4
XGB_WEIGHT = 0.6


def add_seed_pair(X, is_w, seed_lookup_m, seed_lookup_w,
                  base_table_per_season_m, base_table_per_season_w,
                  base_full_m, base_full_w):
    vals = []
    for _, r in X.iterrows():
        season = int(r["Season"])
        ta, tb = int(r["TeamA"]), int(r["TeamB"])
        if int(r["is_womens"]) == 0:
            seed_a = seed_lookup_m.get((season, ta), 17)
            seed_b = seed_lookup_m.get((season, tb), 17)
            tbl = base_table_per_season_m.get(season, base_full_m)
        else:
            seed_a = seed_lookup_w.get((season, ta), 17)
            seed_b = seed_lookup_w.get((season, tb), 17)
            tbl = base_table_per_season_w.get(season, base_full_w)
        vals.append(lookup_p_a_wins(tbl, seed_a, seed_b))
    X = X.copy()
    X["seed_pair_winrate"] = vals
    return X


def loso_2nd_place(X, y, is_w, season_arr, feats):
    """LOSO with 2nd place LR(40%) + XGB(60%) blend on combined M+W training."""
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

        # LR with scaling
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xte_s = scaler.transform(Xte)
        lr = LogisticRegression(C=LR_C, max_iter=2000, solver="lbfgs")
        lr.fit(Xtr_s, y[tr])
        p_lr = lr.predict_proba(Xte_s)[:, 1]

        # XGB raw
        xgb_model = xgb.XGBClassifier(**XGB_HPARAMS)
        xgb_model.fit(Xtr.values, y[tr])
        p_xgb = xgb_model.predict_proba(Xte.values)[:, 1]

        p_blend = LR_WEIGHT * p_lr + XGB_WEIGHT * p_xgb
        p_oof[te] = np.clip(p_blend, 0.02, 0.98)

    n_m = (is_w == 0).sum(); n_w = (is_w == 1).sum()
    bs_m = brier_score_loss(y[is_w == 0], p_oof[is_w == 0])
    bs_w = brier_score_loss(y[is_w == 1], p_oof[is_w == 1])
    bs_c = (bs_m * n_m + bs_w * n_w) / (n_m + n_w)
    return p_oof, bs_m, bs_w, bs_c


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]
    print("Loading data...")
    data_m = load_all_mens_data()
    data_w = load_womens_data()

    print("\nBuilding feature matrix...")
    X, y, is_w = build_combined_features(data_m, data_w, seasons)
    season_arr = X["Season"].values

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
    feats = base_feats + ["seed_pair_winrate"]
    print(f"  Total features: {len(feats)}")

    # Note: 2nd place trained combined M+W. We add `is_womens` flag.
    feats_combined = feats + ["is_womens"]

    print(f"\n{'='*70}\n  2nd place LR(40%)+XGB(60%) blend on combined M+W\n{'='*70}")
    p_oof, bs_m, bs_w, bs_c = loso_2nd_place(X, y, is_w, season_arr, feats_combined)
    print(f"  LOSO: men={bs_m:.4f}  women={bs_w:.4f}  combined={bs_c:.4f}")
    print(f"  vs our +seed_pair LR alone:  0.1637")
    print(f"  vs our baseline LR:          0.1638")

    # Compare to LR-only
    print(f"\n{'='*70}\n  LR-only (per gender) for comparison\n{'='*70}")
    p_oof_lr = np.zeros(len(X))
    for is_g in [0, 1]:
        for s in np.unique(season_arr[is_w == is_g]):
            tr = (season_arr != s) & (is_w == is_g)
            te = (season_arr == s) & (is_w == is_g)
            if te.sum() == 0: continue
            feats_use = [c for c in feats if not (is_g == 1 and c.startswith("massey_"))]
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                ("lr", LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs")),
            ])
            pipe.fit(X.loc[tr, feats_use].apply(pd.to_numeric, errors="coerce"), y[tr])
            p_oof_lr[te] = pipe.predict_proba(X.loc[te, feats_use].apply(pd.to_numeric, errors="coerce"))[:, 1]
    bs_m_lr = brier_score_loss(y[is_w == 0], p_oof_lr[is_w == 0])
    bs_w_lr = brier_score_loss(y[is_w == 1], p_oof_lr[is_w == 1])
    n_m = (is_w == 0).sum(); n_w_ = (is_w == 1).sum()
    bs_c_lr = (bs_m_lr * n_m + bs_w_lr * n_w_) / (n_m + n_w_)
    print(f"  LR-only LOSO: men={bs_m_lr:.4f}  women={bs_w_lr:.4f}  combined={bs_c_lr:.4f}")

    # Apply best 2nd-place to 2026 if it wins
    print(f"\n{'='*70}\n  Apply 2nd place recipe to 2026\n{'='*70}")
    X_2026, _, is_w_2026 = build_combined_features_2026(data_m, data_w)
    X_2026 = add_seed_pair(X_2026, is_w_2026.astype(int),
                            seed_lookup_m, seed_lookup_w,
                            {2026: base_full_m}, {2026: base_full_w},
                            base_full_m, base_full_w)

    Xtr = X[feats_combined].apply(pd.to_numeric, errors="coerce").fillna(
        X[feats_combined].apply(pd.to_numeric, errors="coerce").median()
    )
    X_2026_arr = X_2026[feats_combined].apply(pd.to_numeric, errors="coerce").fillna(
        Xtr.median()
    )

    scaler = StandardScaler().fit(Xtr)
    lr = LogisticRegression(C=LR_C, max_iter=2000, solver="lbfgs")
    lr.fit(scaler.transform(Xtr), y)
    p_lr_2026 = lr.predict_proba(scaler.transform(X_2026_arr))[:, 1]

    xgb_model = xgb.XGBClassifier(**XGB_HPARAMS)
    xgb_model.fit(Xtr.values, y)
    p_xgb_2026 = xgb_model.predict_proba(X_2026_arr.values)[:, 1]

    p_2026 = LR_WEIGHT * p_lr_2026 + XGB_WEIGHT * p_xgb_2026
    p_2026 = np.clip(p_2026, 0.02, 0.98)

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

    bs_m_2026, n_m_a = br(actual_m)
    bs_w_2026, n_w_a = br(actual_w)
    bs_c_2026 = (bs_m_2026 * n_m_a + bs_w_2026 * n_w_a) / (n_m_a + n_w_a)

    print(f"  2nd place recipe 2026: men={bs_m_2026:.4f}  women={bs_w_2026:.4f}  combined={bs_c_2026:.4f}")
    print(f"  vs our +seed_pair LR alone (2026 = 0.1255)")
    print(f"  vs Kaggle 2nd place (final = 0.1149)")


if __name__ == "__main__":
    main()
