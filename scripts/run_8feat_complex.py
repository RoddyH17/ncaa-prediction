"""
Try more complex models on the 8-feature set.

Tests:
  (A) Polynomial features (degree 2) + L1/L2 LR
  (B) Smoothing splines via patsy + LR
  (C) LR + XGB blend at varying weights
  (D) Random Forest (shallow)
  (E) Gradient Boosting (sklearn)

All decisions on LOSO; pick best, apply to 2026.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
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


def loso_brier(X, y, is_w, season_arr, feats, model_fn):
    p_oof = np.zeros(len(X))
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        if te.sum() == 0: continue
        pipe = model_fn()
        Xtr = X.loc[tr, feats].apply(pd.to_numeric, errors="coerce")
        Xte = X.loc[te, feats].apply(pd.to_numeric, errors="coerce")
        pipe.fit(Xtr, y[tr])
        p_oof[te] = pipe.predict_proba(Xte)[:, 1]
    n_m = (is_w == 0).sum(); n_w_ = (is_w == 1).sum()
    bs_m = brier_score_loss(y[is_w == 0], p_oof[is_w == 0])
    bs_w = brier_score_loss(y[is_w == 1], p_oof[is_w == 1])
    bs_c = (bs_m * n_m + bs_w * n_w_) / (n_m + n_w_)
    return p_oof, bs_m, bs_w, bs_c


def loso_brier_blend(X, y, is_w, season_arr, feats, lr_C, xgb_params, w_lr):
    """Compute LOSO Brier for LR(scaled) + XGB(raw) blend."""
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
        # LR
        scaler = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=lr_C, max_iter=2000, solver="lbfgs")
        lr.fit(scaler.transform(Xtr), y[tr])
        p_lr = lr.predict_proba(scaler.transform(Xte))[:, 1]
        # XGB
        xm = xgb.XGBClassifier(**xgb_params)
        xm.fit(Xtr.values, y[tr])
        p_xgb = xm.predict_proba(Xte.values)[:, 1]
        p_oof[te] = w_lr * p_lr + (1 - w_lr) * p_xgb
    p_oof = np.clip(p_oof, 0.005, 0.995)
    n_m = (is_w == 0).sum(); n_w_ = (is_w == 1).sum()
    bs_m = brier_score_loss(y[is_w == 0], p_oof[is_w == 0])
    bs_w = brier_score_loss(y[is_w == 1], p_oof[is_w == 1])
    bs_c = (bs_m * n_m + bs_w * n_w_) / (n_m + n_w_)
    return p_oof, bs_m, bs_w, bs_c


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

    print(f"\n  Features (8): {FEATS_8}")
    print(f"  Total games: {len(X)}\n")

    rows = []

    # ======================================================
    # Baseline: vanilla LR
    # ======================================================
    print(f"{'='*70}\n  Baseline: vanilla LR\n{'='*70}")
    for C in [0.1, 0.3, 0.5]:
        _, bs_m, bs_w, bs_c = loso_brier(X, y, is_w, season_arr, FEATS_8,
            lambda C=C: Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                ("lr", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
            ])
        )
        rows.append({"model": f"LR_C={C}", "men": bs_m, "women": bs_w, "combined": bs_c})
        print(f"  LR C={C}: combined={bs_c:.4f}  men={bs_m:.4f}  women={bs_w:.4f}")

    # ======================================================
    # Strategy A: Polynomial features (degree 2)
    # ======================================================
    print(f"\n{'='*70}\n  A: Polynomial features (degree 2)\n{'='*70}")
    for poly_deg in [2]:
        # 8 features → 8 + C(8,2) + 8 = 44 features (with bias=False, interaction_only=False)
        # Use interaction_only to limit explosion: 8 + 28 = 36 features
        for io in [False, True]:
            for C in [0.05, 0.1, 0.3, 1.0]:
                def build(C=C, io=io, poly_deg=poly_deg):
                    return Pipeline([
                        ("imp", SimpleImputer(strategy="median")),
                        ("scl", StandardScaler()),
                        ("poly", PolynomialFeatures(degree=poly_deg, interaction_only=io, include_bias=False)),
                        ("lr", LogisticRegression(C=C, max_iter=3000, solver="lbfgs")),
                    ])
                _, bs_m, bs_w, bs_c = loso_brier(X, y, is_w, season_arr, FEATS_8, build)
                tag = f"poly2_io={io}_C={C}"
                rows.append({"model": tag, "men": bs_m, "women": bs_w, "combined": bs_c})
                print(f"  {tag}: combined={bs_c:.4f}")

    # ======================================================
    # Strategy B: LR + XGB blend (varying weights)
    # ======================================================
    print(f"\n{'='*70}\n  B: LR(scaled) + XGB(raw) blend\n{'='*70}")
    xgb_params = dict(
        n_estimators=300, max_depth=3, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
        random_state=42, tree_method="hist",
    )
    for w_lr in [1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4]:
        _, bs_m, bs_w, bs_c = loso_brier_blend(X, y, is_w, season_arr, FEATS_8,
                                                 lr_C=0.1, xgb_params=xgb_params,
                                                 w_lr=w_lr)
        rows.append({"model": f"LR+XGB_w_lr={w_lr}", "men": bs_m, "women": bs_w, "combined": bs_c})
        print(f"  w_lr={w_lr}: combined={bs_c:.4f}  men={bs_m:.4f}  women={bs_w:.4f}")

    # ======================================================
    # Strategy C: Random Forest
    # ======================================================
    print(f"\n{'='*70}\n  C: Random Forest\n{'='*70}")
    for n_est, max_depth, min_leaf in [(300, None, 5), (500, None, 10), (500, 6, 5), (1000, 4, 5)]:
        def build(n=n_est, d=max_depth, m=min_leaf):
            return Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("rf", RandomForestClassifier(n_estimators=n, max_depth=d,
                                                min_samples_leaf=m,
                                                random_state=42, n_jobs=-1)),
            ])
        _, bs_m, bs_w, bs_c = loso_brier(X, y, is_w, season_arr, FEATS_8, build)
        tag = f"RF_n={n_est}_d={max_depth}_leaf={min_leaf}"
        rows.append({"model": tag, "men": bs_m, "women": bs_w, "combined": bs_c})
        print(f"  {tag}: combined={bs_c:.4f}")

    # ======================================================
    # Strategy D: Gradient Boosting
    # ======================================================
    print(f"\n{'='*70}\n  D: Gradient Boosting\n{'='*70}")
    for n_est, lr_, max_depth in [(300, 0.05, 3), (200, 0.03, 2), (500, 0.01, 3)]:
        def build(n=n_est, lr=lr_, d=max_depth):
            return Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("gb", GradientBoostingClassifier(n_estimators=n, learning_rate=lr,
                                                     max_depth=d, random_state=42)),
            ])
        _, bs_m, bs_w, bs_c = loso_brier(X, y, is_w, season_arr, FEATS_8, build)
        tag = f"GB_n={n_est}_lr={lr_}_d={max_depth}"
        rows.append({"model": tag, "men": bs_m, "women": bs_w, "combined": bs_c})
        print(f"  {tag}: combined={bs_c:.4f}")

    # ======================================================
    # Summary + 2026 evaluation of top 5
    # ======================================================
    df = pd.DataFrame(rows).sort_values("combined")
    df.to_csv("output/8feat_complex_loso.csv", index=False)
    print(f"\n{'='*70}\n  TOP 10 by LOSO combined Brier\n{'='*70}")
    print(df.head(10).to_string(index=False))

    # Apply top 5 to 2026
    print(f"\n{'='*70}\n  Apply top 5 LOSO models to 2026\n{'='*70}")
    X_2026, _, is_w_2026 = build_combined_features_2026(data_m, data_w)
    X_2026.loc[is_w_2026 == 1, massey_cols] = 0.0
    X_2026 = add_seed_pair(X_2026, is_w_2026.astype(int),
                            seed_lookup_m, seed_lookup_w,
                            {2026: base_full_m}, {2026: base_full_w},
                            base_full_m, base_full_w)

    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")

    def eval_2026(p_2026):
        pair_lk = {(int(r["TeamA"]), int(r["TeamB"])): float(p_2026[i])
                   for i, r in X_2026.reset_index(drop=True).iterrows()}
        yt_m, yp_m, yt_w, yp_w = [], [], [], []
        for _, g in actual_m.iterrows():
            w_, l = int(g["WTeamID"]), int(g["LTeamID"])
            key = (w_, l) if w_ < l else (l, w_)
            yt_m.append(1 if w_ < l else 0); yp_m.append(pair_lk.get(key, 0.5))
        for _, g in actual_w.iterrows():
            w_, l = int(g["WTeamID"]), int(g["LTeamID"])
            key = (w_, l) if w_ < l else (l, w_)
            yt_w.append(1 if w_ < l else 0); yp_w.append(pair_lk.get(key, 0.5))
        bs_m = brier_score_loss(yt_m, yp_m)
        bs_w = brier_score_loss(yt_w, yp_w)
        bs_c = (bs_m * 63 + bs_w * 63) / 126
        return bs_m, bs_w, bs_c

    Xtr = X[FEATS_8].apply(pd.to_numeric, errors="coerce").fillna(
        X[FEATS_8].apply(pd.to_numeric, errors="coerce").median()
    )
    X_2026_arr = X_2026[FEATS_8].apply(pd.to_numeric, errors="coerce").fillna(Xtr.median())

    print(f"\n  {'Model':<35s} {'LOSO':>8s} {'Men':>8s} {'Wom':>8s} {'Combined':>10s}")
    for i in range(min(10, len(df))):
        r = df.iloc[i]
        model_name = r["model"]
        # Rebuild model + predict 2026
        if model_name.startswith("LR_C="):
            C = float(model_name.split("=")[1])
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                ("lr", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
            ])
            pipe.fit(X[FEATS_8].apply(pd.to_numeric, errors="coerce"), y)
            p_2026 = pipe.predict_proba(X_2026[FEATS_8].apply(pd.to_numeric, errors="coerce"))[:, 1]
        elif model_name.startswith("poly2"):
            parts = model_name.split("_")
            io = parts[1].split("=")[1] == "True"
            C = float(parts[2].split("=")[1])
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("scl", StandardScaler()),
                ("poly", PolynomialFeatures(degree=2, interaction_only=io, include_bias=False)),
                ("lr", LogisticRegression(C=C, max_iter=3000, solver="lbfgs")),
            ])
            pipe.fit(X[FEATS_8].apply(pd.to_numeric, errors="coerce"), y)
            p_2026 = pipe.predict_proba(X_2026[FEATS_8].apply(pd.to_numeric, errors="coerce"))[:, 1]
        elif model_name.startswith("LR+XGB"):
            w_lr = float(model_name.split("=")[1])
            scaler = StandardScaler().fit(Xtr)
            lr = LogisticRegression(C=0.1, max_iter=2000, solver="lbfgs").fit(scaler.transform(Xtr), y)
            p_lr = lr.predict_proba(scaler.transform(X_2026_arr))[:, 1]
            xm = xgb.XGBClassifier(**xgb_params).fit(Xtr.values, y)
            p_xgb = xm.predict_proba(X_2026_arr.values)[:, 1]
            p_2026 = np.clip(w_lr * p_lr + (1 - w_lr) * p_xgb, 0.005, 0.995)
        elif model_name.startswith("RF"):
            parts = model_name.split("_")
            n = int(parts[1].split("=")[1])
            d_str = parts[2].split("=")[1]
            d = None if d_str == "None" else int(d_str)
            m = int(parts[3].split("=")[1])
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("rf", RandomForestClassifier(n_estimators=n, max_depth=d,
                                                min_samples_leaf=m,
                                                random_state=42, n_jobs=-1)),
            ])
            pipe.fit(X[FEATS_8].apply(pd.to_numeric, errors="coerce"), y)
            p_2026 = pipe.predict_proba(X_2026[FEATS_8].apply(pd.to_numeric, errors="coerce"))[:, 1]
        elif model_name.startswith("GB"):
            parts = model_name.split("_")
            n = int(parts[1].split("=")[1])
            lr_v = float(parts[2].split("=")[1])
            d = int(parts[3].split("=")[1])
            pipe = Pipeline([
                ("imp", SimpleImputer(strategy="median")),
                ("gb", GradientBoostingClassifier(n_estimators=n, learning_rate=lr_v,
                                                     max_depth=d, random_state=42)),
            ])
            pipe.fit(X[FEATS_8].apply(pd.to_numeric, errors="coerce"), y)
            p_2026 = pipe.predict_proba(X_2026[FEATS_8].apply(pd.to_numeric, errors="coerce"))[:, 1]
        else:
            continue
        bs_m_2, bs_w_2, bs_c_2 = eval_2026(np.clip(p_2026, 0.005, 0.995))
        print(f"  {model_name:<35s} {r['combined']:>8.4f} {bs_m_2:>8.4f} {bs_w_2:>8.4f} {bs_c_2:>10.4f}")


if __name__ == "__main__":
    main()
