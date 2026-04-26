"""
Aggressive simplification: can we drop features WITHIN the L1-selected 11
while improving (or maintaining) both LOSO and 2026?

Strategies tested:
  (A) Tighter L1 (smaller C) — natural sparsity progression
  (B) Manual minimal sets — curated subsets of the 11
  (C) PCA-composite of strength ratings — replace 4-5 ratings with 1 PC

All decisions on LOSO; report 2026 actual for transparency.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import _parse_seed_num
from scripts.build_womens_model import load_womens_data
from scripts.run_top3 import (
    build_combined_features, build_combined_features_2026,
    FEATURE_COLS as TOP3_FEATURES,
)
from src.seed_base_rate import compute_base_rate_table, lookup_p_a_wins


L1_11 = ["tempo_diff", "ft_rate_diff", "bart_net_diff", "bart_adjde_diff",
         "elo_diff", "elo_slope_diff", "srs_diff", "massey_mean_diff",
         "massey_min_diff", "harry_diff", "seed_pair_winrate"]


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


def loso_brier(X, y, is_w, season_arr, feats, C, penalty="l2"):
    p_oof = np.zeros(len(X))
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        if te.sum() == 0: continue
        if penalty == "l1":
            lr = LogisticRegression(C=C, penalty="l1", solver="liblinear",
                                     max_iter=3000, random_state=42)
        else:
            lr = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=2000)
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("lr", lr),
        ])
        Xtr = X.loc[tr, feats].apply(pd.to_numeric, errors="coerce")
        Xte = X.loc[te, feats].apply(pd.to_numeric, errors="coerce")
        pipe.fit(Xtr, y[tr])
        p_oof[te] = pipe.predict_proba(Xte)[:, 1]
    n_m = (is_w == 0).sum(); n_w_ = (is_w == 1).sum()
    bs_m = brier_score_loss(y[is_w == 0], p_oof[is_w == 0])
    bs_w = brier_score_loss(y[is_w == 1], p_oof[is_w == 1])
    bs_c = (bs_m * n_m + bs_w * n_w_) / (n_m + n_w_)
    return p_oof, bs_m, bs_w, bs_c


def get_l1_active(X, y, feats, C):
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("lr", LogisticRegression(C=C, penalty="l1", solver="liblinear",
                                   max_iter=3000, random_state=42)),
    ])
    Xt = X[feats].apply(pd.to_numeric, errors="coerce")
    pipe.fit(Xt, y)
    coefs = pipe.named_steps["lr"].coef_[0]
    return [f for f, c in zip(feats, coefs) if abs(c) > 1e-8], coefs


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]
    print("Loading data...")
    data_m = load_all_mens_data()
    data_w = load_womens_data()

    print("\nBuilding feature matrix (combined M+W)...")
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

    # Show feature correlations within L1_11 to identify redundancy
    print(f"\n{'='*70}\n  Feature correlations within L1's 11 features\n{'='*70}")
    corr = X[L1_11].apply(pd.to_numeric, errors="coerce").corr().round(2)
    print(corr.to_string())

    # ============================================================
    # Strategy A: tighter L1 to drop more features
    # ============================================================
    print(f"\n{'='*70}\n  Strategy A: Tighter L1 (more sparse)\n{'='*70}")
    rows = []
    for C in [0.02, 0.03, 0.05, 0.07, 0.1, 0.15]:
        # L1 LOSO Brier
        _, bs_m, bs_w, bs_c = loso_brier(X, y, is_w, season_arr, L1_11, C, penalty="l1")
        active, _ = get_l1_active(X, y, L1_11, C)
        rows.append({"strategy": f"L1_C={C}", "n": len(active),
                     "men": bs_m, "women": bs_w, "combined": bs_c, "feats": ",".join(active)})
        print(f"  L1 C={C}: combined={bs_c:.4f}  active={len(active)}  -> {active}")

    # ============================================================
    # Strategy B: Manual minimal sets (curated)
    # ============================================================
    print(f"\n{'='*70}\n  Strategy B: Manual minimal sets\n{'='*70}")
    manual_sets = {
        "minimal_3": ["seed_pair_winrate", "bart_net_diff", "harry_diff"],
        "minimal_4": ["seed_pair_winrate", "bart_net_diff", "harry_diff", "elo_diff"],
        "minimal_5": ["seed_pair_winrate", "bart_net_diff", "harry_diff", "elo_diff", "srs_diff"],
        "minimal_6": ["seed_pair_winrate", "bart_net_diff", "harry_diff", "elo_diff", "srs_diff", "massey_mean_diff"],
        "no_redundant_8": ["seed_pair_winrate", "bart_net_diff", "harry_diff", "elo_diff",
                            "elo_slope_diff", "srs_diff", "massey_mean_diff", "tempo_diff"],
    }
    for name, feats in manual_sets.items():
        for C in [0.05, 0.1, 0.3]:
            _, bs_m, bs_w, bs_c = loso_brier(X, y, is_w, season_arr, feats, C)
            rows.append({"strategy": f"{name}_C={C}", "n": len(feats),
                         "men": bs_m, "women": bs_w, "combined": bs_c, "feats": ",".join(feats)})
        # Also report best C for this set
        best_for_set = min([r for r in rows if r["strategy"].startswith(name+"_")],
                            key=lambda r: r["combined"])
        print(f"  {name} ({len(feats)} feats): best C={best_for_set['strategy'].split('=')[-1]}, "
              f"Brier={best_for_set['combined']:.4f}")

    # ============================================================
    # Strategy C: PCA composite of ratings
    # ============================================================
    print(f"\n{'='*70}\n  Strategy C: PCA composite of strength ratings\n{'='*70}")
    rating_feats = ["bart_net_diff", "bart_adjde_diff", "elo_diff", "srs_diff",
                    "massey_mean_diff", "massey_min_diff", "harry_diff"]
    other_feats = ["tempo_diff", "ft_rate_diff", "elo_slope_diff", "seed_pair_winrate"]

    # Build PCA on ratings (use full data; LOSO would be more rigorous but PCA is stable)
    Xr = X[rating_feats].apply(pd.to_numeric, errors="coerce").fillna(
        X[rating_feats].apply(pd.to_numeric, errors="coerce").median()
    )
    scaler = StandardScaler().fit(Xr)
    Xr_scaled = scaler.transform(Xr)
    for n_pc in [1, 2]:
        pca = PCA(n_components=n_pc).fit(Xr_scaled)
        pc = pca.transform(Xr_scaled)
        for i in range(n_pc):
            X[f"rating_pc{i+1}"] = pc[:, i]
        feats_pca = other_feats + [f"rating_pc{i+1}" for i in range(n_pc)]
        for C in [0.05, 0.1, 0.3]:
            _, bs_m, bs_w, bs_c = loso_brier(X, y, is_w, season_arr, feats_pca, C)
            rows.append({"strategy": f"PCA_{n_pc}_C={C}", "n": len(feats_pca),
                         "men": bs_m, "women": bs_w, "combined": bs_c, "feats": ",".join(feats_pca)})
        print(f"  PCA {n_pc} component(s) -> {len(feats_pca)} total features, "
              f"explained var: {pca.explained_variance_ratio_}")

    # ============================================================
    # Show top results
    # ============================================================
    df = pd.DataFrame(rows).sort_values("combined")
    df.to_csv("output/aggressive_simplify_loso.csv", index=False)
    print(f"\n{'='*70}\n  Top 15 by LOSO combined Brier\n{'='*70}")
    print(df.drop("feats", axis=1).head(15).to_string(index=False))

    # ============================================================
    # Apply top configurations to 2026
    # ============================================================
    print(f"\n{'='*70}\n  Apply top LOSO configs to 2026\n{'='*70}")
    X_2026, _, is_w_2026 = build_combined_features_2026(data_m, data_w)
    X_2026.loc[is_w_2026 == 1, massey_cols] = 0.0
    X_2026 = add_seed_pair(X_2026, is_w_2026.astype(int),
                            seed_lookup_m, seed_lookup_w,
                            {2026: base_full_m}, {2026: base_full_w},
                            base_full_m, base_full_w)
    # Add PCA features to 2026 too
    Xr_2026 = X_2026[rating_feats].apply(pd.to_numeric, errors="coerce").fillna(
        X_2026[rating_feats].apply(pd.to_numeric, errors="coerce").median()
    )
    Xr_2026_scaled = scaler.transform(Xr_2026)
    for n_pc in [1, 2]:
        pca = PCA(n_components=n_pc).fit(scaler.transform(Xr))
        pc_2026 = pca.transform(Xr_2026_scaled)
        for i in range(n_pc):
            X_2026[f"rating_pc{i+1}"] = pc_2026[:, i]

    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")

    def evaluate(feats, C, penalty="l2"):
        if penalty == "l1":
            lr = LogisticRegression(C=C, penalty="l1", solver="liblinear", max_iter=3000, random_state=42)
        else:
            lr = LogisticRegression(C=C, penalty="l2", solver="lbfgs", max_iter=2000)
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("lr", lr),
        ])
        pipe.fit(X[feats].apply(pd.to_numeric, errors="coerce"), y)
        p_2026 = pipe.predict_proba(X_2026[feats].apply(pd.to_numeric, errors="coerce"))[:, 1]
        p_2026 = np.clip(p_2026, 0.005, 0.995)
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
        bs_m_ = brier_score_loss(yt_m, yp_m)
        bs_w_ = brier_score_loss(yt_w, yp_w)
        bs_c_ = (bs_m_ * len(yt_m) + bs_w_ * len(yt_w)) / (len(yt_m) + len(yt_w))
        return bs_m_, bs_w_, bs_c_

    print(f"\n  Top configs evaluated on 2026 actual:")
    print(f"  {'Strategy':<35s} {'n':>3s} {'LOSO':>8s} {'Men':>8s} {'Wom':>8s} {'Combined':>10s}")
    for i in range(min(10, len(df))):
        r = df.iloc[i]
        feats = r["feats"].split(",")
        # Determine penalty/C
        s = r["strategy"]
        if s.startswith("L1_"):
            penalty = "l1"
            C = float(s.split("=")[-1])
        else:
            penalty = "l2"
            C = float(s.split("=")[-1])
        bs_m_2, bs_w_2, bs_c_2 = evaluate(feats, C, penalty=penalty)
        print(f"  {s:<35s} {r['n']:>3d} {r['combined']:>8.4f} "
              f"{bs_m_2:>8.4f} {bs_w_2:>8.4f} {bs_c_2:>10.4f}")


if __name__ == "__main__":
    main()
