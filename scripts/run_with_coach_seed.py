"""
Add coach features (men's) + seed-pair base rate to unified LR pipeline.

Tests three configurations:
  (A) Baseline: 25 features
  (B) +seed_pair_winrate: 26 features
  (C) +seed_pair_winrate + 5 coach features: 31 features (men's), 26 (women's auto-drop coach)
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
from src.coach_features import build_coach_features, get_team_coach_features


COACH_FEATS = ["coach_apps", "coach_winpct", "coach_pase",
               "coach_won_champ", "coach_school_yrs"]


def add_features(X, is_w, seasons_iter,
                 seed_lookup_m, seed_lookup_w,
                 base_table_per_season_m, base_table_per_season_w,
                 coach_data, base_table_full_m=None, base_table_full_w=None):
    """Add seed_pair_winrate + coach diff features to X (training, LOSO-aware)."""
    seed_pair_vals = []
    coach_apps_diff = []
    coach_winpct_diff = []
    coach_pase_diff = []
    coach_won_champ_diff = []
    coach_school_yrs_diff = []

    for _, r in X.iterrows():
        season = int(r["Season"])
        ta, tb = int(r["TeamA"]), int(r["TeamB"])
        if int(r["is_womens"]) == 0:
            seed_a = seed_lookup_m.get((season, ta), 17)
            seed_b = seed_lookup_m.get((season, tb), 17)
            tbl = base_table_per_season_m.get(season, base_table_full_m)
            seed_pair_vals.append(lookup_p_a_wins(tbl, seed_a, seed_b))
            # Coach (men's only)
            ca = get_team_coach_features(coach_data, season, ta)
            cb = get_team_coach_features(coach_data, season, tb)
            coach_apps_diff.append(ca["apps"] - cb["apps"])
            coach_winpct_diff.append(ca["winpct"] - cb["winpct"])
            coach_pase_diff.append(ca["pase"] - cb["pase"])
            coach_won_champ_diff.append(ca["won_champ"] - cb["won_champ"])
            coach_school_yrs_diff.append(ca["school_yrs"] - cb["school_yrs"])
        else:
            seed_a = seed_lookup_w.get((season, ta), 17)
            seed_b = seed_lookup_w.get((season, tb), 17)
            tbl = base_table_per_season_w.get(season, base_table_full_w)
            seed_pair_vals.append(lookup_p_a_wins(tbl, seed_a, seed_b))
            # No women's coach data: all 0
            coach_apps_diff.append(0); coach_winpct_diff.append(0)
            coach_pase_diff.append(0); coach_won_champ_diff.append(0)
            coach_school_yrs_diff.append(0)

    X = X.copy()
    X["seed_pair_winrate"] = seed_pair_vals
    X["coach_apps"] = coach_apps_diff
    X["coach_winpct"] = coach_winpct_diff
    X["coach_pase"] = coach_pase_diff
    X["coach_won_champ"] = coach_won_champ_diff
    X["coach_school_yrs"] = coach_school_yrs_diff
    return X


def build_lr(C=0.1):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
    ])


def loso_eval(X, y, is_w, season_arr, feats, C):
    p_oof = np.zeros(len(X))
    for is_g in [0, 1]:
        for s in np.unique(season_arr[is_w == is_g]):
            tr = (season_arr != s) & (is_w == is_g)
            te = (season_arr == s) & (is_w == is_g)
            if te.sum() == 0: continue
            feats_use = [c for c in feats if not (is_g == 1 and (c.startswith("massey_") or c.startswith("coach_")))]
            pipe = build_lr(C=C)
            Xtr = X.loc[tr, feats_use].apply(pd.to_numeric, errors="coerce")
            Xte = X.loc[te, feats_use].apply(pd.to_numeric, errors="coerce")
            pipe.fit(Xtr, y[tr])
            p_oof[te] = pipe.predict_proba(Xte)[:, 1]
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

    print("\nBuilding base feature matrix (top3 25 features)...")
    X, y, is_w = build_combined_features(data_m, data_w, seasons)
    season_arr = X["Season"].values

    # Seed lookups
    seeds_m = data_m["seeds"]; seeds_w = data_w["seeds"]
    seed_lookup_m = {(int(r["Season"]), int(r["TeamID"])): _parse_seed_num(r["Seed"])
                     for _, r in seeds_m.iterrows()}
    seed_lookup_w = {(int(r["Season"]), int(r["TeamID"])): _parse_seed_num(r["Seed"])
                     for _, r in seeds_w.iterrows()}

    # Base rate tables (LOSO-aware caches)
    print("\nBuilding LOSO-aware base rate tables...")
    tourney_m = data_m["tourney_compact"]; tourney_w = data_w["tourney_compact"]
    base_full_m = compute_base_rate_table(tourney_m, seeds_m)
    base_full_w = compute_base_rate_table(tourney_w, seeds_w)
    base_per_season_m = {s: compute_base_rate_table(tourney_m, seeds_m, exclude_season=s)
                         for s in seasons}
    base_per_season_w = {s: compute_base_rate_table(tourney_w, seeds_w, exclude_season=s)
                         for s in seasons}

    # Coach data (men's only)
    print("\nBuilding coach features...")
    coaches = pd.read_csv(DATA_DIR / "MTeamCoaches.csv")
    coach_data = build_coach_features(coaches, tourney_m, seeds_m)
    print(f"  Coaches with games: {len(coach_data['games_per_coach'])}")

    # Add features
    print("\nInjecting features into training matrix...")
    X = add_features(X, is_w, seasons, seed_lookup_m, seed_lookup_w,
                     base_per_season_m, base_per_season_w, coach_data,
                     base_full_m, base_full_w)

    # Define feature sets
    base_feats = [c for c in TOP3_FEATURES if c in X.columns]
    feats_with_seed = base_feats + ["seed_pair_winrate"]
    feats_with_seed_coach = feats_with_seed + COACH_FEATS

    print(f"\n  Baseline: {len(base_feats)} features")
    print(f"  +seed_pair: {len(feats_with_seed)} features")
    print(f"  +seed_pair + coach: {len(feats_with_seed_coach)} features (men's), women's drops coach")

    # ---- LOSO comparison ----
    print(f"\n{'='*70}\n  LOSO comparison\n{'='*70}")
    rows = []
    for feats, label in [
        (base_feats, "baseline_25"),
        (feats_with_seed, "+seed_pair_26"),
        (feats_with_seed_coach, "+seed_pair+coach_31"),
    ]:
        for C in [0.05, 0.1, 0.3, 0.5]:
            _, bs_m, bs_w, bs_c = loso_eval(X, y, is_w, season_arr, feats, C)
            rows.append({"variant": label, "C": C, "men": bs_m, "women": bs_w, "combined": bs_c})
            print(f"  {label:<25s} C={C}: men={bs_m:.4f}  women={bs_w:.4f}  combined={bs_c:.4f}")

    df = pd.DataFrame(rows).sort_values("combined")
    df.to_csv("output/coach_seed_loso.csv", index=False)
    print(f"\nTop 5:")
    print(df.head(5).to_string(index=False))

    # ---- Apply best to 2026 ----
    best = df.iloc[0]
    print(f"\nBest LOSO: {best['variant']}, C={best['C']} -> Brier {best['combined']:.4f}")

    print("\nBuilding 2026 features...")
    X_2026, _, is_w_2026 = build_combined_features_2026(data_m, data_w)
    X_2026 = add_features(X_2026, is_w_2026.astype(int), [2026],
                           seed_lookup_m, seed_lookup_w,
                           {2026: base_full_m}, {2026: base_full_w}, coach_data,
                           base_full_m, base_full_w)

    feats_chosen = {
        "baseline_25": base_feats,
        "+seed_pair_26": feats_with_seed,
        "+seed_pair+coach_31": feats_with_seed_coach,
    }[best["variant"]]
    feats_use_men = feats_chosen
    feats_use_wom = [c for c in feats_chosen if not (c.startswith("massey_") or c.startswith("coach_"))]

    pipe_m = build_lr(C=best["C"])
    pipe_m.fit(
        X.loc[is_w == 0, feats_use_men].apply(pd.to_numeric, errors="coerce"),
        y[is_w == 0]
    )
    pipe_w = build_lr(C=best["C"])
    pipe_w.fit(
        X.loc[is_w == 1, feats_use_wom].apply(pd.to_numeric, errors="coerce"),
        y[is_w == 1]
    )

    p_2026 = np.zeros(len(X_2026))
    p_2026[is_w_2026 == 0] = pipe_m.predict_proba(
        X_2026.loc[is_w_2026 == 0, feats_use_men].apply(pd.to_numeric, errors="coerce")
    )[:, 1]
    p_2026[is_w_2026 == 1] = pipe_w.predict_proba(
        X_2026.loc[is_w_2026 == 1, feats_use_wom].apply(pd.to_numeric, errors="coerce")
    )[:, 1]
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

    bs_m_2026, n_m_a = br(actual_m)
    bs_w_2026, n_w_a = br(actual_w)
    bs_c_2026 = (bs_m_2026 * n_m_a + bs_w_2026 * n_w_a) / (n_m_a + n_w_a)

    print(f"\n{'='*70}")
    print(f"  2026 RESULTS for {best['variant']} C={best['C']}")
    print(f"{'='*70}")
    print(f"  Men's:    {bs_m_2026:.4f}")
    print(f"  Women's:  {bs_w_2026:.4f}")
    print(f"  Combined: {bs_c_2026:.4f}")
    print(f"\n  vs unified baseline (no seed_pair, no coach):")
    print(f"    LOSO   {best['combined']:.4f} vs 0.1638  -> {best['combined']-0.1638:+.4f}")
    print(f"    2026   {bs_c_2026:.4f} vs 0.1261  -> {bs_c_2026-0.1261:+.4f}")

    # Also evaluate ALL three configs on 2026 (for paper transparency)
    print(f"\n{'='*70}\n  All configs on 2026 (transparency)\n{'='*70}")
    for variant, feats in [
        ("baseline_25", base_feats),
        ("+seed_pair_26", feats_with_seed),
        ("+seed_pair+coach_31", feats_with_seed_coach),
    ]:
        for C in [0.1]:
            feats_men = feats
            feats_wom = [c for c in feats if not (c.startswith("massey_") or c.startswith("coach_"))]
            pipe_m = build_lr(C=C).fit(
                X.loc[is_w == 0, feats_men].apply(pd.to_numeric, errors="coerce"),
                y[is_w == 0]
            )
            pipe_w = build_lr(C=C).fit(
                X.loc[is_w == 1, feats_wom].apply(pd.to_numeric, errors="coerce"),
                y[is_w == 1]
            )
            p_2026_v = np.zeros(len(X_2026))
            p_2026_v[is_w_2026 == 0] = pipe_m.predict_proba(
                X_2026.loc[is_w_2026 == 0, feats_men].apply(pd.to_numeric, errors="coerce")
            )[:, 1]
            p_2026_v[is_w_2026 == 1] = pipe_w.predict_proba(
                X_2026.loc[is_w_2026 == 1, feats_wom].apply(pd.to_numeric, errors="coerce")
            )[:, 1]
            p_2026_v = np.clip(p_2026_v, 0.005, 0.995)
            lk_v = {(int(r["TeamA"]), int(r["TeamB"])): float(p_2026_v[i])
                    for i, r in X_2026.reset_index(drop=True).iterrows()}
            yt_m, yp_m = [], []
            for _, g in actual_m.iterrows():
                w_, l = int(g["WTeamID"]), int(g["LTeamID"])
                key = (w_, l) if w_ < l else (l, w_)
                yt_m.append(1 if w_ < l else 0)
                yp_m.append(lk_v.get(key, 0.5))
            yt_w, yp_w = [], []
            for _, g in actual_w.iterrows():
                w_, l = int(g["WTeamID"]), int(g["LTeamID"])
                key = (w_, l) if w_ < l else (l, w_)
                yt_w.append(1 if w_ < l else 0)
                yp_w.append(lk_v.get(key, 0.5))
            bs_m_v = brier_score_loss(yt_m, yp_m)
            bs_w_v = brier_score_loss(yt_w, yp_w)
            bs_c_v = (bs_m_v * len(yt_m) + bs_w_v * len(yt_w)) / (len(yt_m) + len(yt_w))
            print(f"  {variant:<25s} C={C}: men={bs_m_v:.4f} women={bs_w_v:.4f} combined={bs_c_v:.4f}")


if __name__ == "__main__":
    main()
