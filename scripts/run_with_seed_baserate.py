"""
Add per-seed-pair historical base rate as a single feature to the unified
pipeline and test if it improves both LOSO and 2026.

Base rate computed LOSO-aware: when holding out season s, base rate uses
all OTHER seasons (1985 - 2025 excl. s).
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
from scripts.run_top3 import build_combined_features, build_combined_features_2026
from src.seed_base_rate import compute_base_rate_table, lookup_p_a_wins


# Same 25-feature set as unified pipeline + 1 new feature
EXTRA_FEAT = "seed_pair_winrate"


def add_seed_pair_feature(X: pd.DataFrame, base_rate_table: dict, seeds_df: pd.DataFrame,
                          season: int | None = None) -> pd.DataFrame:
    """Add `seed_pair_winrate` column to X using base_rate_table.

    Looks up each (TeamA, TeamB)'s seeds and returns P(A wins) per row.
    """
    seed_map = {}
    if season is not None:
        sdf = seeds_df[seeds_df["Season"] == season]
    else:
        sdf = seeds_df
    for _, r in sdf.iterrows():
        seed_map[(int(r["Season"]), int(r["TeamID"]))] = _parse_seed_num(r["Seed"])

    out = X.copy()
    vals = []
    for _, r in out.iterrows():
        s_a = seed_map.get((int(r["Season"]), int(r["TeamA"])), 17)
        s_b = seed_map.get((int(r["Season"]), int(r["TeamB"])), 17)
        vals.append(lookup_p_a_wins(base_rate_table, s_a, s_b))
    out[EXTRA_FEAT] = vals
    return out


def build_lr(C=0.1):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
    ])


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]
    print("Loading data...")
    data_m = load_all_mens_data()
    data_w = load_womens_data()

    # Build pooled M+W tournament histories for LOSO-aware base rate
    tourney_m = data_m["tourney_compact"]
    tourney_w = data_w["tourney_compact"]
    seeds_m = data_m["seeds"]
    seeds_w = data_w["seeds"]

    print("\nBuilding feature matrix...")
    X, y, is_w = build_combined_features(data_m, data_w, seasons)
    season_arr = X["Season"].values
    n_m = (is_w == 0).sum(); n_w = (is_w == 1).sum()

    # ---- Compute LOSO-aware base rate table ----
    # Per-gender base rate using gender-specific tournament history
    # Avoid leak: when holding out season s, exclude season s from base rate
    # Build all-seasons table once; then build per-season excluded tables
    all_table_m = compute_base_rate_table(tourney_m, seeds_m, exclude_season=None)
    all_table_w = compute_base_rate_table(tourney_w, seeds_w, exclude_season=None)

    print(f"\n  Men's base rate pairs: {len(all_table_m)}")
    print(f"  Women's base rate pairs: {len(all_table_w)}")

    # ---- Inject feature into X (training data) ----
    # For each row, look up base rate using ALL-OTHER-SEASONS table for its season
    # This is LOSO-correct
    print("\nInjecting LOSO-aware base rate feature...")

    # Pre-build seed maps per season
    seed_lookup_m = {}
    for _, r in seeds_m.iterrows():
        seed_lookup_m[(int(r["Season"]), int(r["TeamID"]))] = _parse_seed_num(r["Seed"])
    seed_lookup_w = {}
    for _, r in seeds_w.iterrows():
        seed_lookup_w[(int(r["Season"]), int(r["TeamID"]))] = _parse_seed_num(r["Seed"])

    # For LOSO honesty: cache table per (gender, exclude_season)
    table_cache_m: dict[int, dict] = {}
    table_cache_w: dict[int, dict] = {}
    for s in seasons:
        table_cache_m[s] = compute_base_rate_table(tourney_m, seeds_m, exclude_season=s)
        table_cache_w[s] = compute_base_rate_table(tourney_w, seeds_w, exclude_season=s)

    p_seed_pair = []
    for _, r in X.iterrows():
        season = int(r["Season"])
        if int(r["is_womens"]) == 0:
            seed_a = seed_lookup_m.get((season, int(r["TeamA"])), 17)
            seed_b = seed_lookup_m.get((season, int(r["TeamB"])), 17)
            tbl = table_cache_m[season]
        else:
            seed_a = seed_lookup_w.get((season, int(r["TeamA"])), 17)
            seed_b = seed_lookup_w.get((season, int(r["TeamB"])), 17)
            tbl = table_cache_w[season]
        p_seed_pair.append(lookup_p_a_wins(tbl, seed_a, seed_b))

    X["seed_pair_winrate"] = p_seed_pair
    print(f"  Done. Feature distribution:")
    print(f"    min={X['seed_pair_winrate'].min():.3f}  "
          f"max={X['seed_pair_winrate'].max():.3f}  "
          f"mean={X['seed_pair_winrate'].mean():.3f}")

    # ---- LOSO baseline (no extra feature, original 25 features) ----
    from scripts.run_top3 import FEATURE_COLS as TOP3_FEATURES
    base_feats = [c for c in TOP3_FEATURES if c in X.columns]
    new_feats = base_feats + [EXTRA_FEAT]
    print(f"\n  Base features: {len(base_feats)}, with seed_pair: {len(new_feats)}")

    def loso_eval(feats, C=0.1):
        p_oof = np.zeros(len(X))
        for is_g in [0, 1]:
            for s in np.unique(season_arr[is_w == is_g]):
                tr = (season_arr != s) & (is_w == is_g)
                te = (season_arr == s) & (is_w == is_g)
                if te.sum() == 0: continue
                feats_use = [c for c in feats if not (is_g == 1 and c.startswith("massey_"))]
                pipe = build_lr(C=C)
                Xtr = X.loc[tr, feats_use].apply(pd.to_numeric, errors="coerce")
                Xte = X.loc[te, feats_use].apply(pd.to_numeric, errors="coerce")
                pipe.fit(Xtr, y[tr])
                p_oof[te] = pipe.predict_proba(Xte)[:, 1]
        bs_m = brier_score_loss(y[is_w == 0], p_oof[is_w == 0])
        bs_w = brier_score_loss(y[is_w == 1], p_oof[is_w == 1])
        bs_c = (bs_m * n_m + bs_w * n_w) / (n_m + n_w)
        return p_oof, bs_m, bs_w, bs_c

    print(f"\n{'='*70}\n  LOSO comparison\n{'='*70}")
    rows = []
    for feats, label in [(base_feats, "baseline_25_features"),
                          (new_feats, "+ seed_pair_winrate (26 features)")]:
        for C in [0.05, 0.1, 0.3, 0.5, 1.0]:
            _, bs_m, bs_w, bs_c = loso_eval(feats, C=C)
            rows.append({"variant": label, "C": C, "men": bs_m, "women": bs_w, "combined": bs_c})
            print(f"  {label:<35s} C={C}: men={bs_m:.4f}  women={bs_w:.4f}  combined={bs_c:.4f}")

    df = pd.DataFrame(rows).sort_values("combined")
    print(f"\nTop 5:")
    print(df.head(5).to_string(index=False))

    df.to_csv("output/seed_pair_loso.csv", index=False)

    # ---- Apply best variant to 2026 ----
    best = df.iloc[0]
    use_seed_pair = ("seed_pair_winrate" in best["variant"])
    best_C = best["C"]
    print(f"\nBest variant: {best['variant']}, C={best_C}")
    print(f"  LOSO Brier: {best['combined']:.4f}")

    print("\nBuilding 2026 features...")
    X_2026, _, is_w_2026 = build_combined_features_2026(data_m, data_w)

    # Inject seed_pair feature for 2026
    p_seed_pair_2026 = []
    for _, r in X_2026.iterrows():
        if int(r["is_womens"]) == 0:
            seed_a = seed_lookup_m.get((2026, int(r["TeamA"])), 17)
            seed_b = seed_lookup_m.get((2026, int(r["TeamB"])), 17)
            tbl = all_table_m  # use all-history table for 2026 prediction
        else:
            seed_a = seed_lookup_w.get((2026, int(r["TeamA"])), 17)
            seed_b = seed_lookup_w.get((2026, int(r["TeamB"])), 17)
            tbl = all_table_w
        p_seed_pair_2026.append(lookup_p_a_wins(tbl, seed_a, seed_b))
    X_2026["seed_pair_winrate"] = p_seed_pair_2026

    # Train final per-gender models on full 2014-2025
    feats_use_men = [c for c in (new_feats if use_seed_pair else base_feats)]
    feats_use_wom = [c for c in feats_use_men if not c.startswith("massey_")]
    pipe_m = build_lr(C=best_C)
    pipe_m.fit(
        X.loc[is_w == 0, feats_use_men].apply(pd.to_numeric, errors="coerce"),
        y[is_w == 0]
    )
    pipe_w = build_lr(C=best_C)
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

    # Build pair lookup
    pair_lk = {(int(r["TeamA"]), int(r["TeamB"])): float(p_2026[i])
               for i, r in X_2026.reset_index(drop=True).iterrows()}

    # Evaluate
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
    print(f"  2026 RESULTS (best LOSO config: {best['variant']}, C={best_C})")
    print(f"{'='*70}")
    print(f"  Men's:    {bs_m_2026:.4f}")
    print(f"  Women's:  {bs_w_2026:.4f}")
    print(f"  Combined: {bs_c_2026:.4f}")
    print(f"\n  Comparison to unified baseline (no seed_pair):")
    print(f"  LOSO   {best['combined']:.4f} vs 0.1638  -> {best['combined']-0.1638:+.4f}")
    print(f"  2026   {bs_c_2026:.4f} vs 0.1261  -> {bs_c_2026-0.1261:+.4f}")


if __name__ == "__main__":
    main()
