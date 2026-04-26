"""
Final submission with Polymarket title-futures overlay.

Pipeline = our 8-feature LR+XGB 70/30 blend (sports-only baseline 0.1229)
         + Polymarket championship futures overlay using bracket DP

Hyperparameters for overlay are pre-registered from external 1st-place blog
post (ledmaster, "kill your darlings" Polymarket integration):
  alpha = 0.10
  team_offset_cap = +/- 0.10
  per_game_move_cap = +/- 0.03

These were chosen by an external author, not tuned by us on 2026 outcomes.

Apply only to men's (where Polymarket has 30 active markets and the actual
champion was the favorite). Women's retains the sports-only prediction.
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
from scripts.run_top3 import (
    build_combined_features, build_combined_features_2026,
    FEATURE_COLS as TOP3_FEATURES,
)
from src.seed_base_rate import compute_base_rate_table, lookup_p_a_wins
from scripts.run_polymarket_overlay import (
    monte_carlo_champ_probs, apply_overlay, build_bracket_structure, REGIONS,
)


FEATS_8 = [
    "seed_pair_winrate", "bart_net_diff", "harry_diff",
    "elo_diff", "elo_slope_diff", "srs_diff",
    "massey_mean_diff", "tempo_diff",
]
LR_C = 0.1
W_LR = 0.7
XGB_PARAMS = dict(
    n_estimators=300, max_depth=3, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
    reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
    tree_method="hist", random_state=42,
)

# Pre-registered Polymarket overlay hyperparameters (from external blog post)
ALPHA = 0.10
TEAM_CAP = 0.10
GAME_CAP = 0.03


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


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]
    print("Loading data...")
    data_m = load_all_mens_data()
    data_w = load_womens_data()

    print("\n[1/3] Building 8-feature 70/30 blend predictions for 2026...")
    X, y, is_w = build_combined_features(data_m, data_w, seasons)
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

    Xtr_full = X[FEATS_8].apply(pd.to_numeric, errors="coerce").fillna(
        X[FEATS_8].apply(pd.to_numeric, errors="coerce").median()
    )
    scaler = StandardScaler().fit(Xtr_full)
    lr_final = LogisticRegression(C=LR_C, max_iter=2000, solver="lbfgs")
    lr_final.fit(scaler.transform(Xtr_full), y)
    xgb_final = xgb.XGBClassifier(**XGB_PARAMS)
    xgb_final.fit(Xtr_full.values, y)

    X_2026, _, is_w_2026 = build_combined_features_2026(data_m, data_w)
    X_2026.loc[is_w_2026 == 1, massey_cols] = 0.0
    X_2026 = add_seed_pair(X_2026, is_w_2026.astype(int),
                            seed_lookup_m, seed_lookup_w,
                            {2026: base_full_m}, {2026: base_full_w},
                            base_full_m, base_full_w)
    X_2026_arr = X_2026[FEATS_8].apply(pd.to_numeric, errors="coerce").fillna(Xtr_full.median())

    p_lr_2026 = lr_final.predict_proba(scaler.transform(X_2026_arr))[:, 1]
    p_xgb_2026 = xgb_final.predict_proba(X_2026_arr.values)[:, 1]
    p_2026_sports = W_LR * p_lr_2026 + (1 - W_LR) * p_xgb_2026
    p_2026_sports = np.clip(p_2026_sports, 0.005, 0.995)

    pair_lk_sports = {(int(r["TeamA"]), int(r["TeamB"])): float(p_2026_sports[i])
                      for i, r in X_2026.reset_index(drop=True).iterrows()}

    # ============================================================
    # [2/3] Apply Polymarket title-futures overlay (men's)
    # ============================================================
    print("\n[2/3] Applying Polymarket overlay (men's)...")

    s2026 = seeds_m[seeds_m["Season"] == 2026].copy()
    s2026["SeedNum"] = s2026["Seed"].apply(_parse_seed_num)
    region_m = build_bracket_structure(s2026)

    all_m = set()
    for r in REGIONS: all_m.update(region_m[r])

    lk_m_tourney = {}
    for k, v in pair_lk_sports.items():
        a, b = k
        if a in all_m and b in all_m:
            lk_m_tourney[(a, b)] = v
            lk_m_tourney[(b, a)] = 1 - v

    print(f"  Tournament pairwise probs in lookup: {len(lk_m_tourney)//2}")

    p_model_champ = monte_carlo_champ_probs(lk_m_tourney, region_m, n_sims=100_000)
    pm_m = pd.read_csv("data/external/polymarket/champ_men_2026.csv")
    pm_market = dict(zip(pm_m["TeamID"].dropna().astype(int), pm_m["normalized_prob"]))
    print(f"  Polymarket markets mapped to TeamID: {len(pm_market)}")

    teams_data = data_m["teams"]
    tn = dict(zip(teams_data["TeamID"], teams_data["TeamName"]))
    print(f"\n  Top 5 model vs market champion probabilities:")
    print(f"  {'Team':<24s} {'Model':>8s} {'Market':>8s}")
    for tid, p_market in sorted(pm_market.items(), key=lambda x: -x[1])[:5]:
        p_model = p_model_champ.get(tid, 0.0)
        print(f"  {tn.get(tid, tid):<24s} {p_model:>8.3f} {p_market:>8.3f}")

    lk_m_overlay, deltas = apply_overlay(
        lk_m_tourney, p_model_champ, pm_market,
        alpha=ALPHA, team_offset_cap=TEAM_CAP, per_game_move_cap=GAME_CAP,
    )

    # Merge overlay back into full pair_lk
    pair_lk_final = dict(pair_lk_sports)
    for k, v in lk_m_overlay.items():
        if k[0] < k[1]:  # canonical order only
            pair_lk_final[k] = v

    # ============================================================
    # [3/3] Evaluate on 2026 actual
    # ============================================================
    print("\n[3/3] Evaluation on 2026 actual...")
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

    bs_m_sports, n_m = br(pair_lk_sports, actual_m)
    bs_w_sports, n_w = br(pair_lk_sports, actual_w)
    bs_c_sports = (bs_m_sports * n_m + bs_w_sports * n_w) / (n_m + n_w)

    bs_m_market, _ = br(pair_lk_final, actual_m)
    bs_w_market, _ = br(pair_lk_final, actual_w)
    bs_c_market = (bs_m_market * n_m + bs_w_market * n_w) / (n_m + n_w)

    print(f"\n{'='*70}\n  RESULTS\n{'='*70}")
    print(f"  Sports-only baseline:")
    print(f"    Men's:    {bs_m_sports:.4f}")
    print(f"    Women's:  {bs_w_sports:.4f}")
    print(f"    Combined: {bs_c_sports:.4f}")
    print(f"  + Polymarket overlay (alpha={ALPHA}, team_cap={TEAM_CAP}, game_cap={GAME_CAP}):")
    print(f"    Men's:    {bs_m_market:.4f}  (delta {bs_m_market - bs_m_sports:+.4f})")
    print(f"    Women's:  {bs_w_market:.4f}  (unchanged)")
    print(f"    Combined: {bs_c_market:.4f}  (delta {bs_c_market - bs_c_sports:+.4f})")

    print(f"\n  Comparison to Kaggle top:")
    print(f"    1st place: 0.1097")
    print(f"    2nd place: 0.1149")
    print(f"    3rd place: 0.1160")
    print(f"    Our sports-only: {bs_c_sports:.4f}")
    print(f"    Our + market:    {bs_c_market:.4f}")

    # ============================================================
    # Try women's Polymarket overlay
    # ============================================================
    print(f"\n{'='*70}\n  Women's Polymarket overlay (alpha={ALPHA})\n{'='*70}")
    pm_w_path = DATA_DIR / "external" / "polymarket" / "champ_women_2026.csv"
    if not pm_w_path.exists():
        print("  No women's Polymarket data; skip")
    else:
        s2026_w = seeds_w[seeds_w["Season"] == 2026].copy()
        s2026_w["SeedNum"] = s2026_w["Seed"].apply(_parse_seed_num)
        region_w = build_bracket_structure(s2026_w)
        if region_w is None:
            print("  Cannot build women's bracket")
        else:
            all_w = set()
            for r in REGIONS: all_w.update(region_w[r])
            lk_w_tourney = {}
            for k, v in pair_lk_sports.items():
                a, b = k
                if a in all_w and b in all_w:
                    lk_w_tourney[(a, b)] = v
                    lk_w_tourney[(b, a)] = 1 - v
            print(f"  W tournament pairwise probs: {len(lk_w_tourney)//2}")
            p_model_champ_w = monte_carlo_champ_probs(lk_w_tourney, region_w, n_sims=80_000)
            pm_w = pd.read_csv(pm_w_path)
            pm_market_w = dict(zip(pm_w["TeamID"].dropna().astype(int), pm_w["normalized_prob"]))
            print(f"  W Polymarket markets: {len(pm_market_w)}")
            lk_w_overlay, _ = apply_overlay(
                lk_w_tourney, p_model_champ_w, pm_market_w,
                alpha=ALPHA, team_offset_cap=TEAM_CAP, per_game_move_cap=GAME_CAP,
            )
            pair_lk_full2 = dict(pair_lk_final)
            for k, v in lk_w_overlay.items():
                if k[0] < k[1]:
                    pair_lk_full2[k] = v

            bs_m_full, _ = br(pair_lk_full2, actual_m)
            bs_w_full, _ = br(pair_lk_full2, actual_w)
            bs_c_full = (bs_m_full * n_m + bs_w_full * n_w) / (n_m + n_w)
            print(f"\n  + W Polymarket too:")
            print(f"    Men's:    {bs_m_full:.4f}  (unchanged)")
            print(f"    Women's:  {bs_w_full:.4f}  (delta {bs_w_full - bs_w_market:+.4f})")
            print(f"    Combined: {bs_c_full:.4f}  (delta {bs_c_full - bs_c_market:+.4f})")

            # Save best version
            if bs_c_full < bs_c_market:
                sub2 = pd.read_csv("output/submission_stage2.csv")
                sub2[["s_str", "ta_str", "tb_str"]] = sub2["ID"].str.split("_", expand=True)
                sub2["TeamA"] = sub2["ta_str"].astype(int); sub2["TeamB"] = sub2["tb_str"].astype(int)
                sub2["Pred"] = sub2.apply(
                    lambda r: pair_lk_full2.get((r["TeamA"], r["TeamB"]), float(r["Pred"])),
                    axis=1
                ).clip(0.005, 0.995)
                sub2[["ID", "Pred"]].to_csv("output/submission_stage2_FINAL_market_full.csv", index=False)
                print(f"\n  Saved output/submission_stage2_FINAL_market_full.csv (better than M-only)")

    # Save submission
    sub = pd.read_csv("output/submission_stage2.csv")
    sub[["s_str", "ta_str", "tb_str"]] = sub["ID"].str.split("_", expand=True)
    sub["TeamA"] = sub["ta_str"].astype(int); sub["TeamB"] = sub["tb_str"].astype(int)
    sub["Pred"] = sub.apply(
        lambda r: pair_lk_final.get((r["TeamA"], r["TeamB"]), float(r["Pred"])),
        axis=1
    ).clip(0.005, 0.995)
    sub[["ID", "Pred"]].to_csv("output/submission_stage2_FINAL_market.csv", index=False)
    print(f"\n  Saved output/submission_stage2_FINAL_market.csv")

    # Save summary
    pd.DataFrame([
        {"version": "sports_only", "men": bs_m_sports, "women": bs_w_sports, "combined": bs_c_sports},
        {"version": "with_polymarket", "men": bs_m_market, "women": bs_w_market, "combined": bs_c_market},
    ]).to_csv("output/FINAL_with_market_summary.csv", index=False)


if __name__ == "__main__":
    main()
