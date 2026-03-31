"""
Generate Kaggle submission CSV and 2026 bracket predictions.

Covers Phase 2 (Kaggle submission) and Phase 4 (2026 predictions + trading).

Usage:
    python scripts/generate_kaggle_submission.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import (
    make_build_features_fn, build_rating_features_for_season,
    build_efficiency_for_season, build_four_factors_for_season,
    build_momentum_for_season, _parse_seed_num, _ID_COLS,
)
from src.models import MultiFeatureLogistic
from src.kalshi_trader import SyntheticMarketPrices, TradingStrategy


def build_submission_features(data: dict, season: int, submission_path: str):
    """
    Build features for ALL matchups in the Kaggle submission file.
    Tournament-team pairs get full features; others get NaN (predicted as 0.5).
    """
    # Read submission template
    sub = pd.read_csv(submission_path)
    sub[["season_str", "TeamA_str", "TeamB_str"]] = sub["ID"].str.split("_", expand=True)
    sub["Season"] = sub["season_str"].astype(int)
    sub["TeamA"] = sub["TeamA_str"].astype(int)
    sub["TeamB"] = sub["TeamB_str"].astype(int)
    sub = sub[sub["Season"] == season].copy()

    # Get tournament team IDs
    seeds = data["seeds"]
    tourney_seeds = seeds[seeds["Season"] == season].copy()
    tourney_seeds["SeedNum"] = tourney_seeds["Seed"].apply(_parse_seed_num)
    seed_map = dict(zip(tourney_seeds["TeamID"], tourney_seeds["SeedNum"]))
    tourney_teams = set(seed_map.keys())

    # Identify tournament-vs-tournament pairs
    sub["is_tourney"] = sub.apply(
        lambda r: r["TeamA"] in tourney_teams and r["TeamB"] in tourney_teams, axis=1
    )
    tourney_pairs = sub[sub["is_tourney"]].copy()
    print(f"  Total submission rows: {len(sub)}")
    print(f"  Tournament-vs-tournament pairs: {len(tourney_pairs)}")

    if len(tourney_pairs) == 0:
        return sub[["ID"]], np.full(len(sub), 0.5), pd.DataFrame()

    # Build features for tournament pairs
    ratings = build_rating_features_for_season(data, season)
    efficiency = build_efficiency_for_season(data, season)
    four_factors = build_four_factors_for_season(data, season)
    all_team_ids = list(tourney_teams)
    from src.features import build_momentum_features
    momentum_data = {}
    for tid in all_team_ids:
        momentum_data[tid] = build_momentum_features(
            data["regular_compact"], tid, season, 10
        )

    # Check for Barttorvik data
    bart = None
    try:
        bart_path = DATA_DIR / "external" / f"barttorvik_{season}.csv"
        if bart_path.exists():
            bart = pd.read_csv(bart_path).set_index("TeamID")
    except:
        pass

    # Check for coach data
    coach_exp = {}
    if "coaches" in data:
        coaches = data["coaches"]
        for tid in all_team_ids:
            tc = coaches[(coaches["Season"] <= season) & (coaches["TeamID"] == tid)]
            coach_exp[tid] = len(tc)

    features_list = []
    for _, row in tourney_pairs.iterrows():
        ta, tb = row["TeamA"], row["TeamB"]
        feat = {
            "Season": season, "TeamA": ta, "TeamB": tb,
            "seed_diff": seed_map.get(ta, 16) - seed_map.get(tb, 16),
            "seed_A": seed_map.get(ta, 16),
            "seed_B": seed_map.get(tb, 16),
        }

        # Ratings
        for sys_name in ["POM", "SAG", "MOR", "DOL", "COL", "AP", "USA", "WOL", "RPI"]:
            ra = ratings.loc[ta, sys_name] if (not ratings.empty and ta in ratings.index and sys_name in ratings.columns) else np.nan
            rb = ratings.loc[tb, sys_name] if (not ratings.empty and tb in ratings.index and sys_name in ratings.columns) else np.nan
            feat[f"rank_diff_{sys_name}"] = ra - rb

        # Efficiency
        for col in ["off_eff", "def_eff", "net_eff", "tempo"]:
            va = efficiency.loc[ta, col] if (not efficiency.empty and ta in efficiency.index) else np.nan
            vb = efficiency.loc[tb, col] if (not efficiency.empty and tb in efficiency.index) else np.nan
            feat[f"{col}_diff"] = va - vb
            if col == "net_eff":
                feat[f"{col}_A"] = va
                feat[f"{col}_B"] = vb

        # Four Factors
        ff_cols = ["efg_pct", "to_pct", "or_pct", "ft_rate",
                   "opp_efg_pct", "opp_to_pct", "opp_or_pct", "opp_ft_rate"]
        for col in ff_cols:
            va = four_factors.loc[ta, col] if (not four_factors.empty and ta in four_factors.index) else np.nan
            vb = four_factors.loc[tb, col] if (not four_factors.empty and tb in four_factors.index) else np.nan
            feat[f"{col}_diff"] = va - vb

        # Barttorvik (match exact column names from pipeline)
        if bart is not None:
            bart_mapping = {
                "AdjOE": "bart_adjoe_diff",
                "AdjDE": "bart_adjde_diff",
                "NetRtg": "bart_net_diff",
                "Barthag": "bart_barthag_diff",
                "AdjTempo": "bart_tempo_diff",
            }
            for src_col, feat_name in bart_mapping.items():
                va = bart.loc[ta, src_col] if ta in bart.index else np.nan
                vb = bart.loc[tb, src_col] if tb in bart.index else np.nan
                feat[feat_name] = va - vb

        # Momentum
        ma = momentum_data.get(ta, {"momentum_win_pct": 0.5, "momentum_avg_margin": 0.0})
        mb = momentum_data.get(tb, {"momentum_win_pct": 0.5, "momentum_avg_margin": 0.0})
        feat["momentum_winpct_diff"] = ma.get("momentum_win_pct", 0.5) - mb.get("momentum_win_pct", 0.5)
        feat["momentum_margin_diff"] = ma.get("momentum_avg_margin", 0.0) - mb.get("momentum_avg_margin", 0.0)

        # Coach (match pipeline column names)
        feat["coach_tourney_apps_diff"] = coach_exp.get(ta, 0) - coach_exp.get(tb, 0)
        feat["coach_seasons_diff"] = coach_exp.get(ta, 0) - coach_exp.get(tb, 0)

        features_list.append(feat)

    X_tourney = pd.DataFrame(features_list)
    return sub, X_tourney, tourney_pairs


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)

    # Train model on all available historical data
    train_seasons = [s for s in range(2014, 2026) if s != 2020]
    print(f"\nTraining MultiFeatureLogistic on {len(train_seasons)} seasons...")
    X_train, y_train = build_fn(train_seasons)
    model = MultiFeatureLogistic(C=0.5)
    model.fit(X_train, y_train)
    print(f"  Trained on {len(X_train)} games, {len(model.cols_)} features")

    # Build 2026 submission features
    print("\nBuilding 2026 submission features...")
    sub_path = str(DATA_DIR / "SampleSubmissionStage2.csv")
    sub_df, X_tourney, tourney_pairs = build_submission_features(data, 2026, sub_path)

    # Predict tournament pairs
    print("\nGenerating predictions...")
    tourney_probs = model.predict_proba(X_tourney)[:, 1]

    # Build full submission
    sub_df["Pred"] = 0.5  # default for non-tournament pairs
    tourney_idx = sub_df[sub_df.get("is_tourney", False) == True].index
    sub_df.loc[tourney_idx, "Pred"] = tourney_probs

    # Clip to avoid extreme predictions
    sub_df["Pred"] = sub_df["Pred"].clip(0.01, 0.99)

    # Save submission
    output = sub_df[["ID", "Pred"]]
    output.to_csv("output/submission_stage2.csv", index=False)
    print(f"\nSaved submission: {len(output)} rows to output/submission_stage2.csv")

    # === 2026 Bracket Predictions ===
    print("\n" + "=" * 60)
    print("  2026 TOURNAMENT PREDICTIONS")
    print("=" * 60)

    seeds = data["seeds"]
    s2026 = seeds[seeds["Season"] == 2026].copy()
    s2026["SeedNum"] = s2026["Seed"].apply(_parse_seed_num)
    seed_map = dict(zip(s2026["TeamID"], s2026["SeedNum"]))
    seed_str_map = dict(zip(s2026["TeamID"], s2026["Seed"]))
    teams = data["teams"]
    team_names = dict(zip(teams["TeamID"], teams["TeamName"]))

    # Show first-round matchup predictions (1v16, 2v15, etc.)
    slots = data.get("slots")
    if slots is not None:
        s2026_slots = slots[slots["Season"] == 2026]

    # Find matchups by seed pairing
    print(f"\n{'Matchup':<45} {'P(Higher Seed)':<15} {'Upset Prob'}")
    print("-" * 70)

    # Pair up seeds: W01vW16, W02vW15, etc.
    regions = ["W", "X", "Y", "Z"]
    for region in regions:
        for hi_seed in range(1, 9):
            lo_seed = 17 - hi_seed
            hi_teams = s2026[s2026["Seed"].str.startswith(f"{region}{hi_seed:02d}")]
            lo_teams = s2026[s2026["Seed"].str.startswith(f"{region}{lo_seed:02d}")]

            if hi_teams.empty or lo_teams.empty:
                continue

            hi_id = hi_teams.iloc[0]["TeamID"]
            lo_id = lo_teams.iloc[0]["TeamID"]
            hi_name = team_names.get(hi_id, str(hi_id))
            lo_name = team_names.get(lo_id, str(lo_id))

            # Find prediction (canonical order: lower ID first)
            if hi_id < lo_id:
                mask = (X_tourney["TeamA"] == hi_id) & (X_tourney["TeamB"] == lo_id)
                if mask.any():
                    idx = mask.idxmax()
                    p_hi = tourney_probs[X_tourney.index.get_loc(idx)]
                else:
                    p_hi = 0.5
            else:
                mask = (X_tourney["TeamA"] == lo_id) & (X_tourney["TeamB"] == hi_id)
                if mask.any():
                    idx = mask.idxmax()
                    p_hi = 1 - tourney_probs[X_tourney.index.get_loc(idx)]
                else:
                    p_hi = 0.5

            upset_prob = 1 - p_hi
            matchup = f"({hi_seed}) {hi_name} vs ({lo_seed}) {lo_name}"
            print(f"{matchup:<45} {p_hi:.3f}           {upset_prob:.3f}")

    # === 2026 Trading Signals ===
    print("\n" + "=" * 60)
    print("  2026 TRADING SIGNALS (DRY RUN)")
    print("=" * 60)

    market_gen = SyntheticMarketPrices(data, noise_std=0.04, market_alpha_std=0.03)
    strategy = TradingStrategy(threshold=0.03, kelly_fraction=0.25,
                               max_per_game=0.05, max_total_exposure=0.20,
                               max_contracts=100, spread=0.03)

    # Generate synthetic market prices for tournament pairs
    from src.pipeline import build_tourney_matchups
    matchups_2026 = build_tourney_matchups(data, 2026)

    if matchups_2026.empty:
        # No actual tournament results yet, use seed-paired matchups
        print("\nNo 2026 tournament results in data. Using first-round seed pairings.")

        # Build first-round matchup features from X_tourney
        # Just use all tourney pairs and show top signals
        p_market = market_gen.generate_for_matchups(
            pd.DataFrame({"SeedA": X_tourney["seed_A"], "SeedB": X_tourney["seed_B"]}),
            p_model=tourney_probs, seed=2026
        )

        game_ids = list(range(len(tourney_probs)))
        signals = strategy.generate_signals(tourney_probs, p_market, game_ids, 10000)

        if signals.empty:
            print("No signals generated at current threshold.")
        else:
            print(f"\n{len(signals)} trading signals generated:\n")
            for _, sig in signals.iterrows():
                idx = int(sig["game_id"])
                ta = int(X_tourney.iloc[idx]["TeamA"])
                tb = int(X_tourney.iloc[idx]["TeamB"])
                name_a = team_names.get(ta, str(ta))
                name_b = team_names.get(tb, str(tb))
                print(f"  {sig['side'].upper():3s} {sig['contracts']:.0f}x "
                      f"{name_a} vs {name_b} "
                      f"@ {sig['price']:.3f} | edge={sig['edge']:+.3f} "
                      f"| size=${sig['position_size']:.0f}")
    else:
        print(f"\n2026 tournament has {len(matchups_2026)} games in data.")

    print("\nDone.")


if __name__ == "__main__":
    main()
