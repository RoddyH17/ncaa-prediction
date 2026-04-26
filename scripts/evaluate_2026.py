"""
Evaluate our 2026 predictions and strategies against actual tournament results.

Computes:
  1. Brier score on actual 2026 games
  2. ESPN bracket score for chalk vs MAP vs sample strategies
  3. Realized P&L of trading dry-run signals against actual outcomes
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import make_build_features_fn, _parse_seed_num
from src.models import MultiFeatureLogistic, BarttovikLogistic, KenPomLogistic, SeedLogistic
from src.kalshi_trader import SyntheticMarketPrices, TradingStrategy


ROUND_POINTS = [10, 20, 40, 80, 160, 320]


def main():
    print("Loading data...")
    data = load_all_mens_data()
    actual = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    print(f"  Actual 2026 games: {len(actual)}")

    teams = data["teams"]
    team_names = dict(zip(teams["TeamID"], teams["TeamName"]))
    seeds = data["seeds"]
    s2026 = seeds[seeds["Season"] == 2026].copy()
    s2026["SeedNum"] = s2026["Seed"].apply(_parse_seed_num)
    seed_map = dict(zip(s2026["TeamID"], s2026["SeedNum"]))

    # Train models on 2014-2025
    build_fn = make_build_features_fn(data)
    train_seasons = [s for s in range(2014, 2026) if s != 2020]
    print(f"\nTraining models on {len(train_seasons)} seasons...")
    X_train, y_train = build_fn(train_seasons)

    models = {
        "Seed Logistic":         SeedLogistic(),
        "KenPom Logistic":       KenPomLogistic(),
        "Barttorvik Logistic":   BarttovikLogistic(),
        "Multi-Feature Logistic": MultiFeatureLogistic(C=0.5),
    }
    for name, m in models.items():
        m.fit(X_train, y_train)

    # Build all 2026 tournament-pair features
    from scripts.generate_kaggle_submission import build_submission_features
    sub_path = str(DATA_DIR / "SampleSubmissionStage2.csv")
    _, X_tourney, _ = build_submission_features(data, 2026, sub_path)

    # Build prediction lookup: (team_a, team_b) -> P(team_a wins)
    pred_lookups = {}
    for name, m in models.items():
        p = m.predict_proba(X_tourney)[:, 1]
        lookup = {}
        for i, (_, row) in enumerate(X_tourney.iterrows()):
            ta, tb = int(row["TeamA"]), int(row["TeamB"])
            lookup[(ta, tb)] = p[i]
            lookup[(tb, ta)] = 1 - p[i]
        pred_lookups[name] = lookup

    # === 1. Brier on actual games ===
    print(f"\n{'='*60}\n  1. BRIER SCORE ON ACTUAL 2026 GAMES\n{'='*60}")
    rows = []
    for name, lookup in pred_lookups.items():
        y_true = []
        y_pred = []
        for _, g in actual.iterrows():
            w, l = g["WTeamID"], g["LTeamID"]
            if w < l:
                p_a_wins = lookup.get((w, l), 0.5)
                y_true.append(1)
            else:
                p_a_wins = lookup.get((l, w), 0.5)
                y_true.append(0)
            y_pred.append(p_a_wins)
        bs = brier_score_loss(y_true, y_pred)
        rows.append({"model": name, "brier_2026": bs, "n_games": len(y_true)})
        print(f"  {name:<28s} Brier = {bs:.4f}")
    brier_df = pd.DataFrame(rows)
    brier_df.to_csv("output/eval_2026_brier.csv", index=False)

    # === 2. Bracket scoring ===
    print(f"\n{'='*60}\n  2. BRACKET STRATEGY EVALUATION\n{'='*60}")

    # Build the actual "true" bracket from results.
    # Sort actual games by number of "rounds played" — round 1 has 32 games, round 2 has 16, etc.
    # Easiest: group by season and sort by score sum (proxy for round) — but DayNum is missing.
    # Use seed-based grouping: round 1 winners are the 32 winners with TeamID still in tournament,
    # round 2 winners have played 1 game already, etc.
    # Simpler approach: use the games and rebuild round-by-round
    actual_winners = set(actual["WTeamID"].tolist())  # all winners (32+16+8+4+2+1=63)
    actual_losers_per_round = []  # 32, 16, 8, 4, 2, 1 losers per round

    # Build round 1: each first-round game is a 1v16, 2v15, 8v9, etc. matchup
    # Identify first-round games: those where one team is a higher seed (1-8) playing seed (9-16)
    r1_games = actual[
        ((actual["WSeed"] + actual["LSeed"]) == 17) |
        # Handle cases where seeds don't sum to 17 (play-in games)
        False
    ]
    # Simpler: count games each team played. Round 1 losers played 1 game.
    games_played = {}
    for _, g in actual.iterrows():
        for tid in [g["WTeamID"], g["LTeamID"]]:
            games_played[tid] = games_played.get(tid, 0) + 1

    # Round losers per number-of-games-played:
    # Lost in R1 → played 1 game, R2 → 2, S16 → 3, E8 → 4, F4 → 5, Final → 6
    # Champion played 6 games and won all
    rounds_lost_at = {}
    for _, g in actual.iterrows():
        tid = g["LTeamID"]
        rounds_lost_at[tid] = games_played[tid]

    # Build true bracket: list of winners per round
    # Round k winners = teams that played at least k+1 games (or won the final)
    true_bracket = []
    for round_idx in range(6):
        n_played = round_idx + 1
        # Winners of round (round_idx+1) advanced past it, so they played > n_played games
        # OR they won the championship (played 6 games, won 6)
        round_winners = [
            tid for tid in games_played
            if games_played[tid] > n_played or
               (games_played[tid] == 6 and tid not in rounds_lost_at)
        ]
        true_bracket.append(round_winners)

    # Champion is the team that won 6 games and never lost
    champ = [tid for tid in games_played
             if games_played[tid] == 6 and tid not in rounds_lost_at]
    print(f"  2026 Champion: {team_names.get(champ[0], champ[0])} ({seed_map.get(champ[0])})") if champ else None
    print(f"  R1 winners: {len(true_bracket[0])}, R2: {len(true_bracket[1])}, "
          f"S16: {len(true_bracket[2])}, E8: {len(true_bracket[3])}, "
          f"F4: {len(true_bracket[4])}, Final: {len(true_bracket[5])}")

    # Simulate strategies
    def fill_bracket(p_func, mode, rng=None):
        """Fill a bracket using strategy. Returns winners per round."""
        regions = ["W", "X", "Y", "Z"]
        first_pairs = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
        bracket = []
        region_winners = {}
        for region in regions:
            wins = []
            for hi, lo in first_pairs:
                hi_keys = [k for k in s2026["Seed"] if k.startswith(f"{region}{hi:02d}")]
                lo_keys = [k for k in s2026["Seed"] if k.startswith(f"{region}{lo:02d}")]
                if not hi_keys or not lo_keys:
                    continue
                hi_team = s2026[s2026["Seed"] == hi_keys[0]]["TeamID"].iloc[0]
                lo_team = s2026[s2026["Seed"] == lo_keys[0]]["TeamID"].iloc[0]
                if mode == "chalk":
                    winner = hi_team
                else:
                    p = p_func(hi_team, lo_team)
                    if mode == "map":
                        winner = hi_team if p >= 0.5 else lo_team
                    else:
                        winner = hi_team if rng.random() < p else lo_team
                wins.append(winner)
            region_winners[region] = wins
        bracket.append([w for r in regions for w in region_winners[r]])

        for round_idx in range(3):
            new = {}
            for region in regions:
                prev = region_winners[region]
                wins = []
                for i in range(0, len(prev), 2):
                    if i + 1 >= len(prev):
                        wins.append(prev[i])
                        continue
                    a, b = prev[i], prev[i+1]
                    if mode == "chalk":
                        winner = a if seed_map.get(a, 16) <= seed_map.get(b, 16) else b
                    else:
                        p = p_func(a, b)
                        if mode == "map":
                            winner = a if p >= 0.5 else b
                        else:
                            winner = a if rng.random() < p else b
                    wins.append(winner)
                new[region] = wins
            region_winners = new
            bracket.append([w for r in regions for w in region_winners[r]])

        f4 = [region_winners[r][0] for r in regions]
        sf = [(f4[0], f4[1]), (f4[2], f4[3])]
        f_winners = []
        for a, b in sf:
            if mode == "chalk":
                winner = a if seed_map.get(a, 16) <= seed_map.get(b, 16) else b
            else:
                p = p_func(a, b)
                if mode == "map":
                    winner = a if p >= 0.5 else b
                else:
                    winner = a if rng.random() < p else b
            f_winners.append(winner)
        bracket.append(f_winners)

        a, b = f_winners
        if mode == "chalk":
            ch = a if seed_map.get(a, 16) <= seed_map.get(b, 16) else b
        else:
            p = p_func(a, b)
            if mode == "map":
                ch = a if p >= 0.5 else b
            else:
                ch = a if rng.random() < p else b
        bracket.append([ch])
        return bracket

    def score_bracket(my_bracket, true_bracket):
        score = 0
        for r in range(min(len(my_bracket), len(true_bracket))):
            pts = ROUND_POINTS[r]
            score += len(set(my_bracket[r]) & set(true_bracket[r])) * pts
        return score

    bracket_results = []
    for model_name, lookup in pred_lookups.items():
        def p_func(a, b, lk=lookup):
            return lk.get((a, b), 0.5)
        chalk_b = fill_bracket(p_func, "chalk")
        map_b = fill_bracket(p_func, "map")

        chalk_score = score_bracket(chalk_b, true_bracket)
        map_score = score_bracket(map_b, true_bracket)

        # 1000 sample brackets
        rng = np.random.default_rng(42)
        sample_scores = []
        for _ in range(1000):
            sb = fill_bracket(p_func, "sample", rng)
            sample_scores.append(score_bracket(sb, true_bracket))
        sample_arr = np.array(sample_scores)

        bracket_results.append({
            "model": model_name,
            "chalk_score": chalk_score,
            "map_score": map_score,
            "sample_mean": sample_arr.mean(),
            "sample_p95": np.percentile(sample_arr, 95),
        })
        print(f"  {model_name:<28s} chalk={chalk_score:>4d}  MAP={map_score:>4d}  "
              f"sample mean={sample_arr.mean():.0f}, 95%={np.percentile(sample_arr, 95):.0f}")

    bracket_df = pd.DataFrame(bracket_results)
    bracket_df.to_csv("output/eval_2026_brackets.csv", index=False)

    # === 3. Trading P&L ===
    print(f"\n{'='*60}\n  3. REALIZED TRADING P&L (Multi-Feature Logistic)\n{'='*60}")
    market_gen = SyntheticMarketPrices(data, noise_std=0.04, market_alpha_std=0.03)
    strategy = TradingStrategy(threshold=0.03, kelly_fraction=0.25,
                               max_per_game=0.05, max_total_exposure=0.20,
                               max_contracts=100, spread=0.03)

    # Predictions for all 2026 tournament pairs
    p_model_full = pred_lookups["Multi-Feature Logistic"]
    # Generate market prices for actual games only
    actual_pairs = []
    actual_y = []
    actual_p_model = []
    for _, g in actual.iterrows():
        w, l = g["WTeamID"], g["LTeamID"]
        # Canonical (lower TeamID = TeamA)
        if w < l:
            ta, tb = w, l
            outcome = 1
        else:
            ta, tb = l, w
            outcome = 0
        actual_pairs.append((ta, tb))
        actual_y.append(outcome)
        actual_p_model.append(p_model_full.get((ta, tb), 0.5))
    actual_p_model = np.array(actual_p_model)

    # Generate synthetic market prices for actual games
    pairs_df = pd.DataFrame({
        "SeedA": [seed_map.get(p[0], 16) for p in actual_pairs],
        "SeedB": [seed_map.get(p[1], 16) for p in actual_pairs],
    })
    p_market = market_gen.generate_for_matchups(pairs_df, p_model=actual_p_model, seed=2026)

    # Generate signals
    game_ids = list(range(len(actual_pairs)))
    signals = strategy.generate_signals(actual_p_model, p_market, game_ids, 10000)

    pnl = 0
    wins = 0
    losses = 0
    if not signals.empty:
        for _, sig in signals.iterrows():
            idx = int(sig["game_id"])
            outcome = actual_y[idx]
            if sig["side"] == "yes":
                correct = (outcome == 1)
            else:
                correct = (outcome == 0)
            n = sig["contracts"]
            cost = sig["position_size"]
            if correct:
                pnl += n - cost
                wins += 1
            else:
                pnl -= cost
                losses += 1

        print(f"  Trades placed: {len(signals)}")
        print(f"  Wins: {wins}, Losses: {losses}")
        print(f"  Win rate: {wins/len(signals):.1%}")
        print(f"  Realized P&L: ${pnl:+.2f}")
        print(f"  ROI: {pnl/10000*100:+.1f}%")
    else:
        print("  No signals generated.")

    pd.DataFrame([{"model": "Multi-Feature Logistic",
                   "n_trades": len(signals), "wins": wins,
                   "losses": losses, "pnl": pnl,
                   "roi_pct": pnl / 10000 * 100}]).to_csv(
        "output/eval_2026_trading.csv", index=False)


if __name__ == "__main__":
    main()
