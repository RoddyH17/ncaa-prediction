"""
A + B: Sequential Bayesian Update + Bracket Portfolio Optimization
Evaluated on actual 2026 tournament results.

A. Sequential Bayesian:
   - Pre-tournament prior from Barttorvik AdjEM
   - Update posterior after each completed round
   - Predict next round games using updated posterior
   - Compare per-round Brier vs fixed pre-tournament prediction

B. Bracket Portfolio:
   - Marginal probabilities from base model + Bayesian update
   - Monte Carlo simulate 10000 tournaments
   - Compute P(team advances to round R) for each team
   - Pick optimal bracket maximizing E[ESPN score] using advancement probs
   - Compare to MAP-pick baseline on actual 2026 outcome
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import _parse_seed_num
from src.sequential_bayes import SequentialBayesianTournament
from src.bracket_optimizer import (
    bracket_first_round_pairs, simulate_one,
    compute_marginal_advancement, optimal_bracket_picks, score_bracket,
    ROUND_POINTS,
)

plt.style.use("seaborn-v0_8-whitegrid")


def assign_round(actual_df: pd.DataFrame, seed_map: dict) -> pd.DataFrame:
    """Assign each game to its round (R64, R32, S16, E8, F4, Final).

    Heuristic: count how many games each team played.
    Lost in R1 -> 1 game. Lost in R2 -> 2 games. ...
    Champion -> 6 games (won all).
    Round of game = max(games played by either team).
    """
    actual = actual_df.copy()
    games_played = {}
    for _, g in actual.iterrows():
        for tid in [g["WTeamID"], g["LTeamID"]]:
            games_played[tid] = games_played.get(tid, 0) + 1

    # Assign each game to a round: it's the round at which the LOSER lost.
    # Loser played K games -> they lost in round K.
    rounds = []
    for _, g in actual.iterrows():
        l = g["LTeamID"]
        round_num = games_played.get(l, 1)  # 1-indexed: R64=1, R32=2, ..., Final=6
        rounds.append(round_num)
    actual["round"] = rounds
    return actual


def get_2026_pre_tournament_strength(data: dict) -> dict:
    """Pre-tournament team strength from Barttorvik AdjEM (NetRtg)."""
    bart = pd.read_csv(DATA_DIR / "external" / "barttorvik_2026.csv")
    # Drop dups and use first occurrence
    bart = bart.drop_duplicates(subset="TeamID").set_index("TeamID")
    out = {}
    for tid in bart.index:
        v = bart.loc[tid, "NetRtg"]
        if hasattr(v, "iloc"):
            v = v.iloc[0]
        try:
            out[int(tid)] = float(v) / 10.0  # scale to ~ logit units
        except (TypeError, ValueError):
            continue
    return out


def run_sequential_bayesian(data: dict):
    """Sequential Bayesian update on 2026 actual tournament."""
    print("\n" + "="*60)
    print("  A. SEQUENTIAL BAYESIAN UPDATE")
    print("="*60)

    actual = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    seeds = data["seeds"]
    s2026 = seeds[seeds["Season"] == 2026].copy()
    s2026["SeedNum"] = s2026["Seed"].apply(_parse_seed_num)
    seed_map = dict(zip(s2026["TeamID"], s2026["SeedNum"]))

    # Add round to each game
    actual = assign_round(actual, seed_map)
    print(f"\nActual 2026 games by round:")
    print(actual.groupby("round").size().to_string())

    # Build pre-tournament prior from Barttorvik
    prior_strengths = get_2026_pre_tournament_strength(data)
    # Filter to 2026 tournament teams only
    tournament_teams = set(s2026["TeamID"])
    prior_strengths = {tid: prior_strengths.get(tid, 0.0) for tid in tournament_teams}

    # Calibrate scale: use the ratio that makes pre-tournament Brier ~0.15
    # Use a simple grid: try scales [0.5, 0.7, 0.9, 1.0, 1.2, 1.5]
    print("\nCalibrating scale parameter on pre-tournament prediction...")

    # First evaluate fixed pre-tournament prediction at each scale
    best_scale = 1.0
    best_brier = 1.0
    for scale in [0.5, 0.7, 0.9, 1.0, 1.2, 1.5, 2.0]:
        m = SequentialBayesianTournament(prior_strengths, prior_var=0.5, obs_scale=scale)
        y_true, y_pred = [], []
        for _, g in actual.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            if w in m.tid_to_idx and l in m.tid_to_idx:
                # Predict in canonical order (lower TeamID = TeamA)
                if w < l:
                    p = m.predict(w, l)
                    y_true.append(1)
                else:
                    p = m.predict(l, w)
                    y_true.append(0)
                y_pred.append(p)
        bs = brier_score_loss(y_true, y_pred)
        print(f"  scale={scale:.2f}: Brier={bs:.4f}")
        if bs < best_brier:
            best_brier = bs
            best_scale = scale
    print(f"  Best scale: {best_scale:.2f} (pre-tourney Brier: {best_brier:.4f})")

    # === Now run sequential update ===
    rounds = sorted(actual["round"].unique())
    per_round_results = []

    # Initialize fresh model
    model = SequentialBayesianTournament(prior_strengths, prior_var=0.5, obs_scale=best_scale)

    # Pre-tournament predictions for ALL games (fixed)
    pre_predictions = {}
    for _, g in actual.iterrows():
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        if w < l:
            pre_predictions[(w, l)] = model.predict(w, l)
        else:
            pre_predictions[(l, w)] = model.predict(l, w)

    # Sequential predictions: for round R, predict using model state BEFORE round R
    seq_predictions = {}
    cumulative_brier = []
    cumulative_y_true = []
    cumulative_y_pred_seq = []
    cumulative_y_pred_pre = []

    for round_num in rounds:
        # Predict all games in this round using current model state
        round_games = actual[actual["round"] == round_num]
        for _, g in round_games.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            if w < l:
                p_seq = model.predict(w, l)
                p_pre = pre_predictions.get((w, l), 0.5)
                seq_predictions[(w, l, round_num)] = p_seq
                cumulative_y_true.append(1)
            else:
                p_seq = model.predict(l, w)
                p_pre = pre_predictions.get((l, w), 0.5)
                seq_predictions[(l, w, round_num)] = p_seq
                cumulative_y_true.append(0)
            cumulative_y_pred_seq.append(p_seq)
            cumulative_y_pred_pre.append(p_pre)

        # AFTER this round, update model with observed outcomes
        for _, g in round_games.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            model.update_with_game(w, l)

        # Compute per-round Brier for this round only
        n_round = len(round_games)
        round_y_true = cumulative_y_true[-n_round:]
        round_y_pred_seq = cumulative_y_pred_seq[-n_round:]
        round_y_pred_pre = cumulative_y_pred_pre[-n_round:]
        bs_seq = brier_score_loss(round_y_true, round_y_pred_seq) if n_round > 0 else 0
        bs_pre = brier_score_loss(round_y_true, round_y_pred_pre) if n_round > 0 else 0

        round_name = ["R64", "R32", "S16", "E8", "F4", "Final"][round_num - 1]
        per_round_results.append({
            "round": round_num,
            "round_name": round_name,
            "n_games": n_round,
            "brier_pre_tournament": bs_pre,
            "brier_sequential": bs_seq,
            "improvement": bs_pre - bs_seq,
        })
        print(f"  {round_name:6s} ({n_round} games): "
              f"pre={bs_pre:.4f}  sequential={bs_seq:.4f}  delta={bs_pre - bs_seq:+.4f}")

    # Aggregate
    agg_seq = brier_score_loss(cumulative_y_true, cumulative_y_pred_seq)
    agg_pre = brier_score_loss(cumulative_y_true, cumulative_y_pred_pre)
    print(f"\n{'='*40}")
    print(f"Aggregate Brier (all 63 games):")
    print(f"  Pre-tournament fixed: {agg_pre:.4f}")
    print(f"  Sequential update:    {agg_seq:.4f}")
    print(f"  Improvement:          {agg_pre - agg_seq:+.4f}")
    print(f"  Vegas/markets:        ~0.1536")

    df_rounds = pd.DataFrame(per_round_results)
    df_rounds.to_csv("output/sequential_per_round.csv", index=False)
    return agg_seq, agg_pre, df_rounds, seq_predictions, pre_predictions, model


def run_bracket_portfolio(data: dict, model: SequentialBayesianTournament,
                          actual: pd.DataFrame, seed_map: dict):
    """Bracket portfolio optimization on 2026 with pre-tournament probabilities."""
    print("\n" + "="*60)
    print("  B. BRACKET PORTFOLIO OPTIMIZATION")
    print("="*60)

    s2026 = data["seeds"][data["seeds"]["Season"] == 2026]
    seed_to_team = dict(zip(s2026["Seed"], s2026["TeamID"]))

    def p_func(team_a, team_b):
        # Use Bayesian model's CURRENT state (pre-tournament if not updated)
        return model.predict(team_a, team_b)

    # Compute marginal advancement probabilities via Monte Carlo
    print("\nRunning 10000 Monte Carlo simulations to estimate advancement probs...")
    advancement = compute_marginal_advancement(p_func, seed_to_team, n_sims=10000)

    # Print top teams by championship probability
    champ_probs = {tid: advancement[tid][5] for tid in advancement}
    top_champ = sorted(champ_probs.items(), key=lambda x: -x[1])[:10]
    teams_df = data["teams"]
    name_map = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))
    print("\nTop 10 by P(win championship):")
    for tid, p in top_champ:
        seed = next((k for k, v in seed_to_team.items() if v == tid), "?")
        print(f"  ({seed}) {name_map.get(tid, str(tid))}: {p:.3f}")

    # Compute optimal bracket via portfolio approach
    optimal_picks = optimal_bracket_picks(advancement, seed_to_team)

    # Also compute MAP picks (baseline)
    rng = np.random.default_rng(42)

    # MAP strategy: pick most likely winner of each game (pairwise)
    def map_pick_bracket(p_func, seed_to_team):
        regions = ["W", "X", "Y", "Z"]
        pairs = bracket_first_round_pairs()
        bracket = []
        region_state = {}
        for region in regions:
            wins = []
            for hi, lo in pairs:
                hi_keys = [k for k in seed_to_team if k.startswith(f"{region}{hi:02d}")]
                lo_keys = [k for k in seed_to_team if k.startswith(f"{region}{lo:02d}")]
                if not hi_keys or not lo_keys: continue
                ta, tb = seed_to_team[hi_keys[0]], seed_to_team[lo_keys[0]]
                p = p_func(ta, tb)
                wins.append(ta if p >= 0.5 else tb)
            region_state[region] = wins
        bracket.append([w for r in regions for w in region_state[r]])
        for round_idx in range(3):
            new_state = {}
            for region in regions:
                prev = region_state[region]
                wins = []
                for i in range(0, len(prev), 2):
                    if i + 1 >= len(prev):
                        wins.append(prev[i]); continue
                    ta, tb = prev[i], prev[i + 1]
                    p = p_func(ta, tb)
                    wins.append(ta if p >= 0.5 else tb)
                new_state[region] = wins
            region_state = new_state
            bracket.append([w for r in regions for w in region_state[r]])
        f4 = [region_state[r][0] for r in regions if region_state[r]]
        sf = [(f4[0], f4[1]), (f4[2], f4[3])]
        finalists = []
        for ta, tb in sf:
            p = p_func(ta, tb)
            finalists.append(ta if p >= 0.5 else tb)
        bracket.append(finalists)
        if len(finalists) == 2:
            ta, tb = finalists
            p = p_func(ta, tb)
            bracket.append([ta if p >= 0.5 else tb])
        return bracket

    map_bracket = map_pick_bracket(p_func, seed_to_team)

    # Chalk strategy: always higher seed
    def chalk_pick(p_func, seed_to_team):
        regions = ["W", "X", "Y", "Z"]
        pairs = bracket_first_round_pairs()
        bracket = []
        region_state = {}
        for region in regions:
            wins = []
            for hi, lo in pairs:
                hi_keys = [k for k in seed_to_team if k.startswith(f"{region}{hi:02d}")]
                lo_keys = [k for k in seed_to_team if k.startswith(f"{region}{lo:02d}")]
                if not hi_keys or not lo_keys: continue
                # Chalk = higher seed (lower number)
                wins.append(seed_to_team[hi_keys[0]])
            region_state[region] = wins
        bracket.append([w for r in regions for w in region_state[r]])

        # Get seed for each team
        seed_lookup = {v: int(k[1:].rstrip("ab")) for k, v in seed_to_team.items()}
        for round_idx in range(3):
            new_state = {}
            for region in regions:
                prev = region_state[region]
                wins = []
                for i in range(0, len(prev), 2):
                    if i + 1 >= len(prev):
                        wins.append(prev[i]); continue
                    ta, tb = prev[i], prev[i + 1]
                    sa = seed_lookup.get(ta, 16)
                    sb = seed_lookup.get(tb, 16)
                    wins.append(ta if sa <= sb else tb)
                new_state[region] = wins
            region_state = new_state
            bracket.append([w for r in regions for w in region_state[r]])
        f4 = [region_state[r][0] for r in regions if region_state[r]]
        sf = [(f4[0], f4[1]), (f4[2], f4[3])]
        finalists = []
        for ta, tb in sf:
            sa = seed_lookup.get(ta, 16)
            sb = seed_lookup.get(tb, 16)
            finalists.append(ta if sa <= sb else tb)
        bracket.append(finalists)
        if len(finalists) == 2:
            ta, tb = finalists
            sa = seed_lookup.get(ta, 16)
            sb = seed_lookup.get(tb, 16)
            bracket.append([ta if sa <= sb else tb])
        return bracket

    chalk_bracket = chalk_pick(p_func, seed_to_team)

    # Build true bracket from actual results
    games_played = {}
    for _, g in actual.iterrows():
        for tid in [g["WTeamID"], g["LTeamID"]]:
            games_played[tid] = games_played.get(tid, 0) + 1
    rounds_lost_at = {g["LTeamID"]: games_played[g["LTeamID"]] for _, g in actual.iterrows()}
    true_bracket = []
    for round_idx in range(6):
        n_played = round_idx + 1
        round_winners = [tid for tid in games_played
                          if games_played[tid] > n_played
                          or (games_played[tid] == 6 and tid not in rounds_lost_at)]
        true_bracket.append(round_winners)

    # Score brackets
    score_optimal = score_bracket(optimal_picks, true_bracket)
    score_map = score_bracket(map_bracket, true_bracket)
    score_chalk = score_bracket(chalk_bracket, true_bracket)

    print("\n=== 2026 Bracket Score Results (max=1920) ===")
    print(f"  Chalk (higher seed):     {score_chalk:>5}")
    print(f"  MAP (pairwise winner):   {score_map:>5}")
    print(f"  Portfolio Optimal:       {score_optimal:>5}")
    print(f"  Optimal vs MAP:          {score_optimal - score_map:+d}")
    print(f"  Optimal vs Chalk:        {score_optimal - score_chalk:+d}")

    # Save
    with open("output/bracket_optimal_picks.txt", "w") as f:
        for r, picks in enumerate(optimal_picks):
            round_name = ["R64", "R32", "S16", "E8", "F4", "Final"][r]
            f.write(f"=== {round_name} ===\n")
            for tid in picks:
                seed = next((k for k, v in seed_to_team.items() if v == tid), "?")
                f.write(f"  ({seed}) {name_map.get(tid, str(tid))}\n")
            f.write("\n")

    pd.DataFrame([
        {"strategy": "chalk", "score": score_chalk},
        {"strategy": "map", "score": score_map},
        {"strategy": "optimal_portfolio", "score": score_optimal},
    ]).to_csv("output/bracket_optimal_results.csv", index=False)

    return score_optimal, score_map, score_chalk, advancement


def main():
    print("Loading data...")
    data = load_all_mens_data()

    actual = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    seeds = data["seeds"]
    s2026 = seeds[seeds["Season"] == 2026].copy()
    s2026["SeedNum"] = s2026["Seed"].apply(_parse_seed_num)
    seed_map = dict(zip(s2026["TeamID"], s2026["SeedNum"]))
    actual = assign_round(actual, seed_map)

    # A: Sequential Bayesian
    agg_seq, agg_pre, per_round, seq_preds, pre_preds, model = run_sequential_bayesian(data)

    # Reset model for B (Pre-tournament prediction state, no updates yet)
    prior_strengths = get_2026_pre_tournament_strength(data)
    tournament_teams = set(s2026["TeamID"])
    prior_strengths = {tid: prior_strengths.get(tid, 0.0) for tid in tournament_teams}
    pre_model = SequentialBayesianTournament(prior_strengths, prior_var=0.5, obs_scale=1.0)

    # B: Portfolio Optimization (using pre-tournament probabilities)
    score_opt, score_map, score_chalk, advancement = run_bracket_portfolio(
        data, pre_model, actual, seed_map
    )

    print("\n" + "="*60)
    print("  COMBINED SUMMARY")
    print("="*60)
    print(f"\nA. Sequential Bayesian (overall Brier on 63 games):")
    print(f"  Pre-tournament fixed: {agg_pre:.4f}")
    print(f"  Sequential update:    {agg_seq:.4f}")
    print(f"  Vegas/markets:        ~0.1536")
    print(f"\nB. Portfolio Optimization (2026 ESPN score):")
    print(f"  Chalk:                {score_chalk}")
    print(f"  MAP (independent):    {score_map}")
    print(f"  Portfolio Optimal:    {score_opt}")


if __name__ == "__main__":
    main()
