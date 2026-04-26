"""
LOTO evaluation of Sequential Bayesian: does it beat fixed pre-tournament
predictions across 11 historical seasons?

Also computes expected bracket score via Monte Carlo to validate portfolio
optimization.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import _parse_seed_num
from src.sequential_bayes import SequentialBayesianTournament
from src.bracket_optimizer import (
    bracket_first_round_pairs, simulate_one,
    compute_marginal_advancement, optimal_bracket_picks, score_bracket,
    ROUND_POINTS,
)


def assign_round(actual: pd.DataFrame) -> pd.DataFrame:
    """Round = number of games loser played."""
    games_played = {}
    for _, g in actual.iterrows():
        for tid in [g["WTeamID"], g["LTeamID"]]:
            games_played[tid] = games_played.get(tid, 0) + 1
    actual = actual.copy()
    actual["round"] = [games_played.get(g["LTeamID"], 1) for _, g in actual.iterrows()]
    return actual


def get_prior_strengths(season: int) -> dict:
    """Per-team prior strength from Barttorvik NetRtg, scaled."""
    bart = pd.read_csv(DATA_DIR / "external" / f"barttorvik_{season}.csv")
    bart = bart.drop_duplicates(subset="TeamID").set_index("TeamID")
    out = {}
    for tid in bart.index:
        v = bart.loc[tid, "NetRtg"]
        if hasattr(v, "iloc"):
            v = v.iloc[0]
        try:
            out[int(tid)] = float(v) / 10.0
        except (TypeError, ValueError):
            continue
    return out


def calibrate_scale_loto(seasons_train: list, data: dict) -> float:
    """Pick scale that minimizes mean Brier across training seasons (no test data)."""
    best_scale, best_brier = 1.0, 1.0
    tourney = data["tourney_compact"]
    for scale in [0.7, 0.85, 1.0, 1.15, 1.3, 1.5]:
        all_y, all_p = [], []
        for s in seasons_train:
            try:
                priors = get_prior_strengths(s)
            except Exception:
                continue
            if not priors:
                continue
            m = SequentialBayesianTournament(priors, prior_var=0.5, obs_scale=scale)
            sg = tourney[tourney["Season"] == s]
            for _, g in sg.iterrows():
                w, l = int(g["WTeamID"]), int(g["LTeamID"])
                if w in m.tid_to_idx and l in m.tid_to_idx:
                    if w < l:
                        all_p.append(m.predict(w, l)); all_y.append(1)
                    else:
                        all_p.append(m.predict(l, w)); all_y.append(0)
        if all_y:
            bs = brier_score_loss(all_y, all_p)
            if bs < best_brier:
                best_brier, best_scale = bs, scale
    return best_scale


def main():
    print("Loading data...")
    data = load_all_mens_data()
    seasons = [s for s in range(2014, 2026) if s != 2020]
    tourney = data["tourney_compact"]

    # Need 2026 results; for historical we use Kaggle's MNCAATourneyCompactResults
    # Add 2026 from external file
    actual_2026 = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    historical = tourney[tourney["Season"].isin(seasons[:-1])]  # 2014-2025
    # Standardize columns
    actual_2026_std = actual_2026[["Season", "WTeamID", "WScore", "LTeamID", "LScore"]].copy()
    actual_2026_std["DayNum"] = 134  # placeholder
    actual_2026_std["WLoc"] = "N"
    actual_2026_std["NumOT"] = 0

    all_actual = pd.concat([
        historical[["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc", "NumOT"]],
        actual_2026_std[["Season", "DayNum", "WTeamID", "WScore", "LTeamID", "LScore", "WLoc", "NumOT"]]
    ], ignore_index=True)

    # === A. Sequential Bayesian LOTO ===
    print(f"\n{'='*70}\n  A. SEQUENTIAL BAYESIAN LOTO (2014-2026)\n{'='*70}")

    results = []
    for test_season in seasons:
        train_seasons = [s for s in seasons if s != test_season]
        scale = calibrate_scale_loto(train_seasons, data)
        try:
            priors = get_prior_strengths(test_season)
        except FileNotFoundError:
            continue
        if not priors:
            continue

        season_games = all_actual[all_actual["Season"] == test_season]
        season_games = assign_round(season_games)

        # Pre-tournament predictions
        pre_model = SequentialBayesianTournament(priors, prior_var=0.5, obs_scale=scale)
        pre_y, pre_p = [], []
        seq_y, seq_p = [], []

        # Sequential model maintained separately
        seq_model = SequentialBayesianTournament(priors, prior_var=0.5, obs_scale=scale)
        rounds = sorted(season_games["round"].unique())
        for round_num in rounds:
            round_games = season_games[season_games["round"] == round_num]
            for _, g in round_games.iterrows():
                w, l = int(g["WTeamID"]), int(g["LTeamID"])
                if w not in pre_model.tid_to_idx or l not in pre_model.tid_to_idx:
                    continue
                if w < l:
                    p_pre = pre_model.predict(w, l)
                    p_seq = seq_model.predict(w, l)
                    y = 1
                else:
                    p_pre = pre_model.predict(l, w)
                    p_seq = seq_model.predict(l, w)
                    y = 0
                pre_y.append(y); pre_p.append(p_pre)
                seq_y.append(y); seq_p.append(p_seq)
            # Update sequential model with this round's outcomes
            for _, g in round_games.iterrows():
                w, l = int(g["WTeamID"]), int(g["LTeamID"])
                if w in seq_model.tid_to_idx and l in seq_model.tid_to_idx:
                    seq_model.update_with_game(w, l)

        bs_pre = brier_score_loss(pre_y, pre_p) if pre_y else np.nan
        bs_seq = brier_score_loss(seq_y, seq_p) if seq_y else np.nan
        results.append({
            "season": test_season,
            "scale": scale,
            "n_games": len(pre_y),
            "brier_pretourney": bs_pre,
            "brier_sequential": bs_seq,
            "delta": bs_pre - bs_seq,
        })
        print(f"  {test_season}: scale={scale:.2f}  pre={bs_pre:.4f}  seq={bs_seq:.4f}  "
              f"delta={bs_pre-bs_seq:+.4f}  ({len(pre_y)} games)")

    df = pd.DataFrame(results)
    print(f"\n  MEAN: pre={df['brier_pretourney'].mean():.4f}  "
          f"seq={df['brier_sequential'].mean():.4f}  "
          f"delta={df['delta'].mean():+.4f}")

    df.to_csv("output/sequential_bayes_loto.csv", index=False)

    # === B. Portfolio Optimization vs MAP via Monte Carlo ===
    print(f"\n{'='*70}\n  B. PORTFOLIO OPTIMIZATION via Monte Carlo simulation\n{'='*70}")

    # Use 2026 as illustrative case
    s2026 = data["seeds"][data["seeds"]["Season"] == 2026]
    seed_to_team = dict(zip(s2026["Seed"], s2026["TeamID"]))
    seed_lookup = {v: int(k[1:].rstrip("ab")) for k, v in seed_to_team.items()}
    priors = get_prior_strengths(2026)
    tournament_teams = set(s2026["TeamID"])
    priors_t = {tid: priors.get(tid, 0.0) for tid in tournament_teams}
    model_2026 = SequentialBayesianTournament(priors_t, prior_var=0.5, obs_scale=1.5)

    def p_func(a, b):
        return model_2026.predict(a, b)

    rng = np.random.default_rng(42)

    # Simulate N tournaments to estimate E[score] under each strategy
    n_sims = 5000
    print(f"\nRunning {n_sims} simulated tournaments...")

    # Strategies as functions
    def map_strategy():
        """Pre-fill bracket using MAP picks (deterministic from p_func)."""
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
                    if i + 1 >= len(prev): wins.append(prev[i]); continue
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

    def chalk_strategy():
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
                wins.append(seed_to_team[hi_keys[0]])
            region_state[region] = wins
        bracket.append([w for r in regions for w in region_state[r]])
        for round_idx in range(3):
            new_state = {}
            for region in regions:
                prev = region_state[region]
                wins = []
                for i in range(0, len(prev), 2):
                    if i + 1 >= len(prev): wins.append(prev[i]); continue
                    ta, tb = prev[i], prev[i + 1]
                    sa = seed_lookup.get(ta, 16); sb = seed_lookup.get(tb, 16)
                    wins.append(ta if sa <= sb else tb)
                new_state[region] = wins
            region_state = new_state
            bracket.append([w for r in regions for w in region_state[r]])
        f4 = [region_state[r][0] for r in regions if region_state[r]]
        sf = [(f4[0], f4[1]), (f4[2], f4[3])]
        finalists = []
        for ta, tb in sf:
            sa = seed_lookup.get(ta, 16); sb = seed_lookup.get(tb, 16)
            finalists.append(ta if sa <= sb else tb)
        bracket.append(finalists)
        if len(finalists) == 2:
            ta, tb = finalists
            sa = seed_lookup.get(ta, 16); sb = seed_lookup.get(tb, 16)
            bracket.append([ta if sa <= sb else tb])
        return bracket

    map_bracket = map_strategy()
    chalk_bracket = chalk_strategy()

    # Optimal portfolio: use marginal advancement probs
    advancement = compute_marginal_advancement(p_func, seed_to_team, n_sims=20000)
    optimal_bracket = optimal_bracket_picks(advancement, seed_to_team)

    # Score each strategy across simulated true brackets
    map_scores = []
    chalk_scores = []
    optimal_scores = []
    for _ in range(n_sims):
        true_b = simulate_one(p_func, seed_to_team, rng)
        map_scores.append(score_bracket(map_bracket, true_b))
        chalk_scores.append(score_bracket(chalk_bracket, true_b))
        optimal_scores.append(score_bracket(optimal_bracket, true_b))

    map_arr = np.array(map_scores)
    chalk_arr = np.array(chalk_scores)
    opt_arr = np.array(optimal_scores)

    print(f"\n  Strategy            E[score]   median   95th%ile   Win vs MAP")
    print(f"  Chalk             {chalk_arr.mean():>7.0f}   {np.median(chalk_arr):>5.0f}   "
          f"{np.percentile(chalk_arr, 95):>8.0f}   {(chalk_arr > map_arr).mean():.1%}")
    print(f"  MAP (independent) {map_arr.mean():>7.0f}   {np.median(map_arr):>5.0f}   "
          f"{np.percentile(map_arr, 95):>8.0f}     -")
    print(f"  Portfolio Optimal {opt_arr.mean():>7.0f}   {np.median(opt_arr):>5.0f}   "
          f"{np.percentile(opt_arr, 95):>8.0f}   {(opt_arr > map_arr).mean():.1%}")

    # Score against ACTUAL 2026
    games_played = {}
    for _, g in actual_2026.iterrows():
        for tid in [g["WTeamID"], g["LTeamID"]]:
            games_played[tid] = games_played.get(tid, 0) + 1
    rounds_lost_at = {g["LTeamID"]: games_played[g["LTeamID"]] for _, g in actual_2026.iterrows()}
    true_2026 = []
    for round_idx in range(6):
        n_played = round_idx + 1
        round_winners = [tid for tid in games_played
                          if games_played[tid] > n_played
                          or (games_played[tid] == 6 and tid not in rounds_lost_at)]
        true_2026.append(round_winners)

    print(f"\n  2026 ACTUAL bracket scores:")
    print(f"    Chalk:           {score_bracket(chalk_bracket, true_2026)}")
    print(f"    MAP:             {score_bracket(map_bracket, true_2026)}")
    print(f"    Portfolio Opt:   {score_bracket(optimal_bracket, true_2026)}")

    pd.DataFrame({
        "strategy": ["chalk", "map", "optimal"],
        "expected_score": [chalk_arr.mean(), map_arr.mean(), opt_arr.mean()],
        "median": [np.median(chalk_arr), np.median(map_arr), np.median(opt_arr)],
        "p95": [np.percentile(chalk_arr, 95), np.percentile(map_arr, 95),
                np.percentile(opt_arr, 95)],
        "actual_2026": [score_bracket(chalk_bracket, true_2026),
                         score_bracket(map_bracket, true_2026),
                         score_bracket(optimal_bracket, true_2026)],
    }).to_csv("output/portfolio_strategy_comparison.csv", index=False)


if __name__ == "__main__":
    main()
