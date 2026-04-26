"""
Honest LOTO evaluation of Bracket Portfolio Optimization vs MAP vs Chalk
on 11 historical seasons of ACTUAL bracket outcomes.

For each season:
  1. Train Bayesian Bradley-Terry on pre-tournament Barttorvik
  2. Compute marginal advancement probabilities via 10000 simulations
  3. Generate Optimal Portfolio bracket
  4. Generate MAP bracket and Chalk bracket
  5. Score each against ACTUAL outcomes that season

Aggregate over 11 seasons to assess statistical significance.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from scipy import stats

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import _parse_seed_num
from src.sequential_bayes import SequentialBayesianTournament
from src.bracket_optimizer import (
    bracket_first_round_pairs, simulate_one,
    compute_marginal_advancement, optimal_bracket_picks, score_bracket,
    ROUND_POINTS,
)


def get_prior_strengths(season: int) -> dict:
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


def fill_bracket_map(p_func, seed_to_team):
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
            wins.append(ta if p_func(ta, tb) >= 0.5 else tb)
        region_state[region] = wins
    bracket.append([w for r in regions for w in region_state[r]])
    for round_idx in range(3):
        new = {}
        for region in regions:
            prev = region_state[region]
            wins = []
            for i in range(0, len(prev), 2):
                if i + 1 >= len(prev): wins.append(prev[i]); continue
                ta, tb = prev[i], prev[i + 1]
                wins.append(ta if p_func(ta, tb) >= 0.5 else tb)
            new[region] = wins
        region_state = new
        bracket.append([w for r in regions for w in region_state[r]])
    f4 = [region_state[r][0] for r in regions if region_state[r]]
    if len(f4) >= 4:
        sf = [(f4[0], f4[1]), (f4[2], f4[3])]
        finalists = [a if p_func(a, b) >= 0.5 else b for a, b in sf]
        bracket.append(finalists)
        if len(finalists) == 2:
            ta, tb = finalists
            bracket.append([ta if p_func(ta, tb) >= 0.5 else tb])
    return bracket


def fill_bracket_chalk(seed_to_team, seed_lookup):
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
        new = {}
        for region in regions:
            prev = region_state[region]
            wins = []
            for i in range(0, len(prev), 2):
                if i + 1 >= len(prev): wins.append(prev[i]); continue
                ta, tb = prev[i], prev[i + 1]
                sa = seed_lookup.get(ta, 16); sb = seed_lookup.get(tb, 16)
                wins.append(ta if sa <= sb else tb)
            new[region] = wins
        region_state = new
        bracket.append([w for r in regions for w in region_state[r]])
    f4 = [region_state[r][0] for r in regions if region_state[r]]
    if len(f4) >= 4:
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


def build_true_bracket(actual_games: pd.DataFrame) -> list:
    games_played = {}
    for _, g in actual_games.iterrows():
        for tid in [g["WTeamID"], g["LTeamID"]]:
            games_played[tid] = games_played.get(tid, 0) + 1
    rounds_lost_at = {g["LTeamID"]: games_played[g["LTeamID"]] for _, g in actual_games.iterrows()}
    bracket = []
    for round_idx in range(6):
        n_played = round_idx + 1
        winners = [tid for tid in games_played
                    if games_played[tid] > n_played
                    or (games_played[tid] == 6 and tid not in rounds_lost_at)]
        bracket.append(winners)
    return bracket


def main():
    print("Loading data...")
    data = load_all_mens_data()
    seasons = [s for s in range(2014, 2026) if s != 2020]
    tourney = data["tourney_compact"]

    actual_2026 = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")

    results = []

    for season in seasons:
        # Get actual outcomes
        if season < 2026:
            actual_games = tourney[tourney["Season"] == season]
        else:
            actual_games = actual_2026
        if len(actual_games) < 60:
            print(f"Skipping {season}: only {len(actual_games)} games")
            continue

        # Build seed map
        s_season = data["seeds"][data["seeds"]["Season"] == season]
        seed_to_team = dict(zip(s_season["Seed"], s_season["TeamID"]))
        seed_lookup = {v: int(k[1:].rstrip("ab")) for k, v in seed_to_team.items()}
        tournament_teams = set(s_season["TeamID"])

        # Build prior model from Barttorvik
        try:
            priors = get_prior_strengths(season)
        except Exception as e:
            print(f"Skipping {season}: prior unavailable ({e})")
            continue
        priors_t = {tid: priors.get(tid, 0.0) for tid in tournament_teams}
        model = SequentialBayesianTournament(priors_t, prior_var=0.5, obs_scale=1.3)

        def p_func(a, b):
            return model.predict(a, b)

        # Compute strategies
        rng = np.random.default_rng(season)
        advancement = compute_marginal_advancement(p_func, seed_to_team, n_sims=8000, rng_seed=season)
        optimal = optimal_bracket_picks(advancement, seed_to_team)
        map_b = fill_bracket_map(p_func, seed_to_team)
        chalk_b = fill_bracket_chalk(seed_to_team, seed_lookup)

        # True bracket
        true_b = build_true_bracket(actual_games)

        s_chalk = score_bracket(chalk_b, true_b)
        s_map = score_bracket(map_b, true_b)
        s_opt = score_bracket(optimal, true_b)

        results.append({
            "season": season, "chalk": s_chalk, "map": s_map, "optimal": s_opt,
            "opt_minus_map": s_opt - s_map,
        })
        print(f"  {season}: chalk={s_chalk}  map={s_map}  optimal={s_opt}  "
              f"opt-map={s_opt - s_map:+d}")

    df = pd.DataFrame(results)
    print(f"\n{'='*60}")
    print(f"  AGGREGATE OVER {len(df)} SEASONS (REAL OUTCOMES)")
    print(f"{'='*60}")
    print(f"  Mean Chalk:    {df['chalk'].mean():.0f}")
    print(f"  Mean MAP:      {df['map'].mean():.0f}")
    print(f"  Mean Optimal:  {df['optimal'].mean():.0f}")
    print(f"  Mean (Opt - MAP): {df['opt_minus_map'].mean():+.1f}")
    print(f"  Optimal beats MAP in: {(df['opt_minus_map'] > 0).sum()}/{len(df)} seasons")

    # Statistical test: is optimal > map?
    t_stat, p_val = stats.ttest_rel(df["optimal"], df["map"])
    print(f"  Paired t-test (Opt vs MAP): t={t_stat:.2f}, p={p_val:.4f}")

    # Wilcoxon signed-rank (nonparametric, more robust for small n)
    if len(df) >= 6:
        w_stat, w_p = stats.wilcoxon(df["optimal"] - df["map"])
        print(f"  Wilcoxon signed-rank: W={w_stat:.0f}, p={w_p:.4f}")

    df.to_csv("output/portfolio_loto.csv", index=False)


if __name__ == "__main__":
    main()
