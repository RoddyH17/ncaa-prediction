"""
Game-Theoretic Bracket Optimization (Landgraf 2017 style):
Optimize for TOP-K finish probability in a competitor pool, not E[score].

Key insight: in a bracket pool, you compete against other entries.
- If everyone picks chalk, picking chalk = tied with the field
- Picking an upset that hits = jumping ahead of the field
- Optimal strategy depends on COMPETITOR distribution

Approach:
1. Simulate "field" of N=10000 competitor brackets
   - Most casual players pick chalkier than optimal (familiar names)
   - Some pick a few upsets they like
2. For each candidate strategy, compute P(top-X% of field)
3. Compare:
   - Chalk strategy
   - MAP picking (independent expectation)
   - Portfolio Optimal (E[score] maximization)
   - Contrarian: deliberately pick non-consensus upsets identified by upset model
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import _parse_seed_num
from src.sequential_bayes import SequentialBayesianTournament
from src.bracket_optimizer import (
    bracket_first_round_pairs, simulate_one,
    compute_marginal_advancement, optimal_bracket_picks, score_bracket,
    ROUND_POINTS,
)

plt.style.use("seaborn-v0_8-whitegrid")


def get_priors(season: int) -> dict:
    bart = pd.read_csv(DATA_DIR / "external" / f"barttorvik_{season}.csv")
    bart = bart.drop_duplicates(subset="TeamID").set_index("TeamID")
    out = {}
    for tid in bart.index:
        v = bart.loc[tid, "NetRtg"]
        if hasattr(v, "iloc"): v = v.iloc[0]
        try:
            out[int(tid)] = float(v) / 10.0
        except (TypeError, ValueError): continue
    return out


def fill_bracket_chalk(seed_to_team, seed_lookup):
    """All higher seeds advance."""
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


def fill_bracket_with_pfunc(p_func, seed_to_team, mode="map", rng=None):
    """Fill bracket using probability function. mode in {'map', 'sample'}."""
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
            if mode == "map":
                wins.append(ta if p >= 0.5 else tb)
            else:
                wins.append(ta if rng.random() < p else tb)
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
                p = p_func(ta, tb)
                if mode == "map":
                    wins.append(ta if p >= 0.5 else tb)
                else:
                    wins.append(ta if rng.random() < p else tb)
            new[region] = wins
        region_state = new
        bracket.append([w for r in regions for w in region_state[r]])
    f4 = [region_state[r][0] for r in regions if region_state[r]]
    if len(f4) >= 4:
        sf = [(f4[0], f4[1]), (f4[2], f4[3])]
        finalists = []
        for ta, tb in sf:
            p = p_func(ta, tb)
            if mode == "map":
                finalists.append(ta if p >= 0.5 else tb)
            else:
                finalists.append(ta if rng.random() < p else tb)
        bracket.append(finalists)
        if len(finalists) == 2:
            ta, tb = finalists
            p = p_func(ta, tb)
            if mode == "map":
                bracket.append([ta if p >= 0.5 else tb])
            else:
                bracket.append([ta if rng.random() < p else tb])
    return bracket


def simulate_competitor_field(p_func, seed_to_team, seed_lookup, n_competitors=10000,
                              chalk_bias=0.65, rng_seed=42):
    """Simulate field of competitor brackets.

    Each competitor is a casual player who:
    - Picks chalk most of the time (chalk_bias = 65% chalk, 35% sample)
    - The "sample" picks use seed-based probabilities (overconfident on lower seeds)
    """
    rng = np.random.default_rng(rng_seed)

    # Casual player's perceived probabilities: closer to seed-based prior
    def casual_p_func(a, b):
        sa = seed_lookup.get(a, 16)
        sb = seed_lookup.get(b, 16)
        # Casual player thinks 1-seed beats 16-seed 99%, 8 vs 9 is 60% etc.
        # Use sigmoid on seed_diff
        diff = sa - sb
        return 1.0 / (1.0 + np.exp(0.4 * diff))

    field = []
    for i in range(n_competitors):
        if rng.random() < chalk_bias:
            # Pure chalk pick
            field.append(fill_bracket_chalk(seed_to_team, seed_lookup))
        else:
            # Sample from casual player perception
            field.append(fill_bracket_with_pfunc(casual_p_func, seed_to_team,
                                                  mode="sample", rng=rng))
    return field


def evaluate_strategy_in_field(my_bracket, competitor_field, true_bracket):
    """Compute my percentile rank vs competitor field given true outcomes."""
    my_score = score_bracket(my_bracket, true_bracket)
    field_scores = [score_bracket(b, true_bracket) for b in competitor_field]
    n_below = sum(1 for s in field_scores if s < my_score)
    percentile = n_below / len(field_scores)
    return my_score, percentile, field_scores


def topk_probability(strategy_brackets_per_sim, competitor_field_per_sim,
                      true_bracket_per_sim, top_k_pct=0.05):
    """Across simulated tournaments, compute P(my strategy in top K% of field)."""
    in_top_k_count = 0
    for my_bracket, field, true_b in zip(strategy_brackets_per_sim,
                                           competitor_field_per_sim,
                                           true_bracket_per_sim):
        my_score, percentile, _ = evaluate_strategy_in_field(my_bracket, field, true_b)
        if percentile >= 1 - top_k_pct:
            in_top_k_count += 1
    return in_top_k_count / len(strategy_brackets_per_sim)


def main():
    print("Loading data...")
    data = load_all_mens_data()

    # Use 2025 as our headline test case (latest with full historical data)
    test_season = 2025
    season_seeds = data["seeds"][data["seeds"]["Season"] == test_season]
    seed_to_team = dict(zip(season_seeds["Seed"], season_seeds["TeamID"]))
    seed_lookup = {v: int(k[1:].rstrip("ab")) for k, v in seed_to_team.items()}
    tournament_teams = set(season_seeds["TeamID"])

    priors = get_priors(test_season)
    priors_t = {tid: priors.get(tid, 0.0) for tid in tournament_teams}
    model = SequentialBayesianTournament(priors_t, prior_var=0.5, obs_scale=1.3)

    def p_func(a, b):
        return model.predict(a, b)

    n_sims = 500  # number of simulated tournaments
    n_competitors = 1000  # field size per pool

    print(f"\nSimulating {n_sims} tournaments × {n_competitors} competitors")

    # Pre-compute strategies (deterministic for chalk and MAP/optimal)
    chalk_b = fill_bracket_chalk(seed_to_team, seed_lookup)
    map_b = fill_bracket_with_pfunc(p_func, seed_to_team, mode="map")

    # Optimal Portfolio
    advancement = compute_marginal_advancement(p_func, seed_to_team, n_sims=10000, rng_seed=42)
    optimal_b = optimal_bracket_picks(advancement, seed_to_team)

    # Contrarian: same as optimal but force a few high-EV upsets others won't pick
    # Heuristic: replace 2 R64 picks with the strongest upsets per region
    # This is a simple contrarian strategy; can be sophisticated.
    def contrarian_strategy(p_func, seed_to_team, seed_lookup):
        """Optimal portfolio but flip 2-3 highest-EV upsets."""
        # Start from MAP
        bracket = fill_bracket_with_pfunc(p_func, seed_to_team, mode="map")
        # Find top R64 upset opportunities (close games where lower seed has decent prob)
        regions = ["W", "X", "Y", "Z"]
        pairs = bracket_first_round_pairs()
        upset_candidates = []
        for region in regions:
            for hi, lo in pairs:
                hi_keys = [k for k in seed_to_team if k.startswith(f"{region}{hi:02d}")]
                lo_keys = [k for k in seed_to_team if k.startswith(f"{region}{lo:02d}")]
                if not hi_keys or not lo_keys: continue
                ta, tb = seed_to_team[hi_keys[0]], seed_to_team[lo_keys[0]]
                p = p_func(ta, tb)  # P(higher seed wins)
                # Upset opportunity: p < 0.5 means lower seed actually favored (rare)
                # Otherwise, "value upset" = lower seed has p_upset > 0.3 but field thinks p_upset < 0.2
                if p > 0.5 and p < 0.7:  # close game, slight favorite
                    upset_candidates.append((region, hi, lo, ta, tb, p, 1 - p))
        # Pick top 2 most attractive contrarian picks (highest 1-p)
        upset_candidates.sort(key=lambda x: -x[6])
        flipped = 0
        for cand in upset_candidates:
            region, hi_seed, lo_seed, ta, tb, p, p_upset = cand
            if flipped >= 2: break
            # Find this game in bracket and flip
            # Game position depends on first-round-pair index
            pair_idx = pairs.index((hi_seed, lo_seed))
            region_idx = regions.index(region)
            global_idx = region_idx * 8 + pair_idx
            if global_idx < len(bracket[0]):
                # Flip the pick
                bracket[0][global_idx] = tb if bracket[0][global_idx] == ta else ta
                flipped += 1
        return bracket

    contrarian_b = contrarian_strategy(p_func, seed_to_team, seed_lookup)

    print(f"\n  Strategies built. Running tournament + field simulations...")

    rng = np.random.default_rng(0)
    strategies_results = {
        "Chalk": [], "MAP": [], "Portfolio Optimal": [], "Contrarian": [],
    }
    raw_scores = {name: [] for name in strategies_results}
    raw_percentiles = {name: [] for name in strategies_results}

    for sim_idx in range(n_sims):
        # Sample a "true" tournament outcome
        true_b = simulate_one(p_func, seed_to_team, rng)
        # Sample competitor field for this tournament
        field = simulate_competitor_field(p_func, seed_to_team, seed_lookup,
                                            n_competitors=n_competitors,
                                            chalk_bias=0.5, rng_seed=sim_idx)
        for name, brk in [("Chalk", chalk_b), ("MAP", map_b),
                           ("Portfolio Optimal", optimal_b),
                           ("Contrarian", contrarian_b)]:
            score, pct, _ = evaluate_strategy_in_field(brk, field, true_b)
            raw_scores[name].append(score)
            raw_percentiles[name].append(pct)

    print(f"\n{'='*70}")
    print(f"  RESULTS over {n_sims} simulated tournaments × {n_competitors}-bracket field")
    print(f"{'='*70}")
    print(f"\n  Strategy           E[score]   E[percentile]   P(top 5%)   P(top 1%)")
    out_rows = []
    for name in ["Chalk", "MAP", "Portfolio Optimal", "Contrarian"]:
        scores = np.array(raw_scores[name])
        pcts = np.array(raw_percentiles[name])
        p_top5 = (pcts >= 0.95).mean()
        p_top1 = (pcts >= 0.99).mean()
        print(f"  {name:<18s} {scores.mean():>7.0f}   {pcts.mean():>11.1%}   "
              f"{p_top5:>9.1%}   {p_top1:>9.1%}")
        out_rows.append({
            "strategy": name, "E_score": scores.mean(),
            "E_percentile": pcts.mean(), "P_top5": p_top5, "P_top1": p_top1,
        })

    pd.DataFrame(out_rows).to_csv("output/game_theoretic_bracket.csv", index=False)
    print("\nSaved game_theoretic_bracket.csv")


if __name__ == "__main__":
    main()
