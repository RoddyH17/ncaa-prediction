"""
Bracket portfolio optimization for ESPN-style scoring.

Given pre-tournament marginal P(team_a beats team_b) for all matchups,
we sample N tournament realizations and find the bracket strategy that
maximizes expected ESPN score (10/20/40/80/160/320 per round).

Optimal strategy at each bracket node: pick the team most likely to ADVANCE
through that node (which depends on joint outcomes of prior rounds).

This differs from MAP picking: a 1-seed has high P(reach E8) but lower
P(win against another 1-seed in F4). Optimal F4 pick may differ from
the "most likely to win each individual game" heuristic.
"""

import numpy as np
import pandas as pd

ROUND_POINTS = [10, 20, 40, 80, 160, 320]  # R64, R32, S16, E8, F4, Final


def bracket_first_round_pairs():
    """Standard NCAA first-round seed pairings within a region."""
    return [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]


def get_2026_team_for_seed(s2026, region, seed):
    """Get TeamID for a (region, seed) in 2026."""
    keys = [k for k in s2026["Seed"] if k.startswith(f"{region}{seed:02d}")]
    if not keys:
        return None
    return s2026[s2026["Seed"] == keys[0]]["TeamID"].iloc[0]


def simulate_tournament(p_func, seed_to_team: dict, rng: np.random.Generator):
    """Simulate one tournament realization. Return list of winners per round.

    p_func(team_a, team_b) -> P(team_a wins)
    seed_to_team: dict of seed -> team_id

    Returns: bracket = [round_winners] where round_winners[i] is list of teams advancing
             after round i. round_winners[0] = R64 winners (32 teams), etc.
    """
    regions = ["W", "X", "Y", "Z"]
    region_teams = {}  # region -> ordered list of 16 teams (matching first-round pairs)
    for region in regions:
        teams = []
        for hi, lo in bracket_first_round_pairs():
            hi_id = next((s2026_lookup[s] for s in s2026_lookup
                          if s.startswith(f"{region}{hi:02d}")), None)
            lo_id = next((s2026_lookup[s] for s in s2026_lookup
                          if s.startswith(f"{region}{lo:02d}")), None)
            if hi_id is not None:
                teams.append(hi_id)
            if lo_id is not None:
                teams.append(lo_id)
        region_teams[region] = teams

    # ... fill rest


def build_seed_to_team_lookup(s2026):
    """Map seed string -> team ID."""
    return dict(zip(s2026["Seed"], s2026["TeamID"]))


def simulate_one(p_func, seed_to_team, rng):
    """Simulate single tournament; return winners per round (lists of TeamIDs)."""
    regions = ["W", "X", "Y", "Z"]
    pairs = bracket_first_round_pairs()

    # Round of 64: each region has 8 games, total 32 games -> 32 winners
    region_state = {}
    for region in regions:
        winners = []
        for hi, lo in pairs:
            hi_keys = [k for k in seed_to_team if k.startswith(f"{region}{hi:02d}")]
            lo_keys = [k for k in seed_to_team if k.startswith(f"{region}{lo:02d}")]
            if not hi_keys or not lo_keys:
                continue
            ta, tb = seed_to_team[hi_keys[0]], seed_to_team[lo_keys[0]]
            p = p_func(ta, tb)
            winner = ta if rng.random() < p else tb
            winners.append(winner)
        region_state[region] = winners

    bracket = [[w for r in regions for w in region_state[r]]]

    # Rounds 2, 3, 4 within each region
    for round_idx in range(3):
        new_state = {}
        for region in regions:
            prev = region_state[region]
            winners = []
            for i in range(0, len(prev), 2):
                if i + 1 >= len(prev):
                    winners.append(prev[i])
                    continue
                ta, tb = prev[i], prev[i + 1]
                p = p_func(ta, tb)
                w = ta if rng.random() < p else tb
                winners.append(w)
            new_state[region] = winners
        region_state = new_state
        bracket.append([w for r in regions for w in region_state[r]])

    # Final 4 -> National Final
    f4 = [region_state[r][0] for r in regions if region_state[r]]
    sf_pairs = [(f4[0], f4[1]), (f4[2], f4[3])]
    finalists = []
    for ta, tb in sf_pairs:
        p = p_func(ta, tb)
        w = ta if rng.random() < p else tb
        finalists.append(w)
    bracket.append(finalists)

    # Championship
    if len(finalists) == 2:
        ta, tb = finalists
        p = p_func(ta, tb)
        champ = ta if rng.random() < p else tb
        bracket.append([champ])
    else:
        bracket.append([finalists[0] if finalists else 0])

    return bracket


def compute_marginal_advancement(p_func, seed_to_team, n_sims=10000, rng_seed=42):
    """Run N simulations to estimate P(team T advances to round R) for each team.

    Returns dict mapping team_id -> [P(reach R32), P(reach S16), ..., P(win champ)]
    (indexed 0-5 for the 6 rounds of advancement)
    """
    rng = np.random.default_rng(rng_seed)
    all_teams = set(seed_to_team.values())
    advancement = {tid: np.zeros(6) for tid in all_teams}

    for _ in range(n_sims):
        bracket = simulate_one(p_func, seed_to_team, rng)
        for round_idx, winners in enumerate(bracket):
            for tid in winners:
                advancement[tid][round_idx] += 1

    for tid in advancement:
        advancement[tid] /= n_sims

    return advancement


def optimal_bracket_picks(advancement: dict, seed_to_team: dict):
    """Compute the bracket that maximizes E[ESPN score] given advancement probs.

    For each game in the bracket structure, pick the team most likely to ADVANCE
    past that game (i.e., to be in the corresponding "next round winners" set).
    """
    regions = ["W", "X", "Y", "Z"]
    pairs = bracket_first_round_pairs()
    picks_per_round = []

    # Round 1: for each pairing, pick team with higher P(advancing past R64)
    region_picks = {}
    for region in regions:
        rp = []
        for hi, lo in pairs:
            hi_keys = [k for k in seed_to_team if k.startswith(f"{region}{hi:02d}")]
            lo_keys = [k for k in seed_to_team if k.startswith(f"{region}{lo:02d}")]
            if not hi_keys or not lo_keys:
                continue
            ta, tb = seed_to_team[hi_keys[0]], seed_to_team[lo_keys[0]]
            # P(team advances past R64) = advancement[team][0]
            p_a = advancement.get(ta, np.zeros(6))[0]
            p_b = advancement.get(tb, np.zeros(6))[0]
            pick = ta if p_a >= p_b else tb
            rp.append((ta, tb, pick))
        region_picks[region] = rp
    picks_per_round.append([p[2] for region in regions for p in region_picks[region]])

    # Rounds 2-4: pick team with higher P(reach R+1) given they're in the matchup
    # We need to know the BRACKET STRUCTURE to know which two teams could meet.
    # Use first-round pairings to derive R2 matchups.
    for round_idx in range(3):
        round_picks = []
        new_region_picks = {}
        for region in regions:
            prev = region_picks[region]
            rp = []
            for i in range(0, len(prev), 2):
                if i + 1 >= len(prev):
                    continue
                # Matchup between two pairs from R1
                team_a_candidates = [prev[i][0], prev[i][1]]  # could be either of R1 game's teams
                team_b_candidates = [prev[i+1][0], prev[i+1][1]]
                # For each candidate pair (a, b), pick team with highest P(reach round_idx+1)
                # But in optimal bracket strategy, we just pick one team to advance from each side
                # We want the team most likely to BE in the round_idx+1 winners set
                target_round = round_idx + 1
                best_a = max(team_a_candidates,
                             key=lambda t: advancement.get(t, np.zeros(6))[target_round])
                best_b = max(team_b_candidates,
                             key=lambda t: advancement.get(t, np.zeros(6))[target_round])
                pick = best_a if (advancement.get(best_a, np.zeros(6))[target_round] >=
                                   advancement.get(best_b, np.zeros(6))[target_round]) else best_b
                rp.append((best_a, best_b, pick))
            new_region_picks[region] = rp
        region_picks = new_region_picks
        round_picks = [p[2] for r in regions for p in region_picks[r]]
        picks_per_round.append(round_picks)

    # Final 4: regional champs face off (W vs X, Y vs Z)
    f4_picks = []
    for region in regions:
        if region_picks[region]:
            # Most likely team to be regional champion = E8 winner
            candidates = set([t for game in region_picks[region]
                              for t in (game[0], game[1])])
            best = max(candidates,
                       key=lambda t: advancement.get(t, np.zeros(6))[3])  # P(reach E8 winner = F4)
            f4_picks.append(best)
    picks_per_round.append(f4_picks)

    # Championship: pick team with highest P(win championship)
    if len(f4_picks) >= 2:
        best_champ = max(f4_picks,
                          key=lambda t: advancement.get(t, np.zeros(6))[5])
        picks_per_round.append([best_champ])

    return picks_per_round


def score_bracket(my_bracket: list, true_bracket: list) -> int:
    """ESPN scoring."""
    score = 0
    for r in range(min(len(my_bracket), len(true_bracket))):
        pts = ROUND_POINTS[r]
        my_set = set(my_bracket[r])
        true_set = set(true_bracket[r])
        score += len(my_set & true_set) * pts
    return score
