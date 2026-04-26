"""
Monte Carlo bracket simulation: how would our 2026 predictions score in
an ESPN-style bracket pool?

Round point values: R64=10, R32=20, S16=40, E8=80, F4=160, Champ=320
Max possible bracket score = 1920

Strategies compared:
  - chalk: always pick higher seed
  - probability_sampled: sample winner from model probability
  - map: pick most likely winner deterministically (=highest predicted prob)

Usage:
    python scripts/run_bracket_simulation.py
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from src.data_collection import load_all_mens_data
from src.pipeline import make_build_features_fn, _parse_seed_num
from src.models import MultiFeatureLogistic

plt.style.use("seaborn-v0_8-whitegrid")

ROUND_POINTS = [10, 20, 40, 80, 160, 320]


def get_2026_bracket(data: dict) -> dict:
    """Return dict mapping seed (e.g. 'W01') to TeamID for 2026."""
    seeds = data["seeds"]
    s2026 = seeds[seeds["Season"] == 2026].copy()
    return dict(zip(s2026["Seed"], s2026["TeamID"]))


def bracket_first_round_pairs():
    """Standard NCAA bracket first-round seed pairings within a region."""
    # Pairs as (high seed, low seed): 1v16, 2v15, ... 8v9
    pairs = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]
    return pairs


def simulate_bracket(p_func, seed_to_team: dict, rng: np.random.Generator,
                     mode: str = "sample"):
    """Simulate a single bracket using p_func(team_a, team_b) -> P(team_a wins).

    mode: 'sample' (random by prob), 'map' (highest prob), 'chalk' (lower seed)

    Returns: list of winners per round, [round1_winners, round2_winners, ...]
    """
    regions = ["W", "X", "Y", "Z"]
    bracket = []

    # Round 1: each region produces 8 winners from 16 teams
    region_winners = {}
    for region in regions:
        pairs = bracket_first_round_pairs()
        round_results = []
        for hi_seed, lo_seed in pairs:
            hi_keys = [k for k in seed_to_team if k.startswith(f"{region}{hi_seed:02d}")]
            lo_keys = [k for k in seed_to_team if k.startswith(f"{region}{lo_seed:02d}")]
            if not hi_keys or not lo_keys:
                continue
            hi_team = seed_to_team[hi_keys[0]]
            lo_team = seed_to_team[lo_keys[0]]

            if mode == "chalk":
                winner = hi_team
            else:
                p_hi = p_func(hi_team, lo_team)
                if mode == "map":
                    winner = hi_team if p_hi >= 0.5 else lo_team
                else:  # sample
                    winner = hi_team if rng.random() < p_hi else lo_team
            round_results.append(winner)
        region_winners[region] = round_results

    bracket.append([w for region in regions for w in region_winners[region]])

    # Rounds 2-4: within each region
    for round_idx in range(3):
        new_winners = {}
        for region in regions:
            prev = region_winners[region]
            round_results = []
            for i in range(0, len(prev), 2):
                if i + 1 >= len(prev):
                    round_results.append(prev[i])
                    continue
                a, b = prev[i], prev[i + 1]
                if mode == "chalk":
                    # Use seed numbers (find via seed_to_team reverse lookup)
                    seed_a = next((s for s, t in seed_to_team.items() if t == a), "16")
                    seed_b = next((s for s, t in seed_to_team.items() if t == b), "16")
                    winner = a if _parse_seed_num(seed_a) <= _parse_seed_num(seed_b) else b
                else:
                    p_a = p_func(a, b)
                    if mode == "map":
                        winner = a if p_a >= 0.5 else b
                    else:
                        winner = a if rng.random() < p_a else b
                round_results.append(winner)
            new_winners[region] = round_results
        region_winners = new_winners
        bracket.append([w for region in regions for w in region_winners[region]])

    # Final 4: one team per region
    f4 = [region_winners[r][0] for r in regions]

    # Semifinals: W vs X, Y vs Z
    sf_pairs = [(f4[0], f4[1]), (f4[2], f4[3])]
    f_winners = []
    for a, b in sf_pairs:
        if mode == "chalk":
            seed_a = next((s for s, t in seed_to_team.items() if t == a), "16")
            seed_b = next((s for s, t in seed_to_team.items() if t == b), "16")
            winner = a if _parse_seed_num(seed_a) <= _parse_seed_num(seed_b) else b
        else:
            p_a = p_func(a, b)
            winner = a if (mode == "map" and p_a >= 0.5) or (mode == "sample" and rng.random() < p_a) else b
        f_winners.append(winner)
    bracket.append(f_winners)

    # Championship
    a, b = f_winners
    if mode == "chalk":
        seed_a = next((s for s, t in seed_to_team.items() if t == a), "16")
        seed_b = next((s for s, t in seed_to_team.items() if t == b), "16")
        champ = a if _parse_seed_num(seed_a) <= _parse_seed_num(seed_b) else b
    else:
        p_a = p_func(a, b)
        champ = a if (mode == "map" and p_a >= 0.5) or (mode == "sample" and rng.random() < p_a) else b
    bracket.append([champ])

    return bracket


def score_bracket(my_bracket: list, true_bracket: list) -> int:
    """ESPN scoring: 10/20/40/80/160/320 per round."""
    score = 0
    for r, (my_round, true_round) in enumerate(zip(my_bracket, true_bracket)):
        pts = ROUND_POINTS[r]
        my_set = set(my_round)
        true_set = set(true_round)
        score += len(my_set & true_set) * pts
    return score


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    seasons_train = [s for s in range(2014, 2026) if s != 2020]

    print("Training Multi-Feature Logistic on 2014-2025...")
    X_train, y_train = build_fn(seasons_train)
    model = MultiFeatureLogistic(C=0.5)
    model.fit(X_train, y_train)

    # Build all 2026 tournament-pair predictions in a lookup table
    print("Building 2026 prediction table...")
    seed_to_team = get_2026_bracket(data)
    teams_2026 = sorted(seed_to_team.values())

    # Use pipeline.build_submission_features-style approach: build features for all pairs
    from scripts.generate_kaggle_submission import build_submission_features
    from src.data_collection import DATA_DIR
    sub_path = str(DATA_DIR / "SampleSubmissionStage2.csv")
    _, X_tourney, _ = build_submission_features(data, 2026, sub_path)
    p_tourney = model.predict_proba(X_tourney)[:, 1]

    # Build lookup: (team_a, team_b) -> P(team_a beats team_b)
    pred_lookup = {}
    for i, (_, row) in enumerate(X_tourney.iterrows()):
        ta, tb = int(row["TeamA"]), int(row["TeamB"])
        pred_lookup[(ta, tb)] = p_tourney[i]
        pred_lookup[(tb, ta)] = 1 - p_tourney[i]

    def p_func(team_a, team_b):
        if team_a == team_b:
            return 0.5
        return pred_lookup.get((team_a, team_b), 0.5)

    # Generate "true" bracket via sampling (since we don't have actual 2026 results)
    # We'll evaluate strategies against many sampled true brackets
    print("\nRunning Monte Carlo simulation...")
    rng = np.random.default_rng(42)
    n_sims = 5000

    strategies = ["chalk", "map", "sample"]
    scores = {s: [] for s in strategies}

    for sim in range(n_sims):
        # Sample a "true" bracket from model probabilities
        true_bracket = simulate_bracket(p_func, seed_to_team, rng, mode="sample")

        for strat in strategies:
            my_bracket = simulate_bracket(p_func, seed_to_team, rng, mode=strat)
            scores[strat].append(score_bracket(my_bracket, true_bracket))

    # Summary stats
    rows = []
    for strat, vals in scores.items():
        arr = np.array(vals)
        rows.append({
            "strategy": strat,
            "mean_score": arr.mean(),
            "std_score": arr.std(),
            "p25": np.percentile(arr, 25),
            "median": np.median(arr),
            "p75": np.percentile(arr, 75),
            "p95": np.percentile(arr, 95),
        })

    summary = pd.DataFrame(rows).sort_values("mean_score", ascending=False)
    print(f"\n{'='*70}\n  BRACKET STRATEGY COMPARISON ({n_sims} simulations)\n{'='*70}")
    print(summary.to_string(index=False))
    summary.to_csv("output/bracket_strategies.csv", index=False)

    # Distribution plot
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = {"chalk": "#94a3b8", "map": "#2563eb", "sample": "#16a34a"}
    for strat, vals in scores.items():
        ax.hist(vals, bins=40, alpha=0.5, label=strat, color=colors[strat],
                edgecolor="black", linewidth=0.3)
    ax.set_xlabel("Bracket score (ESPN scoring, max 1920)")
    ax.set_ylabel("Frequency")
    ax.set_title(f"Bracket Score Distribution ({n_sims} Monte Carlo simulations)")
    ax.legend()
    plt.tight_layout()
    plt.savefig("output/bracket_distribution.png", dpi=150, bbox_inches="tight")
    print("\nSaved bracket_distribution.png and bracket_strategies.csv")

    # Win-rate analysis: how often does each strategy beat chalk?
    chalk_scores = np.array(scores["chalk"])
    print(f"\n--- Win rate vs chalk strategy ---")
    for strat in strategies:
        if strat == "chalk":
            continue
        s = np.array(scores[strat])
        wins = (s > chalk_scores).mean()
        print(f"  {strat} beats chalk: {wins:.1%} of simulations")


if __name__ == "__main__":
    main()
