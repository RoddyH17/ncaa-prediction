"""
Per-seed-pair historical base rate feature.

Captures non-linear seed-pair upset patterns directly from 40+ years of
NCAA tournament history (1985-current) instead of letting LR learn it
from seed_diff alone.

Computes: P(better-seeded team beats worse-seeded team) per (s_lo, s_hi) pair.

Usage:
    base_rate_table = compute_base_rate_table(tourney_compact, seeds_df, exclude_season=None)
    # base_rate_table[(s_lo, s_hi)] = float in [0, 1]
"""

from __future__ import annotations

import pandas as pd

from src.pipeline import _parse_seed_num


def compute_base_rate_table(
    tourney_compact: pd.DataFrame,
    seeds_df: pd.DataFrame,
    exclude_season: int | None = None,
    min_count: int = 3,
) -> dict[tuple[int, int], float]:
    """For each unordered (s_lo, s_hi) seed pair, return P(s_lo wins).

    Uses all tournament games in `tourney_compact`. If `exclude_season` is set,
    that season's games are dropped (LOSO-correct).

    Pairs with fewer than `min_count` historical observations get the smoothed
    estimate using a Beta(alpha, beta) prior with alpha=beta=2 (mild "no info"
    prior).
    """
    seed_map: dict[tuple[int, int], int] = {}
    for _, r in seeds_df.iterrows():
        seed_map[(int(r["Season"]), int(r["TeamID"]))] = _parse_seed_num(r["Seed"])

    counts: dict[tuple[int, int], int] = {}
    wins_lo: dict[tuple[int, int], int] = {}

    for _, g in tourney_compact.iterrows():
        season = int(g["Season"])
        if exclude_season is not None and season == exclude_season:
            continue
        w_seed = seed_map.get((season, int(g["WTeamID"])))
        l_seed = seed_map.get((season, int(g["LTeamID"])))
        if w_seed is None or l_seed is None or w_seed == l_seed:
            continue
        s_lo, s_hi = min(w_seed, l_seed), max(w_seed, l_seed)
        key = (s_lo, s_hi)
        counts[key] = counts.get(key, 0) + 1
        if w_seed == s_lo:
            wins_lo[key] = wins_lo.get(key, 0) + 1
        else:
            wins_lo.setdefault(key, 0)

    # Beta(2, 2) smoothing: rate = (wins + 2) / (count + 4)
    out: dict[tuple[int, int], float] = {}
    for k, n in counts.items():
        w = wins_lo.get(k, 0)
        if n >= min_count:
            out[k] = w / n
        else:
            out[k] = (w + 2) / (n + 4)
    return out


def lookup_p_a_wins(
    base_table: dict[tuple[int, int], float],
    seed_a: int,
    seed_b: int,
    fallback_beta: float = 0.13,
) -> float:
    """Return P(team A beats team B) given each team's seed.

    Falls back to seed_diff sigmoid when (s_lo, s_hi) not in table.
    """
    import numpy as np
    if seed_a == seed_b:
        return 0.5
    s_lo, s_hi = min(seed_a, seed_b), max(seed_a, seed_b)
    p_lo_wins = base_table.get((s_lo, s_hi))
    if p_lo_wins is None:
        # Fallback: simple seed_diff sigmoid
        diff = seed_a - seed_b
        return 1.0 / (1.0 + np.exp(fallback_beta * diff))
    if seed_a == s_lo:
        return p_lo_wins
    return 1.0 - p_lo_wins
