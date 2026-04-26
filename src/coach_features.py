"""
Coach features for tournament prediction (men's only — no women's coach data
in Kaggle dataset).

Computes per (Season, TeamID):
  coach_apps        — career tournament appearances at start of season
  coach_winpct      — career tournament W% at start of season
  coach_pase        — Performance Above Seed Expectation: career sum of
                      (actual_wins - base_rate_implied_wins) normalized
  coach_won_champ   — binary: coach has won a championship before season
  coach_school_yrs  — consecutive seasons at current school

All features are LOSO-aware: when computing for season s, use only data
from prior seasons (< s).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src.pipeline import _parse_seed_num
from src.seed_base_rate import compute_base_rate_table, lookup_p_a_wins


def get_team_coach_at_season_end(coaches_df: pd.DataFrame) -> dict[tuple[int, int], str]:
    """For each (Season, TeamID), return the coach with the latest LastDayNum
    (i.e., the one coaching going into the tournament).
    """
    out = {}
    for (season, team), grp in coaches_df.groupby(["Season", "TeamID"]):
        end_coach = grp.sort_values("LastDayNum").iloc[-1]["CoachName"]
        out[(int(season), int(team))] = str(end_coach)
    return out


def build_coach_features(
    coaches_df: pd.DataFrame,
    tourney_compact: pd.DataFrame,
    seeds_df: pd.DataFrame,
) -> dict[tuple[int, int, str], dict]:
    """Build per (Season, CoachName) coach feature dict.

    Returns:
      coach_feat[(season, coach)] = {
        'apps', 'winpct', 'pase', 'won_champ', 'school_yrs_team_X' (...)
      }

    All features are computed as of START of season s (using prior seasons only).
    """
    # Index coach by (Season, TeamID) at season end (going into tournament)
    team_coach = get_team_coach_at_season_end(coaches_df)

    # Build base rate from all available history (we'll re-compute LOSO-aware
    # base rate when needed; for coach PASE we use cumulative-up-to-s)
    seed_lookup = {(int(r["Season"]), int(r["TeamID"])): _parse_seed_num(r["Seed"])
                   for _, r in seeds_df.iterrows()}

    # For each tournament game, identify (season, winner_coach, loser_coach)
    games = tourney_compact.copy()
    games = games.sort_values("Season")
    games_per_coach: dict[str, list] = {}
    champ_games: dict[int, dict] = {}  # season -> {champion: team_id}

    for _, g in games.iterrows():
        season = int(g["Season"])
        w_id, l_id = int(g["WTeamID"]), int(g["LTeamID"])
        w_coach = team_coach.get((season, w_id))
        l_coach = team_coach.get((season, l_id))
        w_seed = seed_lookup.get((season, w_id))
        l_seed = seed_lookup.get((season, l_id))
        if w_coach is None or l_coach is None or w_seed is None or l_seed is None:
            continue
        games_per_coach.setdefault(w_coach, []).append({
            "season": season, "won": 1, "own_seed": w_seed, "opp_seed": l_seed,
        })
        games_per_coach.setdefault(l_coach, []).append({
            "season": season, "won": 0, "own_seed": l_seed, "opp_seed": w_seed,
        })

    # Champions: teams that played 6 games and won all in that season
    for season in games["Season"].unique():
        season_games = games[games["Season"] == season]
        played = {}
        losers = set()
        for _, g in season_games.iterrows():
            for tid in [int(g["WTeamID"]), int(g["LTeamID"])]:
                played[tid] = played.get(tid, 0) + 1
            losers.add(int(g["LTeamID"]))
        for tid, n in played.items():
            if n == 6 and tid not in losers:
                champ_games[int(season)] = {"team": tid, "coach": team_coach.get((int(season), tid))}

    # School tenure: per (TeamID, CoachName) -> first season coached this team
    school_first: dict[tuple[int, str], int] = {}
    for (season, team), coach in team_coach.items():
        key = (team, coach)
        if key not in school_first or school_first[key] > season:
            school_first[key] = season

    return {
        "team_coach": team_coach,
        "games_per_coach": games_per_coach,
        "champ_games": champ_games,
        "school_first": school_first,
    }


def get_coach_feats_at_season(
    coach_data: dict, season: int, coach: str, team_id: int,
) -> dict[str, float]:
    """Return coach features for `coach` at the START of `season`.

    Uses only data from seasons strictly < season (LOSO-aware).
    """
    games = [g for g in coach_data["games_per_coach"].get(coach, [])
             if g["season"] < season]
    apps = len(set(g["season"] for g in games))
    n_g = len(games)
    n_w = sum(g["won"] for g in games)
    winpct = n_w / max(n_g, 1)

    # PASE: sum (actual_won - expected_winrate_from_seedpair_history)
    # Use a 1985..(season-1) base rate table (we'll compute it once outside)
    pase = 0.0
    if n_g > 0:
        for g in games:
            # Approximation: use seed-diff-implied prob (no need for full base rate here)
            seed_diff = g["own_seed"] - g["opp_seed"]
            expected = 1.0 / (1.0 + np.exp(0.13 * seed_diff))
            pase += (g["won"] - expected)
        pase /= n_g

    # Won championship before
    won_champ = 0
    for s_prev in range(1985, season):
        c = coach_data["champ_games"].get(s_prev)
        if c and c.get("coach") == coach:
            won_champ = 1
            break

    # School tenure: how long has this coach been at this team?
    first_at_school = coach_data["school_first"].get((team_id, coach), season)
    school_yrs = season - first_at_school

    return {
        "apps": apps,
        "winpct": winpct,
        "pase": pase,
        "won_champ": won_champ,
        "school_yrs": school_yrs,
    }


def get_team_coach_features(
    coach_data: dict, season: int, team_id: int,
) -> dict[str, float]:
    """Wrapper: lookup the coach for (season, team_id) then return features."""
    coach = coach_data["team_coach"].get((season, team_id))
    if coach is None:
        return {"apps": 0, "winpct": 0.0, "pase": 0.0,
                "won_champ": 0, "school_yrs": 0}
    return get_coach_feats_at_season(coach_data, season, coach, team_id)
