"""
Build per-team season game sequences for the Transformer encoder.

Each team's regular season is encoded as a sequence of game embeddings.
Game embedding features (~14 dims):
  - Opponent POM rank (1)
  - Score margin, normalized (1)
  - Location: +1 home, 0 neutral, -1 away (1)
  - Box score: FG%, 3P%, FT%, reb rate, ast, to, stl, blk (8)
  - Day number, normalized to [0,1] (1)
  - Win indicator (1)
  - Opponent seed if available (1)
"""

import numpy as np
import pandas as pd


def _get_pom_ranks(massey_df, season, day_cutoff=133):
    """Get latest POM ranks for a season."""
    df = massey_df[
        (massey_df["Season"] == season)
        & (massey_df["SystemName"] == "POM")
        & (massey_df["RankingDayNum"] <= day_cutoff)
    ]
    if df.empty:
        return {}
    latest_day = df["RankingDayNum"].max()
    df = df[df["RankingDayNum"] == latest_day]
    return dict(zip(df["TeamID"], df["OrdinalRank"]))


def build_team_season_sequence(data, team_id, season, max_games=35):
    """
    Build game embedding sequence for one team-season.

    Returns:
        np.ndarray of shape (n_games, 14), padded/truncated to max_games
        int: actual number of games (before padding)
    """
    detail = data.get("regular_detail")
    if detail is None:
        return np.zeros((max_games, 14)), 0

    massey = data.get("massey")
    pom_ranks = _get_pom_ranks(massey, season) if massey is not None else {}
    max_rank = 363  # approximate max D1 teams

    # Get all games for this team in this season (from detailed results)
    won = detail[(detail["Season"] == season) & (detail["WTeamID"] == team_id)].copy()
    lost = detail[(detail["Season"] == season) & (detail["LTeamID"] == team_id)].copy()

    games = []

    for _, g in won.iterrows():
        opp = g["LTeamID"]
        margin = g["WScore"] - g["LScore"]
        loc_val = 1.0 if g.get("WLoc", "N") == "H" else (0.0 if g.get("WLoc", "N") == "N" else -1.0)

        fga = max(g["WFGA"], 1)
        fga3 = max(g["WFGA3"], 1)
        fta = max(g["WFTA"], 1)
        total_reb = g["WOR"] + g["WDR"]
        opp_total_reb = g["LOR"] + g["LDR"]
        reb_rate = total_reb / max(total_reb + opp_total_reb, 1)

        games.append({
            "day": g["DayNum"],
            "opp_rank": pom_ranks.get(opp, max_rank) / max_rank,
            "margin": margin / 40.0,  # normalize by ~max margin
            "location": loc_val,
            "fg_pct": g["WFGM"] / fga,
            "fg3_pct": g["WFGM3"] / fga3,
            "ft_pct": g["WFTM"] / fta,
            "reb_rate": reb_rate,
            "ast": g["WAst"] / 30.0,
            "to": g["WTO"] / 25.0,
            "stl": g["WStl"] / 15.0,
            "blk": g["WBlk"] / 10.0,
            "win": 1.0,
            "day_norm": 0.0,  # filled below
        })

    for _, g in lost.iterrows():
        opp = g["WTeamID"]
        margin = g["LScore"] - g["WScore"]  # negative
        loc_val = -1.0 if g.get("WLoc", "N") == "H" else (0.0 if g.get("WLoc", "N") == "N" else 1.0)

        fga = max(g["LFGA"], 1)
        fga3 = max(g["LFGA3"], 1)
        fta = max(g["LFTA"], 1)
        total_reb = g["LOR"] + g["LDR"]
        opp_total_reb = g["WOR"] + g["WDR"]
        reb_rate = total_reb / max(total_reb + opp_total_reb, 1)

        games.append({
            "day": g["DayNum"],
            "opp_rank": pom_ranks.get(opp, max_rank) / max_rank,
            "margin": margin / 40.0,
            "location": loc_val,
            "fg_pct": g["LFGM"] / fga,
            "fg3_pct": g["LFGM3"] / fga3,
            "ft_pct": g["LFTM"] / fta,
            "reb_rate": reb_rate,
            "ast": g["LAst"] / 30.0,
            "to": g["LTO"] / 25.0,
            "stl": g["LStl"] / 15.0,
            "blk": g["LBlk"] / 10.0,
            "win": 0.0,
            "day_norm": 0.0,
        })

    if not games:
        return np.zeros((max_games, 14)), 0

    # Sort by day
    games.sort(key=lambda x: x["day"])

    # Normalize day to [0, 1]
    min_day = games[0]["day"]
    max_day = games[-1]["day"]
    day_range = max(max_day - min_day, 1)
    for g in games:
        g["day_norm"] = (g["day"] - min_day) / day_range

    # Convert to array
    feature_keys = [
        "opp_rank", "margin", "location",
        "fg_pct", "fg3_pct", "ft_pct", "reb_rate",
        "ast", "to", "stl", "blk",
        "win", "day_norm",
    ]
    n_games = len(games)
    arr = np.zeros((max_games, len(feature_keys)))
    for i, g in enumerate(games[:max_games]):
        arr[i] = [g[k] for k in feature_keys]

    return arr, min(n_games, max_games)


def build_matchup_sequences(data, team_a, team_b, season, max_games=35):
    """Build sequences for a matchup pair. Returns (seq_a, len_a, seq_b, len_b)."""
    seq_a, len_a = build_team_season_sequence(data, team_a, season, max_games)
    seq_b, len_b = build_team_season_sequence(data, team_b, season, max_games)
    return seq_a, len_a, seq_b, len_b
