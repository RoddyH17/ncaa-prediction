"""
Custom rating systems used by 2nd and 3rd place 2026 Kaggle solutions:

  - Carry-over Elo with margin-of-victory multiplier (FiveThirtyEight-style)
  - Colley Matrix rating (Bias-Free Method, 2002)
  - SRS (Simple Rating System): iterative avg_margin + mean(opp_ratings)

All three produce one rating per (Season, TeamID) computed from regular season
results only. They are orthogonal to Barttorvik / Massey and provide useful
signal beyond W/L records.

References:
  Carry-over Elo: 2nd place writeup, 538 NFL methodology
  Colley: https://www.colleyrankings.com/matrate.pdf
  SRS: simple iterative scheme; converges in ~30 iterations
"""

from __future__ import annotations

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Carry-over Elo with MoV multiplier
# ---------------------------------------------------------------------------

def compute_carryover_elo(
    compact_df: pd.DataFrame,
    k: float = 20.0,
    home_adv: float = 100.0,
    mean_reversion: float = 0.75,
    base: float = 1500.0,
) -> dict:
    """Compute end-of-season Elo for each (Season, TeamID).

    Carry-over: at the start of season s+1, team's Elo is reset to
        elo_new = mean_reversion * elo_end + (1 - mean_reversion) * base

    Margin-of-victory multiplier (538-style):
        mov_mult = ln(|mov| + 1) * 2.2 / (0.001 * |w_elo - l_elo| + 2.2)

    Returns:
      dict (Season, TeamID) -> Elo
      and dict (Season, TeamID) -> SeasonStartElo (the post-reversion value)
    """
    elo: dict[int, float] = {}
    end_of_season: dict[tuple[int, int], float] = {}
    season_start: dict[tuple[int, int], float] = {}
    elo_slope: dict[tuple[int, int], float] = {}

    seasons = sorted(compact_df["Season"].unique())
    for season in seasons:
        # Carry-over reversion at season start
        for team in elo:
            elo[team] = mean_reversion * elo[team] + (1 - mean_reversion) * base
        # Record season-start Elo per team that we already know
        for team in elo:
            season_start[(season, team)] = elo[team]

        season_games = compact_df[compact_df["Season"] == season].sort_values("DayNum")
        # Track first-game Elo for slope
        first_elo: dict[int, float] = {}

        for _, g in season_games.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            if w not in elo: elo[w] = base
            if l not in elo: elo[l] = base
            w_elo, l_elo = elo[w], elo[l]

            w_eff = w_elo + (home_adv if g.get("WLoc") == "H" else 0)
            l_eff = l_elo + (home_adv if g.get("WLoc") == "A" else 0)

            w_exp = 1.0 / (1.0 + 10 ** ((l_eff - w_eff) / 400.0))
            mov = float(g["WScore"]) - float(g["LScore"])
            mov_mag = max(abs(mov), 1.0)
            mov_mult = np.log(mov_mag + 1) * (2.2 / (abs(w_eff - l_eff) * 0.001 + 2.2))

            update = k * mov_mult * (1 - w_exp)
            elo[w] += update
            elo[l] -= update

            if w not in first_elo: first_elo[w] = w_elo
            if l not in first_elo: first_elo[l] = l_elo

        for team, end_elo in elo.items():
            end_of_season[(season, team)] = end_elo
            if (season, team) in season_start:
                elo_slope[(season, team)] = end_elo - season_start[(season, team)]
            elif team in first_elo:
                elo_slope[(season, team)] = end_elo - first_elo[team]
            else:
                elo_slope[(season, team)] = 0.0

    return {"end_of_season": end_of_season,
            "season_start": season_start,
            "elo_slope": elo_slope}


def elo_dict_to_df(elo_dict: dict) -> pd.DataFrame:
    """Convert dict {(Season, TeamID): val} to long DataFrame."""
    rows = []
    for (s, t), v in elo_dict.items():
        rows.append({"Season": int(s), "TeamID": int(t), "value": float(v)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Colley Matrix rating
# ---------------------------------------------------------------------------

def compute_colley_rating(compact_df: pd.DataFrame) -> dict:
    """Solve Colley's bias-free system per season.

    For each team i:
        (2 + n_i) r_i - sum_j(games_ij) r_j = 1 + (w_i - l_i) / 2

    Equivalently, solve (2I + diag(n) - C) r = 1 + (w - l)/2 where C[i,j] = #games i vs j
    (here n_i = total games, so coefficient on r_i is (2 + n_i) - C[i,i]; we set C[i,i] = 0).

    Returns dict (Season, TeamID) -> Colley rating in [0, 1].
    """
    out: dict[tuple[int, int], float] = {}
    for season, season_games in compact_df.groupby("Season"):
        teams = sorted(set(season_games["WTeamID"]) | set(season_games["LTeamID"]))
        n = len(teams)
        if n == 0:
            continue
        idx = {t: i for i, t in enumerate(teams)}

        wins = np.zeros(n)
        losses = np.zeros(n)
        adj = np.zeros((n, n))  # adj[i, j] = number of games between i and j

        for _, g in season_games.iterrows():
            i, j = idx[int(g["WTeamID"])], idx[int(g["LTeamID"])]
            wins[i] += 1
            losses[j] += 1
            adj[i, j] += 1
            adj[j, i] += 1

        total_games = adj.sum(axis=1)
        # Colley matrix: (2 + n_i) on diagonal, -adj[i,j] off-diagonal
        C = -adj
        np.fill_diagonal(C, 2 + total_games)
        b = 1 + (wins - losses) / 2.0
        try:
            r = np.linalg.solve(C, b)
        except np.linalg.LinAlgError:
            r = np.full(n, 0.5)
        for t, i in idx.items():
            out[(int(season), int(t))] = float(r[i])
    return out


# ---------------------------------------------------------------------------
# SRS: Simple Rating System
# ---------------------------------------------------------------------------

def compute_srs(compact_df: pd.DataFrame, max_iter: int = 100, tol: float = 1e-6) -> dict:
    """Iterative SRS: rating_i = avg_margin_i + mean(opp_ratings).

    Mean-centered each iteration (so ratings are relative to the league mean).
    Converges within ~30 iterations.

    Returns dict (Season, TeamID) -> SRS.
    """
    out: dict[tuple[int, int], float] = {}
    for season, season_games in compact_df.groupby("Season"):
        teams = sorted(set(season_games["WTeamID"]) | set(season_games["LTeamID"]))
        if not teams:
            continue
        idx = {t: i for i, t in enumerate(teams)}
        n = len(teams)

        # Per-team avg margin and list of opponents
        margin_sum = np.zeros(n)
        n_games = np.zeros(n)
        opps: list[list[int]] = [[] for _ in range(n)]

        for _, g in season_games.iterrows():
            i = idx[int(g["WTeamID"])]
            j = idx[int(g["LTeamID"])]
            mov = float(g["WScore"]) - float(g["LScore"])
            margin_sum[i] += mov
            margin_sum[j] -= mov
            n_games[i] += 1
            n_games[j] += 1
            opps[i].append(j)
            opps[j].append(i)

        avg_margin = margin_sum / np.maximum(n_games, 1)
        r = avg_margin.copy()

        for it in range(max_iter):
            opp_avg = np.zeros(n)
            for i in range(n):
                if opps[i]:
                    opp_avg[i] = np.mean([r[j] for j in opps[i]])
            new_r = avg_margin + opp_avg
            new_r -= new_r.mean()  # mean-center
            if np.max(np.abs(new_r - r)) < tol:
                r = new_r
                break
            r = new_r
        for t, i in idx.items():
            out[(int(season), int(t))] = float(r[i])
    return out


# ---------------------------------------------------------------------------
# Convenience: build all three rating dicts for a dataset
# ---------------------------------------------------------------------------

def build_extra_ratings(compact_df: pd.DataFrame) -> dict:
    """Return dict with 'elo_end', 'elo_slope', 'colley', 'srs'."""
    elo_data = compute_carryover_elo(compact_df)
    colley = compute_colley_rating(compact_df)
    srs = compute_srs(compact_df)
    return {
        "elo_end": elo_data["end_of_season"],
        "elo_slope": elo_data["elo_slope"],
        "colley": colley,
        "srs": srs,
    }
