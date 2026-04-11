"""
Opponent-adjusted efficiency ratings computed from box score data.

Replicates the core of KenPom/Barttorvik methodology:
1. Compute per-game offensive/defensive efficiency (points per 100 possessions)
2. Iteratively adjust for opponent strength until convergence
3. Output: AdjOE, AdjDE, AdjEM per team per season

Reference: Pomeroy (2004), Oliver (2004)
"""

import numpy as np
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
CACHE_DIR = DATA_DIR / "external"


def _compute_possessions(fga, orb, to, fta):
    """Estimate possessions using Dean Oliver formula."""
    return fga - orb + to + 0.44 * fta


def compute_adjusted_efficiency(
    detail_df: pd.DataFrame,
    season: int,
    n_iterations: int = 20,
    convergence_threshold: float = 0.01,
) -> pd.DataFrame:
    """
    Compute opponent-adjusted offensive/defensive efficiency for one season.

    Args:
        detail_df: MRegularSeasonDetailedResults with box score columns
        season: Season year to compute
        n_iterations: Max iterations for adjustment
        convergence_threshold: Stop when max change < this

    Returns:
        DataFrame with columns: TeamID, AdjOE, AdjDE, AdjEM, RawOE, RawDE, n_games
    """
    sdf = detail_df[detail_df["Season"] == season].copy()
    if sdf.empty:
        return pd.DataFrame()

    # Compute possessions per game (average of both teams' estimates)
    w_poss = _compute_possessions(sdf["WFGA"], sdf["WOR"], sdf["WTO"], sdf["WFTA"])
    l_poss = _compute_possessions(sdf["LFGA"], sdf["LOR"], sdf["LTO"], sdf["LFTA"])
    sdf["poss"] = (w_poss + l_poss) / 2

    # Filter out games with unreasonable possessions (data quality)
    sdf = sdf[(sdf["poss"] > 40) & (sdf["poss"] < 120)]

    # Build game-level DataFrame: one row per team per game
    w_rows = pd.DataFrame({
        "TeamID": sdf["WTeamID"].values,
        "OppID": sdf["LTeamID"].values,
        "pts_for": sdf["WScore"].values,
        "pts_against": sdf["LScore"].values,
        "poss": sdf["poss"].values,
    })
    l_rows = pd.DataFrame({
        "TeamID": sdf["LTeamID"].values,
        "OppID": sdf["WTeamID"].values,
        "pts_for": sdf["LScore"].values,
        "pts_against": sdf["WScore"].values,
        "poss": sdf["poss"].values,
    })
    gdf = pd.concat([w_rows, l_rows], ignore_index=True)

    # Per-100 possession efficiency
    gdf["OE"] = gdf["pts_for"] / gdf["poss"] * 100
    gdf["DE"] = gdf["pts_against"] / gdf["poss"] * 100

    # League averages
    total_pts = gdf["pts_for"].sum()
    total_poss = gdf["poss"].sum()
    league_avg = total_pts / total_poss * 100  # OE == DE at league level

    # Initialize team efficiency as possession-weighted averages
    team_stats = gdf.groupby("TeamID").apply(
        lambda g: pd.Series({
            "OE": (g["pts_for"].sum() / g["poss"].sum() * 100),
            "DE": (g["pts_against"].sum() / g["poss"].sum() * 100),
            "n_games": len(g),
        })
    )

    adj_oe = team_stats["OE"].to_dict()
    adj_de = team_stats["DE"].to_dict()

    # Pre-compute game arrays for vectorized iteration
    team_arr = gdf["TeamID"].values
    opp_arr = gdf["OppID"].values
    oe_arr = gdf["OE"].values
    de_arr = gdf["DE"].values
    poss_arr = gdf["poss"].values
    team_ids = np.array(sorted(adj_oe.keys()))
    tid_to_idx = {t: i for i, t in enumerate(team_ids)}

    # Convert to arrays indexed by team
    n_teams = len(team_ids)
    oe_vec = np.array([adj_oe[t] for t in team_ids])
    de_vec = np.array([adj_de[t] for t in team_ids])

    # Map game-level arrays to team indices
    team_idx = np.array([tid_to_idx[t] for t in team_arr])
    opp_idx = np.array([tid_to_idx[t] for t in opp_arr])

    # Iterative adjustment
    for iteration in range(n_iterations):
        # For each game: adjust OE by opponent's DE, adjust DE by opponent's OE
        # Clamp to avoid division by zero (epsilon guard)
        safe_de = np.maximum(de_vec[opp_idx], 1.0)
        safe_oe = np.maximum(oe_vec[opp_idx], 1.0)
        opp_de_factors = league_avg / safe_de
        opp_oe_factors = league_avg / safe_oe

        adj_oe_game = oe_arr * opp_de_factors
        adj_de_game = de_arr * opp_oe_factors

        # Weighted average per team
        new_oe = np.zeros(n_teams)
        new_de = np.zeros(n_teams)
        total_p = np.zeros(n_teams)

        np.add.at(new_oe, team_idx, adj_oe_game * poss_arr)
        np.add.at(new_de, team_idx, adj_de_game * poss_arr)
        np.add.at(total_p, team_idx, poss_arr)

        new_oe /= total_p
        new_de /= total_p

        # Check convergence
        max_diff = max(np.max(np.abs(new_oe - oe_vec)), np.max(np.abs(new_de - de_vec)))
        oe_vec = new_oe
        de_vec = new_de

        if max_diff < convergence_threshold:
            break

    # Build result
    result = pd.DataFrame({
        "TeamID": team_ids,
        "AdjOE": oe_vec,
        "AdjDE": de_vec,
        "AdjEM": oe_vec - de_vec,
        "RawOE": [adj_oe.get(t, league_avg) for t in team_ids],  # original raw
        "RawDE": [adj_de.get(t, league_avg) for t in team_ids],
        "n_games": [team_stats.loc[t, "n_games"] if t in team_stats.index else 0 for t in team_ids],
    })
    result["Season"] = season

    return result


def compute_all_seasons(
    detail_df: pd.DataFrame,
    seasons: list[int] | None = None,
    cache: bool = True,
) -> pd.DataFrame:
    """
    Compute adjusted efficiency for all seasons. Cache to CSV.

    Args:
        detail_df: Full MRegularSeasonDetailedResults
        seasons: List of seasons to compute (default: all available)
        cache: Whether to save/load from cache

    Returns:
        DataFrame with all seasons stacked
    """
    cache_path = CACHE_DIR / "adjusted_efficiency.csv"
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if seasons is None:
        seasons = sorted(detail_df["Season"].unique())

    # Load existing cache and check which seasons need computing
    cached = None
    if cache and cache_path.exists():
        cached = pd.read_csv(cache_path)
        cached_seasons = set(cached["Season"].unique())
        missing_seasons = [s for s in seasons if s not in cached_seasons]
        if not missing_seasons:
            return cached[cached["Season"].isin(seasons)]
    else:
        missing_seasons = seasons

    # Compute only missing seasons
    new_results = []
    for season in missing_seasons:
        result = compute_adjusted_efficiency(detail_df, season)
        if not result.empty:
            new_results.append(result)

    if not new_results and cached is not None:
        return cached[cached["Season"].isin(seasons)]
    elif not new_results:
        return pd.DataFrame()

    new_combined = pd.concat(new_results, ignore_index=True)

    # Merge with existing cache (don't overwrite other seasons)
    if cached is not None:
        # Remove seasons we just recomputed, keep the rest
        kept = cached[~cached["Season"].isin(missing_seasons)]
        combined = pd.concat([kept, new_combined], ignore_index=True)
    else:
        combined = new_combined

    if cache:
        combined.to_csv(cache_path, index=False)

    return combined


if __name__ == "__main__":
    detail = pd.read_csv(DATA_DIR / "MRegularSeasonDetailedResults.csv")
    teams = pd.read_csv(DATA_DIR / "MTeams.csv")

    # Test on 2025
    result = compute_adjusted_efficiency(detail, 2025)
    result = result.merge(teams[["TeamID", "TeamName"]], on="TeamID", how="left")
    result = result.sort_values("AdjEM", ascending=False)

    print("2025 Top 15 by AdjEM:")
    print(result.head(15)[["TeamName", "AdjOE", "AdjDE", "AdjEM", "n_games"]].round(2).to_string(index=False))
    print(f"\nTotal teams: {len(result)}")
    print(f"AdjEM range: {result['AdjEM'].min():.1f} to {result['AdjEM'].max():.1f}")
