"""
Feature engineering for NCAA tournament prediction.

Feature layers:
  L1: Static team ratings (KenPom, seed, conference)
  L2: Matchup features (rating diff, tempo mismatch, geography)
  L3: Temporal / momentum (rolling form, late-season trend)
  L4: Market features (odds-implied probability, disagreement signal)
"""

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Layer 1: Static team ratings
# ---------------------------------------------------------------------------

def build_seed_features(seeds_df: pd.DataFrame) -> pd.DataFrame:
    """Extract numeric seed from string (e.g., 'W01' -> 1, 'Z16a' -> 16)."""
    df = seeds_df.copy()
    df["SeedNum"] = df["Seed"].str.extract(r"(\d+)").astype(int)
    return df


def build_rating_features(massey_df: pd.DataFrame, season: int, day_cutoff: int = 133) -> pd.DataFrame:
    """
    Build team rating features from Massey ordinals.
    Use latest pre-tournament snapshot (day <= day_cutoff).
    Key systems: 'POM' (KenPom), 'SAG' (Sagarin), 'MOR' (Massey), 'DOL' (Dolphin).
    """
    df = massey_df[(massey_df["Season"] == season) & (massey_df["RankingDayNum"] <= day_cutoff)]
    # Get latest day for each system
    latest = df.groupby("SystemName")["RankingDayNum"].max().reset_index()
    latest.columns = ["SystemName", "LatestDay"]
    df = df.merge(latest, on="SystemName")
    df = df[df["RankingDayNum"] == df["LatestDay"]]
    # Pivot: rows=TeamID, cols=SystemName, values=OrdinalRank
    pivot = df.pivot_table(index="TeamID", columns="SystemName", values="OrdinalRank")
    return pivot


# ---------------------------------------------------------------------------
# Layer 2: Matchup features
# ---------------------------------------------------------------------------

def build_matchup_features(team_a_id: int, team_b_id: int, ratings: pd.DataFrame, seeds: pd.DataFrame) -> dict:
    """Build feature vector for a single matchup."""
    features = {}
    # Seed diff
    seed_a = seeds.loc[seeds["TeamID"] == team_a_id, "SeedNum"].values
    seed_b = seeds.loc[seeds["TeamID"] == team_b_id, "SeedNum"].values
    features["seed_diff"] = (seed_a[0] - seed_b[0]) if len(seed_a) and len(seed_b) else 0

    # Rating diffs (for each system available)
    for system in ["POM", "SAG", "MOR", "DOL"]:
        if system in ratings.columns:
            rank_a = ratings.loc[team_a_id, system] if team_a_id in ratings.index else np.nan
            rank_b = ratings.loc[team_b_id, system] if team_b_id in ratings.index else np.nan
            features[f"rank_diff_{system}"] = rank_a - rank_b

    return features


# ---------------------------------------------------------------------------
# Layer 3: Temporal / momentum
# ---------------------------------------------------------------------------

def build_momentum_features(results_df: pd.DataFrame, team_id: int, season: int, last_n: int = 10) -> dict:
    """Rolling win% and margin over last N games."""
    team_games = results_df[
        (results_df["Season"] == season) &
        ((results_df["WTeamID"] == team_id) | (results_df["LTeamID"] == team_id))
    ].sort_values("DayNum", ascending=False).head(last_n)

    if len(team_games) == 0:
        return {"momentum_win_pct": 0.5, "momentum_avg_margin": 0.0}

    wins = (team_games["WTeamID"] == team_id).sum()
    margins = []
    for _, row in team_games.iterrows():
        if row["WTeamID"] == team_id:
            margins.append(row["WScore"] - row["LScore"])
        else:
            margins.append(row["LScore"] - row["WScore"])

    return {
        "momentum_win_pct": wins / len(team_games),
        "momentum_avg_margin": np.mean(margins),
    }


# ---------------------------------------------------------------------------
# Layer 4: Market features
# ---------------------------------------------------------------------------

def build_market_features(model_prob: float, market_prob: float) -> dict:
    """Disagreement signal between model and market."""
    return {
        "market_prob": market_prob,
        "model_market_diff": model_prob - market_prob,
        "abs_disagreement": abs(model_prob - market_prob),
    }
