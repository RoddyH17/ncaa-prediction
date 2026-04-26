"""
Massey Ordinals composite using all systems in last 2 weeks of regular season.

Per 2nd and 3rd place 2026 Kaggle solutions, this is the strongest non-seed
men's signal: aggregate 28+ independent ranking systems into mean/median/min
filtered to the final 2 weeks before Selection Sunday (i.e., systems are
"peaking" toward tournament time).

Returns one DataFrame per season with columns:
    TeamID, MasseyMean, MasseyMedian, MasseyMin

Women's ordinals are not in the Kaggle dataset; we return defaults if asked.
"""

from __future__ import annotations

import pandas as pd
import numpy as np


def get_massey_composite(
    massey_df: pd.DataFrame,
    season: int,
    days_window: int = 14,
) -> pd.DataFrame:
    """Aggregate Massey systems for a season into mean/median/min ordinals.

    Filters to RankingDayNum in [max_day - days_window, max_day].
    """
    if massey_df is None or massey_df.empty:
        return pd.DataFrame(columns=["TeamID", "MasseyMean", "MasseyMedian", "MasseyMin"])

    so = massey_df[massey_df["Season"] == season]
    if so.empty:
        return pd.DataFrame(columns=["TeamID", "MasseyMean", "MasseyMedian", "MasseyMin"])

    max_day = so["RankingDayNum"].max()
    late = so[so["RankingDayNum"] >= max_day - days_window]

    agg = late.groupby("TeamID")["OrdinalRank"].agg(
        MasseyMean="mean", MasseyMedian="median", MasseyMin="min"
    ).reset_index()
    return agg


def build_massey_lookup_all_seasons(massey_df: pd.DataFrame) -> dict:
    """Per (Season, TeamID): mean, median, min ordinal across all systems
    in the last 2 weeks of the regular season.
    """
    out: dict[tuple[int, int], dict[str, float]] = {}
    if massey_df is None or massey_df.empty:
        return out
    for season in massey_df["Season"].unique():
        agg = get_massey_composite(massey_df, int(season))
        for _, r in agg.iterrows():
            out[(int(season), int(r["TeamID"]))] = {
                "mean": float(r["MasseyMean"]),
                "median": float(r["MasseyMedian"]),
                "min": float(r["MasseyMin"]),
            }
    return out
