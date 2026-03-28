"""
End-to-end pipeline: load data → build features → construct train/test matrices.

Usage:
    from src.pipeline import build_dataset
    X_train, y_train, X_test, y_test = build_dataset(train_seasons=[2014,...], test_seasons=[2026])
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_collection import DATA_DIR, load_csv, load_all_mens_data
from src.features import build_seed_features, build_rating_features, build_momentum_features


def _parse_seed_num(seed_str: str) -> int:
    """Extract numeric seed: 'W01' -> 1, 'Z16a' -> 16."""
    import re
    m = re.search(r"(\d+)", seed_str)
    return int(m.group(1)) if m else 16


def build_tourney_matchups(data: dict, season: int) -> pd.DataFrame:
    """
    Build one row per tournament game for a given season.
    Uses compact results (available for all years) with seed info.
    """
    tourney = data["tourney_compact"]
    seeds = data["seeds"]

    games = tourney[tourney["Season"] == season].copy()
    season_seeds = seeds[seeds["Season"] == season].copy()
    season_seeds["SeedNum"] = season_seeds["Seed"].apply(_parse_seed_num)
    seed_map = dict(zip(season_seeds["TeamID"], season_seeds["SeedNum"]))

    rows = []
    for _, g in games.iterrows():
        w, l = g["WTeamID"], g["LTeamID"]
        # Canonical ordering: lower TeamID = TeamA
        if w < l:
            team_a, team_b, result = w, l, 1
        else:
            team_a, team_b, result = l, w, 0

        rows.append({
            "Season": season,
            "TeamA": team_a,
            "TeamB": team_b,
            "SeedA": seed_map.get(team_a, 16),
            "SeedB": seed_map.get(team_b, 16),
            "Result": result,  # 1 if TeamA won
        })

    return pd.DataFrame(rows)


def build_rating_features_for_season(data: dict, season: int, day_cutoff: int = 133) -> pd.DataFrame:
    """
    Build team-level rating features from Massey ordinals for one season.
    Returns DataFrame indexed by TeamID with columns for key ranking systems.
    """
    massey = data["massey"]
    key_systems = ["POM", "SAG", "MOR", "DOL", "COL", "AP", "USA", "WOL", "RPI"]

    df = massey[(massey["Season"] == season) & (massey["RankingDayNum"] <= day_cutoff)]
    df = df[df["SystemName"].isin(key_systems)]

    if df.empty:
        return pd.DataFrame()

    # Latest available day per system
    latest = df.groupby("SystemName")["RankingDayNum"].max().reset_index()
    latest.columns = ["SystemName", "LatestDay"]
    df = df.merge(latest, on="SystemName")
    df = df[df["RankingDayNum"] == df["LatestDay"]]

    pivot = df.pivot_table(index="TeamID", columns="SystemName", values="OrdinalRank")
    return pivot


def build_momentum_for_season(data: dict, season: int, team_ids: list, last_n: int = 10) -> pd.DataFrame:
    """Build momentum features for all teams in a season."""
    results = data["regular_compact"]
    rows = []
    for tid in team_ids:
        feats = build_momentum_features(results, tid, season, last_n)
        feats["TeamID"] = tid
        feats["Season"] = season
        rows.append(feats)
    return pd.DataFrame(rows)


def build_feature_matrix(data: dict, seasons: list[int]) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Build full feature matrix for given seasons.
    Returns (X, y) where X is a DataFrame of features and y is binary outcome.
    """
    all_matchups = []
    all_features = []

    for season in seasons:
        matchups = build_tourney_matchups(data, season)
        if matchups.empty:
            continue

        ratings = build_rating_features_for_season(data, season)
        all_team_ids = list(set(matchups["TeamA"].tolist() + matchups["TeamB"].tolist()))
        momentum = build_momentum_for_season(data, season, all_team_ids)
        mom_map = {row["TeamID"]: row for _, row in momentum.iterrows()}

        for _, m in matchups.iterrows():
            feat = {
                "Season": season,
                "TeamA": m["TeamA"],
                "TeamB": m["TeamB"],
                # L1: Seeds
                "seed_diff": m["SeedA"] - m["SeedB"],
                "seed_A": m["SeedA"],
                "seed_B": m["SeedB"],
            }

            # L1: Rating diffs (fixed set of systems for consistent columns)
            for sys_name in ["POM", "SAG", "MOR", "DOL", "COL", "AP", "USA", "WOL", "RPI"]:
                rank_a = np.nan
                rank_b = np.nan
                if not ratings.empty and sys_name in ratings.columns:
                    rank_a = ratings.loc[m["TeamA"], sys_name] if m["TeamA"] in ratings.index else np.nan
                    rank_b = ratings.loc[m["TeamB"], sys_name] if m["TeamB"] in ratings.index else np.nan
                feat[f"rank_diff_{sys_name}"] = rank_a - rank_b
                feat[f"rank_A_{sys_name}"] = rank_a
                feat[f"rank_B_{sys_name}"] = rank_b

            # L3: Momentum
            mom_a = mom_map.get(m["TeamA"], {})
            mom_b = mom_map.get(m["TeamB"], {})
            feat["momentum_winpct_A"] = mom_a.get("momentum_win_pct", 0.5)
            feat["momentum_winpct_B"] = mom_b.get("momentum_win_pct", 0.5)
            feat["momentum_winpct_diff"] = feat["momentum_winpct_A"] - feat["momentum_winpct_B"]
            feat["momentum_margin_A"] = mom_a.get("momentum_avg_margin", 0.0)
            feat["momentum_margin_B"] = mom_b.get("momentum_avg_margin", 0.0)
            feat["momentum_margin_diff"] = feat["momentum_margin_A"] - feat["momentum_margin_B"]

            all_features.append(feat)
            all_matchups.append(m["Result"])

    X = pd.DataFrame(all_features)
    y = np.array(all_matchups)
    return X, y


_ID_COLS = ["Season", "TeamA", "TeamB"]


def make_build_features_fn(data: dict):
    """Closure adapter for evaluation.py's LOTO interface.
    Returns a function with signature: (seasons) -> (X, y)
    """
    def build_features_fn(seasons):
        return build_feature_matrix(data, seasons)
    return build_features_fn


def build_dataset(
    data: dict | None = None,
    train_seasons: list[int] | None = None,
    test_seasons: list[int] | None = None,
):
    """
    Convenience function: load data, build train/test splits.
    Default: train on 2003-2025, test on 2026.
    """
    if data is None:
        data = load_all_mens_data()

    if train_seasons is None:
        train_seasons = list(range(2003, 2026))
    if test_seasons is None:
        test_seasons = [2026]

    X_train, y_train = build_feature_matrix(data, train_seasons)
    X_test, y_test = build_feature_matrix(data, test_seasons)

    print(f"Train: {X_train.shape[0]} games across {len(train_seasons)} seasons")
    print(f"Test:  {X_test.shape[0]} games across {len(test_seasons)} seasons")

    return X_train, y_train, X_test, y_test


if __name__ == "__main__":
    data = load_all_mens_data()
    X, y = build_feature_matrix(data, list(range(2014, 2027)))
    print(f"\nTotal feature matrix: {X.shape}")
    print(f"Win rate (TeamA): {y.mean():.3f}")
    print(f"\nFeature columns:\n{list(X.columns)}")
    print(f"\nSample row:\n{X.iloc[0]}")
