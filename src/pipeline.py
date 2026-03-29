"""
End-to-end pipeline: load data → build features → construct train/test matrices.

Usage:
    from src.pipeline import build_dataset
    X_train, y_train, X_test, y_test = build_dataset(train_seasons=[2014,...], test_seasons=[2026])
"""

import numpy as np
import pandas as pd
from pathlib import Path

from src.data_collection import DATA_DIR, load_csv, load_all_mens_data, fetch_barttorvik_ratings
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


def build_efficiency_for_season(data: dict, season: int) -> pd.DataFrame:
    """
    Compute tempo-adjusted efficiency metrics from box scores.
    This is our own version of KenPom-style efficiency ratings.

    Returns DataFrame indexed by TeamID with columns:
      off_eff, def_eff, net_eff, tempo
      (plus SOS-adjusted versions using iterative adjustment)
    """
    detail = data.get("regular_detail")
    if detail is None:
        return pd.DataFrame()

    sg = detail[detail["Season"] == season]
    if sg.empty:
        return pd.DataFrame()

    # Accumulate per-team stats
    team_stats = {}

    for _, g in sg.iterrows():
        w, l = g["WTeamID"], g["LTeamID"]

        # Estimate possessions for each team
        w_poss = g["WFGA"] - g["WOR"] + g["WTO"] + 0.44 * g["WFTA"]
        l_poss = g["LFGA"] - g["LOR"] + g["LTO"] + 0.44 * g["LFTA"]
        # Average possessions (should be roughly equal)
        avg_poss = (w_poss + l_poss) / 2
        avg_poss = max(avg_poss, 1)

        for tid, pts_for, pts_against in [(w, g["WScore"], g["LScore"]),
                                           (l, g["LScore"], g["WScore"])]:
            if tid not in team_stats:
                team_stats[tid] = {"games": 0, "poss": 0,
                                   "pts_for": 0, "pts_against": 0,
                                   "opponents": []}
            ts = team_stats[tid]
            ts["games"] += 1
            ts["poss"] += avg_poss
            ts["pts_for"] += pts_for
            ts["pts_against"] += pts_against
            opp = l if tid == w else w
            ts["opponents"].append(opp)

    # Phase 1: raw efficiency
    raw = {}
    for tid, ts in team_stats.items():
        total_poss = max(ts["poss"], 1)
        raw[tid] = {
            "off_eff": ts["pts_for"] / total_poss * 100,
            "def_eff": ts["pts_against"] / total_poss * 100,
            "tempo": ts["poss"] / ts["games"],
        }
        raw[tid]["net_eff"] = raw[tid]["off_eff"] - raw[tid]["def_eff"]

    # Phase 2: SOS-adjusted efficiency via iterative fixed-point method
    # Solve: AdjO_i = RawO_i + mean(AdjD_opponents - league_avg) * alpha
    # Iterate until convergence (like KenPom's approach)
    adj = {tid: dict(r) for tid, r in raw.items()}
    league_off = np.mean([r["off_eff"] for r in raw.values()])
    league_def = np.mean([r["def_eff"] for r in raw.values()])

    for iteration in range(50):
        new_adj = {}
        for tid, ts in team_stats.items():
            opp_adj_def = np.mean([adj[o]["def_eff"] for o in ts["opponents"] if o in adj])
            opp_adj_off = np.mean([adj[o]["off_eff"] for o in ts["opponents"] if o in adj])

            # How much harder/easier was the schedule?
            sos_off = opp_adj_def - league_def  # faced tough defenses -> boost offense
            sos_def = opp_adj_off - league_off  # faced tough offenses -> credit defense

            new_adj[tid] = {
                "off_eff": raw[tid]["off_eff"] + sos_off,
                "def_eff": raw[tid]["def_eff"] - sos_def,
                "tempo": raw[tid]["tempo"],
            }
            new_adj[tid]["net_eff"] = new_adj[tid]["off_eff"] - new_adj[tid]["def_eff"]
        adj = new_adj

    rows = []
    for tid in adj:
        rows.append({
            "TeamID": tid,
            "off_eff": adj[tid]["off_eff"],
            "def_eff": adj[tid]["def_eff"],
            "net_eff": adj[tid]["net_eff"],
            "tempo": adj[tid]["tempo"],
        })
    return pd.DataFrame(rows).set_index("TeamID")


def build_four_factors_for_season(data: dict, season: int) -> pd.DataFrame:
    """
    Compute Dean Oliver's Four Factors from season box scores.
    Returns DataFrame indexed by TeamID with columns:
      efg_pct, to_pct, or_pct, ft_rate (offensive)
      opp_efg_pct, opp_to_pct, opp_or_pct, opp_ft_rate (defensive)
    """
    detail = data.get("regular_detail")
    if detail is None:
        return pd.DataFrame()

    sg = detail[detail["Season"] == season]
    if sg.empty:
        return pd.DataFrame()

    records = {}

    for _, g in sg.iterrows():
        w, l = g["WTeamID"], g["LTeamID"]

        # Winner offense
        w_fga = max(g["WFGA"], 1)
        w_fta = max(g["WFTA"], 1)
        w_off = {
            "efg": (g["WFGM"] + 0.5 * g["WFGM3"]) / w_fga,
            "to":  g["WTO"] / max(w_fga + 0.44 * w_fta + g["WTO"], 1),
            "or":  g["WOR"] / max(g["WOR"] + g["LDR"], 1),
            "ftr": g["WFTM"] / w_fga,
        }
        # Loser offense
        l_fga = max(g["LFGA"], 1)
        l_fta = max(g["LFTA"], 1)
        l_off = {
            "efg": (g["LFGM"] + 0.5 * g["LFGM3"]) / l_fga,
            "to":  g["LTO"] / max(l_fga + 0.44 * l_fta + g["LTO"], 1),
            "or":  g["LOR"] / max(g["LOR"] + g["WDR"], 1),
            "ftr": g["LFTM"] / l_fga,
        }

        for tid, off, opp_off in [(w, w_off, l_off), (l, l_off, w_off)]:
            if tid not in records:
                records[tid] = {"games": 0, "efg": 0, "to": 0, "or": 0, "ftr": 0,
                                "opp_efg": 0, "opp_to": 0, "opp_or": 0, "opp_ftr": 0}
            r = records[tid]
            r["games"] += 1
            r["efg"] += off["efg"]
            r["to"] += off["to"]
            r["or"] += off["or"]
            r["ftr"] += off["ftr"]
            r["opp_efg"] += opp_off["efg"]
            r["opp_to"] += opp_off["to"]
            r["opp_or"] += opp_off["or"]
            r["opp_ftr"] += opp_off["ftr"]

    rows = []
    for tid, r in records.items():
        n = r["games"]
        rows.append({
            "TeamID": tid,
            "efg_pct": r["efg"] / n,
            "to_pct": r["to"] / n,
            "or_pct": r["or"] / n,
            "ft_rate": r["ftr"] / n,
            "opp_efg_pct": r["opp_efg"] / n,
            "opp_to_pct": r["opp_to"] / n,
            "opp_or_pct": r["opp_or"] / n,
            "opp_ft_rate": r["opp_ftr"] / n,
        })
    return pd.DataFrame(rows).set_index("TeamID")


def build_barttorvik_for_season(season: int) -> pd.DataFrame:
    """
    Load Barttorvik continuous efficiency ratings for a season.
    Returns DataFrame indexed by TeamID with columns:
      AdjOE, AdjDE, NetRtg, Barthag, AdjTempo
    """
    df = fetch_barttorvik_ratings(season)
    if df is None or df.empty:
        return pd.DataFrame()
    # Drop duplicates, keeping the higher-rated entry
    df = df.sort_values("NetRtg", ascending=False).drop_duplicates(subset="TeamID", keep="first")
    return df.set_index("TeamID")


def build_coach_features_for_season(data: dict, season: int) -> pd.DataFrame:
    """
    Build coach tournament experience features.
    Returns DataFrame indexed by TeamID with columns:
      coach_tourney_apps: number of prior tournament appearances as head coach
      coach_seasons: total seasons as head coach at any school
    """
    coaches = data.get("coaches")
    seeds = data.get("seeds")
    if coaches is None or seeds is None:
        return pd.DataFrame()

    # Current coaches for this season
    current = coaches[(coaches["Season"] == season)]
    if current.empty:
        return pd.DataFrame()

    # Count prior tournament appearances per coach
    # A coach "appeared" if their team got a seed in a prior year
    past_coaches = coaches[coaches["Season"] < season]
    past_seeds = seeds[seeds["Season"] < season]

    # Merge to find which coach-seasons had tournament bids
    if not past_coaches.empty and not past_seeds.empty:
        merged = past_coaches.merge(past_seeds[["Season", "TeamID"]], on=["Season", "TeamID"], how="inner")
        coach_apps = merged.groupby("CoachName").size().to_dict()
    else:
        coach_apps = {}

    # Total coaching seasons per coach
    coach_seasons = past_coaches.groupby("CoachName").size().to_dict()

    rows = []
    for _, row in current.iterrows():
        coach = row["CoachName"]
        rows.append({
            "TeamID": row["TeamID"],
            "coach_tourney_apps": coach_apps.get(coach, 0),
            "coach_seasons": coach_seasons.get(coach, 0),
        })

    return pd.DataFrame(rows).drop_duplicates(subset="TeamID", keep="last").set_index("TeamID")


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
        efficiency = build_efficiency_for_season(data, season)
        four_factors = build_four_factors_for_season(data, season)
        barttorvik = build_barttorvik_for_season(season)
        coach_feats = build_coach_features_for_season(data, season)
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

            # L1b: Barttorvik continuous efficiency ratings
            bart_cols = [("AdjOE", "bart_adjoe"), ("AdjDE", "bart_adjde"),
                         ("NetRtg", "bart_net"), ("Barthag", "bart_barthag"),
                         ("AdjTempo", "bart_tempo")]
            for src_col, feat_name in bart_cols:
                val_a = barttorvik.loc[m["TeamA"], src_col] if (not barttorvik.empty and m["TeamA"] in barttorvik.index) else np.nan
                val_b = barttorvik.loc[m["TeamB"], src_col] if (not barttorvik.empty and m["TeamB"] in barttorvik.index) else np.nan
                feat[f"{feat_name}_diff"] = val_a - val_b

            # L2a: Tempo-adjusted efficiency (our own AdjEM)
            for col in ["off_eff", "def_eff", "net_eff", "tempo"]:
                val_a = efficiency.loc[m["TeamA"], col] if (not efficiency.empty and m["TeamA"] in efficiency.index) else np.nan
                val_b = efficiency.loc[m["TeamB"], col] if (not efficiency.empty and m["TeamB"] in efficiency.index) else np.nan
                feat[f"{col}_diff"] = val_a - val_b
                if col == "net_eff":
                    feat[f"{col}_A"] = val_a
                    feat[f"{col}_B"] = val_b

            # L2b: Four Factors (continuous efficiency metrics)
            ff_cols = ["efg_pct", "to_pct", "or_pct", "ft_rate",
                       "opp_efg_pct", "opp_to_pct", "opp_or_pct", "opp_ft_rate"]
            for col in ff_cols:
                val_a = four_factors.loc[m["TeamA"], col] if (not four_factors.empty and m["TeamA"] in four_factors.index) else np.nan
                val_b = four_factors.loc[m["TeamB"], col] if (not four_factors.empty and m["TeamB"] in four_factors.index) else np.nan
                feat[f"{col}_diff"] = val_a - val_b

            # L3: Momentum
            mom_a = mom_map.get(m["TeamA"], {})
            mom_b = mom_map.get(m["TeamB"], {})
            feat["momentum_winpct_A"] = mom_a.get("momentum_win_pct", 0.5)
            feat["momentum_winpct_B"] = mom_b.get("momentum_win_pct", 0.5)
            feat["momentum_winpct_diff"] = feat["momentum_winpct_A"] - feat["momentum_winpct_B"]
            feat["momentum_margin_A"] = mom_a.get("momentum_avg_margin", 0.0)
            feat["momentum_margin_B"] = mom_b.get("momentum_avg_margin", 0.0)
            feat["momentum_margin_diff"] = feat["momentum_margin_A"] - feat["momentum_margin_B"]

            # L4: Coach tournament experience
            for col in ["coach_tourney_apps", "coach_seasons"]:
                val_a = coach_feats.loc[m["TeamA"], col] if (not coach_feats.empty and m["TeamA"] in coach_feats.index) else 0
                val_b = coach_feats.loc[m["TeamB"], col] if (not coach_feats.empty and m["TeamB"] in coach_feats.index) else 0
                feat[f"{col}_diff"] = val_a - val_b

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
