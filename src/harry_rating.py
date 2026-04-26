"""
harry_Rating: 1st-place 2026 Kaggle hand-tuned team rating.

Computes per-(Season, TeamID) custom ratings:

  harry_Rating = NetEff * (1 + opp_pts_minmax) * power_conf_minmax * top12_minmax

Where:
  NetEff:   2% trim-mean season-average net efficiency (offense - defense)
  opp_pts:  total opponent quality points across the regular season
            quality tiers: T2_seed<=4 -> 6, T2_seed<=16 -> 4, NIT -> 2, else 0.25
  power_conf: 1 if conference in [big_ten, acc, sec, big_twelve, big_east, pac_twelve]
  top12:    1 if AP poll week-6 ranking <= 12 (men's only; we approximate via
            Massey AP system at RankingDayNum closest to 50)

Provides also `opp_qlty_pts_won_diff`: differential of quality wins.

Reference: 1st place writeup
  https://www.kaggle.com/competitions/march-machine-learning-mania-2026/writeups/march-machine-learning-mania-2026-1st-place-solut
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import trim_mean
from sklearn.preprocessing import MinMaxScaler


POWER_CONFS = ["big_ten", "acc", "sec", "big_twelve", "big_east", "pac_twelve"]


def _possessions(fga, fgm, oreb, to, fta):
    return fga - oreb + to + 0.475 * fta


def compute_team_season_efficiency(regular_detail: pd.DataFrame) -> pd.DataFrame:
    """For each (Season, TeamID), compute 2%-trimmed mean net efficiency.

    Aggregates per-game possession-normalized offensive and defensive efficiency
    over the regular season. Returns DataFrame with cols [Season, TeamID, NetEff].
    """
    if regular_detail is None or regular_detail.empty:
        return pd.DataFrame(columns=["Season", "TeamID", "NetEff"])

    df = regular_detail.copy()
    df["WPoss"] = _possessions(df["WFGA"], df["WFGM"], df["WOR"], df["WTO"], df["WFTA"])
    df["LPoss"] = _possessions(df["LFGA"], df["LFGM"], df["LOR"], df["LTO"], df["LFTA"])
    # Per-team rows (T1 perspective)
    rows = []
    for _, g in df.iterrows():
        # Winner perspective
        if g["WPoss"] > 0 and g["LPoss"] > 0:
            rows.append({"Season": g["Season"], "TeamID": g["WTeamID"],
                         "OffEff": (g["WScore"] / g["WPoss"]) * 70,
                         "DefEff": (g["LScore"] / g["LPoss"]) * 70})
            rows.append({"Season": g["Season"], "TeamID": g["LTeamID"],
                         "OffEff": (g["LScore"] / g["LPoss"]) * 70,
                         "DefEff": (g["WScore"] / g["WPoss"]) * 70})
    if not rows:
        return pd.DataFrame(columns=["Season", "TeamID", "NetEff"])
    pg = pd.DataFrame(rows)
    pg["NetEff_game"] = pg["OffEff"] - pg["DefEff"]

    # 2% trim-mean per (Season, TeamID)
    out = pg.groupby(["Season", "TeamID"])["NetEff_game"].agg(
        lambda x: trim_mean(x, 0.02)
    ).reset_index().rename(columns={"NetEff_game": "NetEff"})
    return out


def opp_quality_points(season_seed_map: dict, season_nit_set: set) -> callable:
    """Return f(opp_team_id) -> quality points, given seed/NIT info for the season."""
    def f(opp):
        s = season_seed_map.get(opp)
        if s is not None and s <= 4:
            return 6.0
        if s is not None and s <= 16:
            return 4.0
        if opp in season_nit_set:
            return 2.0
        return 0.25
    return f


def compute_opp_quality_features(
    regular_compact: pd.DataFrame,
    seeds: pd.DataFrame,
    nit_seeds: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Per (Season, TeamID): compute opp_qlty_pts (sum across all games) and
    opp_qlty_pts_won (sum across wins only).
    """
    rows = []
    for season in regular_compact["Season"].unique():
        season_games = regular_compact[regular_compact["Season"] == season]
        season_seeds = seeds[seeds["Season"] == season]
        seed_map = {}
        for _, sr in season_seeds.iterrows():
            seed_map[int(sr["TeamID"])] = int(_seed_num(sr["Seed"]))
        nit_set = set()
        if nit_seeds is not None and not nit_seeds.empty:
            n = nit_seeds[nit_seeds["Season"] == season]
            nit_set = set(int(t) for t in n["TeamID"].unique())
        f = opp_quality_points(seed_map, nit_set)

        agg = {}  # team_id -> (total_pts, won_pts)
        for _, g in season_games.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            agg.setdefault(w, [0.0, 0.0])
            agg.setdefault(l, [0.0, 0.0])
            pts_for_w = f(l)
            pts_for_l = f(w)
            agg[w][0] += pts_for_w
            agg[w][1] += pts_for_w  # won
            agg[l][0] += pts_for_l
            # not won, so don't add to [1]
        for tid, (total, won) in agg.items():
            rows.append({"Season": season, "TeamID": tid,
                         "opp_qlty_pts": total, "opp_qlty_pts_won": won})
    return pd.DataFrame(rows)


def _seed_num(seed_str: str) -> int:
    import re
    m = re.search(r"(\d+)", str(seed_str))
    return int(m.group(1)) if m else 16


def add_power_conf_flag(df: pd.DataFrame, conferences: pd.DataFrame) -> pd.DataFrame:
    """Add 'Power' column to df via (Season, TeamID) -> conference -> in_power_set."""
    if conferences is None or conferences.empty:
        df = df.copy()
        df["Power"] = 1
        return df
    conf_map = {}
    for _, r in conferences.iterrows():
        conf_map[(int(r["Season"]), int(r["TeamID"]))] = str(r["ConfAbbrev"]).lower()
    df = df.copy()
    df["Power"] = df.apply(
        lambda r: 1 if conf_map.get((int(r["Season"]), int(r["TeamID"])), "") in POWER_CONFS else 0,
        axis=1,
    )
    return df


def add_top12_flag(df: pd.DataFrame, massey: pd.DataFrame) -> pd.DataFrame:
    """Add 'Top12' column based on AP poll at RankingDayNum closest to 50.

    For seasons where AP isn't in the Massey file, defaults to 0.
    """
    df = df.copy()
    df["Top12"] = 0
    if massey is None or massey.empty:
        return df
    ap = massey[massey["SystemName"] == "AP"]
    if ap.empty:
        return df
    # Pick ranking day closest to 50 (week 6)
    ap = ap.copy()
    ap["dist"] = (ap["RankingDayNum"] - 50).abs()
    # For each season, find min dist
    season_day = {}
    for season in ap["Season"].unique():
        sub = ap[ap["Season"] == season]
        best_day = sub.loc[sub["dist"].idxmin(), "RankingDayNum"]
        season_day[int(season)] = int(best_day)
    top12_set = set()
    for season, day in season_day.items():
        sub = ap[(ap["Season"] == season) & (ap["RankingDayNum"] == day)]
        for _, r in sub.iterrows():
            if r["OrdinalRank"] <= 12:
                top12_set.add((int(r["Season"]), int(r["TeamID"])))
    df["Top12"] = df.apply(
        lambda r: 1 if (int(r["Season"]), int(r["TeamID"])) in top12_set else 0,
        axis=1,
    )
    return df


def compute_harry_rating(
    eff: pd.DataFrame,
    opp_q: pd.DataFrame,
    conferences: pd.DataFrame | None,
    massey: pd.DataFrame | None = None,
    is_womens: bool = False,
) -> pd.DataFrame:
    """Combine NetEff with hand-tuned MinMax scalers to produce harry_Rating.

    Per the 1st place writeup, scalers are season-stratified within gender.
    We apply per-season MinMax to get a normalized rating that's comparable
    across years.
    """
    df = eff.merge(opp_q, on=["Season", "TeamID"], how="left")
    df = add_power_conf_flag(df, conferences)
    df = add_top12_flag(df, massey)
    df = df.fillna({"opp_qlty_pts": df["opp_qlty_pts"].median(),
                    "opp_qlty_pts_won": df["opp_qlty_pts_won"].median()})

    if is_womens:
        opp_range = (-0.5, 0.5)
        conf_range = (1.0, 1.1)
        top12_range = None
    else:
        opp_range = (-0.55, 0.55)
        conf_range = (1.0, 1.3)
        top12_range = (1.0, 1.2)

    df["opp_pts_mm"] = 1.0
    df["power_mm"] = 1.0
    df["top12_mm"] = 1.0

    for season, sub in df.groupby("Season"):
        idx = sub.index
        if len(sub) >= 2:
            sc = MinMaxScaler(feature_range=opp_range)
            df.loc[idx, "opp_pts_mm"] = sc.fit_transform(sub[["opp_qlty_pts"]]).flatten()
            sc2 = MinMaxScaler(feature_range=conf_range)
            df.loc[idx, "power_mm"] = sc2.fit_transform(sub[["Power"]]).flatten() \
                if sub["Power"].nunique() > 1 else conf_range[0]
            if top12_range is not None and sub["Top12"].nunique() > 1:
                sc3 = MinMaxScaler(feature_range=top12_range)
                df.loc[idx, "top12_mm"] = sc3.fit_transform(sub[["Top12"]]).flatten()

    df["harry_rating"] = (
        df["NetEff"] * (1 + df["opp_pts_mm"]) * df["power_mm"] * df["top12_mm"]
    )
    return df[["Season", "TeamID", "NetEff", "opp_qlty_pts", "opp_qlty_pts_won",
               "Power", "Top12", "harry_rating"]]


def build_harry_features(data: dict, seasons: list[int], is_womens: bool = False) -> pd.DataFrame:
    """Compute harry_Rating + opp_qlty_pts_won for all teams across seasons."""
    rd = data.get("regular_detail")
    rc = data.get("regular_compact")
    seeds = data.get("seeds")
    confs = data.get("conferences")
    massey = data.get("massey")

    eff = compute_team_season_efficiency(rd)
    eff = eff[eff["Season"].isin(seasons)]
    opp_q = compute_opp_quality_features(rc[rc["Season"].isin(seasons)], seeds, nit_seeds=None)
    return compute_harry_rating(eff, opp_q, confs, massey, is_womens=is_womens)
