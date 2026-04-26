"""
Build women's tournament model using the same pipeline.

Reuses pipeline functions by passing a women's data dict with the same keys
as the men's dict. Skips features that are men's-only (Massey rankings, coaches).

Generates:
  - LOTO Brier evaluation on women's tournaments 2014-2025
  - 2026 women's predictions
  - Updated submission CSV with women's predictions filled in
"""

import sys
sys.path.insert(0, ".")

import re
import requests
import io
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss
from rapidfuzz import process, fuzz

from src.data_collection import DATA_DIR
from src.pipeline import (
    _parse_seed_num, build_efficiency_for_season, build_four_factors_for_season,
    build_momentum_for_season, build_tourney_matchups,
)


def load_womens_data() -> dict:
    """Load all women's CSVs into a dict matching the men's structure."""
    files = {
        "regular_detail": "WRegularSeasonDetailedResults.csv",
        "regular_compact": "WRegularSeasonCompactResults.csv",
        "tourney_detail": "WNCAATourneyDetailedResults.csv",
        "tourney_compact": "WNCAATourneyCompactResults.csv",
        "seeds": "WNCAATourneySeeds.csv",
        "slots": "WNCAATourneySlots.csv",
        "teams": "WTeams.csv",
    }
    data = {}
    for key, fname in files.items():
        path = DATA_DIR / fname
        if path.exists():
            data[key] = pd.read_csv(path)
    # Empty placeholders for men's-only keys
    data["massey"] = pd.DataFrame(columns=["Season", "RankingDayNum", "SystemName",
                                            "TeamID", "OrdinalRank"])
    data["coaches"] = pd.DataFrame(columns=["Season", "TeamID", "CoachName"])
    return data


def fetch_womens_barttorvik(season: int, end_date: str = None) -> pd.DataFrame | None:
    """Scrape Barttorvik women's data, optionally with pre-tournament cutoff."""
    cache_path = DATA_DIR / "external" / f"barttorvik_w_{season}.csv"
    if cache_path.exists():
        return pd.read_csv(cache_path).set_index("TeamID")

    if end_date is None:
        end_date = f"{season}0315"  # Selection Sunday approx
    begin_date = f"{season-1}1101"

    session = requests.Session()
    session.headers.update({"User-Agent": "Mozilla/5.0"})
    url = (f"https://barttorvik.com/ncaaw/trank.php?year={season}"
           f"&conyes=0&begin={begin_date}&end={end_date}")
    session.get(url, timeout=15)
    resp = session.post(url, data={"js_test_submitted": "1"}, timeout=15)
    if resp.status_code != 200 or len(resp.text) < 1000:
        return None
    tables = pd.read_html(io.StringIO(resp.text))
    if not tables:
        return None
    raw = tables[0]
    raw.columns = [c[1] if isinstance(c, tuple) else c for c in raw.columns]

    # Build name map
    spell = pd.read_csv(DATA_DIR / "WTeamSpellings.csv", encoding="latin-1")
    teams = pd.read_csv(DATA_DIR / "WTeams.csv")
    spell_map = {}
    for _, r in spell.iterrows():
        spell_map[str(r["TeamNameSpelling"]).lower().strip()] = int(r["TeamID"])
    for _, r in teams.iterrows():
        spell_map[str(r["TeamName"]).lower().strip()] = int(r["TeamID"])

    def extract(v):
        if pd.isna(v): return None
        m = re.search(r"(-?\d+\.\d+|\d+)", str(v))
        if not m: return None
        try: return float(m.group(1))
        except: return None

    rows = []
    for _, row in raw.iterrows():
        name = str(row.get("Team", "")).strip()
        name_clean = re.sub(r"\s+\d+\s*-\s*\d+.*$", "", name).strip()
        adjoe = extract(row.get("AdjOE"))
        adjde = extract(row.get("AdjDE"))
        barthag = extract(row.get("Barthag"))
        tempo = extract(row.get("Adj T."))
        if adjoe is None or adjde is None: continue

        tid = spell_map.get(name_clean.lower())
        if tid is None:
            result = process.extractOne(name_clean.lower(), list(spell_map.keys()),
                                         scorer=fuzz.WRatio, score_cutoff=80)
            if result:
                tid = spell_map[result[0]]
        if tid is None: continue

        rows.append({"TeamID": tid, "AdjOE": adjoe, "AdjDE": adjde,
                     "Barthag": barthag if barthag else 0.5,
                     "AdjTempo": tempo if tempo else 67,
                     "NetRtg": adjoe - adjde})

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    return df.set_index("TeamID")


def build_womens_features(data: dict, seasons: list[int]) -> tuple:
    """Build feature matrix for women's tournament games."""
    all_features = []
    all_labels = []

    for season in seasons:
        matchups = build_tourney_matchups(data, season)
        if matchups.empty:
            continue
        efficiency = build_efficiency_for_season(data, season)
        four_factors = build_four_factors_for_season(data, season)
        team_ids = list(set(matchups["TeamA"]) | set(matchups["TeamB"]))
        momentum = build_momentum_for_season(data, season, team_ids)
        mom_map = {row["TeamID"]: row for _, row in momentum.iterrows()}

        bart = fetch_womens_barttorvik(season)

        for _, m in matchups.iterrows():
            ta, tb = m["TeamA"], m["TeamB"]
            feat = {
                "Season": season, "TeamA": ta, "TeamB": tb,
                "seed_diff": m["SeedA"] - m["SeedB"],
                "seed_A": m["SeedA"], "seed_B": m["SeedB"],
            }
            for col in ["off_eff", "def_eff", "net_eff", "tempo"]:
                va = efficiency.loc[ta, col] if (not efficiency.empty and ta in efficiency.index) else np.nan
                vb = efficiency.loc[tb, col] if (not efficiency.empty and tb in efficiency.index) else np.nan
                feat[f"{col}_diff"] = va - vb
            for col in ["efg_pct", "to_pct", "or_pct", "ft_rate",
                        "opp_efg_pct", "opp_to_pct", "opp_or_pct", "opp_ft_rate"]:
                va = four_factors.loc[ta, col] if (not four_factors.empty and ta in four_factors.index) else np.nan
                vb = four_factors.loc[tb, col] if (not four_factors.empty and tb in four_factors.index) else np.nan
                feat[f"{col}_diff"] = va - vb
            if bart is not None:
                for src, dst in [("AdjOE", "bart_adjoe_diff"), ("AdjDE", "bart_adjde_diff"),
                                 ("NetRtg", "bart_net_diff"), ("Barthag", "bart_barthag_diff"),
                                 ("AdjTempo", "bart_tempo_diff")]:
                    va = bart.loc[ta, src] if ta in bart.index else np.nan
                    vb = bart.loc[tb, src] if tb in bart.index else np.nan
                    feat[dst] = va - vb
            mom_a = mom_map.get(ta, {})
            mom_b = mom_map.get(tb, {})
            feat["momentum_winpct_diff"] = mom_a.get("momentum_win_pct", 0.5) - mom_b.get("momentum_win_pct", 0.5)
            feat["momentum_margin_diff"] = mom_a.get("momentum_avg_margin", 0.0) - mom_b.get("momentum_avg_margin", 0.0)

            all_features.append(feat)
            all_labels.append(m["Result"])

    X = pd.DataFrame(all_features)
    y = np.array(all_labels)
    return X, y


# Women's-specific feature columns (no POM, no coaches)
WOMENS_FEATURES = [
    "seed_diff", "net_eff_diff", "off_eff_diff", "def_eff_diff", "tempo_diff",
    "efg_pct_diff", "to_pct_diff", "or_pct_diff", "ft_rate_diff",
    "opp_efg_pct_diff", "opp_to_pct_diff", "opp_or_pct_diff", "opp_ft_rate_diff",
    "momentum_margin_diff", "momentum_winpct_diff",
    "bart_net_diff", "bart_adjoe_diff", "bart_adjde_diff", "bart_barthag_diff",
]


class WomensLogistic(BaseEstimator, ClassifierMixin):
    def __init__(self, C=0.5):
        self.C = C
        self.pipe = None

    def fit(self, X, y):
        cols = [c for c in WOMENS_FEATURES if c in X.columns]
        self.cols_ = cols
        # Force numeric (some Barttorvik values may have come in as object/string)
        Xn = X[cols].apply(pd.to_numeric, errors="coerce")
        self.pipe = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(C=self.C, max_iter=2000, solver="lbfgs")),
        ])
        self.pipe.fit(Xn, y)
        return self

    def predict_proba(self, X):
        Xn = X[self.cols_].apply(pd.to_numeric, errors="coerce")
        return self.pipe.predict_proba(Xn)


def main():
    print("Loading women's data...")
    data = load_womens_data()
    seasons = [s for s in range(2014, 2026) if s != 2020]

    # Pre-cache Barttorvik for all training seasons (will be slow first time)
    print("Pre-caching Barttorvik for training seasons...")
    for s in seasons:
        bart = fetch_womens_barttorvik(s)
        if bart is not None:
            print(f"  W Barttorvik {s}: {len(bart)} teams")

    # Build full training matrix
    print("\nBuilding women's training features...")
    X, y = build_womens_features(data, seasons)
    print(f"  {len(X)} games, {X.shape[1]} cols")

    # LOTO evaluation
    print(f"\n{'='*60}\n  WOMEN'S LOTO BACKTEST\n{'='*60}")
    rows = []
    for holdout in seasons:
        X_train = X[X["Season"] != holdout]
        y_train = y[X["Season"] != holdout]
        X_test = X[X["Season"] == holdout]
        y_test = y[X["Season"] == holdout]
        if len(X_test) == 0: continue

        m = WomensLogistic(C=0.5)
        m.fit(X_train, y_train)
        p = m.predict_proba(X_test)[:, 1]
        bs = brier_score_loss(y_test, p)
        rows.append({"season": holdout, "brier": bs, "n_games": len(y_test)})
        print(f"  Season {holdout}: Brier = {bs:.4f} ({len(y_test)} games)")

    loto_df = pd.DataFrame(rows)
    print(f"\n  Mean Brier: {loto_df['brier'].mean():.4f} ± {loto_df['brier'].std():.4f}")
    loto_df.to_csv("output/loto_womens.csv", index=False)

    # Train final model on all 2014-2025 for 2026 predictions
    print(f"\n{'='*60}\n  TRAINING FINAL WOMEN'S MODEL\n{'='*60}")
    final_model = WomensLogistic(C=0.5)
    final_model.fit(X, y)
    print(f"  Trained on {len(X)} games")

    # Build 2026 women's submission features
    print("\nBuilding 2026 women's submission features...")
    sub = pd.read_csv(DATA_DIR / "SampleSubmissionStage2.csv")
    sub[["s", "ta", "tb"]] = sub["ID"].str.split("_", expand=True)
    sub["Season"] = sub["s"].astype(int)
    sub["TeamA"] = sub["ta"].astype(int)
    sub["TeamB"] = sub["tb"].astype(int)
    # Women's are TeamID >= 3000
    w_sub = sub[(sub["TeamA"] >= 3000) & (sub["TeamB"] >= 3000) & (sub["Season"] == 2026)].copy()
    print(f"  Women's submission rows: {len(w_sub)}")

    seeds = data["seeds"]
    w_seeds_2026 = seeds[seeds["Season"] == 2026].copy()
    w_seeds_2026["SeedNum"] = w_seeds_2026["Seed"].apply(_parse_seed_num)
    seed_map = dict(zip(w_seeds_2026["TeamID"], w_seeds_2026["SeedNum"]))
    tourney_teams = set(seed_map.keys())
    w_sub["is_tourney"] = w_sub.apply(
        lambda r: r["TeamA"] in tourney_teams and r["TeamB"] in tourney_teams, axis=1)
    tourney_pairs = w_sub[w_sub["is_tourney"]].copy()
    print(f"  Tournament-vs-tournament pairs: {len(tourney_pairs)}")

    # Build features for 2026
    season = 2026
    efficiency = build_efficiency_for_season(data, season)
    four_factors = build_four_factors_for_season(data, season)
    team_ids = list(tourney_teams)
    momentum = build_momentum_for_season(data, season, team_ids)
    mom_map = {row["TeamID"]: row for _, row in momentum.iterrows()}
    bart = fetch_womens_barttorvik(2026)

    X_2026 = []
    for _, row in tourney_pairs.iterrows():
        ta, tb = row["TeamA"], row["TeamB"]
        feat = {
            "Season": 2026, "TeamA": ta, "TeamB": tb,
            "seed_diff": seed_map.get(ta, 16) - seed_map.get(tb, 16),
            "seed_A": seed_map.get(ta, 16), "seed_B": seed_map.get(tb, 16),
        }
        for col in ["off_eff", "def_eff", "net_eff", "tempo"]:
            va = efficiency.loc[ta, col] if (not efficiency.empty and ta in efficiency.index) else np.nan
            vb = efficiency.loc[tb, col] if (not efficiency.empty and tb in efficiency.index) else np.nan
            feat[f"{col}_diff"] = va - vb
        for col in ["efg_pct", "to_pct", "or_pct", "ft_rate",
                    "opp_efg_pct", "opp_to_pct", "opp_or_pct", "opp_ft_rate"]:
            va = four_factors.loc[ta, col] if (not four_factors.empty and ta in four_factors.index) else np.nan
            vb = four_factors.loc[tb, col] if (not four_factors.empty and tb in four_factors.index) else np.nan
            feat[f"{col}_diff"] = va - vb
        if bart is not None:
            for src, dst in [("AdjOE", "bart_adjoe_diff"), ("AdjDE", "bart_adjde_diff"),
                             ("NetRtg", "bart_net_diff"), ("Barthag", "bart_barthag_diff"),
                             ("AdjTempo", "bart_tempo_diff")]:
                va = bart.loc[ta, src] if ta in bart.index else np.nan
                vb = bart.loc[tb, src] if tb in bart.index else np.nan
                feat[dst] = va - vb
        mom_a = mom_map.get(ta, {})
        mom_b = mom_map.get(tb, {})
        feat["momentum_winpct_diff"] = mom_a.get("momentum_win_pct", 0.5) - mom_b.get("momentum_win_pct", 0.5)
        feat["momentum_margin_diff"] = mom_a.get("momentum_avg_margin", 0.0) - mom_b.get("momentum_avg_margin", 0.0)
        X_2026.append(feat)

    X_2026 = pd.DataFrame(X_2026)
    p_2026 = final_model.predict_proba(X_2026)[:, 1]
    print(f"  Generated predictions for {len(X_2026)} pairs")

    # Update the submission CSV with women's predictions
    print("\nUpdating submission CSV with women's predictions...")
    submission = pd.read_csv("output/submission_stage2.csv")
    # Build lookup
    pred_map = {}
    for i, (_, row) in enumerate(X_2026.iterrows()):
        ta, tb = int(row["TeamA"]), int(row["TeamB"])
        # Canonical lower-id-first format used in submission file
        pred_map[(ta, tb) if ta < tb else (tb, ta)] = p_2026[i] if ta < tb else 1 - p_2026[i]

    submission[["s", "ta", "tb"]] = submission["ID"].str.split("_", expand=True)
    submission["ta"] = submission["ta"].astype(int)
    submission["tb"] = submission["tb"].astype(int)

    n_updated = 0
    for idx, row in submission.iterrows():
        if (row["ta"], row["tb"]) in pred_map:
            submission.at[idx, "Pred"] = float(np.clip(pred_map[(row["ta"], row["tb"])], 0.01, 0.99))
            n_updated += 1
    print(f"  Updated {n_updated} women's predictions")

    submission[["ID", "Pred"]].to_csv("output/submission_stage2.csv", index=False)
    print("Saved to output/submission_stage2.csv")


if __name__ == "__main__":
    main()
