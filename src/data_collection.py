"""
Data collection pipeline for NCAA March Madness prediction.

Sources:
  - Kaggle: march-machine-learning-mania-2026 (~35 CSVs)
  - KenPom: kenpom.com (AdjEM/AdjO/AdjD/AdjT, $24.95/yr)
  - Barttorvik: barttorvik.com (T-Rank, free)
  - Massey: included in Kaggle MMasseyOrdinals.csv
  - BigQuery: bigquery-public-data.ncaa_basketball
  - Kalshi: docs.kalshi.com (live odds)
"""

import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

DATA_DIR = Path(__file__).parent.parent / "data"
EXTERNAL_DIR = DATA_DIR / "external"
ODDS_DIR = DATA_DIR / "odds"


def load_csv(filename: str):
    """Load a CSV file from data/."""
    import pandas as pd
    path = DATA_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Unzip Kaggle data into data/.")
    return pd.read_csv(path)


def load_all_mens_data() -> dict:
    """Load all core Men's tournament data into a dict of DataFrames."""
    import pandas as pd
    files = {
        "regular_detail": "MRegularSeasonDetailedResults.csv",
        "regular_compact": "MRegularSeasonCompactResults.csv",
        "tourney_detail": "MNCAATourneyDetailedResults.csv",
        "tourney_compact": "MNCAATourneyCompactResults.csv",
        "seeds": "MNCAATourneySeeds.csv",
        "slots": "MNCAATourneySlots.csv",
        "massey": "MMasseyOrdinals.csv",
        "teams": "MTeams.csv",
        "seasons": "MSeasons.csv",
        "coaches": "MTeamCoaches.csv",
        "conferences": "MTeamConferences.csv",
        "game_cities": "MGameCities.csv",
    }
    data = {}
    for key, fname in files.items():
        path = DATA_DIR / fname
        if path.exists():
            print(f"Loading {fname}...")
            data[key] = pd.read_csv(path)
        else:
            print(f"Warning: {fname} not found, skipping.")
    return data


def load_massey_ordinals(season: int | None = None):
    """Load Massey composite rankings. Optionally filter by season."""
    df = load_csv("MMasseyOrdinals.csv")
    if season:
        df = df[df["Season"] == season]
    return df


def fetch_kenpom_ratings(season: int):
    """Fetch KenPom ratings for a season (requires subscription)."""
    try:
        from kenpompy.utils import login
        from kenpompy.summary import get_efficiency
        browser = login(
            os.environ.get("KENPOM_EMAIL", ""),
            os.environ.get("KENPOM_PASSWORD", ""),
        )
        return get_efficiency(browser, season=str(season))
    except ImportError:
        print("kenpompy not installed. Install with: pip install kenpompy")
        return None


def _build_spelling_map() -> dict[str, int]:
    """Build lowercase team name -> TeamID mapping from MTeamSpellings + MTeams."""
    mapping = {}
    sp_path = DATA_DIR / "MTeamSpellings.csv"
    teams_path = DATA_DIR / "MTeams.csv"

    if sp_path.exists():
        sp = pd.read_csv(sp_path, encoding="latin-1")
        for _, row in sp.iterrows():
            name = str(row["TeamNameSpelling"]).strip().lower()
            mapping[name] = int(row["TeamID"])

    if teams_path.exists():
        teams = pd.read_csv(teams_path)
        for _, row in teams.iterrows():
            name = str(row["TeamName"]).strip().lower()
            mapping[name] = int(row["TeamID"])

    return mapping


_SPELLING_MAP: dict[str, int] | None = None


def _get_spelling_map() -> dict[str, int]:
    global _SPELLING_MAP
    if _SPELLING_MAP is None:
        _SPELLING_MAP = _build_spelling_map()
    return _SPELLING_MAP


def _normalize_barttorvik_name(name: str) -> str:
    """Normalize a Barttorvik team name for matching."""
    s = name.strip().lower()
    # Remove seed/tournament info like "1 seed, Finals"
    s = re.sub(r"\s+\d+\s+seed.*$", "", s)
    # Remove trailing state abbreviations in parens for some teams
    s = re.sub(r"\s*\(.*?\)\s*$", "", s)
    # Normalize punctuation
    s = s.replace(".", "").replace("'", "'").replace("`", "'").replace("-", " ")
    s = re.sub(r"\s+", " ", s).strip()
    return s


def match_barttorvik_to_kaggle(bart_name: str, spelling_map: dict[str, int]) -> int | None:
    """Match a Barttorvik team name to a Kaggle TeamID."""
    norm = _normalize_barttorvik_name(bart_name)

    # Direct lookup
    if norm in spelling_map:
        return spelling_map[norm]

    # Try with common substitutions
    variants = [
        norm,
        norm.replace("st ", "state "),
        norm.replace("state ", "st "),
        norm.replace("'", ""),
        norm.replace(" ", "-"),
    ]
    for v in variants:
        if v in spelling_map:
            return spelling_map[v]

    # Fuzzy: try matching on first word(s)
    try:
        from rapidfuzz import process, fuzz
        result = process.extractOne(norm, spelling_map.keys(), scorer=fuzz.ratio, score_cutoff=80)
        if result:
            return spelling_map[result[0]]
    except ImportError:
        pass

    return None


def _extract_bart_value(s) -> float | None:
    """Extract numeric value from Barttorvik cell like '124.7 11' -> 124.7."""
    if pd.isna(s):
        return None
    s = str(s).strip()
    m = re.match(r"([+-]?\d*\.?\d+)", s)
    return float(m.group(1)) if m else None


def fetch_barttorvik_ratings(season: int) -> pd.DataFrame | None:
    """
    Scrape Barttorvik T-Rank data (free).
    Returns DataFrame with columns: TeamID, AdjOE, AdjDE, Barthag, AdjTempo, NetRtg.
    Cached to data/external/barttorvik_{season}.csv.
    """
    import io
    import requests

    EXTERNAL_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = EXTERNAL_DIR / f"barttorvik_{season}.csv"

    # Return cached if available
    if cache_path.exists():
        df = pd.read_csv(cache_path)
        if len(df) > 100:
            return df

    # Scrape from Barttorvik
    try:
        session = requests.Session()
        session.headers.update({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
        })
        url = f"https://barttorvik.com/trank.php?year={season}&conyes=0"

        # Handle JS browser verification
        session.get(url, timeout=15)
        resp = session.post(url, data={"js_test_submitted": "1"}, timeout=15)

        if resp.status_code != 200 or len(resp.text) < 1000:
            print(f"  Barttorvik {season}: bad response (status={resp.status_code})")
            return None

        tables = pd.read_html(io.StringIO(resp.text))
        if not tables:
            print(f"  Barttorvik {season}: no tables found")
            return None

        raw = tables[0]
        # Flatten multi-level columns
        raw.columns = [c[1] if isinstance(c, tuple) else c for c in raw.columns]

    except Exception as e:
        print(f"  Barttorvik {season}: scrape failed ({e})")
        return None

    # Parse and clean
    spelling_map = _get_spelling_map()
    rows = []
    unmatched = []

    for _, row in raw.iterrows():
        team_name = str(row.get("Team", ""))
        adjoe = _extract_bart_value(row.get("AdjOE"))
        adjde = _extract_bart_value(row.get("AdjDE"))
        barthag = _extract_bart_value(row.get("Barthag"))
        adj_tempo = _extract_bart_value(row.get("Adj T."))

        if adjoe is None or adjde is None:
            continue

        team_id = match_barttorvik_to_kaggle(team_name, spelling_map)
        if team_id is None:
            unmatched.append(_normalize_barttorvik_name(team_name))
            continue

        rows.append({
            "TeamID": team_id,
            "AdjOE": adjoe,
            "AdjDE": adjde,
            "Barthag": barthag if barthag is not None else adjoe / (adjoe + adjde),
            "AdjTempo": adj_tempo if adj_tempo is not None else 68.0,
            "NetRtg": adjoe - adjde,
        })

    if unmatched:
        print(f"  Barttorvik {season}: {len(unmatched)} unmatched teams: {unmatched[:5]}...")

    if not rows:
        return None

    df = pd.DataFrame(rows)
    df.to_csv(cache_path, index=False)
    print(f"  Barttorvik {season}: {len(df)} teams cached to {cache_path.name}")
    return df


def fetch_kalshi_ncaa_markets():
    """Fetch current NCAA markets from Kalshi API."""
    import httpx
    resp = httpx.get(
        "https://api.elections.kalshi.com/trade-api/v2/markets",
        params={"series_ticker": "NCAAM", "limit": 200},
    )
    if resp.status_code == 200:
        return resp.json()
    return None


if __name__ == "__main__":
    data = load_all_mens_data()
    for key, df in data.items():
        print(f"  {key}: {df.shape}")
    print("Done.")
