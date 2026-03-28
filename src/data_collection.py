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
from pathlib import Path

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


def fetch_barttorvik_ratings(season: int):
    """Scrape Barttorvik T-Rank data (free)."""
    import pandas as pd
    url = f"https://barttorvik.com/trank.php?year={season}&conyes=1&sort=&top=0&conlimit=#"
    try:
        tables = pd.read_html(url)
        if tables:
            return tables[0]
    except Exception as e:
        print(f"Failed to fetch Barttorvik data: {e}")
    return None


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
