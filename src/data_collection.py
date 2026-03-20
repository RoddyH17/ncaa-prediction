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
KAGGLE_DIR = DATA_DIR / "kaggle"
KENPOM_DIR = DATA_DIR / "kenpom"
BARTTORVIK_DIR = DATA_DIR / "barttorvik"
ODDS_DIR = DATA_DIR / "odds"


def download_kaggle_data(competition: str = "march-machine-learning-mania-2026"):
    """Download competition data via Kaggle CLI."""
    KAGGLE_DIR.mkdir(parents=True, exist_ok=True)
    os.system(f"kaggle competitions download -c {competition} -p {KAGGLE_DIR}")
    # Unzip
    import zipfile
    for f in KAGGLE_DIR.glob("*.zip"):
        with zipfile.ZipFile(f) as z:
            z.extractall(KAGGLE_DIR)
        f.unlink()


def load_kaggle_csv(filename: str):
    """Load a single Kaggle CSV file."""
    import pandas as pd
    return pd.read_csv(KAGGLE_DIR / filename)


def load_massey_ordinals(season: int | None = None):
    """Load Massey composite rankings. Optionally filter by season."""
    df = load_kaggle_csv("MMasseyOrdinals.csv")
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
    print("Downloading Kaggle data...")
    download_kaggle_data()
    print("Done.")
