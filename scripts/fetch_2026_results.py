"""
Scrape 2026 NCAA tournament results from Sports Reference and write a CSV
in MNCAATourneyCompactResults.csv format (Season, DayNum, WTeamID, WScore,
LTeamID, LScore, WLoc, NumOT).

Outputs: data/external/tourney_2026_results.csv
"""

import sys, re, requests
sys.path.insert(0, ".")
import pandas as pd
from pathlib import Path
from bs4 import BeautifulSoup

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import _parse_seed_num


URL = "https://www.sports-reference.com/cbb/postseason/men/2026-ncaa.html"


# Manual mapping for Sports Reference URL slugs to Kaggle TeamIDs
# We resolve these by name lookup against MTeamSpellings.csv
def build_name_to_id_map(data: dict) -> dict:
    """Build a map from lowercased team name spellings to TeamID."""
    spellings = pd.read_csv(DATA_DIR / "MTeamSpellings.csv", encoding="latin-1")
    name_map = {}
    for _, row in spellings.iterrows():
        name_map[row["TeamNameSpelling"].lower().strip()] = int(row["TeamID"])
    teams = data["teams"]
    for _, row in teams.iterrows():
        name_map[row["TeamName"].lower().strip()] = int(row["TeamID"])
    return name_map


# Sports Reference uses some unusual names; manual overrides:
SR_NAME_OVERRIDES = {
    "St. John's (NY)": "St John's",
    "Texas Christian": "TCU",
    "Saint Mary's (CA)": "St Mary's CA",
    "Brigham Young": "BYU",
    "Southern Methodist": "SMU",
    "North Carolina State": "NC State",
    "UCLA": "UCLA",
    "UCF": "UCF",
    "VCU": "VCU",
    "Mississippi State": "Mississippi St",
    "Louisiana State": "LSU",
    "Long Island University": "LIU Brooklyn",
    "American": "American Univ",
    "Penn": "Pennsylvania",
    "Cal Baptist": "California Baptist",
    "Florida Atlantic": "Florida Atlantic",
    "Texas A&M": "TX A&M",
    "Wright State": "Wright St",
}


def normalize_name(sr_name: str, name_map: dict) -> int | None:
    """Look up a Sports Reference team name in the Kaggle map."""
    sr_name = sr_name.strip()
    # Try direct
    candidates = [
        sr_name,
        SR_NAME_OVERRIDES.get(sr_name, sr_name),
    ]
    # Strip parentheticals
    if "(" in sr_name:
        candidates.append(sr_name.split("(")[0].strip())
    # Replace abbreviations
    candidates.append(sr_name.replace("State", "St").strip())
    candidates.append(sr_name.replace(" State", " St").strip())
    # Try lower
    for c in candidates:
        if c.lower() in name_map:
            return name_map[c.lower()]
    # Try fuzzy via rapidfuzz
    try:
        from rapidfuzz import process, fuzz
        match = process.extractOne(sr_name.lower(), list(name_map.keys()),
                                   scorer=fuzz.WRatio, score_cutoff=85)
        if match:
            return name_map[match[0]]
    except ImportError:
        pass
    return None


def parse_bracket(html: str, name_map: dict) -> list[dict]:
    """Parse the Sports Reference bracket HTML, return list of game dicts."""
    soup = BeautifulSoup(html, "html.parser")
    games = []
    unmatched = set()

    # Each region is a div with id in [east, midwest, south, west, national]
    # Plus the championship at the end
    for region_id in ["east", "midwest", "south", "west", "national"]:
        region_div = soup.find("div", id=region_id)
        if not region_div:
            continue
        # Each game is <div> <!-- game --> with two team divs inside
        # The first team has class="winner" if it won
        for game_div in region_div.find_all("div", recursive=True):
            # A "game" div has exactly 2 team divs as direct children
            team_divs = [c for c in game_div.find_all("div", recursive=False)
                         if c.find("span") and c.find("a")]
            if len(team_divs) != 2:
                continue

            teams = []
            for td in team_divs:
                seed_span = td.find("span")
                team_link = td.find("a")
                score_link = td.find_all("a")[-1] if len(td.find_all("a")) > 1 else None
                if not (seed_span and team_link and score_link):
                    continue
                try:
                    seed = int(seed_span.text.strip())
                    name = team_link.text.strip()
                    score = int(score_link.text.strip())
                except (ValueError, AttributeError):
                    continue
                tid = normalize_name(name, name_map)
                if tid is None:
                    unmatched.add(name)
                teams.append({
                    "seed": seed,
                    "name": name,
                    "tid": tid,
                    "score": score,
                    "won": "winner" in (td.get("class") or []),
                })

            if len(teams) == 2 and teams[0]["tid"] and teams[1]["tid"]:
                w = next((t for t in teams if t["won"]), None)
                l = next((t for t in teams if not t["won"]), None)
                if w and l:
                    games.append({
                        "Season": 2026,
                        "WTeamID": w["tid"],
                        "WScore": w["score"],
                        "LTeamID": l["tid"],
                        "LScore": l["score"],
                        "WSeed": w["seed"],
                        "LSeed": l["seed"],
                        "WName": w["name"],
                        "LName": l["name"],
                    })

    if unmatched:
        print(f"\n  Unmatched team names ({len(unmatched)}):")
        for n in sorted(unmatched):
            print(f"    - {n}")

    return games


def main():
    print("Loading Kaggle team data for name mapping...")
    data = load_all_mens_data()
    name_map = build_name_to_id_map(data)
    print(f"  Built name map with {len(name_map)} entries")

    print("\nFetching Sports Reference 2026 bracket...")
    resp = requests.get(URL, headers={"User-Agent": "Mozilla/5.0"})
    if resp.status_code != 200:
        print(f"Failed: HTTP {resp.status_code}")
        return
    print(f"  Got {len(resp.text)} bytes")

    games = parse_bracket(resp.text, name_map)
    print(f"\n  Parsed {len(games)} games")

    # Deduplicate (the same game may show up under multiple region containers)
    seen = set()
    deduped = []
    for g in games:
        key = (g["WTeamID"], g["LTeamID"], g["WScore"], g["LScore"])
        if key not in seen:
            seen.add(key)
            deduped.append(g)
    print(f"  After dedup: {len(deduped)} games")

    df = pd.DataFrame(deduped)
    out_path = DATA_DIR / "external" / "tourney_2026_results.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")
    print(f"\nSample games:")
    print(df[["WSeed", "WName", "WScore", "LSeed", "LName", "LScore"]].head(15).to_string(index=False))


if __name__ == "__main__":
    main()
