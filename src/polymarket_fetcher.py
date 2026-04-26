"""
Polymarket data fetcher for NCAA Tournament 2026 prediction.

Replicates the data ingestion path used by the 2026 Kaggle 1st place solution
("Kill your darlings"): pulls championship futures from the Gamma API and
historical price series from the CLOB API, filters to the day before each
tournament round, and maps Polymarket team labels to Kaggle TeamIDs.

Two market sources:
  (1) Title-futures board: 2026-ncaa-tournament-winner (men's, 90 markets)
                           2026-womens-ncaa-tournament-winner (women's)
      Each market resolves YES if a specific team wins the championship.
      Used for an exact-bracket-DP overlay on team-level model probabilities.

  (2) Game-level H2H markets (search via public-search endpoint).
      Each market is a binary "Team A beats Team B" outcome.
      Used for direct moneyline replacement when liquidity is high.

API endpoints (no auth required):
  GET https://gamma-api.polymarket.com/events?slug=<slug>
  GET https://gamma-api.polymarket.com/public-search?q=<query>&limit=...
  GET https://clob.polymarket.com/prices-history?market=<token_id>&interval=max&fidelity=1440
"""

from __future__ import annotations

import json
import time
import re
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from rapidfuzz import process, fuzz

from src.data_collection import DATA_DIR


GAMMA = "https://gamma-api.polymarket.com"
CLOB = "https://clob.polymarket.com"
CACHE_DIR = DATA_DIR / "external" / "polymarket"
CACHE_DIR.mkdir(parents=True, exist_ok=True)


# Selection Sunday 2026 = Mar 15. We want closing prices on Mar 15 (just before
# the tournament starts) so the markets reflect pre-tournament information only,
# matching the way our model is trained.
SELECTION_SUNDAY_2026_TS = int(pd.Timestamp("2026-03-15 23:59:00", tz="UTC").timestamp())


# ---------------------------------------------------------------------------
# HTTP helpers
# ---------------------------------------------------------------------------

def _get(url: str, params: dict | None = None, retries: int = 3) -> dict | list:
    last = None
    for i in range(retries):
        try:
            r = requests.get(url, params=params, timeout=15)
            if r.status_code == 200:
                return r.json()
            last = r.status_code
        except Exception as e:
            last = repr(e)
        time.sleep(0.5 * (2 ** i))
    raise RuntimeError(f"GET {url} {params} failed after {retries} attempts: {last}")


# ---------------------------------------------------------------------------
# Gamma API: events / markets
# ---------------------------------------------------------------------------

def fetch_event_by_slug(slug: str) -> dict | None:
    cache = CACHE_DIR / f"event_{slug}.json"
    if cache.exists():
        return json.loads(cache.read_text())
    out = _get(f"{GAMMA}/events", params={"slug": slug})
    if isinstance(out, list) and out:
        cache.write_text(json.dumps(out[0]))
        return out[0]
    return None


def search_events(query: str, limit: int = 50) -> list[dict]:
    out = _get(f"{GAMMA}/public-search", params={"q": query, "limit": limit})
    if isinstance(out, dict):
        return out.get("events", [])
    return []


def parse_clob_token_ids(market: dict) -> list[str]:
    raw = market.get("clobTokenIds", "[]")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    return list(raw) if isinstance(raw, list) else []


def parse_outcomes(market: dict) -> list[str]:
    raw = market.get("outcomes", "[]")
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except Exception:
            return []
    return list(raw) if isinstance(raw, list) else []


# ---------------------------------------------------------------------------
# CLOB API: price history
# ---------------------------------------------------------------------------

def fetch_price_history(token_id: str, fidelity: int = 1440) -> pd.DataFrame:
    """Returns DataFrame with columns [t (Unix), p (price)] for token's full life."""
    cache = CACHE_DIR / f"prices_{token_id[:24]}.csv"
    if cache.exists():
        return pd.read_csv(cache)
    out = _get(f"{CLOB}/prices-history",
               params={"market": token_id, "interval": "max", "fidelity": fidelity})
    if not isinstance(out, dict) or "history" not in out:
        return pd.DataFrame(columns=["t", "p"])
    df = pd.DataFrame(out["history"])
    if df.empty:
        return df
    df = df.sort_values("t").reset_index(drop=True)
    df.to_csv(cache, index=False)
    return df


def closing_price_at(token_id: str, target_ts: int) -> float | None:
    """Return latest price at or before target_ts (Unix seconds)."""
    df = fetch_price_history(token_id)
    if df.empty:
        return None
    before = df[df["t"] <= target_ts]
    if before.empty:
        return None
    return float(before.iloc[-1]["p"])


# ---------------------------------------------------------------------------
# Team-name mapping
# ---------------------------------------------------------------------------

def _team_question_to_name(question: str) -> str | None:
    """
    Polymarket championship futures questions look like:
        "Will Florida win the 2026 NCAA Tournament?"
        "Will UConn win the 2026 NCAA Tournament?"
    Extract the team name.
    """
    m = re.search(r"Will\s+(.+?)\s+win\s+the\s+2026", question, re.IGNORECASE)
    if m:
        return m.group(1).strip()
    return None


def build_team_name_map(womens: bool = False) -> dict[str, int]:
    """Map lowercase team names to Kaggle TeamID using TeamSpellings."""
    teams_file = "WTeams.csv" if womens else "MTeams.csv"
    spell_file = "WTeamSpellings.csv" if womens else "MTeamSpellings.csv"
    teams = pd.read_csv(DATA_DIR / teams_file)
    try:
        spell = pd.read_csv(DATA_DIR / spell_file, encoding="latin-1")
    except FileNotFoundError:
        spell = None
    name_map = {}
    for _, r in teams.iterrows():
        name_map[str(r["TeamName"]).lower().strip()] = int(r["TeamID"])
    if spell is not None:
        for _, r in spell.iterrows():
            name_map[str(r["TeamNameSpelling"]).lower().strip()] = int(r["TeamID"])
    return name_map


def match_team_name(name: str, name_map: dict[str, int]) -> int | None:
    n = name.lower().strip()
    # Common aliases
    aliases = {
        "uconn": "connecticut",
        "u conn": "connecticut",
        "saint mary's": "st mary's ca",
        "saint marys": "st mary's ca",
        "st. mary's": "st mary's ca",
        "ole miss": "mississippi",
        "ucla": "ucla",
        "byu": "byu",
        "smu": "smu",
        "lsu": "lsu",
        "vcu": "vcu",
        "tcu": "tcu",
    }
    if n in aliases:
        n = aliases[n]
    if n in name_map:
        return name_map[n]
    # Strip punctuation / "university" / "state"
    n2 = re.sub(r"[^\w\s]", "", n).strip()
    if n2 in name_map:
        return name_map[n2]
    # Fuzzy
    best = process.extractOne(n, list(name_map.keys()), scorer=fuzz.WRatio, score_cutoff=85)
    if best:
        return name_map[best[0]]
    return None


# ---------------------------------------------------------------------------
# High-level: championship futures snapshot at Selection Sunday
# ---------------------------------------------------------------------------

def fetch_championship_futures(
    womens: bool = False, target_ts: int = SELECTION_SUNDAY_2026_TS
) -> pd.DataFrame:
    """For each team in the 2026 championship futures board, return:
        TeamID, team_name, raw_yes_price, normalized_prob

    Prices are taken at `target_ts` (default: Selection Sunday 2026).
    Normalization: divide each YES price by the sum so that probabilities
    sum to 1.0 (the "fair" implied championship distribution).
    """
    slug = "2026-womens-ncaa-tournament-winner" if womens else "2026-ncaa-tournament-winner"
    event = fetch_event_by_slug(slug)
    if event is None:
        raise RuntimeError(f"Event {slug} not found")
    name_map = build_team_name_map(womens=womens)

    rows = []
    markets = event.get("markets", [])
    print(f"  Found {len(markets)} markets in {slug}")
    for i, m in enumerate(markets):
        team_name = _team_question_to_name(m.get("question", ""))
        if team_name is None:
            continue
        token_ids = parse_clob_token_ids(m)
        outcomes = parse_outcomes(m)
        if not token_ids or not outcomes:
            continue
        # YES is the first outcome; its token is token_ids[0]
        yes_token = token_ids[0]
        try:
            p = closing_price_at(yes_token, target_ts)
        except Exception as e:
            print(f"    skip {team_name}: {e}")
            continue
        if p is None:
            continue
        tid = match_team_name(team_name, name_map)
        rows.append({
            "team_name": team_name,
            "TeamID": tid,
            "yes_price_raw": p,
            "market_id": m.get("id"),
        })
        if (i + 1) % 20 == 0:
            print(f"    fetched {i+1}/{len(markets)}")
        time.sleep(0.05)

    df = pd.DataFrame(rows)
    if df.empty:
        return df
    df["normalized_prob"] = df["yes_price_raw"] / df["yes_price_raw"].sum()
    return df.sort_values("normalized_prob", ascending=False).reset_index(drop=True)


# ---------------------------------------------------------------------------
# High-level: game-level H2H markets
# ---------------------------------------------------------------------------

def fetch_h2h_markets(query: str = "march madness", limit: int = 200) -> list[dict]:
    """Search Polymarket for game-level H2H markets matching `query`.

    Returns list of event dicts; each event has a list of markets.
    """
    events = search_events(query, limit=limit)
    h2h = []
    for e in events:
        slug = e.get("slug", "").lower()
        if any(x in slug for x in ["march-madness", "ncaa-march", "ncaa-tournament"]):
            h2h.append(e)
    return h2h
