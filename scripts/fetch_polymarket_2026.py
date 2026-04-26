"""
Fetch and cache Polymarket prices for the 2026 NCAA tournament.

Pulls:
  - Men's championship futures (2026-ncaa-tournament-winner) at Selection Sunday
  - Women's championship futures (2026-womens-ncaa-tournament-winner) at SS
  - Game-level H2H markets (search-based) where available

Outputs:
  data/external/polymarket/champ_men_2026.csv
  data/external/polymarket/champ_women_2026.csv
  data/external/polymarket/h2h_men_2026.csv (if available)
"""

import sys
sys.path.insert(0, ".")

import json
import time
import pandas as pd

from src.polymarket_fetcher import (
    fetch_championship_futures, fetch_h2h_markets,
    parse_clob_token_ids, parse_outcomes,
    closing_price_at, build_team_name_map, match_team_name,
    SELECTION_SUNDAY_2026_TS, CACHE_DIR,
)


def main():
    print("="*70)
    print("  MEN'S championship futures (2026-ncaa-tournament-winner)")
    print("="*70)
    df_m = fetch_championship_futures(womens=False)
    print(f"\n  Got {len(df_m)} markets")
    print(f"  Mapped to TeamID: {df_m['TeamID'].notna().sum()}")
    print(f"\n  Top 15 men's title contenders by Polymarket implied prob:")
    print(df_m.head(15).to_string(index=False))
    df_m.to_csv(CACHE_DIR / "champ_men_2026.csv", index=False)
    print(f"\n  Saved {CACHE_DIR / 'champ_men_2026.csv'}")

    print("\n" + "="*70)
    print("  WOMEN'S championship futures (2026-womens-ncaa-tournament-winner)")
    print("="*70)
    try:
        df_w = fetch_championship_futures(womens=True)
        print(f"\n  Got {len(df_w)} markets")
        print(f"  Mapped to TeamID: {df_w['TeamID'].notna().sum()}")
        print(f"\n  Top 15 women's title contenders:")
        print(df_w.head(15).to_string(index=False))
        df_w.to_csv(CACHE_DIR / "champ_women_2026.csv", index=False)
        print(f"\n  Saved {CACHE_DIR / 'champ_women_2026.csv'}")
    except Exception as e:
        print(f"  Women's not available: {e}")

    print("\n" + "="*70)
    print("  H2H game markets")
    print("="*70)
    events = fetch_h2h_markets(query="march madness", limit=200)
    print(f"  Found {len(events)} march-madness related events")
    name_map_m = build_team_name_map(womens=False)
    name_map_w = build_team_name_map(womens=True)

    h2h_rows = []
    for e in events:
        slug = e.get("slug", "")
        # Skip the championship futures events themselves
        if "tournament-winner" in slug or "sum-of" in slug or "region" in slug:
            continue
        markets = e.get("markets", [])
        for m in markets:
            q = m.get("question", "")
            token_ids = parse_clob_token_ids(m)
            outcomes = parse_outcomes(m)
            if len(token_ids) < 2 or len(outcomes) < 2:
                continue
            # H2H markets typically: "Will TeamA beat TeamB?" or "TeamA vs TeamB - winner: TeamA"
            # Outcomes might be ["Yes","No"] or [TeamA, TeamB]
            try:
                yes_price = closing_price_at(token_ids[0], SELECTION_SUNDAY_2026_TS)
            except Exception:
                yes_price = None
            volume = float(m.get("volume", 0) or 0)
            h2h_rows.append({
                "event_slug": slug,
                "market_id": m.get("id"),
                "question": q,
                "outcomes": outcomes,
                "yes_price_at_SS": yes_price,
                "volume": volume,
                "closed": m.get("closed"),
            })
            time.sleep(0.05)
    h2h_df = pd.DataFrame(h2h_rows)
    if not h2h_df.empty:
        print(f"  Total H2H markets: {len(h2h_df)}")
        print(f"  With volume > 1000: {(h2h_df['volume'] > 1000).sum()}")
        print(f"  With volume > 10000: {(h2h_df['volume'] > 10000).sum()}")
        h2h_df.to_csv(CACHE_DIR / "h2h_raw_2026.csv", index=False)
        print(f"  Saved {CACHE_DIR / 'h2h_raw_2026.csv'}")
    else:
        print("  No H2H markets found.")


if __name__ == "__main__":
    main()
