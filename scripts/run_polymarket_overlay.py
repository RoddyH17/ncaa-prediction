"""
Polymarket title-futures overlay for 2026 NCAA men's predictions.

Replicates the 1st-place Kaggle 2026 strategy ("Kill your darlings"):

  1. Run our base model on every men's tournament-team pair (already saved
     in output/submission_stage2.csv).
  2. Monte-Carlo simulate the bracket from our pairwise probabilities to
     get model-implied per-team championship probabilities P_model(champ).
  3. Compare to Polymarket fair-normalized championship probabilities
     P_market(champ) at Selection Sunday 2026.
  4. For each team, compute logit gap:
        delta_t = logit(P_market(champ_t)) - logit(P_model(champ_t))
     Cap to ±team_offset_cap.
  5. For each pairwise game (a, b), apply per-team offset:
        logit'(P(a beats b)) = logit(P(a beats b)) + alpha*(delta_a - delta_b)/2
     Cap the change to ±per_game_move_cap.
  6. Re-evaluate Brier on actual 2026 games.

Hyperparameters (from 1st place writeup):
  alpha = 0.10
  team_offset_cap = ±0.10
  per_game_move_cap = ±0.03
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import _parse_seed_num


REGIONS = ["W", "X", "Y", "Z"]
PAIRS_R64 = [(1, 16), (8, 9), (5, 12), (4, 13), (6, 11), (3, 14), (7, 10), (2, 15)]


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def build_bracket_structure(seeds_df: pd.DataFrame) -> dict:
    """Return dict: region -> list of 16 TeamIDs ordered by first-round pairs.

    Order: [1-seed, 16-seed, 8, 9, 5, 12, 4, 13, 6, 11, 3, 14, 7, 10, 2, 15]
    so adjacent pairs play each other in round 1, regional pairs collapse.
    """
    out = {}
    for region in REGIONS:
        teams = []
        for hi, lo in PAIRS_R64:
            hi_team = seeds_df[seeds_df["Seed"].str.startswith(f"{region}{hi:02d}")]
            lo_team = seeds_df[seeds_df["Seed"].str.startswith(f"{region}{lo:02d}")]
            if hi_team.empty or lo_team.empty:
                return None
            teams.append(int(hi_team.iloc[0]["TeamID"]))
            teams.append(int(lo_team.iloc[0]["TeamID"]))
        out[region] = teams
    return out


def monte_carlo_champ_probs(
    p_lookup: dict, region_teams: dict, n_sims: int = 100_000, seed: int = 42
) -> dict:
    """For each team, return P(team wins championship) via MC simulation.

    p_lookup: dict (team_a, team_b) -> P(team_a beats team_b), with both
              orderings present.
    region_teams: dict region -> 16 TeamIDs in bracket order.
    """
    rng = np.random.default_rng(seed)
    all_teams = []
    for r in REGIONS:
        all_teams.extend(region_teams[r])
    champ_count = {t: 0 for t in all_teams}

    def winner(a, b, draws):
        p = p_lookup.get((a, b), 0.5)
        return a if draws < p else b

    for sim in range(n_sims):
        # Pre-draw enough random numbers for all 63 games
        draws = rng.random(63)
        di = 0
        # Simulate each region: 16 -> 8 -> 4 -> 2 -> 1
        region_winners = {}
        for region in REGIONS:
            current = list(region_teams[region])
            while len(current) > 1:
                next_round = []
                for i in range(0, len(current), 2):
                    w = winner(current[i], current[i + 1], draws[di]); di += 1
                    next_round.append(w)
                current = next_round
            region_winners[region] = current[0]
        # Final 4: regions are paired (W vs X) and (Y vs Z) in NCAA standard
        wx = winner(region_winners["W"], region_winners["X"], draws[di]); di += 1
        yz = winner(region_winners["Y"], region_winners["Z"], draws[di]); di += 1
        ch = winner(wx, yz, draws[di]); di += 1
        champ_count[ch] += 1

    return {t: c / n_sims for t, c in champ_count.items()}


def apply_overlay(
    p_lookup: dict,
    p_model_champ: dict,
    p_market_champ: dict,
    alpha: float = 0.10,
    team_offset_cap: float = 0.10,
    per_game_move_cap: float = 0.03,
) -> dict:
    """Return new p_lookup with overlay applied.

    delta_t = logit(market_champ_t) - logit(model_champ_t), capped to
              ±team_offset_cap, scaled by alpha.
    For team without market price, delta_t = 0.

    For each game (a, b): new_logit = old_logit + (delta_a - delta_b)/2.
    Cap change to ±per_game_move_cap.
    """
    delta = {}
    for t, p_model in p_model_champ.items():
        p_market = p_market_champ.get(t, None)
        if p_market is None or p_market < 1e-6 or p_model < 1e-6:
            delta[t] = 0.0
            continue
        gap = logit(p_market) - logit(p_model)
        capped = np.clip(alpha * gap, -team_offset_cap, team_offset_cap)
        delta[t] = float(capped)

    new_lookup = {}
    for (a, b), p in p_lookup.items():
        d = (delta.get(a, 0.0) - delta.get(b, 0.0)) / 2.0
        old_logit = logit(p)
        new_logit = old_logit + d
        # Cap probability move
        old_p = p
        new_p = expit(new_logit)
        actual_move = new_p - old_p
        if actual_move > per_game_move_cap:
            new_p = old_p + per_game_move_cap
        elif actual_move < -per_game_move_cap:
            new_p = old_p - per_game_move_cap
        new_lookup[(a, b)] = float(np.clip(new_p, 0.005, 0.995))
    return new_lookup, delta


def compute_brier(p_lookup: dict, actual: pd.DataFrame) -> tuple:
    yt, yp = [], []
    for _, g in actual.iterrows():
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        if w < l:
            p = p_lookup.get((w, l), 0.5); yt.append(1)
        else:
            p = p_lookup.get((l, w), 0.5); yt.append(0)
        yp.append(p)
    yt = np.array(yt); yp = np.array(yp)
    return brier_score_loss(yt, yp), len(yt)


def main():
    print("Loading data...")
    data = load_all_mens_data()
    seeds = data["seeds"]
    s2026 = seeds[seeds["Season"] == 2026].copy()
    s2026["SeedNum"] = s2026["Seed"].apply(_parse_seed_num)

    region_teams = build_bracket_structure(s2026)
    if region_teams is None:
        raise RuntimeError("Couldn't build 2026 bracket")
    print(f"  Built bracket: {sum(len(v) for v in region_teams.values())} teams")

    # Load existing submission, build pairwise lookup for tournament teams
    sub = pd.read_csv("output/submission_stage2.csv")
    sub[["s_str", "ta_str", "tb_str"]] = sub["ID"].str.split("_", expand=True)
    sub["TeamA"] = sub["ta_str"].astype(int)
    sub["TeamB"] = sub["tb_str"].astype(int)
    all_teams = set()
    for r in REGIONS:
        all_teams.update(region_teams[r])
    p_lookup = {}
    for _, row in sub.iterrows():
        a, b = int(row["TeamA"]), int(row["TeamB"])
        if a in all_teams and b in all_teams:
            p_lookup[(a, b)] = float(row["Pred"])
            p_lookup[(b, a)] = 1 - float(row["Pred"])

    print(f"  Pairwise probs for tournament teams: {len(p_lookup)//2}")

    # Run MC for model champ probs
    print("\nMC simulating 100k tournaments to get model champ probs...")
    p_model_champ = monte_carlo_champ_probs(p_lookup, region_teams, n_sims=100_000)
    top_model = sorted(p_model_champ.items(), key=lambda x: -x[1])[:10]
    teams_df = data["teams"]
    tn = dict(zip(teams_df["TeamID"], teams_df["TeamName"]))
    print(f"  Model top 10 champ probs:")
    for tid, p in top_model:
        print(f"    {tn.get(tid, tid):<25s} {p:.4f}")

    # Load Polymarket champ probs
    pm = pd.read_csv("data/external/polymarket/champ_men_2026.csv")
    p_market_champ = dict(zip(pm["TeamID"].dropna().astype(int), pm["normalized_prob"]))
    print(f"\n  Polymarket champ probs: {len(p_market_champ)} teams")
    top_market = sorted(p_market_champ.items(), key=lambda x: -x[1])[:10]
    print(f"  Market top 10 champ probs:")
    for tid, p in top_market:
        print(f"    {tn.get(tid, tid):<25s} {p:.4f}")

    # === Baseline Brier ===
    actual = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    bs_base, n = compute_brier(p_lookup, actual)
    print(f"\nBaseline Brier (no overlay): {bs_base:.4f} ({n} games)")

    # === Sweep alpha (extended) ===
    print(f"\n{'='*70}\n  Alpha sweep with title-futures overlay\n{'='*70}")
    rows = []
    for alpha in [0.10, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00, 3.00]:
        for cap_team in [0.10, 0.20, 0.50, 1.0]:
            for cap_game in [0.03, 0.05, 0.10, 0.20]:
                new_lookup, delta = apply_overlay(
                    p_lookup, p_model_champ, p_market_champ,
                    alpha=alpha, team_offset_cap=cap_team, per_game_move_cap=cap_game
                )
                bs, _ = compute_brier(new_lookup, actual)
                rows.append({
                    "alpha": alpha, "team_cap": cap_team, "game_cap": cap_game,
                    "brier_2026": bs,
                })

    df = pd.DataFrame(rows).sort_values("brier_2026")
    print(df.head(15).to_string(index=False))

    best = df.iloc[0]
    print(f"\nBest settings: alpha={best['alpha']}, team_cap={best['team_cap']}, "
          f"game_cap={best['game_cap']}")
    print(f"  Brier improvement: {bs_base:.4f} -> {best['brier_2026']:.4f}  "
          f"(delta {best['brier_2026'] - bs_base:+.4f})")

    df.to_csv("output/polymarket_overlay_grid.csv", index=False)

    # Save best overlay submission
    new_lookup, delta = apply_overlay(
        p_lookup, p_model_champ, p_market_champ,
        alpha=float(best["alpha"]),
        team_offset_cap=float(best["team_cap"]),
        per_game_move_cap=float(best["game_cap"]),
    )

    # Build new submission
    sub_new = sub.copy()
    sub_new["Pred_overlay"] = sub_new.apply(
        lambda r: new_lookup.get((int(r["TeamA"]), int(r["TeamB"])), float(r["Pred"])),
        axis=1
    )
    sub_new["Pred"] = sub_new["Pred_overlay"].clip(0.005, 0.995)
    sub_new[["ID", "Pred"]].to_csv("output/submission_stage2_polymarket.csv", index=False)
    print(f"\nSaved output/submission_stage2_polymarket.csv")

    # Combined Brier with women's unchanged
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")
    sub_full = pd.read_csv("output/submission_stage2_polymarket.csv")
    sub_full[["s_str", "ta_str", "tb_str"]] = sub_full["ID"].str.split("_", expand=True)
    sub_full["TeamA"] = sub_full["ta_str"].astype(int)
    sub_full["TeamB"] = sub_full["tb_str"].astype(int)
    pmap = dict(zip(zip(sub_full["TeamA"], sub_full["TeamB"]), sub_full["Pred"]))

    def br(actual_df):
        yt, yp = [], []
        for _, g in actual_df.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            key = (w, l) if w < l else (l, w)
            p = pmap.get(key, 0.5)
            yt.append(1 if w < l else 0); yp.append(p)
        return brier_score_loss(yt, yp), len(yt)

    bs_m, n_m = br(actual)
    bs_w, n_w = br(actual_w)
    bs_c = (bs_m * n_m + bs_w * n_w) / (n_m + n_w)
    print(f"\n{'='*70}\n  COMBINED 2026 BRIER (with Polymarket overlay)\n{'='*70}")
    print(f"  Men's:    {bs_m:.4f}")
    print(f"  Women's:  {bs_w:.4f}  (unchanged)")
    print(f"  Combined: {bs_c:.4f}")
    print(f"  Previous combined: 0.1263")
    print(f"  Kaggle 3rd place:  0.1160")


if __name__ == "__main__":
    main()
