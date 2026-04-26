"""
Apply Polymarket title-futures overlay to BOTH men's and women's predictions
and report two flavors of result:

  (A) Honest: use the 1st-place 2026 published hyperparameters
        alpha=0.10, team_cap=0.10, game_cap=0.03
      These were chosen WITHOUT seeing 2026 results. Numbers reported here
      are valid out-of-sample.

  (B) Oracle: in-sample tuned by sweeping alpha/cap on 2026 actual
      Reported only to show the upper bound of what the overlay could give.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import _parse_seed_num
from scripts.run_polymarket_overlay import (
    monte_carlo_champ_probs, apply_overlay, compute_brier,
    build_bracket_structure, REGIONS,
)
from scripts.build_womens_model import load_womens_data


def build_pair_lookup(sub_df: pd.DataFrame, all_teams: set) -> dict:
    out = {}
    for _, row in sub_df.iterrows():
        a, b = int(row["TeamA"]), int(row["TeamB"])
        if a in all_teams and b in all_teams:
            out[(a, b)] = float(row["Pred"])
            out[(b, a)] = 1 - float(row["Pred"])
    return out


def evaluate_overlay_on_actual(
    sub_df: pd.DataFrame,
    region_teams: dict,
    p_market_champ: dict,
    actual: pd.DataFrame,
    alpha: float, team_cap: float, game_cap: float,
    n_sims: int = 80_000,
):
    all_teams = set()
    for r in REGIONS:
        all_teams.update(region_teams[r])
    p_lookup = build_pair_lookup(sub_df, all_teams)
    p_model_champ = monte_carlo_champ_probs(p_lookup, region_teams, n_sims=n_sims)
    new_lookup, _ = apply_overlay(
        p_lookup, p_model_champ, p_market_champ,
        alpha=alpha, team_offset_cap=team_cap, per_game_move_cap=game_cap
    )
    bs_base, _ = compute_brier(p_lookup, actual)
    bs_new, _ = compute_brier(new_lookup, actual)
    return bs_base, bs_new, new_lookup, p_lookup


def main():
    print("Loading mens data...")
    data = load_all_mens_data()
    seeds_m = data["seeds"]
    s2026_m = seeds_m[seeds_m["Season"] == 2026].copy()
    s2026_m["SeedNum"] = s2026_m["Seed"].apply(_parse_seed_num)
    region_m = build_bracket_structure(s2026_m)

    print("Loading womens data...")
    data_w = load_womens_data()
    seeds_w = data_w["seeds"]
    s2026_w = seeds_w[seeds_w["Season"] == 2026].copy()
    s2026_w["SeedNum"] = s2026_w["Seed"].apply(_parse_seed_num)
    region_w = build_bracket_structure(s2026_w)
    if region_w is None:
        print("  Women's bracket not buildable, skipping women's overlay")

    sub = pd.read_csv("output/submission_stage2.csv")
    sub[["s_str", "ta_str", "tb_str"]] = sub["ID"].str.split("_", expand=True)
    sub["TeamA"] = sub["ta_str"].astype(int)
    sub["TeamB"] = sub["tb_str"].astype(int)

    pm_m = pd.read_csv("data/external/polymarket/champ_men_2026.csv")
    pm_market_m = dict(zip(pm_m["TeamID"].dropna().astype(int), pm_m["normalized_prob"]))

    pm_w_path = DATA_DIR / "external" / "polymarket" / "champ_women_2026.csv"
    if pm_w_path.exists():
        pm_w = pd.read_csv(pm_w_path)
        pm_market_w = dict(zip(pm_w["TeamID"].dropna().astype(int), pm_w["normalized_prob"]))
    else:
        pm_market_w = {}

    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")

    # ----- Men's evaluation -----
    print(f"\n{'='*70}\n  MEN'S overlay (1st place hyperparams: alpha=0.10, "
          f"caps 0.10/0.03)\n{'='*70}")
    bs_base_m, bs_honest_m, lookup_honest_m, lookup_base_m = evaluate_overlay_on_actual(
        sub, region_m, pm_market_m, actual_m,
        alpha=0.10, team_cap=0.10, game_cap=0.03,
    )
    print(f"  Baseline:    {bs_base_m:.4f}")
    print(f"  + Polymarket 1st place params: {bs_honest_m:.4f}  "
          f"(delta {bs_honest_m - bs_base_m:+.4f})")

    # Oracle sweep on men's
    print(f"\n  Oracle in-sample sweep (CHEATS by tuning on actual 2026):")
    best_bs = bs_honest_m; best_params = (0.10, 0.10, 0.03)
    all_teams_m = set()
    for r in REGIONS: all_teams_m.update(region_m[r])
    p_lookup_m = build_pair_lookup(sub, all_teams_m)
    p_model_champ_m = monte_carlo_champ_probs(p_lookup_m, region_m, n_sims=100_000)
    for alpha in [0.10, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]:
        for cap_t in [0.10, 0.30, 1.00]:
            for cap_g in [0.03, 0.05, 0.10, 0.20]:
                lk, _ = apply_overlay(p_lookup_m, p_model_champ_m, pm_market_m,
                                       alpha=alpha, team_offset_cap=cap_t,
                                       per_game_move_cap=cap_g)
                bs, _ = compute_brier(lk, actual_m)
                if bs < best_bs:
                    best_bs = bs; best_params = (alpha, cap_t, cap_g)
    print(f"  Oracle best: alpha={best_params[0]}, team_cap={best_params[1]}, "
          f"game_cap={best_params[2]} -> Brier {best_bs:.4f}")

    bs_oracle_m = best_bs
    lookup_oracle_m, _ = apply_overlay(p_lookup_m, p_model_champ_m, pm_market_m,
                                        alpha=best_params[0],
                                        team_offset_cap=best_params[1],
                                        per_game_move_cap=best_params[2])

    # ----- Women's evaluation -----
    if region_w is not None and pm_market_w:
        print(f"\n{'='*70}\n  WOMEN'S overlay (1st place hyperparams)\n{'='*70}")
        bs_base_w, bs_honest_w, lookup_honest_w, lookup_base_w = evaluate_overlay_on_actual(
            sub, region_w, pm_market_w, actual_w,
            alpha=0.10, team_cap=0.10, game_cap=0.03,
        )
        print(f"  Baseline:    {bs_base_w:.4f}")
        print(f"  + Polymarket 1st place params: {bs_honest_w:.4f}  "
              f"(delta {bs_honest_w - bs_base_w:+.4f})")

        # Oracle sweep on women's
        all_teams_w = set()
        for r in REGIONS: all_teams_w.update(region_w[r])
        p_lookup_w = build_pair_lookup(sub, all_teams_w)
        p_model_champ_w = monte_carlo_champ_probs(p_lookup_w, region_w, n_sims=100_000)
        best_bs_w = bs_honest_w; best_params_w = (0.10, 0.10, 0.03)
        for alpha in [0.10, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]:
            for cap_t in [0.10, 0.30, 1.00]:
                for cap_g in [0.03, 0.05, 0.10, 0.20]:
                    lk, _ = apply_overlay(p_lookup_w, p_model_champ_w, pm_market_w,
                                           alpha=alpha, team_offset_cap=cap_t,
                                           per_game_move_cap=cap_g)
                    bs, _ = compute_brier(lk, actual_w)
                    if bs < best_bs_w:
                        best_bs_w = bs; best_params_w = (alpha, cap_t, cap_g)
        print(f"  Oracle best: alpha={best_params_w[0]}, team_cap={best_params_w[1]}, "
              f"game_cap={best_params_w[2]} -> Brier {best_bs_w:.4f}")

        bs_oracle_w = best_bs_w
        lookup_oracle_w, _ = apply_overlay(p_lookup_w, p_model_champ_w, pm_market_w,
                                            alpha=best_params_w[0],
                                            team_offset_cap=best_params_w[1],
                                            per_game_move_cap=best_params_w[2])
    else:
        bs_base_w = brier_score_loss(*([] + []) , [0.5]*0) if False else None
        # Use uncalibrated baseline
        from sklearn.metrics import brier_score_loss as _bsl
        # baseline: just the existing Pred values
        # We need pmap from sub
        pmap = dict(zip(zip(sub["TeamA"], sub["TeamB"]), sub["Pred"]))
        yt, yp = [], []
        for _, g in actual_w.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            key = (w, l) if w < l else (l, w)
            yt.append(1 if w < l else 0)
            yp.append(pmap.get(key, 0.5))
        bs_base_w = _bsl(yt, yp)
        bs_honest_w = bs_base_w
        bs_oracle_w = bs_base_w

    # ----- Combined report -----
    n_m = len(actual_m); n_w = len(actual_w)
    bs_base_c = (bs_base_m * n_m + bs_base_w * n_w) / (n_m + n_w)
    bs_honest_c = (bs_honest_m * n_m + bs_honest_w * n_w) / (n_m + n_w)
    bs_oracle_c = (bs_oracle_m * n_m + bs_oracle_w * n_w) / (n_m + n_w)

    print(f"\n{'='*70}\n  FINAL COMBINED BRIER REPORT\n{'='*70}")
    print(f"                                Men's    Women's   Combined")
    print(f"  Baseline (no Polymarket):    {bs_base_m:.4f}  {bs_base_w:.4f}   {bs_base_c:.4f}")
    print(f"  Honest (1st place params):   {bs_honest_m:.4f}  {bs_honest_w:.4f}   {bs_honest_c:.4f}")
    print(f"  Oracle (in-sample tuned):    {bs_oracle_m:.4f}  {bs_oracle_w:.4f}   {bs_oracle_c:.4f}")
    print(f"\n  Kaggle 3rd place:            -        -         0.1160")
    print(f"  Honest delta from baseline:  {bs_honest_c - bs_base_c:+.4f}")
    print(f"  Oracle delta from baseline:  {bs_oracle_c - bs_base_c:+.4f}")

    # Save honest submission
    sub_honest = sub.copy()
    sub_honest["Pred"] = sub_honest.apply(
        lambda r: lookup_honest_m.get((int(r["TeamA"]), int(r["TeamB"])),
                                       lookup_honest_w.get((int(r["TeamA"]), int(r["TeamB"])), float(r["Pred"]))
                                       if region_w is not None and pm_market_w else float(r["Pred"])),
        axis=1
    ).clip(0.005, 0.995)
    sub_honest[["ID", "Pred"]].to_csv("output/submission_stage2_polymarket_honest.csv", index=False)
    print(f"\n  Saved output/submission_stage2_polymarket_honest.csv")

    # Summary CSV
    pd.DataFrame([
        {"setting": "baseline", "men": bs_base_m, "women": bs_base_w, "combined": bs_base_c},
        {"setting": "polymarket_honest_1st_place_params", "men": bs_honest_m, "women": bs_honest_w, "combined": bs_honest_c},
        {"setting": "polymarket_oracle_insample", "men": bs_oracle_m, "women": bs_oracle_w, "combined": bs_oracle_c},
    ]).to_csv("output/polymarket_full_summary.csv", index=False)


if __name__ == "__main__":
    main()
