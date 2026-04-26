"""
Phase A final submission: combine everything we have without external markets.

Per-gender best:
  Men's:   triple blend (top3 LR + MultiFeat + XGB+harry) + isotonic
           + Polymarket title-futures overlay (alpha=0.10, honest)
  Women's: XGB+harry alone (no isotonic, no overlay)

Reports honest (1st-place hyperparams) and oracle (in-sample tuned) numbers.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.data_collection import DATA_DIR
from src.pipeline import _parse_seed_num
from scripts.run_polymarket_overlay import (
    build_bracket_structure, monte_carlo_champ_probs, apply_overlay, REGIONS,
)


def br_lk(lk, actual):
    yt, yp = [], []
    for _, g in actual.iterrows():
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        key = (w, l) if w < l else (l, w)
        yt.append(1 if w < l else 0)
        yp.append(lk.get(key, 0.5))
    return brier_score_loss(yt, yp), len(yt)


def main():
    # Start from triple blend submission (best per-gender already done there)
    sub = pd.read_csv("output/submission_stage2_triple.csv")
    sub[["s_str", "ta_str", "tb_str"]] = sub["ID"].str.split("_", expand=True)
    sub["TeamA"] = sub["ta_str"].astype(int)
    sub["TeamB"] = sub["tb_str"].astype(int)
    pmap = dict(zip(zip(sub["TeamA"], sub["TeamB"]), sub["Pred"]))

    # Apply Polymarket overlay to men's only
    seeds = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    s2026 = seeds[seeds["Season"] == 2026].copy()
    s2026["SeedNum"] = s2026["Seed"].apply(_parse_seed_num)
    region_m = build_bracket_structure(s2026)
    all_m = set()
    for r in REGIONS: all_m.update(region_m[r])
    lk_m_tourney = {}
    for k, v in pmap.items():
        a, b = k
        if a in all_m and b in all_m:
            lk_m_tourney[(a, b)] = v
            lk_m_tourney[(b, a)] = 1 - v

    p_model_champ = monte_carlo_champ_probs(lk_m_tourney, region_m, n_sims=100_000)
    pm_m = pd.read_csv("data/external/polymarket/champ_men_2026.csv")
    pm_market = dict(zip(pm_m["TeamID"].dropna().astype(int), pm_m["normalized_prob"]))

    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")

    bs_base_m, n_m = br_lk(lk_m_tourney, actual_m)
    print(f"Triple blend men's baseline: {bs_base_m:.4f}")

    # Honest (1st-place hyperparams from Polymarket-using ledmaster)
    lk_honest, _ = apply_overlay(
        lk_m_tourney, p_model_champ, pm_market,
        alpha=0.10, team_offset_cap=0.10, per_game_move_cap=0.03,
    )
    bs_honest_m, _ = br_lk(lk_honest, actual_m)
    print(f"  + Polymarket overlay (honest, alpha=0.10): {bs_honest_m:.4f}")

    # Oracle in-sample sweep
    best = (None, bs_honest_m)
    for alpha in [0.10, 0.30, 0.50, 0.75, 1.00, 1.50, 2.00]:
        for cap_t in [0.10, 0.30, 0.50, 1.0]:
            for cap_g in [0.03, 0.05, 0.10, 0.20]:
                lk_test, _ = apply_overlay(
                    lk_m_tourney, p_model_champ, pm_market,
                    alpha=alpha, team_offset_cap=cap_t, per_game_move_cap=cap_g,
                )
                bs, _ = br_lk(lk_test, actual_m)
                if bs < best[1]:
                    best = ((alpha, cap_t, cap_g), bs)
    bs_oracle_m = best[1]
    print(f"  + Polymarket overlay (oracle): {bs_oracle_m:.4f}  params={best[0]}")
    lk_oracle, _ = apply_overlay(
        lk_m_tourney, p_model_champ, pm_market,
        alpha=best[0][0], team_offset_cap=best[0][1], per_game_move_cap=best[0][2],
    )

    # Women's: keep triple blend (which already chose XGB+harry per-gender best)
    # This is already in submission_stage2_triple.csv
    bs_w = br_lk(pmap, actual_w)[0]
    n_w = len(actual_w)
    print(f"\nWomen's (XGB+harry, unchanged): {bs_w:.4f}")

    # Build final submissions
    print(f"\n{'='*70}\n  PHASE A FINAL RESULTS\n{'='*70}")
    print(f"{'Strategy':<45} {'Men':>8} {'Women':>8} {'Combined':>10}")

    rows = []
    for label, lk_men in [
        ("Baseline (triple blend, no overlay)", lk_m_tourney),
        ("+ Polymarket honest (alpha=0.10)", lk_honest),
        ("+ Polymarket oracle (in-sample)",   lk_oracle),
    ]:
        bs_m, _ = br_lk(lk_men, actual_m)
        bs_c = (bs_m * n_m + bs_w * n_w) / (n_m + n_w)
        rows.append({"strategy": label, "men": bs_m, "women": bs_w, "combined": bs_c})
        print(f"  {label:<45} {bs_m:>8.4f} {bs_w:>8.4f} {bs_c:>10.4f}")

    # Save honest version as submission
    new_sub = sub.copy()
    new_sub["Pred"] = new_sub.apply(
        lambda r: lk_honest.get((int(r["TeamA"]), int(r["TeamB"])), float(r["Pred"])),
        axis=1,
    ).clip(0.005, 0.995)
    new_sub[["ID", "Pred"]].to_csv("output/submission_stage2_phaseA.csv", index=False)
    print(f"\nSaved output/submission_stage2_phaseA.csv (honest Polymarket overlay)")

    pd.DataFrame(rows).to_csv("output/phase_a_summary.csv", index=False)
    print(f"\n  Phase A trajectory:")
    print(f"    Initial baseline:                    0.1264")
    print(f"    Phase A best (honest):               {rows[1]['combined']:.4f}")
    print(f"    Phase A best (oracle):               {rows[2]['combined']:.4f}")
    print(f"    Kaggle 3rd:                          0.1160")
    print(f"    Kaggle 1st:                          0.1097")


if __name__ == "__main__":
    main()
