"""
Stack the 1st-place tricks on top of our pipeline:

  (a) In-sample isotonic calibration fit on LOTO OOF predictions
      (1st place reported men's CV Brier 0.185 -> 0.182, women's 0.139 -> 0.136)
  (b) Hard sharpening: P >= 0.97 -> 0.999, P <= 0.03 -> 0.001
      (1st place applied to 35 games: 28 first-round + 7 later)
  (c) Optional: stack on top of Polymarket title-futures overlay

We test every combination on the 2026 actual results to find the best stack
and report combined men+women Brier.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import make_build_features_fn, _parse_seed_num
from src.models import MultiFeatureLogistic
from scripts.build_womens_model import (
    load_womens_data, build_womens_features, WomensLogistic,
)
from scripts.generate_kaggle_submission import build_submission_features
from scripts.run_polymarket_overlay import (
    monte_carlo_champ_probs, apply_overlay, build_bracket_structure, REGIONS,
)


def hard_sharpen(p: np.ndarray, low_thresh=0.03, high_thresh=0.97,
                 low_target=0.001, high_target=0.999) -> np.ndarray:
    p = np.array(p, dtype=float)
    out = p.copy()
    out[p >= high_thresh] = high_target
    out[p <= low_thresh] = low_target
    return out


def collect_actual(p_lookup: dict, actual: pd.DataFrame) -> tuple:
    yt, yp = [], []
    for _, g in actual.iterrows():
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        if w < l:
            p = p_lookup.get((w, l), 0.5); yt.append(1)
        else:
            p = p_lookup.get((l, w), 0.5); yt.append(0)
        yp.append(p)
    return np.array(yt), np.array(yp)


def loto_oof_men(X_all, y_all, season_arr):
    p_oof = np.zeros(len(X_all))
    for s in np.unique(season_arr):
        tr = season_arr != s; te = season_arr == s
        m = MultiFeatureLogistic(C=0.5)
        m.fit(X_all.loc[tr], y_all[tr])
        p_oof[te] = m.predict_proba(X_all.loc[te])[:, 1]
    return p_oof


def loto_oof_women(X_all, y_all, season_arr):
    p_oof = np.zeros(len(X_all))
    for s in np.unique(season_arr):
        tr = season_arr != s; te = season_arr == s
        m = WomensLogistic(C=0.5)
        m.fit(X_all.loc[tr], y_all[tr])
        p_oof[te] = m.predict_proba(X_all.loc[te])[:, 1]
    return p_oof


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]

    # ===== MEN'S =====
    print(f"{'='*70}\n  MEN'S: build LOTO OOF + final predictions\n{'='*70}")
    data_m = load_all_mens_data()
    build_fn = make_build_features_fn(data_m)
    X_m, y_m = build_fn(seasons)
    season_m = X_m["Season"].values

    p_oof_m = loto_oof_men(X_m, y_m, season_m)
    print(f"  Men's LOTO OOF Brier: {brier_score_loss(y_m, p_oof_m):.4f}")

    iso_m = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso_m.fit(p_oof_m, y_m)
    p_oof_m_iso = iso_m.predict(p_oof_m)
    print(f"  Men's LOTO OOF Brier (isotonic, in-sample): {brier_score_loss(y_m, p_oof_m_iso):.4f}")

    final_m = MultiFeatureLogistic(C=0.5).fit(X_m, y_m)
    sub_path = str(DATA_DIR / "SampleSubmissionStage2.csv")
    _, X_2026_m, _ = build_submission_features(data_m, 2026, sub_path)
    p_2026_m_raw = final_m.predict_proba(X_2026_m)[:, 1]
    print(f"  Men's 2026 predictions: {len(p_2026_m_raw)} pairs")

    # ===== WOMEN'S =====
    print(f"\n{'='*70}\n  WOMEN'S: build LOTO OOF + final predictions\n{'='*70}")
    data_w = load_womens_data()
    X_w, y_w = build_womens_features(data_w, seasons)
    season_w = X_w["Season"].values

    p_oof_w = loto_oof_women(X_w, y_w, season_w)
    print(f"  Women's LOTO OOF Brier: {brier_score_loss(y_w, p_oof_w):.4f}")

    iso_w = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso_w.fit(p_oof_w, y_w)
    p_oof_w_iso = iso_w.predict(p_oof_w)
    print(f"  Women's LOTO OOF Brier (isotonic, in-sample): {brier_score_loss(y_w, p_oof_w_iso):.4f}")

    # Train final women's model on all data and predict 2026 pairs
    final_w = WomensLogistic(C=0.5).fit(X_w, y_w)

    # Build women's 2026 prediction features (per-team-pair)
    print("  Building women's 2026 features...")
    seeds_w = data_w["seeds"]
    s2026_w = seeds_w[seeds_w["Season"] == 2026].copy()
    s2026_w["SeedNum"] = s2026_w["Seed"].apply(_parse_seed_num)
    tourney_teams_w = list(s2026_w["TeamID"])
    pairs_w = []
    for i, ta in enumerate(tourney_teams_w):
        for tb in tourney_teams_w[i+1:]:
            pairs_w.append({"Season": 2026, "TeamA": ta, "TeamB": tb})
    pairs_w_df = pd.DataFrame(pairs_w)
    # Build features for these pairs using the women's pipeline (re-use build_womens_features)
    # Easiest: hand-build the same features as build_womens_features does
    from src.pipeline import build_efficiency_for_season, build_four_factors_for_season, build_momentum_for_season
    eff_w = build_efficiency_for_season(data_w, 2026)
    ff_w = build_four_factors_for_season(data_w, 2026)
    mom_w = build_momentum_for_season(data_w, 2026, tourney_teams_w)
    mom_map_w = {row["TeamID"]: row for _, row in mom_w.iterrows()}
    bart_w_path = DATA_DIR / "external" / "barttorvik_w_2026.csv"
    bart_w = pd.read_csv(bart_w_path).set_index("TeamID") if bart_w_path.exists() else None

    seed_map_w = dict(zip(s2026_w["TeamID"], s2026_w["SeedNum"]))
    feats_w = []
    for _, p in pairs_w_df.iterrows():
        ta, tb = int(p["TeamA"]), int(p["TeamB"])
        feat = {
            "Season": 2026, "TeamA": ta, "TeamB": tb,
            "seed_diff": seed_map_w.get(ta, 16) - seed_map_w.get(tb, 16),
            "seed_A": seed_map_w.get(ta, 16), "seed_B": seed_map_w.get(tb, 16),
        }
        for col in ["off_eff", "def_eff", "net_eff", "tempo"]:
            va = eff_w.loc[ta, col] if (not eff_w.empty and ta in eff_w.index) else np.nan
            vb = eff_w.loc[tb, col] if (not eff_w.empty and tb in eff_w.index) else np.nan
            feat[f"{col}_diff"] = va - vb
        for col in ["efg_pct", "to_pct", "or_pct", "ft_rate",
                    "opp_efg_pct", "opp_to_pct", "opp_or_pct", "opp_ft_rate"]:
            va = ff_w.loc[ta, col] if (not ff_w.empty and ta in ff_w.index) else np.nan
            vb = ff_w.loc[tb, col] if (not ff_w.empty and tb in ff_w.index) else np.nan
            feat[f"{col}_diff"] = va - vb
        if bart_w is not None:
            for src, dst in [("AdjOE", "bart_adjoe_diff"), ("AdjDE", "bart_adjde_diff"),
                             ("NetRtg", "bart_net_diff"), ("Barthag", "bart_barthag_diff"),
                             ("AdjTempo", "bart_tempo_diff")]:
                va = bart_w.loc[ta, src] if ta in bart_w.index else np.nan
                vb = bart_w.loc[tb, src] if tb in bart_w.index else np.nan
                feat[dst] = va - vb
        ma = mom_map_w.get(ta, {})
        mb = mom_map_w.get(tb, {})
        feat["momentum_winpct_diff"] = ma.get("momentum_win_pct", 0.5) - mb.get("momentum_win_pct", 0.5)
        feat["momentum_margin_diff"] = ma.get("momentum_avg_margin", 0.0) - mb.get("momentum_avg_margin", 0.0)
        feats_w.append(feat)
    X_2026_w = pd.DataFrame(feats_w)
    p_2026_w_raw = final_w.predict_proba(X_2026_w)[:, 1]
    print(f"  Women's 2026 predictions: {len(p_2026_w_raw)} pairs")

    # ===== Build pair lookups =====
    def build_lookup(X_2026, p_2026):
        out = {}
        for i, row in X_2026.reset_index(drop=True).iterrows():
            a, b = int(row["TeamA"]), int(row["TeamB"])
            out[(a, b)] = float(p_2026[i])
            out[(b, a)] = 1 - float(p_2026[i])
        return out

    # Apply isotonic from LOTO OOF to 2026 predictions
    p_2026_m_iso = iso_m.predict(p_2026_m_raw)
    p_2026_w_iso = iso_w.predict(p_2026_w_raw)

    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")

    # ===== Evaluate every combination =====
    print(f"\n{'='*70}\n  Evaluating stacks on 2026 actual\n{'='*70}")

    rows = []
    n_m, n_w = len(actual_m), len(actual_w)

    def evaluate(p_m_arr, p_w_arr, label):
        lk_m = build_lookup(X_2026_m, p_m_arr)
        lk_w = build_lookup(X_2026_w, p_w_arr)
        yt_m, yp_m = collect_actual(lk_m, actual_m)
        yt_w, yp_w = collect_actual(lk_w, actual_w)
        bs_m = brier_score_loss(yt_m, yp_m)
        bs_w = brier_score_loss(yt_w, yp_w)
        bs_c = (bs_m * n_m + bs_w * n_w) / (n_m + n_w)
        rows.append({"strategy": label, "men": bs_m, "women": bs_w, "combined": bs_c})
        return bs_m, bs_w, bs_c

    # 1. Raw baseline
    evaluate(p_2026_m_raw, p_2026_w_raw, "raw_baseline")

    # 2. Isotonic only
    evaluate(p_2026_m_iso, p_2026_w_iso, "isotonic_only")

    # 3. Sharpening only (on raw)
    p_m_sh = hard_sharpen(p_2026_m_raw)
    p_w_sh = hard_sharpen(p_2026_w_raw)
    evaluate(p_m_sh, p_w_sh, "sharpen_0.97/0.03_only")

    # 4. Isotonic + sharpening
    p_m_is_sh = hard_sharpen(p_2026_m_iso)
    p_w_is_sh = hard_sharpen(p_2026_w_iso)
    evaluate(p_m_is_sh, p_w_is_sh, "isotonic+sharpen_0.97/0.03")

    # 5. Sharpening at different thresholds (in-sample tuned)
    for low, high in [(0.05, 0.95), (0.04, 0.96), (0.03, 0.97), (0.02, 0.98), (0.01, 0.99)]:
        p_m_t = hard_sharpen(p_2026_m_iso, low_thresh=low, high_thresh=high,
                              low_target=low/3, high_target=1-low/3)
        p_w_t = hard_sharpen(p_2026_w_iso, low_thresh=low, high_thresh=high,
                              low_target=low/3, high_target=1-low/3)
        evaluate(p_m_t, p_w_t, f"isotonic+sharpen_{low}/{high}")

    # 6. Stack with Polymarket overlay
    pm_m = pd.read_csv("data/external/polymarket/champ_men_2026.csv")
    pm_market_m = dict(zip(pm_m["TeamID"].dropna().astype(int), pm_m["normalized_prob"]))
    pm_w_path = DATA_DIR / "external" / "polymarket" / "champ_women_2026.csv"
    pm_market_w = {}
    if pm_w_path.exists():
        pm_w = pd.read_csv(pm_w_path)
        pm_market_w = dict(zip(pm_w["TeamID"].dropna().astype(int), pm_w["normalized_prob"]))

    seeds_m = data_m["seeds"]
    s2026_m = seeds_m[seeds_m["Season"] == 2026].copy()
    s2026_m["SeedNum"] = s2026_m["Seed"].apply(_parse_seed_num)
    region_m = build_bracket_structure(s2026_m)
    region_w = build_bracket_structure(s2026_w)

    # Polymarket overlay needs lookups; apply on top of isotonic preds
    lk_m_iso = build_lookup(X_2026_m, p_2026_m_iso)
    all_m = set(); [all_m.update(region_m[r]) for r in REGIONS]
    lk_m_iso_tourney = {k: v for k, v in lk_m_iso.items() if k[0] in all_m and k[1] in all_m}
    p_model_champ_m = monte_carlo_champ_probs(lk_m_iso_tourney, region_m, n_sims=80_000)
    lk_m_overlay, _ = apply_overlay(lk_m_iso_tourney, p_model_champ_m, pm_market_m,
                                     alpha=0.10, team_offset_cap=0.10, per_game_move_cap=0.03)
    # Merge overlay back into full lookup
    lk_m_full = dict(lk_m_iso)
    lk_m_full.update(lk_m_overlay)
    # Convert lookup back to per-pair array aligned with X_2026_m
    p_m_iso_overlay = np.array([
        lk_m_full.get((int(r["TeamA"]), int(r["TeamB"])), p_2026_m_iso[i])
        for i, (_, r) in enumerate(X_2026_m.iterrows())
    ])

    if region_w is not None and pm_market_w:
        lk_w_iso = build_lookup(X_2026_w, p_2026_w_iso)
        all_w = set(); [all_w.update(region_w[r]) for r in REGIONS]
        lk_w_iso_tourney = {k: v for k, v in lk_w_iso.items() if k[0] in all_w and k[1] in all_w}
        p_model_champ_w = monte_carlo_champ_probs(lk_w_iso_tourney, region_w, n_sims=80_000)
        lk_w_overlay, _ = apply_overlay(lk_w_iso_tourney, p_model_champ_w, pm_market_w,
                                         alpha=0.10, team_offset_cap=0.10, per_game_move_cap=0.03)
        lk_w_full = dict(lk_w_iso)
        lk_w_full.update(lk_w_overlay)
        p_w_iso_overlay = np.array([
            lk_w_full.get((int(r["TeamA"]), int(r["TeamB"])), p_2026_w_iso[i])
            for i, (_, r) in enumerate(X_2026_w.iterrows())
        ])
    else:
        p_w_iso_overlay = p_2026_w_iso

    evaluate(p_m_iso_overlay, p_w_iso_overlay, "isotonic+polymarket")
    evaluate(hard_sharpen(p_m_iso_overlay), hard_sharpen(p_w_iso_overlay),
             "isotonic+polymarket+sharpen_0.97/0.03")

    # Final report
    df = pd.DataFrame(rows).sort_values("combined")
    print(df.to_string(index=False))
    df.to_csv("output/sharpen_isotonic_summary.csv", index=False)
    print(f"\n  Best: {df.iloc[0]['strategy']} -> combined Brier {df.iloc[0]['combined']:.4f}")
    print(f"  Kaggle 1st place final: 0.1097")
    print(f"  Kaggle 3rd place final: 0.1160")


if __name__ == "__main__":
    main()
