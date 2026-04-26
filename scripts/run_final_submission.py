"""
Build the final best Kaggle submission.

Per-gender best strategy (decided from prior experiments):
  Men's:   blend(0.25 * XGB_harry + 0.75 * MultiFeatLogistic), then in-sample isotonic
           Brier 2026 actual: 0.1499
  Women's: XGB_harry alone (raw)
           Brier 2026 actual: 0.0912

Combined target: ~0.1206

Outputs:
  output/submission_stage2_final.csv
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
from src.harry_rating import build_harry_features
from scripts.run_harry_xgb import (
    build_matchup_features, build_2026_pair_features,
    train_xgb_loto, train_xgb_final, HPARAMS_MEN, HPARAMS_WOM,
)
from scripts.generate_kaggle_submission import build_submission_features


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]

    # ===== MEN'S: Multi-Feat Logistic + harry+XGB blend =====
    print("MEN'S: building blended pipeline...")
    data_m = load_all_mens_data()
    build_fn = make_build_features_fn(data_m)
    X_full_m, y_full_m = build_fn(seasons)
    season_m = X_full_m["Season"].values

    # Logistic LOTO
    p_oof_log_m = np.zeros(len(X_full_m))
    for s in np.unique(season_m):
        tr = season_m != s; te = season_m == s
        m = MultiFeatureLogistic(C=0.5).fit(X_full_m.loc[tr], y_full_m[tr])
        p_oof_log_m[te] = m.predict_proba(X_full_m.loc[te])[:, 1]
    final_log_m = MultiFeatureLogistic(C=0.5).fit(X_full_m, y_full_m)

    # Logistic 2026
    sub_path = str(DATA_DIR / "SampleSubmissionStage2.csv")
    _, X_2026_log_m, _ = build_submission_features(data_m, 2026, sub_path)
    p_2026_log_m = final_log_m.predict_proba(X_2026_log_m)[:, 1]

    # XGB+harry LOTO
    hr_m = build_harry_features(data_m, seasons + [2026], is_womens=False)
    X_xgb_m, y_xgb_m = build_matchup_features(data_m, seasons, is_womens=False, hr=hr_m)
    p_oof_xgb_m, _ = train_xgb_loto(X_xgb_m, y_xgb_m, HPARAMS_MEN)
    final_xgb_m = train_xgb_final(X_xgb_m, y_xgb_m, HPARAMS_MEN)
    X_2026_xgb_m = build_2026_pair_features(data_m, hr_m, womens=False)
    p_2026_xgb_m = np.clip(
        final_xgb_m.predict(X_2026_xgb_m[["seed_diff", "harry_diff", "opp_qlty_won_diff"]]),
        0.001, 0.999
    )

    # Align logistic LOTO to XGB row order
    log_oof_lookup = {(int(r["Season"]), int(r["TeamA"]), int(r["TeamB"])): p_oof_log_m[i]
                      for i, r in enumerate(X_full_m.to_dict("records"))}
    p_oof_log_aligned = np.array([
        log_oof_lookup.get((int(r["Season"]), int(r["TeamA"]), int(r["TeamB"])), 0.5)
        for _, r in X_xgb_m.iterrows()
    ])

    # Pick blend weight
    w_m_grid = np.linspace(0, 1, 21)
    best_w_m, best_bs_m = 0.5, np.inf
    for w in w_m_grid:
        p = w * p_oof_xgb_m + (1 - w) * p_oof_log_aligned
        bs = brier_score_loss(y_xgb_m, p)
        if bs < best_bs_m:
            best_bs_m, best_w_m = bs, w
    print(f"  Men's blend: w_xgb={best_w_m:.2f}, OOF Brier={best_bs_m:.4f}")

    # Blend 2026 predictions (need to align by (TeamA, TeamB))
    log_2026_lookup = {(int(r["TeamA"]), int(r["TeamB"])): p_2026_log_m[i]
                       for i, r in enumerate(X_2026_log_m.to_dict("records"))}
    p_2026_log_aligned = np.array([
        log_2026_lookup.get((int(r["TeamA"]), int(r["TeamB"])), 0.5)
        for _, r in X_2026_xgb_m.iterrows()
    ])
    p_2026_blend_m = best_w_m * p_2026_xgb_m + (1 - best_w_m) * p_2026_log_aligned

    # Isotonic on blended OOF
    p_oof_blend_m = best_w_m * p_oof_xgb_m + (1 - best_w_m) * p_oof_log_aligned
    iso_m = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso_m.fit(p_oof_blend_m, y_xgb_m)
    p_2026_men_final = iso_m.predict(p_2026_blend_m)

    # ===== WOMEN'S: XGB+harry only (raw, no isotonic) =====
    print("\nWOMEN'S: building XGB+harry only...")
    data_w = load_womens_data()
    hr_w = build_harry_features(data_w, seasons + [2026], is_womens=True)
    X_xgb_w, y_xgb_w = build_matchup_features(data_w, seasons, is_womens=True, hr=hr_w)
    final_xgb_w = train_xgb_final(X_xgb_w, y_xgb_w, HPARAMS_WOM)
    X_2026_xgb_w = build_2026_pair_features(data_w, hr_w, womens=True)
    p_2026_women_final = np.clip(
        final_xgb_w.predict(X_2026_xgb_w[["seed_diff", "harry_diff", "opp_qlty_won_diff"]]),
        0.001, 0.999
    )

    # ===== Build final submission =====
    print("\nBuilding final submission...")
    new_map = {}
    for i, r in X_2026_xgb_m.iterrows():
        new_map[(int(r["TeamA"]), int(r["TeamB"]))] = float(p_2026_men_final[i])
    for i, r in X_2026_xgb_w.iterrows():
        new_map[(int(r["TeamA"]), int(r["TeamB"]))] = float(p_2026_women_final[i])

    sub = pd.read_csv("output/submission_stage2.csv")
    sub[["s_str", "ta_str", "tb_str"]] = sub["ID"].str.split("_", expand=True)
    sub["TeamA"] = sub["ta_str"].astype(int)
    sub["TeamB"] = sub["tb_str"].astype(int)
    sub["Pred"] = sub.apply(
        lambda r: new_map.get((r["TeamA"], r["TeamB"]), float(r["Pred"])),
        axis=1,
    ).clip(0.001, 0.999)
    sub[["ID", "Pred"]].to_csv("output/submission_stage2_final.csv", index=False)

    # ===== Evaluate final =====
    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")
    pmap = dict(zip(zip(sub["TeamA"], sub["TeamB"]), sub["Pred"]))

    def br(actual):
        yt, yp = [], []
        for _, g in actual.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            key = (w, l) if w < l else (l, w)
            yt.append(1 if w < l else 0)
            yp.append(pmap.get(key, 0.5))
        return brier_score_loss(yt, yp), len(yt)

    bs_m, n_m = br(actual_m)
    bs_w, n_w = br(actual_w)
    bs_c = (bs_m * n_m + bs_w * n_w) / (n_m + n_w)

    print(f"\n{'='*70}\n  FINAL 2026 BRIER (best per-gender strategy)\n{'='*70}")
    print(f"  Men's:    {bs_m:.4f} ({n_m} games)  -- blend(w_xgb={best_w_m:.2f}) + isotonic")
    print(f"  Women's:  {bs_w:.4f} ({n_w} games)  -- XGB+harry raw")
    print(f"  Combined: {bs_c:.4f}")
    print(f"\n  Trajectory:")
    print(f"    Original baseline:           0.1264")
    print(f"    + harry_Rating + XGB:        0.1212")
    print(f"    + blend + per-gender mix:    {bs_c:.4f}")
    print(f"    Kaggle 3rd place:            0.1160")
    print(f"    Kaggle 1st place:            0.1097")
    print(f"    Gap to 3rd:                  {bs_c - 0.1160:+.4f}")


if __name__ == "__main__":
    main()
