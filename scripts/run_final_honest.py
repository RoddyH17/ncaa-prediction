"""
FINAL HONEST SUBMISSION (sports data only, LOSO-tuned, no test-set selection).

Per-gender approach (chosen by LOSO Brier ONLY — not by 2026 actual):

  Men's:   triple blend
              0.6 * top3_LR + 0.2 * MultiFeatureLogistic + 0.2 * XGB_harry
           No isotonic (in-sample artifact).
           No market data.

  Women's: top3 LR alone (LOSO best at 0.1395)
           No isotonic. No market data.

Both sets of blend weights and the per-gender choice are determined by LOSO
out-of-fold Brier on 2014-2025 (excl. 2020). The 2026 actual results are NOT
used to tune anything.

This is the number we will report in the paper as the Kaggle Brier achievable
with pure sports data, fully out-of-sample.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import make_build_features_fn, _parse_seed_num
from src.models import MultiFeatureLogistic
from scripts.build_womens_model import (
    load_womens_data, build_womens_features, WomensLogistic,
)
from scripts.generate_kaggle_submission import build_submission_features
from scripts.run_harry_xgb import (
    build_matchup_features, build_2026_pair_features,
    train_xgb_loto, train_xgb_final, HPARAMS_MEN, HPARAMS_WOM,
)
from scripts.run_top3 import (
    build_combined_features, build_combined_features_2026,
    fit_lr, predict_lr,
)
from src.harry_rating import build_harry_features


# Hard-coded LOSO-best blend weights — set BEFORE seeing 2026 results.
W_MEN = (0.60, 0.20, 0.20)   # (top3, MultiFeat, XGB+harry)
W_WOM = (1.00, 0.00, 0.00)   # top3 LR only


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]

    print("Loading data...")
    data_m = load_all_mens_data()
    data_w = load_womens_data()

    # ===== Build all three model families =====
    print("\n[1/3] top3 LR with extended features (per gender)")
    X_top3, y_top3, is_w_top3 = build_combined_features(data_m, data_w, seasons)
    pipe_m_top3, cols_m_top3 = fit_lr(X_top3[is_w_top3 == 0], y_top3[is_w_top3 == 0], C=0.5)
    pipe_w_top3, cols_w_top3 = fit_lr(X_top3[is_w_top3 == 1], y_top3[is_w_top3 == 1], C=0.1)

    print("\n[2/3] MultiFeatureLogistic")
    build_fn_m = make_build_features_fn(data_m)
    X_mf_m, y_mf_m = build_fn_m(seasons)
    final_mf_m = MultiFeatureLogistic(C=0.5).fit(X_mf_m, y_mf_m)

    X_mf_w, y_mf_w = build_womens_features(data_w, seasons)
    final_mf_w = WomensLogistic(C=0.5).fit(X_mf_w, y_mf_w)

    print("\n[3/3] XGB+harry")
    hr_m = build_harry_features(data_m, seasons + [2026], is_womens=False)
    hr_w = build_harry_features(data_w, seasons + [2026], is_womens=True)
    X_xgb_m, y_xgb_m = build_matchup_features(data_m, seasons, is_womens=False, hr=hr_m)
    X_xgb_w, y_xgb_w = build_matchup_features(data_w, seasons, is_womens=True, hr=hr_w)
    final_xgb_m = train_xgb_final(X_xgb_m, y_xgb_m, HPARAMS_MEN)
    final_xgb_w = train_xgb_final(X_xgb_w, y_xgb_w, HPARAMS_WOM)

    # ===== Generate 2026 predictions from each model =====
    print("\nGenerating 2026 predictions per model...")
    sub_path = str(DATA_DIR / "SampleSubmissionStage2.csv")
    _, X_2026_mf_m, _ = build_submission_features(data_m, 2026, sub_path)
    p_2026_mf_m = final_mf_m.predict_proba(X_2026_mf_m)[:, 1]

    # Women's MultiFeat 2026 features
    from src.pipeline import build_efficiency_for_season, build_four_factors_for_season, build_momentum_for_season
    seeds_w = data_w["seeds"]
    s2026_w = seeds_w[seeds_w["Season"] == 2026].copy()
    s2026_w["SeedNum"] = s2026_w["Seed"].apply(_parse_seed_num)
    tids_w = sorted(int(t) for t in s2026_w["TeamID"])
    seed_map_w = dict(zip(s2026_w["TeamID"], s2026_w["SeedNum"]))
    eff_w = build_efficiency_for_season(data_w, 2026)
    ff_w = build_four_factors_for_season(data_w, 2026)
    mom_w = build_momentum_for_season(data_w, 2026, tids_w)
    mom_map_w = {row["TeamID"]: row for _, row in mom_w.iterrows()}
    bart_w_path = DATA_DIR / "external" / "barttorvik_w_2026.csv"
    bart_w = pd.read_csv(bart_w_path).set_index("TeamID") if bart_w_path.exists() else None

    feats_mf_w = []
    for i, ta in enumerate(tids_w):
        for tb in tids_w[i + 1:]:
            feat = {"Season": 2026, "TeamA": ta, "TeamB": tb,
                    "seed_diff": seed_map_w.get(ta, 16) - seed_map_w.get(tb, 16),
                    "seed_A": seed_map_w.get(ta, 16), "seed_B": seed_map_w.get(tb, 16)}
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
            ma = mom_map_w.get(ta, {}); mb = mom_map_w.get(tb, {})
            feat["momentum_winpct_diff"] = ma.get("momentum_win_pct", 0.5) - mb.get("momentum_win_pct", 0.5)
            feat["momentum_margin_diff"] = ma.get("momentum_avg_margin", 0.0) - mb.get("momentum_avg_margin", 0.0)
            feats_mf_w.append(feat)
    X_2026_mf_w = pd.DataFrame(feats_mf_w)
    p_2026_mf_w = final_mf_w.predict_proba(X_2026_mf_w)[:, 1]

    # XGB+harry 2026
    X_2026_xgb_m = build_2026_pair_features(data_m, hr_m, womens=False)
    X_2026_xgb_w = build_2026_pair_features(data_w, hr_w, womens=True)
    p_2026_xgb_m = np.clip(final_xgb_m.predict(
        X_2026_xgb_m[["seed_diff", "harry_diff", "opp_qlty_won_diff"]]
    ), 0.005, 0.995)
    p_2026_xgb_w = np.clip(final_xgb_w.predict(
        X_2026_xgb_w[["seed_diff", "harry_diff", "opp_qlty_won_diff"]]
    ), 0.005, 0.995)

    # Top3 2026
    X_2026_top3, _, is_w_2026 = build_combined_features_2026(data_m, data_w)
    p_2026_top3_m = predict_lr(pipe_m_top3, X_2026_top3[is_w_2026 == 0], cols_m_top3)
    p_2026_top3_w = predict_lr(pipe_w_top3, X_2026_top3[is_w_2026 == 1], cols_w_top3)

    # ===== Build per-pair lookups =====
    def lk(X, p):
        d = {}
        for i, r in X.reset_index(drop=True).iterrows():
            d[(int(r["TeamA"]), int(r["TeamB"]))] = float(p[i])
        return d

    lk_top3_m = lk(X_2026_top3[is_w_2026 == 0], p_2026_top3_m)
    lk_top3_w = lk(X_2026_top3[is_w_2026 == 1], p_2026_top3_w)
    lk_mf_m = lk(X_2026_mf_m, p_2026_mf_m)
    lk_mf_w = lk(X_2026_mf_w, p_2026_mf_w)
    lk_xgb_m = lk(X_2026_xgb_m, p_2026_xgb_m)
    lk_xgb_w = lk(X_2026_xgb_w, p_2026_xgb_w)

    # ===== Apply LOSO-tuned blend weights (set BEFORE seeing 2026) =====
    print(f"\nApplying LOSO-tuned blend (set before any 2026 evaluation):")
    print(f"  Men's:   w = {W_MEN}  (top3, MultiFeat, XGB+harry)")
    print(f"  Women's: w = {W_WOM}  (top3, MultiFeat, XGB+harry)")

    final_lk_m = {}
    keys_m = set(lk_top3_m) | set(lk_mf_m) | set(lk_xgb_m)
    for k in keys_m:
        p = (W_MEN[0] * lk_top3_m.get(k, 0.5) +
             W_MEN[1] * lk_mf_m.get(k, 0.5) +
             W_MEN[2] * lk_xgb_m.get(k, 0.5))
        final_lk_m[k] = float(np.clip(p, 0.005, 0.995))

    final_lk_w = {}
    keys_w = set(lk_top3_w) | set(lk_mf_w) | set(lk_xgb_w)
    for k in keys_w:
        p = (W_WOM[0] * lk_top3_w.get(k, 0.5) +
             W_WOM[1] * lk_mf_w.get(k, 0.5) +
             W_WOM[2] * lk_xgb_w.get(k, 0.5))
        final_lk_w[k] = float(np.clip(p, 0.005, 0.995))

    # ===== Build submission =====
    sub = pd.read_csv("output/submission_stage2.csv")
    sub[["s_str", "ta_str", "tb_str"]] = sub["ID"].str.split("_", expand=True)
    sub["TeamA"] = sub["ta_str"].astype(int)
    sub["TeamB"] = sub["tb_str"].astype(int)
    new_map = {**final_lk_m, **final_lk_w}
    sub["Pred"] = sub.apply(
        lambda r: new_map.get((r["TeamA"], r["TeamB"]), float(r["Pred"])),
        axis=1
    ).clip(0.005, 0.995)
    sub[["ID", "Pred"]].to_csv("output/submission_stage2_HONEST.csv", index=False)
    print(f"\nSaved output/submission_stage2_HONEST.csv ({len(sub)} rows)")

    # ===== Evaluate on 2026 actual =====
    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")

    def br(lk, actual):
        yt, yp = [], []
        for _, g in actual.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            key = (w, l) if w < l else (l, w)
            yt.append(1 if w < l else 0)
            yp.append(lk.get(key, 0.5))
        return brier_score_loss(yt, yp), len(yt)

    bs_m, n_m = br(final_lk_m, actual_m)
    bs_w, n_w = br(final_lk_w, actual_w)
    bs_c = (bs_m * n_m + bs_w * n_w) / (n_m + n_w)

    print(f"\n{'='*70}")
    print(f"  STRICTLY HONEST 2026 BRIER (sports data only, LOSO-tuned)")
    print(f"{'='*70}")
    print(f"  Men's:    {bs_m:.4f} ({n_m} games)")
    print(f"  Women's:  {bs_w:.4f} ({n_w} games)")
    print(f"  Combined: {bs_c:.4f}")
    print(f"\n  Comparison points:")
    print(f"    Initial baseline:               0.1264")
    print(f"    Our HONEST result:              {bs_c:.4f}")
    print(f"    Kaggle 3rd (with markets):      0.1160")
    print(f"    Kaggle 1st (with injury data):  0.1097")

    pd.DataFrame([{"version": "honest_sports_only", "men": bs_m, "women": bs_w,
                   "combined": bs_c,
                   "w_men_top3": W_MEN[0], "w_men_mf": W_MEN[1], "w_men_xgb": W_MEN[2],
                   "w_wom_top3": W_WOM[0], "w_wom_mf": W_WOM[1], "w_wom_xgb": W_WOM[2]}]).to_csv(
        "output/final_honest_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
