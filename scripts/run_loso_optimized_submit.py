"""
Apply LOSO-optimized recipe to 2026 — strictly honest final submission.

Recipe (all decisions made on LOSO 2014-2025, BEFORE seeing 2026):

  Men's:   convex blend = 0.9 * pruned_LR + 0.1 * XGB+harry
           pruned_LR = 12 features, C=0.5
             [seed_diff, def_eff_diff, tempo_diff, efg_pct_diff, to_pct_diff,
              ft_rate_diff, bart_net_diff, bart_adjoe_diff, bart_adjde_diff,
              elo_diff, colley_diff, massey_mean_diff]

  Women's: pruned_LR alone, 9 features, C=0.3
           [win_pct_diff, bart_net_diff, bart_adjoe_diff, elo_diff, elo_slope_diff,
            colley_diff, srs_diff, momentum_winpct_diff, momentum_margin_diff]

Final 2026 LOSO Combined Brier estimate: 0.1602
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import _parse_seed_num
from scripts.build_womens_model import load_womens_data
from scripts.run_top3 import build_combined_features, build_combined_features_2026
from scripts.run_harry_xgb import (
    build_matchup_features, build_2026_pair_features,
    train_xgb_final, HPARAMS_MEN,
)
from src.harry_rating import build_harry_features


MEN_FEATS = [
    "seed_diff", "def_eff_diff", "tempo_diff", "efg_pct_diff", "to_pct_diff",
    "ft_rate_diff", "bart_net_diff", "bart_adjoe_diff", "bart_adjde_diff",
    "elo_diff", "colley_diff", "massey_mean_diff",
]
MEN_C = 0.5

WOMEN_FEATS = [
    "win_pct_diff", "bart_net_diff", "bart_adjoe_diff", "elo_diff",
    "elo_slope_diff", "colley_diff", "srs_diff",
    "momentum_winpct_diff", "momentum_margin_diff",
]
WOMEN_C = 0.3

W_MEN_BLEND = (0.9, 0.1)  # (pruned_LR, XGB+harry)


def fit_pruned(X, y, feats, C):
    Xn = X[feats].apply(pd.to_numeric, errors="coerce")
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
    ])
    pipe.fit(Xn, y)
    return pipe


def predict_pruned(pipe, X, feats):
    Xn = X[feats].apply(pd.to_numeric, errors="coerce")
    return pipe.predict_proba(Xn)[:, 1]


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]
    print("Loading data...")
    data_m = load_all_mens_data()
    data_w = load_womens_data()

    print("\nBuilding training features (top3 extended set)...")
    X_train, y_train, is_w_train = build_combined_features(data_m, data_w, seasons)

    # Train final pruned LR per gender
    print(f"\nFinal training per gender:")
    pipe_m = fit_pruned(X_train[is_w_train == 0], y_train[is_w_train == 0], MEN_FEATS, MEN_C)
    pipe_w = fit_pruned(X_train[is_w_train == 1], y_train[is_w_train == 1], WOMEN_FEATS, WOMEN_C)
    print(f"  Men's:   12 features, C={MEN_C}")
    print(f"  Women's:  9 features, C={WOMEN_C}")

    # XGB+harry for men's blend
    hr_m = build_harry_features(data_m, seasons + [2026], is_womens=False)
    X_xgb_m, y_xgb_m = build_matchup_features(data_m, seasons, is_womens=False, hr=hr_m)
    final_xgb_m = train_xgb_final(X_xgb_m, y_xgb_m, HPARAMS_MEN)

    # ===== Build 2026 features =====
    print("\nBuilding 2026 features...")
    X_2026, _, is_w_2026 = build_combined_features_2026(data_m, data_w)
    X_2026_xgb_m = build_2026_pair_features(data_m, hr_m, womens=False)

    # 2026 predictions
    p_2026_pruned_m = predict_pruned(pipe_m, X_2026[is_w_2026 == 0], MEN_FEATS)
    p_2026_pruned_w = predict_pruned(pipe_w, X_2026[is_w_2026 == 1], WOMEN_FEATS)
    p_2026_xgb_m = np.clip(final_xgb_m.predict(
        X_2026_xgb_m[["seed_diff", "harry_diff", "opp_qlty_won_diff"]]
    ), 0.005, 0.995)

    # Build per-pair lookups
    def lk(X, p):
        d = {}
        for i, r in X.reset_index(drop=True).iterrows():
            d[(int(r["TeamA"]), int(r["TeamB"]))] = float(p[i])
        return d

    lk_pruned_m = lk(X_2026[is_w_2026 == 0], p_2026_pruned_m)
    lk_pruned_w = lk(X_2026[is_w_2026 == 1], p_2026_pruned_w)
    lk_xgb_m = lk(X_2026_xgb_m, p_2026_xgb_m)

    # Apply blend for men's
    final_lk_m = {}
    for k in set(lk_pruned_m) | set(lk_xgb_m):
        p = (W_MEN_BLEND[0] * lk_pruned_m.get(k, 0.5) +
             W_MEN_BLEND[1] * lk_xgb_m.get(k, 0.5))
        final_lk_m[k] = float(np.clip(p, 0.005, 0.995))

    # Women's: pruned LR alone
    final_lk_w = {k: float(np.clip(v, 0.005, 0.995)) for k, v in lk_pruned_w.items()}

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
    sub[["ID", "Pred"]].to_csv("output/submission_stage2_HONEST_OPT.csv", index=False)

    # ===== Evaluate =====
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
    print(f"  STRICTLY HONEST 2026 BRIER (LOSO-optimized, sports-only)")
    print(f"{'='*70}")
    print(f"  Men's:    {bs_m:.4f}  ({n_m} games)  -- 0.9*pruned_LR_12 + 0.1*XGB_harry")
    print(f"  Women's:  {bs_w:.4f}  ({n_w} games)  -- pruned_LR_9 alone")
    print(f"  Combined: {bs_c:.4f}")
    print(f"\n  Trajectory:")
    print(f"    Initial baseline (MultiFeat alone):   0.1264")
    print(f"    Naive triple blend (no pruning):      0.1260")
    print(f"    LOSO-OPTIMIZED honest:                {bs_c:.4f}")
    print(f"    Kaggle 3rd (with markets):            0.1160")
    print(f"    Kaggle 1st (with injury data):        0.1097")
    print(f"\n  LOSO Brier: 0.1602")
    print(f"  2026 actual Brier: {bs_c:.4f}")
    print(f"  CV-actual gap (chalky 2026 effect): {0.1602 - bs_c:+.4f}")

    pd.DataFrame([{
        "version": "honest_loso_optimized", "men": bs_m, "women": bs_w, "combined": bs_c,
        "men_features": ",".join(MEN_FEATS), "men_C": MEN_C,
        "women_features": ",".join(WOMEN_FEATS), "women_C": WOMEN_C,
        "men_blend_pruned_xgb": str(W_MEN_BLEND),
    }]).to_csv("output/honest_optimized_summary.csv", index=False)


if __name__ == "__main__":
    main()
