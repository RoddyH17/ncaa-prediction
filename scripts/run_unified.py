"""
Unified methodology for men's and women's tournaments — single recipe applied
to both genders.

Tests three unified strategies on LOSO 2014-2025 and picks the best:

  Strategy A: Combined M+W training. One LR on pooled training set with
              is_womens binary feature. Same predictor for both tournaments.

  Strategy B: Same single model per gender. We test:
              - top3_LR_only (with same C)
              - multifeat_only (with same C)
              - xgb_harry_only (same hyperparams)

  Strategy C: Same blend per gender. e.g., 0.7 * top3_LR + 0.3 * XGB+harry
              applied identically to both.

All decisions made on LOSO Brier. Then evaluate on 2026.
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
from src.pipeline import make_build_features_fn, _parse_seed_num
from src.models import MultiFeatureLogistic
from scripts.build_womens_model import (
    load_womens_data, build_womens_features, WomensLogistic,
)
from scripts.run_top3 import (
    build_combined_features, build_combined_features_2026, FEATURE_COLS as TOP3_FEATURES,
)
from scripts.run_harry_xgb import (
    build_matchup_features, build_2026_pair_features,
    train_xgb_loto, train_xgb_final, HPARAMS_MEN, HPARAMS_WOM,
)
from scripts.generate_kaggle_submission import build_submission_features
from src.harry_rating import build_harry_features


def fit_lr_features(X, y, feats, C=1.0):
    Xn = X[feats].apply(pd.to_numeric, errors="coerce")
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
    ])
    pipe.fit(Xn, y)
    return pipe


def predict_lr(pipe, X, feats):
    return pipe.predict_proba(X[feats].apply(pd.to_numeric, errors="coerce"))[:, 1]


def loso_brier(X, y, season_arr, feats, C, gender_mask=None):
    p_oof = np.zeros(len(X))
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        if te.sum() == 0: continue
        pipe = fit_lr_features(X.loc[tr], y[tr], feats, C=C)
        p_oof[te] = predict_lr(pipe, X.loc[te], feats)
    if gender_mask is not None:
        return p_oof, brier_score_loss(y[gender_mask], p_oof[gender_mask])
    return p_oof, brier_score_loss(y, p_oof)


def combined_brier(p_oof, y, is_w, n_m, n_w):
    bs_m = brier_score_loss(y[is_w == 0], p_oof[is_w == 0])
    bs_w = brier_score_loss(y[is_w == 1], p_oof[is_w == 1])
    return bs_m, bs_w, (bs_m * n_m + bs_w * n_w) / (n_m + n_w)


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]
    print("Loading data...")
    data_m = load_all_mens_data()
    data_w = load_womens_data()

    print("\nBuilding combined feature matrix...")
    X, y, is_w = build_combined_features(data_m, data_w, seasons)
    season_arr = X["Season"].values
    n_m = (is_w == 0).sum(); n_w = (is_w == 1).sum()
    print(f"  Men's: {n_m}, Women's: {n_w}")

    avail = [c for c in TOP3_FEATURES if c in X.columns]
    print(f"  Features available: {len(avail)}")

    rows = []

    # ===========================================================
    # Strategy A: Combined M+W single LR (with is_womens flag)
    # ===========================================================
    print(f"\n{'='*70}\n  Strategy A: Combined M+W training (one LR)\n{'='*70}")
    feats_A = avail + ["is_womens"]
    for C in [0.1, 0.3, 0.5, 1.0, 3.0]:
        p_oof, _ = loso_brier(X, y, season_arr, feats_A, C=C)
        bs_m, bs_w, bs_c = combined_brier(p_oof, y, is_w, n_m, n_w)
        print(f"  C={C}: men={bs_m:.4f}  women={bs_w:.4f}  combined={bs_c:.4f}")
        rows.append({"strategy": "A_combined_LR", "C": C,
                     "men": bs_m, "women": bs_w, "combined": bs_c})

    # ===========================================================
    # Strategy B: Same single model per gender (per-gender training, same hparams)
    # ===========================================================
    print(f"\n{'='*70}\n  Strategy B: Same single model, per-gender training\n{'='*70}")
    # B1: top3 LR alone with same C for both
    print("\n  B1: top3 LR alone, same C for both genders")
    for C in [0.1, 0.3, 0.5, 1.0]:
        p_oof = np.zeros(len(X))
        for is_g in [0, 1]:
            mask = is_w == is_g
            for s in np.unique(season_arr[mask]):
                tr = (season_arr != s) & mask
                te = (season_arr == s) & mask
                if te.sum() == 0: continue
                feats_B = [c for c in avail if not (is_g == 1 and c.startswith("massey_"))]
                pipe = fit_lr_features(X.loc[tr], y[tr], feats_B, C=C)
                p_oof[te] = predict_lr(pipe, X.loc[te], feats_B)
        bs_m, bs_w, bs_c = combined_brier(p_oof, y, is_w, n_m, n_w)
        print(f"  C={C}: men={bs_m:.4f}  women={bs_w:.4f}  combined={bs_c:.4f}")
        rows.append({"strategy": "B1_top3_per_gender", "C": C,
                     "men": bs_m, "women": bs_w, "combined": bs_c})

    # ===========================================================
    # Strategy C: Same blend per gender — top3_LR + XGB+harry
    # ===========================================================
    print(f"\n{'='*70}\n  Strategy C: Same blend (top3 + XGB+harry)\n{'='*70}")

    # First, build OOF for both top3 and XGB+harry per gender (using each's best C)
    # Top3 LR OOF
    p_oof_top3 = np.zeros(len(X))
    for is_g, C_g in [(0, 0.5), (1, 0.1)]:
        mask = is_w == is_g
        for s in np.unique(season_arr[mask]):
            tr = (season_arr != s) & mask
            te = (season_arr == s) & mask
            if te.sum() == 0: continue
            feats_B = [c for c in avail if not (is_g == 1 and c.startswith("massey_"))]
            pipe = fit_lr_features(X.loc[tr], y[tr], feats_B, C=C_g)
            p_oof_top3[te] = predict_lr(pipe, X.loc[te], feats_B)

    # XGB+harry OOF
    print("  Building XGB+harry OOF...")
    hr_m = build_harry_features(data_m, seasons + [2026], is_womens=False)
    hr_w = build_harry_features(data_w, seasons + [2026], is_womens=True)
    X_xgb_m, y_xgb_m = build_matchup_features(data_m, seasons, is_womens=False, hr=hr_m)
    X_xgb_w, y_xgb_w = build_matchup_features(data_w, seasons, is_womens=True, hr=hr_w)
    p_oof_xgb_m, _ = train_xgb_loto(X_xgb_m, y_xgb_m, HPARAMS_MEN)
    p_oof_xgb_w, _ = train_xgb_loto(X_xgb_w, y_xgb_w, HPARAMS_WOM)

    # Align XGB OOF onto same row order as X (top3)
    xgb_lookup = {}
    for i, r in X_xgb_m.reset_index(drop=True).iterrows():
        xgb_lookup[(int(r["Season"]), int(r["TeamA"]), int(r["TeamB"]))] = float(p_oof_xgb_m[i])
    for i, r in X_xgb_w.reset_index(drop=True).iterrows():
        xgb_lookup[(int(r["Season"]), int(r["TeamA"]), int(r["TeamB"]))] = float(p_oof_xgb_w[i])
    p_oof_xgb_aligned = np.array([
        xgb_lookup.get((int(r["Season"]), int(r["TeamA"]), int(r["TeamB"])), 0.5)
        for _, r in X.iterrows()
    ])

    print("\n  Same convex blend (w_top3, 1-w_top3) for both genders:")
    for w in [0.0, 0.2, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        p_oof = w * p_oof_top3 + (1 - w) * p_oof_xgb_aligned
        p_oof = np.clip(p_oof, 0.005, 0.995)
        bs_m, bs_w, bs_c = combined_brier(p_oof, y, is_w, n_m, n_w)
        print(f"  w_top3={w}: men={bs_m:.4f}  women={bs_w:.4f}  combined={bs_c:.4f}")
        rows.append({"strategy": "C_same_blend", "C": w,
                     "men": bs_m, "women": bs_w, "combined": bs_c})

    # ===========================================================
    # Strategy D: Single best LOSO solution = same recipe both genders
    # We'll just take the best from above as "the unified recipe"
    # ===========================================================
    df = pd.DataFrame(rows).sort_values("combined")
    df.to_csv("output/unified_loso.csv", index=False)
    print(f"\n{'='*70}\n  Top 10 unified configs by LOSO Combined Brier\n{'='*70}")
    print(df.head(10).to_string(index=False))

    best = df.iloc[0]
    print(f"\n  -> Best unified: {best['strategy']} with C={best['C']}")
    print(f"     LOSO Brier = {best['combined']:.4f}")

    # ===========================================================
    # Apply best to 2026
    # ===========================================================
    print(f"\n{'='*70}\n  Apply best unified recipe to 2026\n{'='*70}")
    X_2026, _, is_w_2026 = build_combined_features_2026(data_m, data_w)

    if best["strategy"] == "A_combined_LR":
        # Train on combined data
        feats_A = avail + ["is_womens"]
        pipe = fit_lr_features(X, y, feats_A, C=best["C"])
        p_2026 = predict_lr(pipe, X_2026, feats_A)
    elif best["strategy"] == "B1_top3_per_gender":
        # Per-gender LR with same C
        p_2026 = np.zeros(len(X_2026))
        for is_g in [0, 1]:
            mask_train = is_w == is_g
            mask_test = is_w_2026 == is_g
            feats_B = [c for c in avail if not (is_g == 1 and c.startswith("massey_"))]
            pipe = fit_lr_features(X.loc[mask_train], y[mask_train], feats_B, C=best["C"])
            p_2026[mask_test] = predict_lr(pipe, X_2026.loc[mask_test], feats_B)
    else:
        # Same blend
        # Top3 LR per gender 2026
        feats_m = [c for c in avail]
        feats_w = [c for c in avail if not c.startswith("massey_")]
        pipe_m = fit_lr_features(X[is_w == 0], y[is_w == 0], feats_m, C=0.5)
        pipe_w = fit_lr_features(X[is_w == 1], y[is_w == 1], feats_w, C=0.1)
        p_2026_top3 = np.zeros(len(X_2026))
        p_2026_top3[is_w_2026 == 0] = predict_lr(pipe_m, X_2026[is_w_2026 == 0], feats_m)
        p_2026_top3[is_w_2026 == 1] = predict_lr(pipe_w, X_2026[is_w_2026 == 1], feats_w)
        # XGB+harry per gender 2026
        final_xgb_m = train_xgb_final(X_xgb_m, y_xgb_m, HPARAMS_MEN)
        final_xgb_w = train_xgb_final(X_xgb_w, y_xgb_w, HPARAMS_WOM)
        X_2026_xgb_m = build_2026_pair_features(data_m, hr_m, womens=False)
        X_2026_xgb_w = build_2026_pair_features(data_w, hr_w, womens=True)
        p_2026_xgb_m = np.clip(final_xgb_m.predict(
            X_2026_xgb_m[["seed_diff", "harry_diff", "opp_qlty_won_diff"]]
        ), 0.005, 0.995)
        p_2026_xgb_w = np.clip(final_xgb_w.predict(
            X_2026_xgb_w[["seed_diff", "harry_diff", "opp_qlty_won_diff"]]
        ), 0.005, 0.995)
        # Build lookup for XGB
        xgb_2026_lookup = {}
        for i, r in X_2026_xgb_m.reset_index(drop=True).iterrows():
            xgb_2026_lookup[(int(r["TeamA"]), int(r["TeamB"]))] = float(p_2026_xgb_m[i])
        for i, r in X_2026_xgb_w.reset_index(drop=True).iterrows():
            xgb_2026_lookup[(int(r["TeamA"]), int(r["TeamB"]))] = float(p_2026_xgb_w[i])
        p_2026_xgb = np.array([
            xgb_2026_lookup.get((int(r["TeamA"]), int(r["TeamB"])), 0.5)
            for _, r in X_2026.iterrows()
        ])
        w = best["C"]
        p_2026 = w * p_2026_top3 + (1 - w) * p_2026_xgb
        p_2026 = np.clip(p_2026, 0.005, 0.995)

    # Build per-pair lookup
    pair_lookup = {}
    for i, r in X_2026.reset_index(drop=True).iterrows():
        pair_lookup[(int(r["TeamA"]), int(r["TeamB"]))] = float(p_2026[i])

    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")

    def br(actual):
        yt, yp = [], []
        for _, g in actual.iterrows():
            w_, l = int(g["WTeamID"]), int(g["LTeamID"])
            key = (w_, l) if w_ < l else (l, w_)
            yt.append(1 if w_ < l else 0)
            yp.append(pair_lookup.get(key, 0.5))
        return brier_score_loss(yt, yp), len(yt)

    bs_m_2026, n_m_a = br(actual_m)
    bs_w_2026, n_w_a = br(actual_w)
    bs_c_2026 = (bs_m_2026 * n_m_a + bs_w_2026 * n_w_a) / (n_m_a + n_w_a)

    print(f"\n  Best unified recipe: {best['strategy']} with C={best['C']}")
    print(f"  LOSO Brier:    {best['combined']:.4f}")
    print(f"  2026 Brier:    {bs_c_2026:.4f}")
    print(f"    Men's:       {bs_m_2026:.4f}")
    print(f"    Women's:     {bs_w_2026:.4f}")

    # Save submission
    sub = pd.read_csv("output/submission_stage2.csv")
    sub[["s_str", "ta_str", "tb_str"]] = sub["ID"].str.split("_", expand=True)
    sub["TeamA"] = sub["ta_str"].astype(int); sub["TeamB"] = sub["tb_str"].astype(int)
    sub["Pred"] = sub.apply(
        lambda r: pair_lookup.get((r["TeamA"], r["TeamB"]), float(r["Pred"])),
        axis=1
    ).clip(0.005, 0.995)
    sub[["ID", "Pred"]].to_csv("output/submission_stage2_UNIFIED.csv", index=False)
    print(f"\n  Saved output/submission_stage2_UNIFIED.csv")

    pd.DataFrame([{"strategy": best["strategy"], "C": best["C"],
                   "loso_combined": best["combined"],
                   "actual_men": bs_m_2026, "actual_women": bs_w_2026,
                   "actual_combined": bs_c_2026}]).to_csv(
        "output/unified_summary.csv", index=False
    )


if __name__ == "__main__":
    main()
