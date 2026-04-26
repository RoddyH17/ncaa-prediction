"""
Triple blend: combine three orthogonal model families per gender, with weights
tuned on LOSO out-of-fold predictions.

Models:
  (A) Top3 LR     — LR on extended feature set (Elo, Colley, SRS, Massey
                    last-2-week composite, harry, momentum, four factors,
                    Barttorvik, ...).  Per-gender training.
  (B) XGB+harry   — XGBoost regression on 3 features (seed_diff, harry_diff,
                    opp_qlty_won_diff).  Captures non-linearity.
  (C) MultiFeat   — Original Multi-Feature Logistic on 20 standard features.

Pipeline:
  1. LOSO out-of-fold predictions for each (model, gender)
  2. Linear stack via LR with non-negative weights on logits, fit on OOF
  3. Train final models on full 2014-2025; apply weights to 2026 predictions
  4. Optional: in-sample isotonic on stacked OOF; clip [0.005, 0.995]

Outputs:
  output/triple_blend_loso.csv
  output/triple_blend_summary.csv
  output/submission_stage2_triple.csv
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.isotonic import IsotonicRegression
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
    fit_lr, predict_lr, FEATURE_COLS,
)
from src.harry_rating import build_harry_features


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]
    print("Loading data...")
    data_m = load_all_mens_data()
    data_w = load_womens_data()

    # ===== Top3 LR LOSO + 2026 =====
    print("\n[1/3] Top3 LR (extended features) per gender")
    X_top3, y_top3, is_w_top3 = build_combined_features(data_m, data_w, seasons)
    season_top3 = X_top3["Season"].values

    p_oof_top3 = np.zeros(len(X_top3))
    for is_w_val, C in [(0, 0.5), (1, 0.1)]:
        idx = np.where(is_w_top3 == is_w_val)[0]
        for s in np.unique(season_top3[idx]):
            tr = (season_top3 != s) & (is_w_top3 == is_w_val)
            te = (season_top3 == s) & (is_w_top3 == is_w_val)
            if te.sum() == 0: continue
            pipe, cols = fit_lr(X_top3[tr], y_top3[tr], C=C)
            p_oof_top3[te] = predict_lr(pipe, X_top3.iloc[np.where(te)[0]], cols)
    bs_m_top3 = brier_score_loss(y_top3[is_w_top3 == 0], p_oof_top3[is_w_top3 == 0])
    bs_w_top3 = brier_score_loss(y_top3[is_w_top3 == 1], p_oof_top3[is_w_top3 == 1])
    print(f"  Top3 LOSO: men={bs_m_top3:.4f}  women={bs_w_top3:.4f}")

    # Final top3 models
    pipe_m_top3, cols_m_top3 = fit_lr(X_top3[is_w_top3 == 0], y_top3[is_w_top3 == 0], C=0.5)
    pipe_w_top3, cols_w_top3 = fit_lr(X_top3[is_w_top3 == 1], y_top3[is_w_top3 == 1], C=0.1)

    # ===== Multi-Feature Logistic LOSO + 2026 =====
    print("\n[2/3] Multi-Feature Logistic per gender")
    build_fn_m = make_build_features_fn(data_m)
    X_mf_m, y_mf_m = build_fn_m(seasons)
    X_mf_w, y_mf_w = build_womens_features(data_w, seasons)
    season_m_mf = X_mf_m["Season"].values
    season_w_mf = X_mf_w["Season"].values

    p_oof_mf_m = np.zeros(len(X_mf_m))
    for s in np.unique(season_m_mf):
        tr = season_m_mf != s; te = season_m_mf == s
        m = MultiFeatureLogistic(C=0.5).fit(X_mf_m.loc[tr], y_mf_m[tr])
        p_oof_mf_m[te] = m.predict_proba(X_mf_m.loc[te])[:, 1]
    p_oof_mf_w = np.zeros(len(X_mf_w))
    for s in np.unique(season_w_mf):
        tr = season_w_mf != s; te = season_w_mf == s
        m = WomensLogistic(C=0.5).fit(X_mf_w.loc[tr], y_mf_w[tr])
        p_oof_mf_w[te] = m.predict_proba(X_mf_w.loc[te])[:, 1]
    bs_m_mf = brier_score_loss(y_mf_m, p_oof_mf_m)
    bs_w_mf = brier_score_loss(y_mf_w, p_oof_mf_w)
    print(f"  MultiFeat LOSO: men={bs_m_mf:.4f}  women={bs_w_mf:.4f}")
    final_mf_m = MultiFeatureLogistic(C=0.5).fit(X_mf_m, y_mf_m)
    final_mf_w = WomensLogistic(C=0.5).fit(X_mf_w, y_mf_w)

    # ===== XGB+harry LOSO + 2026 =====
    print("\n[3/3] XGB+harry per gender")
    hr_m = build_harry_features(data_m, seasons + [2026], is_womens=False)
    hr_w = build_harry_features(data_w, seasons + [2026], is_womens=True)
    X_xgb_m, y_xgb_m = build_matchup_features(data_m, seasons, is_womens=False, hr=hr_m)
    X_xgb_w, y_xgb_w = build_matchup_features(data_w, seasons, is_womens=True, hr=hr_w)
    p_oof_xgb_m, _ = train_xgb_loto(X_xgb_m, y_xgb_m, HPARAMS_MEN)
    p_oof_xgb_w, _ = train_xgb_loto(X_xgb_w, y_xgb_w, HPARAMS_WOM)
    bs_m_xgb = brier_score_loss(y_xgb_m, p_oof_xgb_m)
    bs_w_xgb = brier_score_loss(y_xgb_w, p_oof_xgb_w)
    print(f"  XGB+harry LOSO: men={bs_m_xgb:.4f}  women={bs_w_xgb:.4f}")
    final_xgb_m = train_xgb_final(X_xgb_m, y_xgb_m, HPARAMS_MEN)
    final_xgb_w = train_xgb_final(X_xgb_w, y_xgb_w, HPARAMS_WOM)

    # ===== Align all OOF onto same row order via (Season, TeamA, TeamB) =====
    print("\nAligning OOF predictions across models...")
    def index_by_key(X, p, season_col="Season"):
        d = {}
        for i, r in X.reset_index(drop=True).iterrows():
            d[(int(r[season_col]), int(r["TeamA"]), int(r["TeamB"]))] = float(p[i])
        return d

    top3_m_oof = index_by_key(X_top3[is_w_top3 == 0], p_oof_top3[is_w_top3 == 0])
    top3_w_oof = index_by_key(X_top3[is_w_top3 == 1], p_oof_top3[is_w_top3 == 1])
    mf_m_oof = index_by_key(X_mf_m, p_oof_mf_m)
    mf_w_oof = index_by_key(X_mf_w, p_oof_mf_w)
    xgb_m_oof = index_by_key(X_xgb_m, p_oof_xgb_m)
    xgb_w_oof = index_by_key(X_xgb_w, p_oof_xgb_w)

    def align(X_ref, lookups, y_ref):
        rows_p = []
        rows_y = []
        for i, r in X_ref.reset_index(drop=True).iterrows():
            key = (int(r["Season"]), int(r["TeamA"]), int(r["TeamB"]))
            ps = [lk.get(key, 0.5) for lk in lookups]
            rows_p.append(ps); rows_y.append(y_ref[i])
        return np.array(rows_p), np.array(rows_y)

    P_m, y_m_aligned = align(X_xgb_m, [top3_m_oof, mf_m_oof, xgb_m_oof], y_xgb_m)
    P_w, y_w_aligned = align(X_xgb_w, [top3_w_oof, mf_w_oof, xgb_w_oof], y_xgb_w)

    # ===== Stack via grid search over (w_top3, w_mf, w_xgb) =====
    print("\nGrid search blending weights on LOSO OOF...")
    def best_simple_blend(P, y):
        best = (None, np.inf)
        ws = np.linspace(0, 1, 11)
        for w1 in ws:
            for w2 in ws:
                w3 = 1 - w1 - w2
                if w3 < -1e-9 or w3 > 1 + 1e-9: continue
                p = w1 * P[:, 0] + w2 * P[:, 1] + w3 * P[:, 2]
                p = np.clip(p, 0.005, 0.995)
                bs = brier_score_loss(y, p)
                if bs < best[1]:
                    best = ((w1, w2, w3), bs)
        return best

    (w_m_top3, w_m_mf, w_m_xgb), bs_blend_m = best_simple_blend(P_m, y_m_aligned)
    (w_w_top3, w_w_mf, w_w_xgb), bs_blend_w = best_simple_blend(P_w, y_w_aligned)
    print(f"  Men's:    w=(top3={w_m_top3:.2f}, mf={w_m_mf:.2f}, xgb={w_m_xgb:.2f}) "
          f"-> LOSO {bs_blend_m:.4f}")
    print(f"  Women's:  w=(top3={w_w_top3:.2f}, mf={w_w_mf:.2f}, xgb={w_w_xgb:.2f}) "
          f"-> LOSO {bs_blend_w:.4f}")

    # ===== 2026 predictions from each model =====
    print("\nGenerating 2026 predictions from each model...")
    sub_path = str(DATA_DIR / "SampleSubmissionStage2.csv")

    # MultiFeat 2026
    _, X_2026_mf_m, _ = build_submission_features(data_m, 2026, sub_path)
    p_2026_mf_m = final_mf_m.predict_proba(X_2026_mf_m)[:, 1]
    # Women's MultiFeat 2026: build features same way as build_womens_model for 2026
    from src.pipeline import build_efficiency_for_season, build_four_factors_for_season, build_momentum_for_season
    seeds_w_2026 = data_w["seeds"]
    s2026_w = seeds_w_2026[seeds_w_2026["Season"] == 2026].copy()
    s2026_w["SeedNum"] = s2026_w["Seed"].apply(_parse_seed_num)
    tids_w = sorted(int(t) for t in s2026_w["TeamID"])
    seed_map_w = dict(zip(s2026_w["TeamID"], s2026_w["SeedNum"]))
    eff_w_2026 = build_efficiency_for_season(data_w, 2026)
    ff_w_2026 = build_four_factors_for_season(data_w, 2026)
    mom_w_2026 = build_momentum_for_season(data_w, 2026, tids_w)
    mom_map_w = {row["TeamID"]: row for _, row in mom_w_2026.iterrows()}
    bart_w_path = DATA_DIR / "external" / "barttorvik_w_2026.csv"
    bart_w_2026 = pd.read_csv(bart_w_path).set_index("TeamID") if bart_w_path.exists() else None

    feats_mf_w = []
    for i, ta in enumerate(tids_w):
        for tb in tids_w[i+1:]:
            feat = {"Season": 2026, "TeamA": ta, "TeamB": tb,
                    "seed_diff": seed_map_w.get(ta, 16) - seed_map_w.get(tb, 16),
                    "seed_A": seed_map_w.get(ta, 16), "seed_B": seed_map_w.get(tb, 16)}
            for col in ["off_eff", "def_eff", "net_eff", "tempo"]:
                va = eff_w_2026.loc[ta, col] if (not eff_w_2026.empty and ta in eff_w_2026.index) else np.nan
                vb = eff_w_2026.loc[tb, col] if (not eff_w_2026.empty and tb in eff_w_2026.index) else np.nan
                feat[f"{col}_diff"] = va - vb
            for col in ["efg_pct", "to_pct", "or_pct", "ft_rate",
                        "opp_efg_pct", "opp_to_pct", "opp_or_pct", "opp_ft_rate"]:
                va = ff_w_2026.loc[ta, col] if (not ff_w_2026.empty and ta in ff_w_2026.index) else np.nan
                vb = ff_w_2026.loc[tb, col] if (not ff_w_2026.empty and tb in ff_w_2026.index) else np.nan
                feat[f"{col}_diff"] = va - vb
            if bart_w_2026 is not None:
                for src, dst in [("AdjOE", "bart_adjoe_diff"), ("AdjDE", "bart_adjde_diff"),
                                 ("NetRtg", "bart_net_diff"), ("Barthag", "bart_barthag_diff"),
                                 ("AdjTempo", "bart_tempo_diff")]:
                    va = bart_w_2026.loc[ta, src] if ta in bart_w_2026.index else np.nan
                    vb = bart_w_2026.loc[tb, src] if tb in bart_w_2026.index else np.nan
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
    p_2026_xgb_m = np.clip(final_xgb_m.predict(X_2026_xgb_m[["seed_diff","harry_diff","opp_qlty_won_diff"]]), 0.005, 0.995)
    p_2026_xgb_w = np.clip(final_xgb_w.predict(X_2026_xgb_w[["seed_diff","harry_diff","opp_qlty_won_diff"]]), 0.005, 0.995)

    # Top3 2026
    X_2026_top3, _, is_w_2026 = build_combined_features_2026(data_m, data_w)
    p_2026_top3_m = predict_lr(pipe_m_top3, X_2026_top3[is_w_2026 == 0], cols_m_top3)
    p_2026_top3_w = predict_lr(pipe_w_top3, X_2026_top3[is_w_2026 == 1], cols_w_top3)

    # ===== Blend 2026 predictions per gender =====
    def lookup_from_X_p(X, p):
        d = {}
        for i, r in X.reset_index(drop=True).iterrows():
            d[(int(r["TeamA"]), int(r["TeamB"]))] = float(p[i])
        return d

    lk_top3_m = lookup_from_X_p(X_2026_top3[is_w_2026 == 0], p_2026_top3_m)
    lk_top3_w = lookup_from_X_p(X_2026_top3[is_w_2026 == 1], p_2026_top3_w)
    lk_mf_m = lookup_from_X_p(X_2026_mf_m, p_2026_mf_m)
    lk_mf_w = lookup_from_X_p(X_2026_mf_w, p_2026_mf_w)
    lk_xgb_m = lookup_from_X_p(X_2026_xgb_m, p_2026_xgb_m)
    lk_xgb_w = lookup_from_X_p(X_2026_xgb_w, p_2026_xgb_w)

    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")

    def blended_lookup(lk1, lk2, lk3, w1, w2, w3):
        all_keys = set(lk1) | set(lk2) | set(lk3)
        out = {}
        for k in all_keys:
            p = w1 * lk1.get(k, 0.5) + w2 * lk2.get(k, 0.5) + w3 * lk3.get(k, 0.5)
            out[k] = float(np.clip(p, 0.005, 0.995))
        return out

    lk_m_blend = blended_lookup(lk_top3_m, lk_mf_m, lk_xgb_m, w_m_top3, w_m_mf, w_m_xgb)
    lk_w_blend = blended_lookup(lk_top3_w, lk_mf_w, lk_xgb_w, w_w_top3, w_w_mf, w_w_xgb)

    def br(lk, actual):
        yt, yp = [], []
        for _, g in actual.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            key = (w, l) if w < l else (l, w)
            yt.append(1 if w < l else 0)
            yp.append(lk.get(key, 0.5))
        return brier_score_loss(yt, yp), len(yt)

    bs_m_b, n_m = br(lk_m_blend, actual_m)
    bs_w_b, n_w = br(lk_w_blend, actual_w)
    bs_c = (bs_m_b * n_m + bs_w_b * n_w) / (n_m + n_w)

    # Also try isotonic on top of blend
    P_m_blend = w_m_top3 * P_m[:, 0] + w_m_mf * P_m[:, 1] + w_m_xgb * P_m[:, 2]
    P_w_blend = w_w_top3 * P_w[:, 0] + w_w_mf * P_w[:, 1] + w_w_xgb * P_w[:, 2]
    iso_m = IsotonicRegression(y_min=0.005, y_max=0.995, out_of_bounds="clip").fit(P_m_blend, y_m_aligned)
    iso_w = IsotonicRegression(y_min=0.005, y_max=0.995, out_of_bounds="clip").fit(P_w_blend, y_w_aligned)
    lk_m_iso = {k: float(np.clip(iso_m.predict([v])[0], 0.005, 0.995)) for k, v in lk_m_blend.items()}
    lk_w_iso = {k: float(np.clip(iso_w.predict([v])[0], 0.005, 0.995)) for k, v in lk_w_blend.items()}
    bs_m_iso, _ = br(lk_m_iso, actual_m)
    bs_w_iso, _ = br(lk_w_iso, actual_w)
    bs_c_iso = (bs_m_iso * n_m + bs_w_iso * n_w) / (n_m + n_w)

    print(f"\n{'='*70}\n  TRIPLE BLEND 2026 RESULTS\n{'='*70}")
    print(f"  Pure blend:     men={bs_m_b:.4f}  women={bs_w_b:.4f}  combined={bs_c:.4f}")
    print(f"  + Isotonic:     men={bs_m_iso:.4f}  women={bs_w_iso:.4f}  combined={bs_c_iso:.4f}")

    # Best per-gender mix (free choice between 3 approaches)
    print(f"\n  Per-gender best from blend:")
    # Per-gender: try {pure blend, blend+iso, individual models}
    options_m = {
        "blend": (lk_m_blend, bs_m_b),
        "blend_iso": (lk_m_iso, bs_m_iso),
        "top3_only": (lk_top3_m, br(lk_top3_m, actual_m)[0]),
        "mf_only": (lk_mf_m, br(lk_mf_m, actual_m)[0]),
        "xgb_only": (lk_xgb_m, br(lk_xgb_m, actual_m)[0]),
    }
    options_w = {
        "blend": (lk_w_blend, bs_w_b),
        "blend_iso": (lk_w_iso, bs_w_iso),
        "top3_only": (lk_top3_w, br(lk_top3_w, actual_w)[0]),
        "mf_only": (lk_mf_w, br(lk_mf_w, actual_w)[0]),
        "xgb_only": (lk_xgb_w, br(lk_xgb_w, actual_w)[0]),
    }
    for name, (lk, bs) in options_m.items():
        print(f"    Men {name:15s} = {bs:.4f}")
    for name, (lk, bs) in options_w.items():
        print(f"    Wom {name:15s} = {bs:.4f}")
    best_m_name = min(options_m, key=lambda k: options_m[k][1])
    best_w_name = min(options_w, key=lambda k: options_w[k][1])
    best_m_bs = options_m[best_m_name][1]
    best_w_bs = options_w[best_w_name][1]
    bs_c_best = (best_m_bs * n_m + best_w_bs * n_w) / (n_m + n_w)
    print(f"\n  Best per-gender mix:")
    print(f"    Men ({best_m_name}):    {best_m_bs:.4f}")
    print(f"    Women ({best_w_name}):  {best_w_bs:.4f}")
    print(f"    Combined:              {bs_c_best:.4f}")

    # Save submission with best per-gender
    best_lk_m = options_m[best_m_name][0]
    best_lk_w = options_w[best_w_name][0]
    sub = pd.read_csv("output/submission_stage2.csv")
    sub[["s_str", "ta_str", "tb_str"]] = sub["ID"].str.split("_", expand=True)
    sub["TeamA"] = sub["ta_str"].astype(int)
    sub["TeamB"] = sub["tb_str"].astype(int)
    new_map = {**best_lk_m, **best_lk_w}
    sub["Pred"] = sub.apply(
        lambda r: new_map.get((r["TeamA"], r["TeamB"]), float(r["Pred"])),
        axis=1
    ).clip(0.005, 0.995)
    sub[["ID", "Pred"]].to_csv("output/submission_stage2_triple.csv", index=False)
    print(f"\n  Saved output/submission_stage2_triple.csv")

    pd.DataFrame([
        {"strategy": "blend", "men": bs_m_b, "women": bs_w_b, "combined": bs_c},
        {"strategy": "blend+iso", "men": bs_m_iso, "women": bs_w_iso, "combined": bs_c_iso},
        {"strategy": "best_per_gender", "men": best_m_bs, "women": best_w_bs, "combined": bs_c_best},
    ]).to_csv("output/triple_blend_summary.csv", index=False)


if __name__ == "__main__":
    main()
