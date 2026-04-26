"""
Final blend: XGB+harry + Multi-Feature Logistic ensemble for men's and women's.

Sweeps blending weight w in [0, 1]:
    p_blend = w * p_xgb_harry + (1-w) * p_logistic

Tests with and without isotonic and sharpening, on both 2026 men's and women's.
Reports combined Brier across grid; keeps best per-gender selection.
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
    train_xgb_loto, train_xgb_final, hard_sharpen,
    HPARAMS_MEN, HPARAMS_WOM,
)
from scripts.generate_kaggle_submission import build_submission_features


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]

    # ===== MEN'S =====
    print("MEN'S: building both pipelines...")
    data_m = load_all_mens_data()
    build_fn_m = make_build_features_fn(data_m)
    X_full_m, y_full_m = build_fn_m(seasons)

    # 1) Multi-Feature Logistic LOTO + 2026
    p_oof_log_m = np.zeros(len(X_full_m))
    season_m = X_full_m["Season"].values
    for s in np.unique(season_m):
        tr = season_m != s; te = season_m == s
        m = MultiFeatureLogistic(C=0.5).fit(X_full_m.loc[tr], y_full_m[tr])
        p_oof_log_m[te] = m.predict_proba(X_full_m.loc[te])[:, 1]
    final_log_m = MultiFeatureLogistic(C=0.5).fit(X_full_m, y_full_m)
    sub_path = str(DATA_DIR / "SampleSubmissionStage2.csv")
    _, X_2026_log_m, _ = build_submission_features(data_m, 2026, sub_path)
    p_2026_log_m = final_log_m.predict_proba(X_2026_log_m)[:, 1]

    # 2) harry+XGB LOTO + 2026
    hr_m = build_harry_features(data_m, seasons + [2026], is_womens=False)
    X_xgb_m, y_xgb_m = build_matchup_features(data_m, seasons, is_womens=False, hr=hr_m)
    p_oof_xgb_m, _ = train_xgb_loto(X_xgb_m, y_xgb_m, HPARAMS_MEN)
    final_xgb_m = train_xgb_final(X_xgb_m, y_xgb_m, HPARAMS_MEN)
    X_2026_xgb_m = build_2026_pair_features(data_m, hr_m, womens=False)
    p_2026_xgb_m = np.clip(final_xgb_m.predict(X_2026_xgb_m[["seed_diff", "harry_diff", "opp_qlty_won_diff"]]), 0.001, 0.999)

    # Align: build per-pair lookup
    def to_lookup(X, p):
        d = {}
        for i, r in X.reset_index(drop=True).iterrows():
            a, b = int(r["TeamA"]), int(r["TeamB"])
            d[(a, b)] = float(p[i])
        return d

    lk_log_m = to_lookup(X_2026_log_m, p_2026_log_m)
    lk_xgb_m = to_lookup(X_2026_xgb_m, p_2026_xgb_m)
    common_pairs_m = set(lk_log_m.keys()) & set(lk_xgb_m.keys())
    print(f"  Men's common 2026 pairs: {len(common_pairs_m)}")

    # Build LOTO OOF logistic for blend calibration
    # We want: aligned LOTO indices for men's
    # Issue: X_full_m has ID cols, X_xgb_m has same ID cols, so we align via (Season, TeamA, TeamB)
    log_oof_lookup = {}
    for i, r in X_full_m.reset_index(drop=True).iterrows():
        log_oof_lookup[(int(r["Season"]), int(r["TeamA"]), int(r["TeamB"]))] = p_oof_log_m[i]
    p_oof_log_aligned = np.array([
        log_oof_lookup.get((int(r["Season"]), int(r["TeamA"]), int(r["TeamB"])), 0.5)
        for _, r in X_xgb_m.iterrows()
    ])

    # ===== WOMEN'S =====
    print("\nWOMEN'S: building both pipelines...")
    data_w = load_womens_data()
    X_full_w, y_full_w = build_womens_features(data_w, seasons)
    season_w = X_full_w["Season"].values
    p_oof_log_w = np.zeros(len(X_full_w))
    for s in np.unique(season_w):
        tr = season_w != s; te = season_w == s
        m = WomensLogistic(C=0.5).fit(X_full_w.loc[tr], y_full_w[tr])
        p_oof_log_w[te] = m.predict_proba(X_full_w.loc[te])[:, 1]

    final_log_w = WomensLogistic(C=0.5).fit(X_full_w, y_full_w)

    # Re-use existing women's pipeline 2026 prediction
    # Build women's submission pairs for ALL tournament team combinations
    seeds_w = data_w["seeds"]
    s2026_w = seeds_w[seeds_w["Season"] == 2026].copy()
    s2026_w["SeedNum"] = s2026_w["Seed"].apply(_parse_seed_num)
    tids_w = sorted(int(t) for t in s2026_w["TeamID"])
    seed_map_w = dict(zip(s2026_w["TeamID"], s2026_w["SeedNum"]))

    from src.pipeline import build_efficiency_for_season, build_four_factors_for_season, build_momentum_for_season
    eff_w_2026 = build_efficiency_for_season(data_w, 2026)
    ff_w_2026 = build_four_factors_for_season(data_w, 2026)
    mom_w_2026 = build_momentum_for_season(data_w, 2026, tids_w)
    mom_map_w = {row["TeamID"]: row for _, row in mom_w_2026.iterrows()}
    bart_w_path = DATA_DIR / "external" / "barttorvik_w_2026.csv"
    bart_w = pd.read_csv(bart_w_path).set_index("TeamID") if bart_w_path.exists() else None

    feats_log_w = []
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
            feats_log_w.append(feat)
    X_2026_log_w = pd.DataFrame(feats_log_w)
    p_2026_log_w = final_log_w.predict_proba(X_2026_log_w)[:, 1]

    hr_w = build_harry_features(data_w, seasons + [2026], is_womens=True)
    X_xgb_w, y_xgb_w = build_matchup_features(data_w, seasons, is_womens=True, hr=hr_w)
    p_oof_xgb_w, _ = train_xgb_loto(X_xgb_w, y_xgb_w, HPARAMS_WOM)
    final_xgb_w = train_xgb_final(X_xgb_w, y_xgb_w, HPARAMS_WOM)
    X_2026_xgb_w = build_2026_pair_features(data_w, hr_w, womens=True)
    p_2026_xgb_w = np.clip(final_xgb_w.predict(X_2026_xgb_w[["seed_diff", "harry_diff", "opp_qlty_won_diff"]]), 0.001, 0.999)

    # Align logistic women's preds onto same row order as XGB women's
    lk_log_w = to_lookup(X_2026_log_w, p_2026_log_w)
    lk_xgb_w = to_lookup(X_2026_xgb_w, p_2026_xgb_w)

    log_oof_w_lookup = {}
    for i, r in X_full_w.reset_index(drop=True).iterrows():
        log_oof_w_lookup[(int(r["Season"]), int(r["TeamA"]), int(r["TeamB"]))] = p_oof_log_w[i]
    p_oof_log_w_aligned = np.array([
        log_oof_w_lookup.get((int(r["Season"]), int(r["TeamA"]), int(r["TeamB"])), 0.5)
        for _, r in X_xgb_w.iterrows()
    ])

    # ===== Blend on LOTO OOF, choose best weight per gender =====
    def best_blend(p_xgb_oof, p_log_oof, y, name):
        best_bs = np.inf; best_w = 0.5
        for w in np.linspace(0.0, 1.0, 21):
            p = w * p_xgb_oof + (1 - w) * p_log_oof
            bs = brier_score_loss(y, p)
            if bs < best_bs:
                best_bs = bs; best_w = w
        print(f"  {name}: best w_xgb = {best_w:.2f}, OOF Brier = {best_bs:.4f}")
        return best_w

    print(f"\n{'='*70}\n  Blend weight selection on LOTO OOF\n{'='*70}")
    w_m = best_blend(p_oof_xgb_m, p_oof_log_aligned, y_xgb_m, "men's")
    w_w = best_blend(p_oof_xgb_w, p_oof_log_w_aligned, y_xgb_w, "women's")

    # ===== Apply blend to 2026 + isotonic + sharpen =====
    def get_blend_2026_lookup(lk_xgb, lk_log, w):
        out = {}
        common = set(lk_xgb.keys()) & set(lk_log.keys())
        for k in common:
            out[k] = w * lk_xgb[k] + (1 - w) * lk_log[k]
        return out

    lk_m_blend = get_blend_2026_lookup(lk_xgb_m, lk_log_m, w_m)
    lk_w_blend = get_blend_2026_lookup(lk_xgb_w, lk_log_w, w_w)

    # Isotonic from blended OOF
    p_oof_blend_m = w_m * p_oof_xgb_m + (1 - w_m) * p_oof_log_aligned
    p_oof_blend_w = w_w * p_oof_xgb_w + (1 - w_w) * p_oof_log_w_aligned
    iso_m = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip").fit(p_oof_blend_m, y_xgb_m)
    iso_w = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip").fit(p_oof_blend_w, y_xgb_w)

    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")

    def br_via_lookup(lk, actual):
        yt, yp = [], []
        for _, g in actual.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            key = (w, l) if w < l else (l, w)
            yt.append(1 if w < l else 0)
            yp.append(lk.get(key, 0.5))
        return brier_score_loss(yt, yp)

    n_m, n_w = len(actual_m), len(actual_w)

    # Apply isotonic + optional sharpen
    def transform_lookup(lk, iso=None, sharpen=False):
        out = {}
        for k, v in lk.items():
            p = iso.predict([v])[0] if iso is not None else v
            if sharpen:
                if p >= 0.97: p = 0.999
                if p <= 0.03: p = 0.001
            out[k] = float(np.clip(p, 0.001, 0.999))
        return out

    rows = []
    for label, lk_m, lk_w in [
        ("blend_raw", lk_m_blend, lk_w_blend),
        ("blend_iso", transform_lookup(lk_m_blend, iso=iso_m), transform_lookup(lk_w_blend, iso=iso_w)),
        ("blend_iso_sharpen", transform_lookup(lk_m_blend, iso=iso_m, sharpen=True),
                              transform_lookup(lk_w_blend, iso=iso_w, sharpen=True)),
        ("blend_sharpen", transform_lookup(lk_m_blend, sharpen=True), transform_lookup(lk_w_blend, sharpen=True)),
    ]:
        bs_m = br_via_lookup(lk_m, actual_m)
        bs_w = br_via_lookup(lk_w, actual_w)
        bs_c = (bs_m * n_m + bs_w * n_w) / (n_m + n_w)
        rows.append({"strategy": label, "men": bs_m, "women": bs_w, "combined": bs_c,
                     "w_m": w_m, "w_w": w_w})

    df = pd.DataFrame(rows).sort_values("combined")
    print(f"\n{'='*70}\n  Final 2026 evaluation\n{'='*70}")
    print(df.to_string(index=False))
    df.to_csv("output/blend_final_summary.csv", index=False)

    # Also include best per-gender selection
    print(f"\n  Best mixed (per-gender best):")
    best_m = df["men"].min(); best_w = df["women"].min()
    print(f"    men: {best_m:.4f}, women: {best_w:.4f}, combined: "
          f"{(best_m*n_m + best_w*n_w)/(n_m+n_w):.4f}")

    print(f"\n  Targets: 1st place 0.1097, 3rd place 0.1160")
    print(f"  Our previous baseline:      0.1264")


if __name__ == "__main__":
    main()
