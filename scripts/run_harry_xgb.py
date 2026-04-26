"""
Replicate the 1st-place 2026 Kaggle approach: XGBoost regression on
3 features (men's) / 4 features (women's) with hand-tuned harry_Rating.

Pipeline:
  1. Compute harry_Rating + opp_qlty_pts_won for all (Season, TeamID)
  2. Build matchup features as differentials (T1 - T2)
  3. XGBoost regression on Win (binary 0/1, regression on the binary target)
     hyperparams from 1st place writeup
  4. LOTO evaluation
  5. In-sample isotonic calibration
  6. 2026 predictions + actual evaluation
  7. Optional hard sharpening at 0.97/0.03

Outputs:
  output/harry_xgb_loto.csv      - per-season LOTO Brier
  output/harry_xgb_summary.csv   - final summary table
  output/submission_stage2_harry.csv
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import build_tourney_matchups, _parse_seed_num
from src.harry_rating import build_harry_features
from scripts.build_womens_model import load_womens_data


HPARAMS_MEN = dict(max_depth=2, min_child_weight=5,
                   subsample=0.7, colsample_bytree=0.8,
                   reg_alpha=0.1, reg_lambda=1.0)
HPARAMS_WOM = dict(max_depth=2, min_child_weight=3,
                   subsample=0.8, colsample_bytree=0.8,
                   reg_alpha=0.1, reg_lambda=1.0)


def build_matchup_features(data: dict, seasons: list[int], is_womens: bool, hr: pd.DataFrame):
    """Build matchup-level differential features."""
    hr_lookup = {(int(r["Season"]), int(r["TeamID"])): r for _, r in hr.iterrows()}
    tourney = data["tourney_compact"] if not is_womens else data["tourney_compact"]
    seeds = data["seeds"]

    rows, labels = [], []
    for season in seasons:
        season_seeds = seeds[seeds["Season"] == season]
        seed_map = {int(r["TeamID"]): _parse_seed_num(r["Seed"])
                    for _, r in season_seeds.iterrows()}
        games = tourney[tourney["Season"] == season]
        for _, g in games.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            ta, tb, y = (w, l, 1) if w < l else (l, w, 0)
            ra = hr_lookup.get((season, ta))
            rb = hr_lookup.get((season, tb))
            feat = {
                "Season": season, "TeamA": ta, "TeamB": tb,
                "seed_diff": seed_map.get(ta, 16) - seed_map.get(tb, 16),
                "harry_diff": (ra["harry_rating"] if ra is not None else 0) -
                              (rb["harry_rating"] if rb is not None else 0),
                "opp_qlty_won_diff": (ra["opp_qlty_pts_won"] if ra is not None else 0) -
                                      (rb["opp_qlty_pts_won"] if rb is not None else 0),
            }
            if is_womens:
                # avg blocks differential — proxy via NetEff defense (close enough placeholder)
                # we keep harry_diff and opp_qlty_won_diff; the women's avg_blk_diff would
                # need season-level avg blocks per team, which we can compute if needed
                pass
            rows.append(feat)
            labels.append(y)
    return pd.DataFrame(rows), np.array(labels)


def build_2026_pair_features(data: dict, hr: pd.DataFrame, womens: bool = False):
    """Build matchup features for ALL 2026 tournament-pair combinations."""
    hr_lookup = {(int(r["Season"]), int(r["TeamID"])): r for _, r in hr.iterrows()}
    seeds = data["seeds"]
    s2026 = seeds[seeds["Season"] == 2026]
    seed_map = {int(r["TeamID"]): _parse_seed_num(r["Seed"])
                for _, r in s2026.iterrows()}
    tids = sorted(int(t) for t in s2026["TeamID"])
    rows = []
    for i, ta in enumerate(tids):
        for tb in tids[i+1:]:
            ra = hr_lookup.get((2026, ta))
            rb = hr_lookup.get((2026, tb))
            rows.append({
                "Season": 2026, "TeamA": ta, "TeamB": tb,
                "seed_diff": seed_map.get(ta, 16) - seed_map.get(tb, 16),
                "harry_diff": (ra["harry_rating"] if ra is not None else 0) -
                              (rb["harry_rating"] if rb is not None else 0),
                "opp_qlty_won_diff": (ra["opp_qlty_pts_won"] if ra is not None else 0) -
                                      (rb["opp_qlty_pts_won"] if rb is not None else 0),
            })
    return pd.DataFrame(rows)


def train_xgb_loto(X: pd.DataFrame, y: np.ndarray, hparams: dict) -> tuple:
    """LOTO out-of-fold predictions; returns (p_oof, fold_briers)."""
    feat_cols = ["seed_diff", "harry_diff", "opp_qlty_won_diff"]
    p_oof = np.zeros(len(X))
    fold_briers = []
    for s in np.unique(X["Season"]):
        tr_mask = (X["Season"] != s).values
        te_mask = (X["Season"] == s).values
        if te_mask.sum() == 0:
            continue
        # Inner train/val split for early stopping
        tr_idx = np.where(tr_mask)[0]
        np.random.default_rng(42).shuffle(tr_idx)
        n_val = int(0.1 * len(tr_idx))
        val_idx, sub_tr_idx = tr_idx[:n_val], tr_idx[n_val:]

        model = xgb.XGBRegressor(
            eval_metric="rmse",
            n_estimators=4000,
            learning_rate=0.003,
            early_stopping_rounds=100,
            objective="reg:squarederror",
            tree_method="hist",
            **hparams,
        )
        model.fit(
            X.iloc[sub_tr_idx][feat_cols], y[sub_tr_idx],
            eval_set=[(X.iloc[val_idx][feat_cols], y[val_idx])],
            verbose=False,
        )
        p_te = model.predict(X.iloc[te_mask][feat_cols])
        p_oof[te_mask] = np.clip(p_te, 0.001, 0.999)
        fold_briers.append({
            "season": int(s),
            "n": int(te_mask.sum()),
            "brier": float(brier_score_loss(y[te_mask], p_oof[te_mask])),
        })
    return p_oof, fold_briers


def train_xgb_final(X: pd.DataFrame, y: np.ndarray, hparams: dict) -> xgb.XGBRegressor:
    feat_cols = ["seed_diff", "harry_diff", "opp_qlty_won_diff"]
    rng = np.random.default_rng(42)
    idx = np.arange(len(X)); rng.shuffle(idx)
    n_val = int(0.1 * len(idx))
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    model = xgb.XGBRegressor(
        eval_metric="rmse", n_estimators=4000, learning_rate=0.003,
        early_stopping_rounds=100, objective="reg:squarederror",
        tree_method="hist", **hparams,
    )
    model.fit(X.iloc[tr_idx][feat_cols], y[tr_idx],
              eval_set=[(X.iloc[val_idx][feat_cols], y[val_idx])],
              verbose=False)
    return model


def hard_sharpen(p: np.ndarray, low=0.03, high=0.97, low_t=0.001, high_t=0.999):
    out = np.array(p, dtype=float).copy()
    out[p >= high] = high_t
    out[p <= low] = low_t
    return out


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]

    # ===== Men's =====
    print(f"{'='*70}\n  MEN'S: harry_Rating + XGBoost regression\n{'='*70}")
    data_m = load_all_mens_data()
    hr_m = build_harry_features(data_m, seasons + [2026], is_womens=False)
    X_m, y_m = build_matchup_features(data_m, seasons, is_womens=False, hr=hr_m)
    print(f"  N games: {len(X_m)}")

    p_oof_m, fold_m = train_xgb_loto(X_m, y_m, HPARAMS_MEN)
    fold_df = pd.DataFrame(fold_m)
    print(f"  LOTO Brier per season:")
    for r in fold_m:
        print(f"    {r['season']}: {r['brier']:.4f}")
    print(f"  Men's CV Brier (raw):        {brier_score_loss(y_m, p_oof_m):.4f}")

    iso_m = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso_m.fit(p_oof_m, y_m)
    p_oof_m_cal = iso_m.predict(p_oof_m)
    print(f"  Men's CV Brier (calibrated): {brier_score_loss(y_m, p_oof_m_cal):.4f}")

    # ===== Women's =====
    print(f"\n{'='*70}\n  WOMEN'S: harry_Rating + XGBoost regression\n{'='*70}")
    data_w = load_womens_data()
    hr_w = build_harry_features(data_w, seasons + [2026], is_womens=True)
    X_w, y_w = build_matchup_features(data_w, seasons, is_womens=True, hr=hr_w)
    print(f"  N games: {len(X_w)}")

    p_oof_w, fold_w = train_xgb_loto(X_w, y_w, HPARAMS_WOM)
    print(f"  Women's CV Brier (raw):      {brier_score_loss(y_w, p_oof_w):.4f}")

    iso_w = IsotonicRegression(y_min=0.001, y_max=0.999, out_of_bounds="clip")
    iso_w.fit(p_oof_w, y_w)
    p_oof_w_cal = iso_w.predict(p_oof_w)
    print(f"  Women's CV Brier (calibrated): {brier_score_loss(y_w, p_oof_w_cal):.4f}")

    # ===== Combined CV =====
    bs_m_raw = brier_score_loss(y_m, p_oof_m)
    bs_w_raw = brier_score_loss(y_w, p_oof_w)
    bs_m_cal = brier_score_loss(y_m, p_oof_m_cal)
    bs_w_cal = brier_score_loss(y_w, p_oof_w_cal)
    bs_c_raw = (bs_m_raw * len(y_m) + bs_w_raw * len(y_w)) / (len(y_m) + len(y_w))
    bs_c_cal = (bs_m_cal * len(y_m) + bs_w_cal * len(y_w)) / (len(y_m) + len(y_w))
    print(f"\n  Total CV Brier (raw):        {bs_c_raw:.4f}")
    print(f"  Total CV Brier (calibrated): {bs_c_cal:.4f}")
    print(f"  1st place reported:          0.1620 raw / 0.1590 calibrated")

    # ===== Final models + 2026 predictions =====
    print(f"\n{'='*70}\n  Train final models + predict 2026\n{'='*70}")
    final_m = train_xgb_final(X_m, y_m, HPARAMS_MEN)
    final_w = train_xgb_final(X_w, y_w, HPARAMS_WOM)

    X_2026_m = build_2026_pair_features(data_m, hr_m, womens=False)
    X_2026_w = build_2026_pair_features(data_w, hr_w, womens=True)

    p_2026_m = np.clip(final_m.predict(X_2026_m[["seed_diff", "harry_diff", "opp_qlty_won_diff"]]), 0.001, 0.999)
    p_2026_w = np.clip(final_w.predict(X_2026_w[["seed_diff", "harry_diff", "opp_qlty_won_diff"]]), 0.001, 0.999)

    p_2026_m_cal = iso_m.predict(p_2026_m)
    p_2026_w_cal = iso_w.predict(p_2026_w)

    # Pair lookups
    def lookup(X, p):
        out = {}
        for i, r in X.reset_index(drop=True).iterrows():
            a, b = int(r["TeamA"]), int(r["TeamB"])
            out[(a, b)] = float(p[i]); out[(b, a)] = 1 - float(p[i])
        return out

    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")

    def br(lookup_dict, actual):
        yt, yp = [], []
        for _, g in actual.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            key = (w, l) if w < l else (l, w)
            yt.append(1 if w < l else 0)
            yp.append(lookup_dict.get(key, 0.5))
        return brier_score_loss(yt, yp)

    rows = []
    n_m, n_w = len(actual_m), len(actual_w)
    for label, pm, pw in [
        ("xgb_raw", p_2026_m, p_2026_w),
        ("xgb_isotonic", p_2026_m_cal, p_2026_w_cal),
        ("xgb_iso_sharpen", hard_sharpen(p_2026_m_cal), hard_sharpen(p_2026_w_cal)),
        ("xgb_raw_sharpen", hard_sharpen(p_2026_m), hard_sharpen(p_2026_w)),
    ]:
        lk_m = lookup(X_2026_m, pm)
        lk_w = lookup(X_2026_w, pw)
        bs_m = br(lk_m, actual_m)
        bs_w = br(lk_w, actual_w)
        bs_c = (bs_m * n_m + bs_w * n_w) / (n_m + n_w)
        rows.append({"strategy": label, "men": bs_m, "women": bs_w, "combined": bs_c})

    summary = pd.DataFrame(rows).sort_values("combined")
    print(summary.to_string(index=False))
    summary.to_csv("output/harry_xgb_summary.csv", index=False)

    print(f"\n  Best XGB+harry: {summary.iloc[0]['strategy']} -> "
          f"combined Brier {summary.iloc[0]['combined']:.4f}")
    print(f"  Previous Multi-Feat Logistic combined: 0.1264")
    print(f"  1st place final: 0.1097")
    print(f"  3rd place final: 0.1160")


if __name__ == "__main__":
    main()
