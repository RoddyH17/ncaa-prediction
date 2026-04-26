"""
Replicate 2nd/3rd-place 2026 Kaggle approach with free data only.

Combines features from both winning writeups:
  - Combined men's + women's training set (2nd place key insight)
  - Last-2-weeks all-system Massey composite (mean/median/min)
  - Carry-over Elo with MoV multiplier (75% reversion)
  - Colley Matrix rating
  - SRS (Simple Rating System)
  - Existing Barttorvik + Four Factors + efficiency features
  - LR with separate hyperparams per gender (C=100 men, C=0.15 women) — sklearn pipeline
  - Probability clip [0.02, 0.98]

LOSO cross-validation across 2014-2025 (excl. 2020).
Final 2026 prediction + evaluation against actual Brier.

Outputs:
  output/top3_loso.csv
  output/top3_summary.csv
  output/submission_stage2_top3.csv
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
from src.pipeline import (
    _parse_seed_num, build_efficiency_for_season, build_four_factors_for_season,
    build_momentum_for_season,
)
from scripts.build_womens_model import load_womens_data
from src.ratings_extra import build_extra_ratings
from src.massey_composite import build_massey_lookup_all_seasons
from src.harry_rating import build_harry_features


def safe_get(d, key, default=np.nan):
    v = d.get(key, default)
    return v if v is not None else default


def _scalar(df_indexed, key, col):
    """Return df.loc[key, col] coerced to a scalar (handles duplicate indexes)."""
    if key not in df_indexed.index:
        return np.nan
    val = df_indexed.loc[key, col]
    if hasattr(val, "iloc"):
        if len(val) == 0:
            return np.nan
        return val.iloc[0]
    return val


def build_combined_features(
    data_m: dict, data_w: dict, seasons: list[int],
) -> tuple[pd.DataFrame, np.ndarray, pd.Series]:
    """Build combined men's + women's matchup feature DataFrame.

    Per-team-season cache:
      - efficiency, four factors, momentum (already in pipeline)
      - Barttorvik (men's: barttorvik_{season}.csv; women's: barttorvik_w_{season}.csv)
      - carry-over Elo, Colley, SRS (computed from regular_compact)
      - last-2-week Massey composite (men's only)
      - harry_Rating

    Per-game features (all differentials Team1 - Team2):
      seed_diff, win_pct_diff,
      net_eff_diff, off_eff_diff, def_eff_diff, tempo_diff,
      efg_pct_diff, to_pct_diff, or_pct_diff, ft_rate_diff,
      bart_net_diff, bart_adjoe_diff, bart_adjde_diff, bart_barthag_diff,
      elo_diff, elo_slope_diff, colley_diff, srs_diff,
      massey_mean_diff, massey_median_diff, massey_min_diff (men's only),
      harry_diff, opp_qlty_won_diff,
      momentum_winpct_diff, momentum_margin_diff,
      is_womens (binary flag for combined model)
    """
    rows = []
    labels = []

    for label, data, is_womens in [("M", data_m, 0), ("W", data_w, 1)]:
        print(f"  Processing {label}'s data...")
        seeds_all = data["seeds"]
        regular_compact = data["regular_compact"]
        regular_detail = data.get("regular_detail")
        massey = data.get("massey")
        tourney = data["tourney_compact"]

        # Build extra ratings (Elo, Colley, SRS) on full regular_compact across all seasons
        extra = build_extra_ratings(regular_compact)
        massey_lookup = build_massey_lookup_all_seasons(massey)

        # Win percentages per team-season
        win_pct: dict[tuple[int, int], float] = {}
        for season, g in regular_compact.groupby("Season"):
            cnt: dict[int, list[int]] = {}
            for _, gm in g.iterrows():
                w, l = int(gm["WTeamID"]), int(gm["LTeamID"])
                cnt.setdefault(w, [0, 0])
                cnt.setdefault(l, [0, 0])
                cnt[w][0] += 1; cnt[w][1] += 1
                cnt[l][1] += 1
            for t, (wins, total) in cnt.items():
                win_pct[(int(season), int(t))] = wins / max(total, 1)

        # harry_Rating (uses regular_detail + conferences)
        try:
            hr = build_harry_features(data, seasons + [2026], is_womens=bool(is_womens))
            hr_lookup = {(int(r["Season"]), int(r["TeamID"])): r for _, r in hr.iterrows()}
        except Exception as e:
            print(f"    harry_rating failed: {e}")
            hr_lookup = {}

        # Per-season feature caches
        eff_cache: dict[int, pd.DataFrame] = {}
        ff_cache: dict[int, pd.DataFrame] = {}
        mom_cache: dict[int, dict] = {}
        bart_cache: dict[int, pd.DataFrame] = {}

        for season in seasons:
            try:
                eff_cache[season] = build_efficiency_for_season(data, season)
            except Exception:
                eff_cache[season] = pd.DataFrame()
            try:
                ff_cache[season] = build_four_factors_for_season(data, season)
            except Exception:
                ff_cache[season] = pd.DataFrame()
            seeds_season = seeds_all[seeds_all["Season"] == season]
            tids = list(seeds_season["TeamID"])
            try:
                mom = build_momentum_for_season(data, season, tids)
                mom_cache[season] = {row["TeamID"]: row for _, row in mom.iterrows()}
            except Exception:
                mom_cache[season] = {}
            # Barttorvik
            bart_path = DATA_DIR / "external" / (
                f"barttorvik_w_{season}.csv" if is_womens else f"barttorvik_{season}.csv"
            )
            if bart_path.exists():
                try:
                    bart_cache[season] = pd.read_csv(bart_path).set_index("TeamID")
                except Exception:
                    bart_cache[season] = pd.DataFrame()

        # Build per-game features
        for season in seasons:
            season_seeds = seeds_all[seeds_all["Season"] == season]
            seed_map = {int(r["TeamID"]): _parse_seed_num(r["Seed"])
                        for _, r in season_seeds.iterrows()}
            games = tourney[tourney["Season"] == season]
            eff = eff_cache.get(season, pd.DataFrame())
            ff = ff_cache.get(season, pd.DataFrame())
            mom = mom_cache.get(season, {})
            bart = bart_cache.get(season)

            for _, g in games.iterrows():
                w, l = int(g["WTeamID"]), int(g["LTeamID"])
                ta, tb, y = (w, l, 1) if w < l else (l, w, 0)

                feat = {
                    "Season": season, "TeamA": ta, "TeamB": tb, "is_womens": is_womens,
                    "seed_diff": seed_map.get(ta, 17) - seed_map.get(tb, 17),
                    "win_pct_diff": (win_pct.get((season, ta), 0.5) -
                                     win_pct.get((season, tb), 0.5)),
                }
                # Efficiency / Four Factors
                for col in ["off_eff", "def_eff", "net_eff", "tempo"]:
                    va = eff.loc[ta, col] if (not eff.empty and ta in eff.index) else np.nan
                    vb = eff.loc[tb, col] if (not eff.empty and tb in eff.index) else np.nan
                    feat[f"{col}_diff"] = va - vb
                for col in ["efg_pct", "to_pct", "or_pct", "ft_rate"]:
                    va = ff.loc[ta, col] if (not ff.empty and ta in ff.index) else np.nan
                    vb = ff.loc[tb, col] if (not ff.empty and tb in ff.index) else np.nan
                    feat[f"{col}_diff"] = va - vb
                # Barttorvik
                if bart is not None and not bart.empty:
                    for src, dst in [("AdjOE", "bart_adjoe_diff"),
                                     ("AdjDE", "bart_adjde_diff"),
                                     ("NetRtg", "bart_net_diff"),
                                     ("Barthag", "bart_barthag_diff")]:
                        va = _scalar(bart, ta, src)
                        vb = _scalar(bart, tb, src)
                        if pd.notna(va) and pd.notna(vb):
                            feat[dst] = float(va) - float(vb)
                        else:
                            feat[dst] = np.nan
                else:
                    for dst in ["bart_adjoe_diff", "bart_adjde_diff",
                                "bart_net_diff", "bart_barthag_diff"]:
                        feat[dst] = np.nan

                # Elo / Colley / SRS
                feat["elo_diff"] = (extra["elo_end"].get((season, ta), 1500.0) -
                                    extra["elo_end"].get((season, tb), 1500.0))
                feat["elo_slope_diff"] = (extra["elo_slope"].get((season, ta), 0.0) -
                                          extra["elo_slope"].get((season, tb), 0.0))
                feat["colley_diff"] = (extra["colley"].get((season, ta), 0.5) -
                                       extra["colley"].get((season, tb), 0.5))
                feat["srs_diff"] = (extra["srs"].get((season, ta), 0.0) -
                                    extra["srs"].get((season, tb), 0.0))

                # Massey composite (men's only — women's defaults to 0 diff)
                if not is_womens:
                    ma = massey_lookup.get((season, ta), {"mean": 200.0, "median": 200.0, "min": 200.0})
                    mb = massey_lookup.get((season, tb), {"mean": 200.0, "median": 200.0, "min": 200.0})
                    feat["massey_mean_diff"] = ma["mean"] - mb["mean"]
                    feat["massey_median_diff"] = ma["median"] - mb["median"]
                    feat["massey_min_diff"] = ma["min"] - mb["min"]
                else:
                    feat["massey_mean_diff"] = 0.0
                    feat["massey_median_diff"] = 0.0
                    feat["massey_min_diff"] = 0.0

                # harry_Rating
                ra = hr_lookup.get((season, ta))
                rb = hr_lookup.get((season, tb))
                feat["harry_diff"] = ((ra["harry_rating"] if ra is not None else 0) -
                                      (rb["harry_rating"] if rb is not None else 0))
                feat["opp_qlty_won_diff"] = ((ra["opp_qlty_pts_won"] if ra is not None else 0) -
                                              (rb["opp_qlty_pts_won"] if rb is not None else 0))

                # Momentum
                ma_mom = mom.get(ta, {})
                mb_mom = mom.get(tb, {})
                feat["momentum_winpct_diff"] = (
                    safe_get(ma_mom, "momentum_win_pct", 0.5) -
                    safe_get(mb_mom, "momentum_win_pct", 0.5)
                )
                feat["momentum_margin_diff"] = (
                    safe_get(ma_mom, "momentum_avg_margin", 0.0) -
                    safe_get(mb_mom, "momentum_avg_margin", 0.0)
                )

                rows.append(feat)
                labels.append(y)

    X = pd.DataFrame(rows)
    y = np.array(labels)
    return X, y, X["is_womens"]


FEATURE_COLS = [
    "seed_diff", "win_pct_diff",
    "off_eff_diff", "def_eff_diff", "net_eff_diff", "tempo_diff",
    "efg_pct_diff", "to_pct_diff", "or_pct_diff", "ft_rate_diff",
    "bart_net_diff", "bart_adjoe_diff", "bart_adjde_diff", "bart_barthag_diff",
    "elo_diff", "elo_slope_diff", "colley_diff", "srs_diff",
    "massey_mean_diff", "massey_median_diff", "massey_min_diff",
    "harry_diff", "opp_qlty_won_diff",
    "momentum_winpct_diff", "momentum_margin_diff",
]


def fit_lr(X, y, C=1.0):
    cols = [c for c in FEATURE_COLS if c in X.columns]
    Xn = X[cols].apply(pd.to_numeric, errors="coerce")
    pipe = Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
    ])
    pipe.fit(Xn, y)
    return pipe, cols


def predict_lr(pipe, X, cols):
    Xn = X[cols].apply(pd.to_numeric, errors="coerce")
    return pipe.predict_proba(Xn)[:, 1]


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]
    print("Loading data...")
    data_m = load_all_mens_data()
    data_w = load_womens_data()

    print("\nBuilding combined feature matrix (M + W)...")
    X, y, is_womens = build_combined_features(data_m, data_w, seasons)
    print(f"  Total games: {len(X)}  (M={len(X[X['is_womens']==0])}, W={len(X[X['is_womens']==1])})")

    season_arr = X["Season"].values
    is_w = X["is_womens"].values

    # ===== LOSO grid over C per gender =====
    print(f"\n{'='*70}\n  LOSO grid: C per gender\n{'='*70}")
    C_grid_m = [0.5, 1.0, 5.0, 10.0, 50.0, 100.0]
    C_grid_w = [0.05, 0.1, 0.15, 0.5, 1.0, 5.0]

    # Also: try combined-train C, plus separate-models per gender
    results = []

    # Separate-model approach (3rd place style)
    print("\n  -- Separate models per gender --")
    for C_m in C_grid_m:
        p_oof_m = np.zeros((is_w == 0).sum())
        idx_m = np.where(is_w == 0)[0]
        for s in np.unique(season_arr[is_w == 0]):
            te_mask_full = (season_arr == s) & (is_w == 0)
            tr_mask_full = (season_arr != s) & (is_w == 0)
            if te_mask_full.sum() == 0:
                continue
            pipe, cols = fit_lr(X[tr_mask_full], y[tr_mask_full], C=C_m)
            te_local = np.where(te_mask_full[idx_m])[0]
            te_global = idx_m[te_local]
            p = predict_lr(pipe, X.iloc[te_global], cols)
            p_oof_m[te_local] = p
        bs_m = brier_score_loss(y[idx_m], np.clip(p_oof_m, 0.02, 0.98))
        print(f"    C_m={C_m}: men's LOSO Brier = {bs_m:.4f}")
        results.append({"approach": "separate_men", "C": C_m, "brier": bs_m})

    for C_w in C_grid_w:
        p_oof_w = np.zeros((is_w == 1).sum())
        idx_w = np.where(is_w == 1)[0]
        for s in np.unique(season_arr[is_w == 1]):
            te_mask_full = (season_arr == s) & (is_w == 1)
            tr_mask_full = (season_arr != s) & (is_w == 1)
            if te_mask_full.sum() == 0:
                continue
            pipe, cols = fit_lr(X[tr_mask_full], y[tr_mask_full], C=C_w)
            te_local = np.where(te_mask_full[idx_w])[0]
            te_global = idx_w[te_local]
            p = predict_lr(pipe, X.iloc[te_global], cols)
            p_oof_w[te_local] = p
        bs_w = brier_score_loss(y[idx_w], np.clip(p_oof_w, 0.005, 0.995))
        print(f"    C_w={C_w}: women's LOSO Brier = {bs_w:.4f}")
        results.append({"approach": "separate_women", "C": C_w, "brier": bs_w})

    # Combined-model approach (2nd place style)
    print("\n  -- Combined model (M + W trained together) --")
    for C in [0.5, 1.0, 5.0, 10.0]:
        p_oof = np.zeros(len(X))
        for s in np.unique(season_arr):
            tr = season_arr != s; te = season_arr == s
            if te.sum() == 0: continue
            pipe, cols = fit_lr(X[tr], y[tr], C=C)
            p_oof[te] = predict_lr(pipe, X.iloc[np.where(te)[0]], cols)
        bs_m = brier_score_loss(y[is_w == 0], np.clip(p_oof[is_w == 0], 0.02, 0.98))
        bs_w = brier_score_loss(y[is_w == 1], np.clip(p_oof[is_w == 1], 0.02, 0.98))
        n_m = (is_w == 0).sum(); n_w_ = (is_w == 1).sum()
        bs_c = (bs_m * n_m + bs_w * n_w_) / (n_m + n_w_)
        print(f"    C={C}: men={bs_m:.4f}  women={bs_w:.4f}  combined={bs_c:.4f}")
        results.append({"approach": "combined", "C": C,
                        "brier_men": bs_m, "brier_women": bs_w, "combined": bs_c})

    pd.DataFrame(results).to_csv("output/top3_loso.csv", index=False)

    # Pick best C per gender (from separate)
    sep_men = [r for r in results if r["approach"] == "separate_men"]
    sep_wom = [r for r in results if r["approach"] == "separate_women"]
    best_C_m = min(sep_men, key=lambda r: r["brier"])["C"]
    best_C_w = min(sep_wom, key=lambda r: r["brier"])["C"]
    best_bs_m = min(sep_men, key=lambda r: r["brier"])["brier"]
    best_bs_w = min(sep_wom, key=lambda r: r["brier"])["brier"]
    print(f"\n  Best LOSO: C_m={best_C_m} (Brier {best_bs_m:.4f}), "
          f"C_w={best_C_w} (Brier {best_bs_w:.4f})")
    n_m = (is_w == 0).sum(); n_w_ = (is_w == 1).sum()
    print(f"  Best LOSO combined: {(best_bs_m*n_m + best_bs_w*n_w_)/(n_m+n_w_):.4f}")

    # ===== Train final + predict 2026 =====
    print(f"\n{'='*70}\n  Train final + predict 2026\n{'='*70}")
    pipe_m, cols_m = fit_lr(X[is_w == 0], y[is_w == 0], C=best_C_m)
    pipe_w, cols_w = fit_lr(X[is_w == 1], y[is_w == 1], C=best_C_w)

    # Build 2026 features: combine men's tournament pairs and women's pairs
    print("  Building 2026 features...")
    X_2026, _, is_w_2026 = build_combined_features_2026(data_m, data_w)
    print(f"  Total 2026 pairs: {len(X_2026)}")

    p_2026_m = predict_lr(pipe_m, X_2026[is_w_2026 == 0], cols_m)
    p_2026_w = predict_lr(pipe_w, X_2026[is_w_2026 == 1], cols_w)
    p_2026_m = np.clip(p_2026_m, 0.02, 0.98)
    p_2026_w = np.clip(p_2026_w, 0.005, 0.995)

    # Pair lookups
    new_map = {}
    X_m_2026 = X_2026[is_w_2026 == 0].reset_index(drop=True)
    X_w_2026 = X_2026[is_w_2026 == 1].reset_index(drop=True)
    for i, r in X_m_2026.iterrows():
        new_map[(int(r["TeamA"]), int(r["TeamB"]))] = float(p_2026_m[i])
    for i, r in X_w_2026.iterrows():
        new_map[(int(r["TeamA"]), int(r["TeamB"]))] = float(p_2026_w[i])

    # Update existing submission template
    sub = pd.read_csv("output/submission_stage2.csv")
    sub[["s_str", "ta_str", "tb_str"]] = sub["ID"].str.split("_", expand=True)
    sub["TeamA"] = sub["ta_str"].astype(int)
    sub["TeamB"] = sub["tb_str"].astype(int)
    sub["Pred"] = sub.apply(
        lambda r: new_map.get((r["TeamA"], r["TeamB"]), float(r["Pred"])),
        axis=1
    ).clip(0.005, 0.995)
    sub[["ID", "Pred"]].to_csv("output/submission_stage2_top3.csv", index=False)

    # Evaluate
    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")
    pmap = dict(zip(zip(sub["TeamA"], sub["TeamB"]), sub["Pred"]))

    def br(actual_df):
        yt, yp = [], []
        for _, g in actual_df.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            key = (w, l) if w < l else (l, w)
            yt.append(1 if w < l else 0)
            yp.append(pmap.get(key, 0.5))
        return brier_score_loss(yt, yp), len(yt)

    bs_m, n_m = br(actual_m); bs_w, n_w_ = br(actual_w)
    bs_c = (bs_m * n_m + bs_w * n_w_) / (n_m + n_w_)

    print(f"\n{'='*70}\n  FINAL 2026 BRIER (top3 replication)\n{'='*70}")
    print(f"  Men's:    {bs_m:.4f} ({n_m} games)")
    print(f"  Women's:  {bs_w:.4f} ({n_w_} games)")
    print(f"  Combined: {bs_c:.4f}")
    print(f"\n  Trajectory:")
    print(f"    Original baseline:           0.1264")
    print(f"    + harry+XGB+blend (prev best): 0.1206")
    print(f"    + 2nd/3rd place tricks:      {bs_c:.4f}")
    print(f"    Kaggle 3rd place:            0.1160")
    print(f"    Kaggle 1st place:            0.1097")

    pd.DataFrame([{"strategy": "top3_replication", "men": bs_m, "women": bs_w,
                   "combined": bs_c, "C_m": best_C_m, "C_w": best_C_w}]).to_csv(
        "output/top3_summary.csv", index=False
    )


def build_combined_features_2026(data_m, data_w):
    """Build 2026 prediction features for all tournament pairs (M + W)."""
    rows = []
    for label, data, is_w in [("M", data_m, 0), ("W", data_w, 1)]:
        seeds_all = data["seeds"]
        s2026 = seeds_all[seeds_all["Season"] == 2026]
        if s2026.empty:
            continue
        regular_compact = data["regular_compact"]
        massey = data.get("massey")
        extra = build_extra_ratings(regular_compact)
        massey_lookup = build_massey_lookup_all_seasons(massey)

        win_pct: dict[tuple[int, int], float] = {}
        for season, g in regular_compact.groupby("Season"):
            cnt: dict[int, list[int]] = {}
            for _, gm in g.iterrows():
                w, l = int(gm["WTeamID"]), int(gm["LTeamID"])
                cnt.setdefault(w, [0, 0]); cnt.setdefault(l, [0, 0])
                cnt[w][0] += 1; cnt[w][1] += 1; cnt[l][1] += 1
            for t, (wins, total) in cnt.items():
                win_pct[(int(season), int(t))] = wins / max(total, 1)

        try:
            hr = build_harry_features(data, [2026], is_womens=bool(is_w))
            hr_lookup = {(int(r["Season"]), int(r["TeamID"])): r for _, r in hr.iterrows()}
        except Exception:
            hr_lookup = {}

        try:
            eff = build_efficiency_for_season(data, 2026)
        except Exception:
            eff = pd.DataFrame()
        try:
            ff = build_four_factors_for_season(data, 2026)
        except Exception:
            ff = pd.DataFrame()
        tids = list(s2026["TeamID"])
        try:
            mom = build_momentum_for_season(data, 2026, tids)
            mom_map = {row["TeamID"]: row for _, row in mom.iterrows()}
        except Exception:
            mom_map = {}
        bart_path = DATA_DIR / "external" / (
            f"barttorvik_w_2026.csv" if is_w else f"barttorvik_2026.csv"
        )
        bart = pd.read_csv(bart_path).set_index("TeamID") if bart_path.exists() else None

        seed_map = {int(r["TeamID"]): _parse_seed_num(r["Seed"])
                    for _, r in s2026.iterrows()}
        ids = sorted(int(t) for t in s2026["TeamID"])
        for i, ta in enumerate(ids):
            for tb in ids[i+1:]:
                feat = {"Season": 2026, "TeamA": ta, "TeamB": tb, "is_womens": is_w,
                        "seed_diff": seed_map.get(ta, 17) - seed_map.get(tb, 17),
                        "win_pct_diff": (win_pct.get((2026, ta), 0.5) -
                                         win_pct.get((2026, tb), 0.5))}
                for col in ["off_eff", "def_eff", "net_eff", "tempo"]:
                    va = eff.loc[ta, col] if (not eff.empty and ta in eff.index) else np.nan
                    vb = eff.loc[tb, col] if (not eff.empty and tb in eff.index) else np.nan
                    feat[f"{col}_diff"] = va - vb
                for col in ["efg_pct", "to_pct", "or_pct", "ft_rate"]:
                    va = ff.loc[ta, col] if (not ff.empty and ta in ff.index) else np.nan
                    vb = ff.loc[tb, col] if (not ff.empty and tb in ff.index) else np.nan
                    feat[f"{col}_diff"] = va - vb
                if bart is not None and not bart.empty:
                    for src, dst in [("AdjOE", "bart_adjoe_diff"),
                                     ("AdjDE", "bart_adjde_diff"),
                                     ("NetRtg", "bart_net_diff"),
                                     ("Barthag", "bart_barthag_diff")]:
                        va = _scalar(bart, ta, src)
                        vb = _scalar(bart, tb, src)
                        if pd.notna(va) and pd.notna(vb):
                            feat[dst] = float(va) - float(vb)
                        else:
                            feat[dst] = np.nan
                else:
                    for dst in ["bart_adjoe_diff", "bart_adjde_diff",
                                "bart_net_diff", "bart_barthag_diff"]:
                        feat[dst] = np.nan
                feat["elo_diff"] = (extra["elo_end"].get((2026, ta), 1500.0) -
                                    extra["elo_end"].get((2026, tb), 1500.0))
                feat["elo_slope_diff"] = (extra["elo_slope"].get((2026, ta), 0.0) -
                                          extra["elo_slope"].get((2026, tb), 0.0))
                feat["colley_diff"] = (extra["colley"].get((2026, ta), 0.5) -
                                       extra["colley"].get((2026, tb), 0.5))
                feat["srs_diff"] = (extra["srs"].get((2026, ta), 0.0) -
                                    extra["srs"].get((2026, tb), 0.0))
                if not is_w:
                    ma = massey_lookup.get((2026, ta), {"mean": 200.0, "median": 200.0, "min": 200.0})
                    mb = massey_lookup.get((2026, tb), {"mean": 200.0, "median": 200.0, "min": 200.0})
                    feat["massey_mean_diff"] = ma["mean"] - mb["mean"]
                    feat["massey_median_diff"] = ma["median"] - mb["median"]
                    feat["massey_min_diff"] = ma["min"] - mb["min"]
                else:
                    feat["massey_mean_diff"] = 0.0
                    feat["massey_median_diff"] = 0.0
                    feat["massey_min_diff"] = 0.0
                ra = hr_lookup.get((2026, ta))
                rb = hr_lookup.get((2026, tb))
                feat["harry_diff"] = ((ra["harry_rating"] if ra is not None else 0) -
                                      (rb["harry_rating"] if rb is not None else 0))
                feat["opp_qlty_won_diff"] = ((ra["opp_qlty_pts_won"] if ra is not None else 0) -
                                              (rb["opp_qlty_pts_won"] if rb is not None else 0))
                ma_mom = mom_map.get(ta, {})
                mb_mom = mom_map.get(tb, {})
                feat["momentum_winpct_diff"] = (
                    safe_get(ma_mom, "momentum_win_pct", 0.5) -
                    safe_get(mb_mom, "momentum_win_pct", 0.5)
                )
                feat["momentum_margin_diff"] = (
                    safe_get(ma_mom, "momentum_avg_margin", 0.0) -
                    safe_get(mb_mom, "momentum_avg_margin", 0.0)
                )
                rows.append(feat)
    X = pd.DataFrame(rows)
    return X, np.zeros(len(X)), X["is_womens"].values


if __name__ == "__main__":
    main()
