"""
Robust optimization that aims to improve BOTH LOSO and 2026 actual:

  (1) Extended training: pre-2014 compact-only history (Barttorvik defaults
      to season-median imputation; effective extra ~520 games per gender)

  (2) Random Fourier Features (RFF): automatic non-linear interactions via
      RBF kernel approximation. Robust in small-N regime; replaces hand-
      designed interactions.

  (3) Bayesian shrinkage toward seed-implied prior:
        p_final = (1-lambda) * p_model + lambda * sigmoid(beta * seed_diff)
      Tuned on LOSO. Single shrinkage parameter -> robust calibration that
      avoids isotonic over-fit.

We test each lever individually and in combination, all on LOSO. The
strategy with best LOSO Brier is then evaluated ONCE on 2026 actual.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.kernel_approximation import RBFSampler
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


SEED_PRIOR_BETA = 0.13  # logistic mapping from seed_diff to p


def seed_prior(seed_diff):
    """Seed-implied probability: sigmoid(-beta * seed_diff). seed_diff > 0
    means TeamA has higher (worse) seed, so lower P(TeamA wins)."""
    return 1.0 / (1.0 + np.exp(SEED_PRIOR_BETA * seed_diff))


# ===========================================================================
# Build features extended to 2003-2025 (compact-only path for pre-2015)
# ===========================================================================

# Features that work without Barttorvik (using season-median imputation pre-2015):
EXTENDED_FEATURES = [
    "seed_diff", "win_pct_diff",
    "elo_diff", "elo_slope_diff", "colley_diff", "srs_diff",
    "bart_net_diff", "bart_adjoe_diff", "bart_adjde_diff", "bart_barthag_diff",
    "massey_mean_diff", "massey_median_diff", "massey_min_diff",
    "harry_diff", "opp_qlty_won_diff",
    "momentum_winpct_diff", "momentum_margin_diff",
    "off_eff_diff", "def_eff_diff", "net_eff_diff", "tempo_diff",
    "efg_pct_diff", "to_pct_diff", "or_pct_diff", "ft_rate_diff",
]


def build_extended_features(
    data_m: dict, data_w: dict, seasons: list[int]
) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Build features for extended-history training.

    Pre-2015 seasons get NaN for Barttorvik (will be imputed by median in pipeline).
    All other features (Elo, Colley, SRS, Massey, harry) computable for all seasons.
    """
    rows = []
    labels = []
    for label, data, is_womens in [("M", data_m, 0), ("W", data_w, 1)]:
        print(f"  Building extended features for {label}'s ({len(seasons)} seasons)...")
        seeds_all = data["seeds"]
        regular_compact = data["regular_compact"]
        regular_detail = data.get("regular_detail")
        massey = data.get("massey")
        tourney = data["tourney_compact"]

        extra = build_extra_ratings(regular_compact)
        massey_lookup = build_massey_lookup_all_seasons(massey)

        # Win pct
        win_pct = {}
        for season, g in regular_compact.groupby("Season"):
            cnt = {}
            for _, gm in g.iterrows():
                w_, l_ = int(gm["WTeamID"]), int(gm["LTeamID"])
                cnt.setdefault(w_, [0, 0]); cnt.setdefault(l_, [0, 0])
                cnt[w_][0] += 1; cnt[w_][1] += 1; cnt[l_][1] += 1
            for t, (wins, total) in cnt.items():
                win_pct[(int(season), int(t))] = wins / max(total, 1)

        # harry_Rating: compute on all seasons regardless of Barttorvik availability
        try:
            hr = build_harry_features(data, seasons + [2026], is_womens=bool(is_womens))
            hr_lookup = {(int(r["Season"]), int(r["TeamID"])): r for _, r in hr.iterrows()}
        except Exception:
            hr_lookup = {}

        # Per-season caches
        eff_cache, ff_cache, mom_cache, bart_cache = {}, {}, {}, {}
        for season in seasons:
            try: eff_cache[season] = build_efficiency_for_season(data, season)
            except: eff_cache[season] = pd.DataFrame()
            try: ff_cache[season] = build_four_factors_for_season(data, season)
            except: ff_cache[season] = pd.DataFrame()
            seeds_season = seeds_all[seeds_all["Season"] == season]
            tids = list(seeds_season["TeamID"])
            try:
                mom = build_momentum_for_season(data, season, tids)
                mom_cache[season] = {row["TeamID"]: row for _, row in mom.iterrows()}
            except:
                mom_cache[season] = {}
            bart_path = DATA_DIR / "external" / (
                f"barttorvik_w_{season}.csv" if is_womens else f"barttorvik_{season}.csv"
            )
            if bart_path.exists():
                try:
                    bart_cache[season] = pd.read_csv(bart_path).set_index("TeamID")
                except:
                    bart_cache[season] = pd.DataFrame()

        for season in seasons:
            seeds_season = seeds_all[seeds_all["Season"] == season]
            seed_map = {int(r["TeamID"]): _parse_seed_num(r["Seed"])
                        for _, r in seeds_season.iterrows()}
            games = tourney[tourney["Season"] == season]
            eff = eff_cache.get(season, pd.DataFrame())
            ff = ff_cache.get(season, pd.DataFrame())
            mom = mom_cache.get(season, {})
            bart = bart_cache.get(season)

            for _, g in games.iterrows():
                w_, l_ = int(g["WTeamID"]), int(g["LTeamID"])
                ta, tb, y = (w_, l_, 1) if w_ < l_ else (l_, w_, 0)
                feat = {
                    "Season": season, "TeamA": ta, "TeamB": tb, "is_womens": is_womens,
                    "seed_diff": seed_map.get(ta, 17) - seed_map.get(tb, 17),
                    "win_pct_diff": (win_pct.get((season, ta), 0.5) -
                                     win_pct.get((season, tb), 0.5)),
                }
                # Efficiency / Four Factors (NaN if not in detail data)
                for col in ["off_eff", "def_eff", "net_eff", "tempo"]:
                    va = eff.loc[ta, col] if (not eff.empty and ta in eff.index) else np.nan
                    vb = eff.loc[tb, col] if (not eff.empty and tb in eff.index) else np.nan
                    feat[f"{col}_diff"] = va - vb
                for col in ["efg_pct", "to_pct", "or_pct", "ft_rate"]:
                    va = ff.loc[ta, col] if (not ff.empty and ta in ff.index) else np.nan
                    vb = ff.loc[tb, col] if (not ff.empty and tb in ff.index) else np.nan
                    feat[f"{col}_diff"] = va - vb
                # Barttorvik: NaN for pre-2015 seasons; pipeline will impute
                if bart is not None and not bart.empty:
                    for src, dst in [("AdjOE", "bart_adjoe_diff"),
                                     ("AdjDE", "bart_adjde_diff"),
                                     ("NetRtg", "bart_net_diff"),
                                     ("Barthag", "bart_barthag_diff")]:
                        va = bart.loc[ta, src] if ta in bart.index else np.nan
                        vb = bart.loc[tb, src] if tb in bart.index else np.nan
                        if hasattr(va, "iloc"): va = va.iloc[0] if len(va) > 0 else np.nan
                        if hasattr(vb, "iloc"): vb = vb.iloc[0] if len(vb) > 0 else np.nan
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

                ra = hr_lookup.get((season, ta))
                rb = hr_lookup.get((season, tb))
                feat["harry_diff"] = ((ra["harry_rating"] if ra is not None else 0) -
                                      (rb["harry_rating"] if rb is not None else 0))
                feat["opp_qlty_won_diff"] = ((ra["opp_qlty_pts_won"] if ra is not None else 0) -
                                              (rb["opp_qlty_pts_won"] if rb is not None else 0))
                ma_mom = mom.get(ta, {})
                mb_mom = mom.get(tb, {})
                feat["momentum_winpct_diff"] = (
                    (ma_mom.get("momentum_win_pct", 0.5) if hasattr(ma_mom, "get") else 0.5) -
                    (mb_mom.get("momentum_win_pct", 0.5) if hasattr(mb_mom, "get") else 0.5)
                )
                feat["momentum_margin_diff"] = (
                    (ma_mom.get("momentum_avg_margin", 0.0) if hasattr(ma_mom, "get") else 0.0) -
                    (mb_mom.get("momentum_avg_margin", 0.0) if hasattr(mb_mom, "get") else 0.0)
                )
                rows.append(feat)
                labels.append(y)
    X = pd.DataFrame(rows)
    y = np.array(labels)
    is_w = X["is_womens"].values
    return X, y, is_w


# ===========================================================================
# Models / pipelines
# ===========================================================================

def build_lr_pipeline(C=0.1):
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
    ])


def build_rff_pipeline(n_components=200, gamma=0.1, C=1.0, seed=42):
    """LR on Random Fourier Features (RBF kernel approximation)."""
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("rff", RBFSampler(gamma=gamma, n_components=n_components,
                            random_state=seed)),
        ("lr", LogisticRegression(C=C, max_iter=3000, solver="lbfgs")),
    ])


def loso_predict(X, y, season_arr, feats, pipeline_factory, gender_mask=None):
    """Generic LOSO; returns out-of-fold predictions and per-season Brier."""
    p_oof = np.zeros(len(X))
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        if te.sum() == 0:
            continue
        pipe = pipeline_factory()
        Xtr = X.loc[tr, feats].apply(pd.to_numeric, errors="coerce")
        Xte = X.loc[te, feats].apply(pd.to_numeric, errors="coerce")
        pipe.fit(Xtr, y[tr])
        p_oof[te] = pipe.predict_proba(Xte)[:, 1]
    if gender_mask is not None:
        return p_oof, brier_score_loss(y[gender_mask], p_oof[gender_mask])
    return p_oof, brier_score_loss(y, p_oof)


def shrink(p_model, seed_diff, lam):
    """p = (1-lam) * model + lam * seed_prior."""
    p_prior = seed_prior(seed_diff)
    return np.clip((1 - lam) * p_model + lam * p_prior, 0.005, 0.995)


# ===========================================================================
# Main experiment
# ===========================================================================

def main():
    print("Loading data...")
    data_m = load_all_mens_data()
    data_w = load_womens_data()

    seasons_short = [s for s in range(2014, 2026) if s != 2020]
    seasons_long = [s for s in range(2003, 2026) if s != 2020]
    print(f"  Short history: {len(seasons_short)} seasons (2014-2025)")
    print(f"  Long history:  {len(seasons_long)} seasons (2003-2025)")

    print("\n[1/2] Build short-history features (2014-2025)...")
    X_s, y_s, is_w_s = build_extended_features(data_m, data_w, seasons_short)
    print(f"  Total games: {len(X_s)}")

    print("\n[2/2] Build long-history features (2003-2025)...")
    X_l, y_l, is_w_l = build_extended_features(data_m, data_w, seasons_long)
    print(f"  Total games: {len(X_l)}")

    feats = [c for c in EXTENDED_FEATURES if c in X_s.columns]
    print(f"\n  Features: {len(feats)}")

    n_m_s = (is_w_s == 0).sum(); n_w_s = (is_w_s == 1).sum()

    rows = []

    # ===========================================================
    # Baseline: short-history LR
    # ===========================================================
    print(f"\n{'='*70}\n  Baseline: short-history LR (2014-2025)\n{'='*70}")
    p_oof_base = np.zeros(len(X_s))
    for is_g in [0, 1]:
        for s in np.unique(X_s.loc[is_w_s == is_g, "Season"]):
            mask_te = (X_s["Season"].values == s) & (is_w_s == is_g)
            mask_tr = (X_s["Season"].values != s) & (is_w_s == is_g)
            if mask_te.sum() == 0: continue
            feats_use = [c for c in feats if not (is_g == 1 and c.startswith("massey_"))]
            pipe = build_lr_pipeline(C=0.1)
            pipe.fit(X_s.loc[mask_tr, feats_use], y_s[mask_tr])
            p_oof_base[mask_te] = pipe.predict_proba(X_s.loc[mask_te, feats_use])[:, 1]
    bs_m_b = brier_score_loss(y_s[is_w_s == 0], p_oof_base[is_w_s == 0])
    bs_w_b = brier_score_loss(y_s[is_w_s == 1], p_oof_base[is_w_s == 1])
    bs_c_b = (bs_m_b * n_m_s + bs_w_b * n_w_s) / (n_m_s + n_w_s)
    print(f"  Baseline: men={bs_m_b:.4f}  women={bs_w_b:.4f}  combined={bs_c_b:.4f}")
    rows.append({"strategy": "baseline_short", "men": bs_m_b, "women": bs_w_b, "combined": bs_c_b})

    # ===========================================================
    # Lever 1: long history LR
    # ===========================================================
    print(f"\n{'='*70}\n  Lever 1: Long-history LR (2003-2025)\n{'='*70}")
    # LOSO on long history but evaluate Brier on short history's seasons (apples to apples)
    p_oof_long = np.zeros(len(X_s))
    season_long_arr = X_l["Season"].values
    season_short_arr = X_s["Season"].values
    is_w_long = X_l["is_womens"].values

    # Re-key OOF to match short-history rows
    long_oof_lookup = {}
    for is_g in [0, 1]:
        for s in seasons_short:
            mask_te = (X_l["Season"].values == s) & (is_w_long == is_g)
            mask_tr = (X_l["Season"].values != s) & (is_w_long == is_g)
            if mask_te.sum() == 0 or mask_tr.sum() == 0:
                continue
            feats_use = [c for c in feats if not (is_g == 1 and c.startswith("massey_"))]
            pipe = build_lr_pipeline(C=0.1)
            pipe.fit(X_l.loc[mask_tr, feats_use], y_l[mask_tr])
            p = pipe.predict_proba(X_l.loc[mask_te, feats_use])[:, 1]
            sub_l = X_l.loc[mask_te].reset_index(drop=True)
            for i, r in sub_l.iterrows():
                long_oof_lookup[(int(r["Season"]), int(r["TeamA"]), int(r["TeamB"]))] = float(p[i])

    p_oof_long = np.array([
        long_oof_lookup.get((int(r["Season"]), int(r["TeamA"]), int(r["TeamB"])), 0.5)
        for _, r in X_s.iterrows()
    ])
    bs_m_l = brier_score_loss(y_s[is_w_s == 0], p_oof_long[is_w_s == 0])
    bs_w_l = brier_score_loss(y_s[is_w_s == 1], p_oof_long[is_w_s == 1])
    bs_c_l = (bs_m_l * n_m_s + bs_w_l * n_w_s) / (n_m_s + n_w_s)
    print(f"  Long-history: men={bs_m_l:.4f}  women={bs_w_l:.4f}  combined={bs_c_l:.4f}")
    rows.append({"strategy": "long_history_lr", "men": bs_m_l, "women": bs_w_l, "combined": bs_c_l})

    # ===========================================================
    # Lever 2: RFF — automatic non-linear interactions
    # ===========================================================
    print(f"\n{'='*70}\n  Lever 2: RFF (automatic interactions)\n{'='*70}")
    # Grid over gamma and n_components
    best_rff = (None, np.inf)
    best_p = None
    for gamma in [0.01, 0.05, 0.1, 0.3, 1.0]:
        for n_comp in [100, 200, 400]:
            for C in [0.1, 1.0]:
                p_oof = np.zeros(len(X_s))
                for is_g in [0, 1]:
                    for s in seasons_short:
                        mask_te = (X_s["Season"].values == s) & (is_w_s == is_g)
                        mask_tr = (X_s["Season"].values != s) & (is_w_s == is_g)
                        if mask_te.sum() == 0: continue
                        feats_use = [c for c in feats if not (is_g == 1 and c.startswith("massey_"))]
                        pipe = build_rff_pipeline(n_components=n_comp, gamma=gamma, C=C)
                        pipe.fit(X_s.loc[mask_tr, feats_use], y_s[mask_tr])
                        p_oof[mask_te] = pipe.predict_proba(X_s.loc[mask_te, feats_use])[:, 1]
                bs_m = brier_score_loss(y_s[is_w_s == 0], p_oof[is_w_s == 0])
                bs_w = brier_score_loss(y_s[is_w_s == 1], p_oof[is_w_s == 1])
                bs_c = (bs_m * n_m_s + bs_w * n_w_s) / (n_m_s + n_w_s)
                rows.append({
                    "strategy": f"rff_g={gamma}_d={n_comp}_C={C}",
                    "men": bs_m, "women": bs_w, "combined": bs_c
                })
                print(f"  gamma={gamma:.2f} D={n_comp} C={C}: combined={bs_c:.4f}")
                if bs_c < best_rff[1]:
                    best_rff = ((gamma, n_comp, C), bs_c)
                    best_p = p_oof.copy()
    print(f"  Best RFF: {best_rff[0]} -> combined={best_rff[1]:.4f}")
    p_oof_rff = best_p

    # ===========================================================
    # Lever 3: Bayesian shrinkage on baseline LR
    # ===========================================================
    print(f"\n{'='*70}\n  Lever 3: Bayesian shrinkage toward seed prior\n{'='*70}")
    seed_diff_s = X_s["seed_diff"].values
    for lam in [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5]:
        p_oof_shrunk = shrink(p_oof_base, seed_diff_s, lam)
        bs_m = brier_score_loss(y_s[is_w_s == 0], p_oof_shrunk[is_w_s == 0])
        bs_w = brier_score_loss(y_s[is_w_s == 1], p_oof_shrunk[is_w_s == 1])
        bs_c = (bs_m * n_m_s + bs_w * n_w_s) / (n_m_s + n_w_s)
        rows.append({"strategy": f"shrink_lam={lam}",
                     "men": bs_m, "women": bs_w, "combined": bs_c})
        print(f"  lambda={lam}: combined={bs_c:.4f}")

    # ===========================================================
    # Combinations
    # ===========================================================
    print(f"\n{'='*70}\n  Combos: lever 1 + 2 + 3\n{'='*70}")
    # Long-history + shrink
    for lam in [0.0, 0.05, 0.1, 0.2]:
        p = shrink(p_oof_long, seed_diff_s, lam)
        bs_m = brier_score_loss(y_s[is_w_s == 0], p[is_w_s == 0])
        bs_w = brier_score_loss(y_s[is_w_s == 1], p[is_w_s == 1])
        bs_c = (bs_m * n_m_s + bs_w * n_w_s) / (n_m_s + n_w_s)
        rows.append({"strategy": f"long+shrink_{lam}", "men": bs_m, "women": bs_w, "combined": bs_c})
        print(f"  long+shrink lambda={lam}: combined={bs_c:.4f}")

    # RFF + shrink
    if p_oof_rff is not None:
        for lam in [0.0, 0.05, 0.1, 0.2]:
            p = shrink(p_oof_rff, seed_diff_s, lam)
            bs_m = brier_score_loss(y_s[is_w_s == 0], p[is_w_s == 0])
            bs_w = brier_score_loss(y_s[is_w_s == 1], p[is_w_s == 1])
            bs_c = (bs_m * n_m_s + bs_w * n_w_s) / (n_m_s + n_w_s)
            rows.append({"strategy": f"rff+shrink_{lam}", "men": bs_m, "women": bs_w, "combined": bs_c})
            print(f"  rff+shrink lambda={lam}: combined={bs_c:.4f}")

    # Long+RFF stacking via simple average
    if p_oof_rff is not None:
        for w in [0.3, 0.5, 0.7]:
            p_stack = w * p_oof_rff + (1 - w) * p_oof_long
            for lam in [0.0, 0.1, 0.2]:
                p = shrink(p_stack, seed_diff_s, lam)
                bs_m = brier_score_loss(y_s[is_w_s == 0], p[is_w_s == 0])
                bs_w = brier_score_loss(y_s[is_w_s == 1], p[is_w_s == 1])
                bs_c = (bs_m * n_m_s + bs_w * n_w_s) / (n_m_s + n_w_s)
                rows.append({"strategy": f"long+rff_{w}+shrink_{lam}",
                             "men": bs_m, "women": bs_w, "combined": bs_c})
                print(f"  long+RFF(w={w})+shrink({lam}): combined={bs_c:.4f}")

    # Save & print
    df = pd.DataFrame(rows).sort_values("combined")
    df.to_csv("output/robust_loso_grid.csv", index=False)
    print(f"\n{'='*70}\n  TOP 15 BY LOSO COMBINED BRIER\n{'='*70}")
    print(df.head(15).to_string(index=False))
    print(f"\n  Baseline: combined={bs_c_b:.4f}")
    print(f"  Best:     combined={df.iloc[0]['combined']:.4f}")
    print(f"  Improvement: {df.iloc[0]['combined'] - bs_c_b:+.4f}")


if __name__ == "__main__":
    main()
