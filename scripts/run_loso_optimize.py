"""
Systematic LOSO-only optimization for the strictly honest sports-only submission.

Pipeline (all decisions made BEFORE looking at 2026 actual):

  Step 1: Backward feature elimination per gender on top3 LR features.
          Greedily remove the feature whose removal most improves LOSO Brier
          until no further improvement.

  Step 2: C grid search on the pruned feature set per gender.

  Step 3: Combined M+W training trial — does training on both genders together
          beat per-gender training on the pruned features?

  Step 4: Logit-space stacking — does a learned stack on logits beat convex
          blending of (top3 LR pruned, MultiFeat, XGB+harry)?

  Step 5: Final blend weight search on LOSO using best top3 LR variant.

All hyperparameter / feature selection decisions are recorded; we then evaluate
on 2026 ONCE and report.

Outputs:
  output/loso_opt_pruning.csv
  output/loso_opt_C_grid.csv
  output/loso_opt_stack.csv
  output/loso_opt_summary.csv
  output/submission_stage2_optimized.csv
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
from scripts.run_top3 import build_combined_features, FEATURE_COLS as TOP3_FEATURES_ALL
from scripts.run_harry_xgb import (
    build_matchup_features, train_xgb_loto, train_xgb_final,
    HPARAMS_MEN, HPARAMS_WOM,
)
from src.harry_rating import build_harry_features


def loso_brier_lr(X, y, season_arr, feature_cols, C=1.0):
    """Compute LOSO Brier for LR with given features and C."""
    p_oof = np.zeros(len(X))
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        if te.sum() == 0:
            continue
        Xtr = X.loc[tr, feature_cols].apply(pd.to_numeric, errors="coerce")
        Xte = X.loc[te, feature_cols].apply(pd.to_numeric, errors="coerce")
        pipe = Pipeline([
            ("imp", SimpleImputer(strategy="median")),
            ("scl", StandardScaler()),
            ("lr", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
        ])
        pipe.fit(Xtr, y[tr])
        p_oof[te] = pipe.predict_proba(Xte)[:, 1]
    return p_oof, brier_score_loss(y, p_oof)


def backward_eliminate(X, y, season_arr, feature_cols, C, min_features=3, verbose=True):
    """Greedy backward elimination minimizing LOSO Brier."""
    current = list(feature_cols)
    _, best_bs = loso_brier_lr(X, y, season_arr, current, C=C)
    history = [{"step": 0, "n_features": len(current), "brier": best_bs,
                "removed": None, "features": list(current)}]
    if verbose:
        print(f"  Start: {len(current)} features, LOSO Brier = {best_bs:.4f}")

    step = 0
    while len(current) > min_features:
        step += 1
        best_drop = None
        best_drop_bs = best_bs
        for f in current:
            trial = [c for c in current if c != f]
            _, bs = loso_brier_lr(X, y, season_arr, trial, C=C)
            if bs < best_drop_bs:
                best_drop_bs = bs
                best_drop = f
        if best_drop is None:
            if verbose:
                print(f"  Step {step}: no improvement found; stop")
            break
        current.remove(best_drop)
        best_bs = best_drop_bs
        history.append({"step": step, "n_features": len(current),
                        "brier": best_bs, "removed": best_drop,
                        "features": list(current)})
        if verbose:
            print(f"  Step {step}: drop {best_drop:30s}  "
                  f"-> {len(current)} features, Brier {best_bs:.4f}")
    return current, best_bs, history


def main():
    seasons = [s for s in range(2014, 2026) if s != 2020]
    print("Loading data...")
    data_m = load_all_mens_data()
    data_w = load_womens_data()

    print("\nBuilding features (top3 extended set)...")
    X, y, is_w = build_combined_features(data_m, data_w, seasons)
    season_arr = X["Season"].values

    # Available features (per gender — women's gets defaults for missing massey)
    avail = [c for c in TOP3_FEATURES_ALL if c in X.columns]
    print(f"  Available top3 features: {len(avail)}")
    print(f"    {avail}")

    # ============================================================
    # Step 1: Backward feature elimination per gender
    # ============================================================
    print(f"\n{'='*70}\n  Step 1: Backward feature elimination\n{'='*70}")

    print("\n[men's]")
    X_m = X[is_w == 0].reset_index(drop=True)
    y_m = y[is_w == 0]
    season_m = X_m["Season"].values
    feats_m, bs_m_pruned, hist_m = backward_eliminate(
        X_m, y_m, season_m, avail, C=0.5, min_features=3
    )

    print("\n[women's]")
    # Women's feature set — drop massey (always 0 for women's, useless)
    avail_w = [c for c in avail if not c.startswith("massey_")]
    X_w = X[is_w == 1].reset_index(drop=True)
    y_w = y[is_w == 1]
    season_w = X_w["Season"].values
    feats_w, bs_w_pruned, hist_w = backward_eliminate(
        X_w, y_w, season_w, avail_w, C=0.1, min_features=3
    )

    pd.DataFrame(hist_m).assign(gender="M").to_csv("output/loso_opt_pruning_men.csv", index=False)
    pd.DataFrame(hist_w).assign(gender="W").to_csv("output/loso_opt_pruning_women.csv", index=False)

    # ============================================================
    # Step 2: C grid search post-pruning
    # ============================================================
    print(f"\n{'='*70}\n  Step 2: C grid on pruned features\n{'='*70}")
    rows_C = []

    print("[men's]")
    best_C_m, best_bs_m = 0.5, bs_m_pruned
    for C in [0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0, 30.0, 100.0]:
        _, bs = loso_brier_lr(X_m, y_m, season_m, feats_m, C=C)
        rows_C.append({"gender": "M", "C": C, "brier": bs})
        print(f"  C={C:>6.2f}: {bs:.4f}")
        if bs < best_bs_m:
            best_bs_m = bs; best_C_m = C
    print(f"  Best: C={best_C_m}, Brier={best_bs_m:.4f}")

    print("\n[women's]")
    best_C_w, best_bs_w = 0.1, bs_w_pruned
    for C in [0.01, 0.05, 0.1, 0.3, 0.5, 1.0, 3.0, 10.0]:
        _, bs = loso_brier_lr(X_w, y_w, season_w, feats_w, C=C)
        rows_C.append({"gender": "W", "C": C, "brier": bs})
        print(f"  C={C:>6.2f}: {bs:.4f}")
        if bs < best_bs_w:
            best_bs_w = bs; best_C_w = C
    print(f"  Best: C={best_C_w}, Brier={best_bs_w:.4f}")
    pd.DataFrame(rows_C).to_csv("output/loso_opt_C_grid.csv", index=False)

    # ============================================================
    # Step 3: Final OOF predictions for top3 pruned per gender
    # ============================================================
    print(f"\n{'='*70}\n  Step 3: Generate final OOF for blending\n{'='*70}")
    p_oof_top3_m, _ = loso_brier_lr(X_m, y_m, season_m, feats_m, C=best_C_m)
    p_oof_top3_w, _ = loso_brier_lr(X_w, y_w, season_w, feats_w, C=best_C_w)
    print(f"  Men's pruned LR LOSO Brier:    {brier_score_loss(y_m, p_oof_top3_m):.4f}")
    print(f"  Women's pruned LR LOSO Brier:  {brier_score_loss(y_w, p_oof_top3_w):.4f}")

    # ============================================================
    # Step 4: MultiFeat + XGB OOF for blending
    # ============================================================
    print(f"\n{'='*70}\n  Step 4: MultiFeat + XGB+harry OOF\n{'='*70}")
    build_fn_m = make_build_features_fn(data_m)
    X_mf_m, y_mf_m = build_fn_m(seasons)
    season_mf_m = X_mf_m["Season"].values
    p_oof_mf_m = np.zeros(len(X_mf_m))
    for s in np.unique(season_mf_m):
        tr = season_mf_m != s; te = season_mf_m == s
        m = MultiFeatureLogistic(C=0.5).fit(X_mf_m.loc[tr], y_mf_m[tr])
        p_oof_mf_m[te] = m.predict_proba(X_mf_m.loc[te])[:, 1]

    X_mf_w, y_mf_w = build_womens_features(data_w, seasons)
    season_mf_w = X_mf_w["Season"].values
    p_oof_mf_w = np.zeros(len(X_mf_w))
    for s in np.unique(season_mf_w):
        tr = season_mf_w != s; te = season_mf_w == s
        m = WomensLogistic(C=0.5).fit(X_mf_w.loc[tr], y_mf_w[tr])
        p_oof_mf_w[te] = m.predict_proba(X_mf_w.loc[te])[:, 1]
    print(f"  MultiFeat M LOSO: {brier_score_loss(y_mf_m, p_oof_mf_m):.4f}")
    print(f"  MultiFeat W LOSO: {brier_score_loss(y_mf_w, p_oof_mf_w):.4f}")

    hr_m = build_harry_features(data_m, seasons + [2026], is_womens=False)
    hr_w = build_harry_features(data_w, seasons + [2026], is_womens=True)
    X_xgb_m, y_xgb_m = build_matchup_features(data_m, seasons, is_womens=False, hr=hr_m)
    X_xgb_w, y_xgb_w = build_matchup_features(data_w, seasons, is_womens=True, hr=hr_w)
    p_oof_xgb_m, _ = train_xgb_loto(X_xgb_m, y_xgb_m, HPARAMS_MEN)
    p_oof_xgb_w, _ = train_xgb_loto(X_xgb_w, y_xgb_w, HPARAMS_WOM)
    print(f"  XGB+harry M LOSO: {brier_score_loss(y_xgb_m, p_oof_xgb_m):.4f}")
    print(f"  XGB+harry W LOSO: {brier_score_loss(y_xgb_w, p_oof_xgb_w):.4f}")

    # Align to top3 row order via (Season, TeamA, TeamB)
    def index_by(X, p):
        return {(int(r["Season"]), int(r["TeamA"]), int(r["TeamB"])): float(p[i])
                for i, r in X.reset_index(drop=True).iterrows()}

    top3_m_oof = index_by(X_m, p_oof_top3_m)
    top3_w_oof = index_by(X_w, p_oof_top3_w)
    mf_m_oof = index_by(X_mf_m, p_oof_mf_m)
    mf_w_oof = index_by(X_mf_w, p_oof_mf_w)
    xgb_m_oof = index_by(X_xgb_m, p_oof_xgb_m)
    xgb_w_oof = index_by(X_xgb_w, p_oof_xgb_w)

    keys_m = sorted(set(top3_m_oof) & set(mf_m_oof) & set(xgb_m_oof))
    keys_w = sorted(set(top3_w_oof) & set(mf_w_oof) & set(xgb_w_oof))

    P_m = np.array([[top3_m_oof[k], mf_m_oof[k], xgb_m_oof[k]] for k in keys_m])
    P_w = np.array([[top3_w_oof[k], mf_w_oof[k], xgb_w_oof[k]] for k in keys_w])
    y_m_aligned = np.array([y_m[i] for i, k in enumerate([
        (int(r["Season"]), int(r["TeamA"]), int(r["TeamB"])) for _, r in X_m.iterrows()
    ]) if k in set(keys_m)])
    # Easier: rebuild via dict
    y_m_aligned = []
    y_m_lookup = {(int(r["Season"]), int(r["TeamA"]), int(r["TeamB"])): y_m[i]
                  for i, r in X_m.reset_index(drop=True).iterrows()}
    for k in keys_m:
        y_m_aligned.append(y_m_lookup[k])
    y_m_aligned = np.array(y_m_aligned)
    y_w_lookup = {(int(r["Season"]), int(r["TeamA"]), int(r["TeamB"])): y_w[i]
                  for i, r in X_w.reset_index(drop=True).iterrows()}
    y_w_aligned = np.array([y_w_lookup[k] for k in keys_w])

    # ============================================================
    # Step 5: Convex blend search
    # ============================================================
    print(f"\n{'='*70}\n  Step 5: Convex blend grid\n{'='*70}")
    def best_blend(P, y):
        best = (None, np.inf)
        ws = np.arange(0, 1.001, 0.05)
        for w1 in ws:
            for w2 in ws:
                w3 = 1 - w1 - w2
                if w3 < -1e-9 or w3 > 1 + 1e-9: continue
                p = w1 * P[:, 0] + w2 * P[:, 1] + w3 * P[:, 2]
                p = np.clip(p, 0.005, 0.995)
                bs = brier_score_loss(y, p)
                if bs < best[1]:
                    best = ((round(w1,2), round(w2,2), round(w3,2)), bs)
        return best

    blend_m = best_blend(P_m, y_m_aligned)
    blend_w = best_blend(P_w, y_w_aligned)
    print(f"  Men's:   weights={blend_m[0]} (top3, mf, xgb)  LOSO Brier={blend_m[1]:.4f}")
    print(f"  Women's: weights={blend_w[0]} (top3, mf, xgb)  LOSO Brier={blend_w[1]:.4f}")

    # ============================================================
    # Step 6: Logit-space stacking (LR on logit features)
    # ============================================================
    print(f"\n{'='*70}\n  Step 6: Logit-space stack with LR\n{'='*70}")
    def logit(p, eps=1e-6):
        p = np.clip(p, eps, 1-eps)
        return np.log(p / (1-p))

    def fit_stack_loso(P, y, season_arr_aligned, C=1.0):
        """LOSO eval of logit-space LR stack."""
        Z = logit(P)
        p_oof = np.zeros(len(y))
        for s in np.unique(season_arr_aligned):
            tr = season_arr_aligned != s
            te = season_arr_aligned == s
            if te.sum() == 0: continue
            lr = LogisticRegression(C=C, max_iter=1000)
            lr.fit(Z[tr], y[tr])
            p_oof[te] = lr.predict_proba(Z[te])[:, 1]
        return p_oof, brier_score_loss(y, p_oof)

    season_m_aligned = np.array([k[0] for k in keys_m])
    season_w_aligned = np.array([k[0] for k in keys_w])

    print("  Men's stack C grid:")
    best_stack_m_C, best_stack_m_bs = None, np.inf
    for C in [0.1, 1.0, 10.0, 100.0]:
        _, bs = fit_stack_loso(P_m, y_m_aligned, season_m_aligned, C=C)
        print(f"    C={C}: {bs:.4f}")
        if bs < best_stack_m_bs:
            best_stack_m_bs, best_stack_m_C = bs, C
    print(f"    Best: C={best_stack_m_C}, Brier={best_stack_m_bs:.4f}")

    print("  Women's stack C grid:")
    best_stack_w_C, best_stack_w_bs = None, np.inf
    for C in [0.1, 1.0, 10.0, 100.0]:
        _, bs = fit_stack_loso(P_w, y_w_aligned, season_w_aligned, C=C)
        print(f"    C={C}: {bs:.4f}")
        if bs < best_stack_w_bs:
            best_stack_w_bs, best_stack_w_C = bs, C
    print(f"    Best: C={best_stack_w_C}, Brier={best_stack_w_bs:.4f}")

    # ============================================================
    # Final selection (LOSO-best per gender)
    # ============================================================
    print(f"\n{'='*70}\n  Final selection summary (LOSO Brier)\n{'='*70}")
    # Per gender, compare: pruned_LR_alone, blend, logit_stack
    m_options = {
        "pruned_LR_only": best_bs_m,
        "convex_blend": blend_m[1],
        "logit_stack": best_stack_m_bs,
    }
    w_options = {
        "pruned_LR_only": best_bs_w,
        "convex_blend": blend_w[1],
        "logit_stack": best_stack_w_bs,
    }
    print("\n  Men's:")
    for name, bs in m_options.items():
        print(f"    {name:15s}: {bs:.4f}")
    print("  Women's:")
    for name, bs in w_options.items():
        print(f"    {name:15s}: {bs:.4f}")

    best_m_name = min(m_options, key=m_options.get)
    best_w_name = min(w_options, key=w_options.get)
    print(f"\n  -> LOSO-best Men's:   {best_m_name} (Brier {m_options[best_m_name]:.4f})")
    print(f"  -> LOSO-best Women's: {best_w_name} (Brier {w_options[best_w_name]:.4f})")

    # Save state for next-step submission generation
    summary = {
        "best_C_m": best_C_m, "best_C_w": best_C_w,
        "feats_m": ",".join(feats_m), "feats_w": ",".join(feats_w),
        "blend_m": str(blend_m[0]), "blend_w": str(blend_w[0]),
        "stack_m_C": best_stack_m_C, "stack_w_C": best_stack_w_C,
        "loso_pruned_m": best_bs_m, "loso_pruned_w": best_bs_w,
        "loso_blend_m": blend_m[1], "loso_blend_w": blend_w[1],
        "loso_stack_m": best_stack_m_bs, "loso_stack_w": best_stack_w_bs,
        "best_m_name": best_m_name, "best_w_name": best_w_name,
    }
    pd.DataFrame([summary]).to_csv("output/loso_opt_summary.csv", index=False)
    print(f"\n  Saved output/loso_opt_summary.csv")

    n_m_loso = (is_w == 0).sum()
    n_w_loso = (is_w == 1).sum()
    best_combined_loso = (m_options[best_m_name] * n_m_loso + w_options[best_w_name] * n_w_loso) / (n_m_loso + n_w_loso)
    print(f"\n  Final LOSO Combined Brier: {best_combined_loso:.4f}")
    print(f"  (will apply same recipe to 2026 in next script)")


if __name__ == "__main__":
    main()
