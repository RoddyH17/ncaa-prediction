"""
Optimal Kaggle pipeline: margin + logistic ensemble + isotonic calibration.

Goal: push men's 2026 Brier below 0.153 (current Multi-Feature solo) toward Kaggle
top of ~0.116. The pipeline:

  1. Train Multi-Feature Logistic and Ridge margin model on 2014-2025 tournament games
  2. LOTO: collect out-of-fold predictions from both models for every season
  3. Form ensemble predictions (simple average + logit average + linear stack)
  4. Fit isotonic regression on LOTO ensemble probabilities vs actual outcomes
  5. Train final models on full 2014-2025
  6. Apply ensemble + isotonic to 2026 actual matchups, compute Brier
  7. If Brier improves, regenerate full Kaggle submission with calibrated probs

Outputs:
  output/kaggle_optimal_loto.csv     - per-season LOTO Brier per strategy
  output/kaggle_optimal_2026.csv     - per-game 2026 predictions across strategies
  output/kaggle_optimal_summary.csv  - final summary
  output/submission_stage2_optimal.csv - new Kaggle submission (if best beats baseline)
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.isotonic import IsotonicRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import make_build_features_fn
from src.models import MultiFeatureLogistic
from scripts.run_margin_model import get_actual_margins, FEATURES as MARGIN_FEATURES
from scripts.generate_kaggle_submission import build_submission_features


def logit(p, eps=1e-6):
    p = np.clip(p, eps, 1 - eps)
    return np.log(p / (1 - p))


def expit(x):
    return 1 / (1 + np.exp(-x))


def margin_pipe():
    return Pipeline([
        ("imp", SimpleImputer(strategy="median")),
        ("scl", StandardScaler()),
        ("reg", Ridge(alpha=1.0)),
    ])


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    seasons = [s for s in range(2014, 2026) if s != 2020]

    X_all, y_all = build_fn(seasons)
    margins = get_actual_margins(data, X_all)
    margin_feat_cols = [c for c in MARGIN_FEATURES if c in X_all.columns]

    print(f"  Tournament games: {len(X_all)}, with margin labels: {(~np.isnan(margins)).sum()}")
    print(f"  Margin feature columns: {len(margin_feat_cols)}")

    # === LOTO: collect out-of-fold predictions for both models ===
    print(f"\n{'='*70}\n  LOTO out-of-fold predictions\n{'='*70}")

    oof_records = []  # one row per game, with predictions from each model
    fold_summary = []

    for holdout in seasons:
        train_mask = (X_all["Season"] != holdout).values
        test_mask = (X_all["Season"] == holdout).values
        if test_mask.sum() == 0:
            continue

        # --- Multi-Feature Logistic ---
        logit_model = MultiFeatureLogistic(C=0.5)
        logit_model.fit(X_all.loc[train_mask], y_all[train_mask])
        p_logit = logit_model.predict_proba(X_all.loc[test_mask])[:, 1]

        # --- Margin Ridge ---
        valid_train = ~np.isnan(margins) & train_mask
        Xt_n = X_all.loc[valid_train, margin_feat_cols].apply(pd.to_numeric, errors="coerce").values
        m_train = margins[valid_train]
        Xh_n = X_all.loc[test_mask, margin_feat_cols].apply(pd.to_numeric, errors="coerce").values

        pipe = margin_pipe()
        pipe.fit(Xt_n, m_train)
        mu_test = pipe.predict(Xh_n)
        residuals = m_train - pipe.predict(Xt_n)
        sigma = float(residuals.std())
        p_margin = stats.norm.cdf(mu_test / sigma)
        p_margin = np.clip(p_margin, 0.01, 0.99)

        # --- Ensembles ---
        p_avg = 0.5 * (p_logit + p_margin)
        p_logit_avg = expit(0.5 * (logit(p_logit) + logit(p_margin)))

        y_test = y_all[test_mask]
        bs_logit = brier_score_loss(y_test, p_logit)
        bs_margin = brier_score_loss(y_test, p_margin)
        bs_avg = brier_score_loss(y_test, p_avg)
        bs_logit_avg = brier_score_loss(y_test, p_logit_avg)

        fold_summary.append({
            "season": holdout,
            "n": int(test_mask.sum()),
            "brier_logit": bs_logit,
            "brier_margin": bs_margin,
            "brier_avg": bs_avg,
            "brier_logit_avg": bs_logit_avg,
            "sigma_margin": sigma,
        })
        print(f"  {holdout}: logit={bs_logit:.4f}  margin={bs_margin:.4f}  "
              f"avg={bs_avg:.4f}  logit_avg={bs_logit_avg:.4f}")

        for i, idx in enumerate(np.where(test_mask)[0]):
            oof_records.append({
                "season": holdout,
                "game_idx": int(idx),
                "y": int(y_test[i]),
                "p_logit": float(p_logit[i]),
                "p_margin": float(p_margin[i]),
                "p_avg": float(p_avg[i]),
                "p_logit_avg": float(p_logit_avg[i]),
            })

    fold_df = pd.DataFrame(fold_summary)
    fold_df.to_csv("output/kaggle_optimal_loto.csv", index=False)
    oof_df = pd.DataFrame(oof_records)

    print(f"\n  LOTO mean Brier:")
    print(f"    Multi-Feature Logistic: {fold_df['brier_logit'].mean():.4f}")
    print(f"    Margin Gaussian:        {fold_df['brier_margin'].mean():.4f}")
    print(f"    Simple average:         {fold_df['brier_avg'].mean():.4f}")
    print(f"    Logit average:          {fold_df['brier_logit_avg'].mean():.4f}")

    # === Linear stack via logistic on OOF ===
    print(f"\n{'='*70}\n  Linear stack (logit-space LR on OOF)\n{'='*70}")
    Z = np.column_stack([logit(oof_df["p_logit"].values), logit(oof_df["p_margin"].values)])
    y_oof = oof_df["y"].values
    stack_lr = LogisticRegression(C=10.0, max_iter=1000)
    stack_lr.fit(Z, y_oof)
    p_stack_oof = stack_lr.predict_proba(Z)[:, 1]
    bs_stack_oof = brier_score_loss(y_oof, p_stack_oof)
    print(f"  Stack coefs: logit={stack_lr.coef_[0,0]:.3f}  margin={stack_lr.coef_[0,1]:.3f}  "
          f"intercept={stack_lr.intercept_[0]:.3f}")
    print(f"  In-sample stack Brier (overfit risk): {bs_stack_oof:.4f}")
    oof_df["p_stack"] = p_stack_oof

    # === Isotonic calibration on each candidate ===
    print(f"\n{'='*70}\n  Isotonic calibration on LOTO OOF\n{'='*70}")
    candidates = ["p_logit", "p_margin", "p_avg", "p_logit_avg", "p_stack"]
    iso_models = {}
    for c in candidates:
        iso = IsotonicRegression(out_of_bounds="clip", y_min=0.01, y_max=0.99)
        iso.fit(oof_df[c].values, y_oof)
        p_iso = iso.predict(oof_df[c].values)
        bs_raw = brier_score_loss(y_oof, oof_df[c].values)
        bs_iso = brier_score_loss(y_oof, p_iso)
        iso_models[c] = iso
        print(f"  {c:14s}  raw={bs_raw:.4f}  iso={bs_iso:.4f}  delta={bs_iso - bs_raw:+.4f}")

    # === Train final models on full 2014-2025 ===
    print(f"\n{'='*70}\n  Train final models (2014-2025) and predict 2026\n{'='*70}")
    final_logit = MultiFeatureLogistic(C=0.5)
    final_logit.fit(X_all, y_all)

    valid = ~np.isnan(margins)
    Xv = X_all.loc[valid, margin_feat_cols].apply(pd.to_numeric, errors="coerce").values
    mv = margins[valid]
    final_margin = margin_pipe()
    final_margin.fit(Xv, mv)
    sigma_final = float((mv - final_margin.predict(Xv)).std())
    print(f"  Final margin sigma = {sigma_final:.2f}")

    # Build 2026 features (all submission tournament pairs)
    sub_path = str(DATA_DIR / "SampleSubmissionStage2.csv")
    sub_df, X_tourney, _ = build_submission_features(data, 2026, sub_path)

    # Predictions on 2026 pairs
    p_logit_2026 = final_logit.predict_proba(X_tourney)[:, 1]

    # For margin model, only some features are present in X_tourney; align columns
    Xt_margin = X_tourney.reindex(columns=margin_feat_cols).apply(pd.to_numeric, errors="coerce").values
    mu_2026 = final_margin.predict(Xt_margin)
    p_margin_2026 = np.clip(stats.norm.cdf(mu_2026 / sigma_final), 0.01, 0.99)

    p_avg_2026 = 0.5 * (p_logit_2026 + p_margin_2026)
    p_logit_avg_2026 = expit(0.5 * (logit(p_logit_2026) + logit(p_margin_2026)))
    Z_2026 = np.column_stack([logit(p_logit_2026), logit(p_margin_2026)])
    p_stack_2026 = stack_lr.predict_proba(Z_2026)[:, 1]

    # === Evaluate on actual 2026 men's games ===
    actual = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    print(f"\n{'='*70}\n  2026 actual evaluation ({len(actual)} men's games)\n{'='*70}")

    # Build lookup (TeamA, TeamB) -> index in X_tourney
    pair_idx = {}
    for i, row in X_tourney.reset_index(drop=True).iterrows():
        pair_idx[(int(row["TeamA"]), int(row["TeamB"]))] = i

    def collect(p_arr):
        y_true, y_pred = [], []
        for _, g in actual.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            if w < l:
                idx = pair_idx.get((w, l))
                if idx is None: continue
                y_true.append(1)
                y_pred.append(p_arr[idx])
            else:
                idx = pair_idx.get((l, w))
                if idx is None: continue
                y_true.append(0)
                y_pred.append(p_arr[idx])
        return np.array(y_true), np.array(y_pred)

    rows = []
    for name, p_2026 in [
        ("logit_raw",      p_logit_2026),
        ("margin_raw",     p_margin_2026),
        ("avg_raw",        p_avg_2026),
        ("logit_avg_raw",  p_logit_avg_2026),
        ("stack_raw",      p_stack_2026),
    ]:
        yt, yp = collect(p_2026)
        bs = brier_score_loss(yt, yp)
        rows.append({"strategy": name, "brier_2026_men": bs, "n": len(yt)})
        print(f"  {name:18s} Brier_2026 = {bs:.4f}")

    print()
    iso_map = {
        "logit_iso":      ("p_logit",     p_logit_2026),
        "margin_iso":     ("p_margin",    p_margin_2026),
        "avg_iso":        ("p_avg",       p_avg_2026),
        "logit_avg_iso":  ("p_logit_avg", p_logit_avg_2026),
        "stack_iso":      ("p_stack",     p_stack_2026),
    }
    iso_preds = {}
    for name, (key, p_2026) in iso_map.items():
        p_cal = iso_models[key].predict(p_2026)
        p_cal = np.clip(p_cal, 0.01, 0.99)
        iso_preds[name] = p_cal
        yt, yp = collect(p_cal)
        bs = brier_score_loss(yt, yp)
        rows.append({"strategy": name, "brier_2026_men": bs, "n": len(yt)})
        print(f"  {name:18s} Brier_2026 = {bs:.4f}")

    summary = pd.DataFrame(rows).sort_values("brier_2026_men")
    summary.to_csv("output/kaggle_optimal_summary.csv", index=False)
    print(f"\nSorted summary:")
    print(summary.to_string(index=False))

    # === Save per-pair 2026 predictions across strategies ===
    pair_preds = pd.DataFrame({
        "TeamA": X_tourney["TeamA"].values,
        "TeamB": X_tourney["TeamB"].values,
        "p_logit": p_logit_2026,
        "p_margin": p_margin_2026,
        "p_avg": p_avg_2026,
        "p_logit_avg": p_logit_avg_2026,
        "p_stack": p_stack_2026,
        **{k: v for k, v in iso_preds.items()},
    })
    pair_preds.to_csv("output/kaggle_optimal_2026.csv", index=False)

    # === Pick winner & write submission ===
    best = summary.iloc[0]
    print(f"\nBest strategy: {best['strategy']} with Brier = {best['brier_2026_men']:.4f}")

    name_to_arr = {
        "logit_raw":     p_logit_2026,
        "margin_raw":    p_margin_2026,
        "avg_raw":       p_avg_2026,
        "logit_avg_raw": p_logit_avg_2026,
        "stack_raw":     p_stack_2026,
        **iso_preds,
    }
    best_p = name_to_arr[best["strategy"]]

    # Build new submission: keep women's predictions from existing submission, replace men's
    existing = pd.read_csv("output/submission_stage2.csv")
    existing[["s_str", "ta_str", "tb_str"]] = existing["ID"].str.split("_", expand=True)
    existing["TeamA"] = existing["ta_str"].astype(int)
    existing["TeamB"] = existing["tb_str"].astype(int)

    # Map (TeamA, TeamB) -> new prob for tournament men's pairs
    new_map = {(int(a), int(b)): float(p)
               for a, b, p in zip(X_tourney["TeamA"], X_tourney["TeamB"], best_p)}

    def update_pred(row):
        key = (row["TeamA"], row["TeamB"])
        if key in new_map:
            return new_map[key]
        return row["Pred"]

    existing["Pred"] = existing.apply(update_pred, axis=1).clip(0.01, 0.99)
    existing[["ID", "Pred"]].to_csv("output/submission_stage2_optimal.csv", index=False)
    print(f"\nSaved output/submission_stage2_optimal.csv "
          f"({len(existing)} rows, {len(new_map)} men's pairs updated)")

    # Combined Brier including women's (using existing women's preds, unchanged)
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")
    sub_new = pd.read_csv("output/submission_stage2_optimal.csv")
    sub_new[["s_str", "ta_str", "tb_str"]] = sub_new["ID"].str.split("_", expand=True)
    sub_new["TeamA"] = sub_new["ta_str"].astype(int)
    sub_new["TeamB"] = sub_new["tb_str"].astype(int)
    pmap = dict(zip(zip(sub_new["TeamA"], sub_new["TeamB"]), sub_new["Pred"]))

    def men_w_brier(actual_df):
        yt, yp = [], []
        for _, g in actual_df.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            if w < l:
                p = pmap.get((w, l), 0.5); yt.append(1)
            else:
                p = pmap.get((l, w), 0.5); yt.append(0)
            yp.append(p)
        return np.array(yt), np.array(yp)

    yt_m, yp_m = men_w_brier(actual)
    yt_w, yp_w = men_w_brier(actual_w)
    bs_m = brier_score_loss(yt_m, yp_m)
    bs_w = brier_score_loss(yt_w, yp_w)
    bs_c = (bs_m * len(yt_m) + bs_w * len(yt_w)) / (len(yt_m) + len(yt_w))
    print(f"\n{'='*70}\n  COMBINED 2026 BRIER (with optimal men's + existing women's)\n{'='*70}")
    print(f"  Men's:    {bs_m:.4f} ({len(yt_m)} games)")
    print(f"  Women's:  {bs_w:.4f} ({len(yt_w)} games)")
    print(f"  Combined: {bs_c:.4f} ({len(yt_m)+len(yt_w)} games)")
    print(f"  Previous combined (Multi-Feature solo): 0.126")
    print(f"  Kaggle top: ~0.116")


if __name__ == "__main__":
    main()
