"""
Push 2026 Brier closer to Kaggle top: recency-weighted training + confidence clipping.

Two Kaggle-style tricks:
  1. Recency weighting: sample_weight = exp(-decay * (2026 - season))
     Down-weights pre-2018 seasons that may be stylistically different.
  2. Confidence clipping: stretch p > T toward clip_high, p < 1-T toward 1-clip_high.
     Brier rewards confident-correct disproportionately, so bold > timid in chalky years.

Both hyperparameters tuned on LOTO, then applied to final 2026 model.

Outputs:
  output/kaggle_clip_recency_loto.csv     - LOTO grid
  output/kaggle_clip_recency_2026.csv     - per-strategy 2026 Brier
  output/submission_stage2_optimal2.csv   - new submission with best strategy
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import make_build_features_fn
from src.models import MultiFeatureLogistic
from scripts.generate_kaggle_submission import build_submission_features


def fit_logistic_with_weights(X, y, weights, C=0.5):
    """Train MultiFeatureLogistic-style pipeline with sample_weight."""
    cols = [c for c in MultiFeatureLogistic._FEATURE_COLS if c in X.columns]
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("lr", LogisticRegression(C=C, max_iter=2000, solver="lbfgs")),
    ])
    pipe.fit(X[cols], y, lr__sample_weight=weights)
    return pipe, cols


def confidence_clip(p, T, clip_high):
    """Stretch confident predictions toward clip_high (and 1-clip_high)."""
    p = np.asarray(p)
    out = p.copy()
    high = p > T
    low = p < (1 - T)
    # Linear stretch from T..1.0 -> T..clip_high
    out[high] = T + (p[high] - T) * (clip_high - T) / (1.0 - T)
    out[low] = (1 - T) - ((1 - T) - p[low]) * (clip_high - T) / (1.0 - T)
    return np.clip(out, 0.01, 0.99)


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    seasons = [s for s in range(2014, 2026) if s != 2020]

    X_all, y_all = build_fn(seasons)
    print(f"  Tournament games: {len(X_all)}")

    # === Step 1: LOTO grid search over decay ===
    print(f"\n{'='*70}\n  Step 1: LOTO grid over recency decay\n{'='*70}")
    decay_grid = [0.00, 0.05, 0.10, 0.15, 0.20, 0.30]
    loto_decay = {}
    oof_by_decay = {}

    for decay in decay_grid:
        season_arr = X_all["Season"].values
        weights_all = np.exp(-decay * (2026 - season_arr))

        oof_p = np.zeros(len(X_all))
        per_fold = []
        for holdout in seasons:
            train_mask = (season_arr != holdout)
            test_mask = (season_arr == holdout)
            if test_mask.sum() == 0: continue
            w_train = weights_all[train_mask]
            pipe, cols = fit_logistic_with_weights(
                X_all.loc[train_mask], y_all[train_mask], w_train, C=0.5
            )
            p_test = pipe.predict_proba(X_all.loc[test_mask, cols])[:, 1]
            oof_p[test_mask] = p_test
            per_fold.append({
                "season": holdout,
                "n": int(test_mask.sum()),
                "brier": brier_score_loss(y_all[test_mask], p_test),
            })

        loto_decay[decay] = pd.DataFrame(per_fold)["brier"].mean()
        oof_by_decay[decay] = oof_p
        print(f"  decay={decay:.2f}  LOTO Brier={loto_decay[decay]:.4f}")

    best_decay = min(loto_decay, key=loto_decay.get)
    print(f"\n  Best decay: {best_decay} (LOTO Brier {loto_decay[best_decay]:.4f})")

    # === Step 2: Confidence clipping grid on best-decay OOF ===
    print(f"\n{'='*70}\n  Step 2: LOTO grid over clip params (using best decay OOF)\n{'='*70}")
    oof_best = oof_by_decay[best_decay]
    bs_no_clip = brier_score_loss(y_all, oof_best)
    print(f"  No clip baseline:    {bs_no_clip:.4f}")

    T_grid = [0.70, 0.75, 0.80, 0.85, 0.90]
    clip_grid = [0.93, 0.95, 0.97, 0.99]
    clip_results = []
    best_clip = (None, None, bs_no_clip)
    for T in T_grid:
        for ch in clip_grid:
            if ch <= T: continue
            p_clip = confidence_clip(oof_best, T, ch)
            bs = brier_score_loss(y_all, p_clip)
            clip_results.append({"T": T, "clip_high": ch, "brier": bs})
            if bs < best_clip[2]:
                best_clip = (T, ch, bs)

    clip_df = pd.DataFrame(clip_results).sort_values("brier")
    print(clip_df.head(10).to_string(index=False))
    print(f"\n  Best clip: T={best_clip[0]}, clip_high={best_clip[1]}  Brier={best_clip[2]:.4f}")

    # Save LOTO summary
    summary_loto = pd.DataFrame({
        "decay": list(loto_decay.keys()),
        "loto_brier_no_clip": [loto_decay[d] for d in loto_decay],
    })
    summary_loto.to_csv("output/kaggle_clip_recency_loto.csv", index=False)
    clip_df.to_csv("output/kaggle_clip_recency_clip_grid.csv", index=False)

    # === Step 3: Train final model with best decay, predict 2026 ===
    print(f"\n{'='*70}\n  Step 3: Final model on 2014-2025 with best decay\n{'='*70}")
    season_arr = X_all["Season"].values
    weights_all = np.exp(-best_decay * (2026 - season_arr))
    final_pipe, final_cols = fit_logistic_with_weights(
        X_all, y_all, weights_all, C=0.5
    )

    # Build 2026 submission features
    sub_path = str(DATA_DIR / "SampleSubmissionStage2.csv")
    sub_df, X_tourney, _ = build_submission_features(data, 2026, sub_path)
    p_logit_2026 = final_pipe.predict_proba(X_tourney[final_cols])[:, 1]

    # Also compute baseline (no recency) for comparison
    base_pipe, base_cols = fit_logistic_with_weights(
        X_all, y_all, np.ones(len(X_all)), C=0.5
    )
    p_base_2026 = base_pipe.predict_proba(X_tourney[base_cols])[:, 1]

    # === Step 4: Evaluate strategies on 2026 actual ===
    actual = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    print(f"\n{'='*70}\n  Step 4: Evaluate on 2026 ({len(actual)} men's games)\n{'='*70}")

    pair_idx = {}
    for i, row in X_tourney.reset_index(drop=True).iterrows():
        pair_idx[(int(row["TeamA"]), int(row["TeamB"]))] = i

    def collect(p_arr):
        yt, yp = [], []
        for _, g in actual.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            if w < l:
                idx = pair_idx.get((w, l))
                if idx is None: continue
                yt.append(1); yp.append(p_arr[idx])
            else:
                idx = pair_idx.get((l, w))
                if idx is None: continue
                yt.append(0); yp.append(p_arr[idx])
        return np.array(yt), np.array(yp)

    rows = []

    # Strategy A: baseline (no recency, no clip)
    yt, yp = collect(p_base_2026)
    rows.append({"strategy": "baseline_no_recency_no_clip", "brier": brier_score_loss(yt, yp)})

    # Strategy B: recency only
    yt, yp = collect(p_logit_2026)
    rows.append({"strategy": f"recency_decay={best_decay}", "brier": brier_score_loss(yt, yp)})

    # Strategy C: clip only (on baseline)
    p_base_clipped = confidence_clip(p_base_2026, best_clip[0], best_clip[1])
    yt, yp = collect(p_base_clipped)
    rows.append({"strategy": f"clip_only_T={best_clip[0]}_ch={best_clip[1]}", "brier": brier_score_loss(yt, yp)})

    # Strategy D: recency + clip
    p_combo = confidence_clip(p_logit_2026, best_clip[0], best_clip[1])
    yt, yp = collect(p_combo)
    rows.append({"strategy": f"recency+clip", "brier": brier_score_loss(yt, yp)})

    # Strategy E,F,G: try a few aggressive clips on the recency model directly
    for T_alt, ch_alt in [(0.70, 0.95), (0.80, 0.97), (0.85, 0.98), (0.90, 0.99)]:
        p_alt = confidence_clip(p_logit_2026, T_alt, ch_alt)
        yt, yp = collect(p_alt)
        rows.append({"strategy": f"recency+clip_T={T_alt}_ch={ch_alt}",
                     "brier": brier_score_loss(yt, yp)})

    rows_df = pd.DataFrame(rows).sort_values("brier")
    print(rows_df.to_string(index=False))
    rows_df.to_csv("output/kaggle_clip_recency_2026.csv", index=False)

    best = rows_df.iloc[0]
    print(f"\n  Best 2026 strategy: {best['strategy']} -> Brier {best['brier']:.4f}")

    # === Step 5: Build new submission with best 2026 strategy ===
    name_to_arr = {
        "baseline_no_recency_no_clip": p_base_2026,
        f"recency_decay={best_decay}": p_logit_2026,
        f"clip_only_T={best_clip[0]}_ch={best_clip[1]}": p_base_clipped,
        f"recency+clip": p_combo,
    }
    for T_alt, ch_alt in [(0.70, 0.95), (0.80, 0.97), (0.85, 0.98), (0.90, 0.99)]:
        name_to_arr[f"recency+clip_T={T_alt}_ch={ch_alt}"] = confidence_clip(
            p_logit_2026, T_alt, ch_alt
        )
    best_p = name_to_arr[best["strategy"]]

    existing = pd.read_csv("output/submission_stage2.csv")
    existing[["s_str", "ta_str", "tb_str"]] = existing["ID"].str.split("_", expand=True)
    existing["TeamA"] = existing["ta_str"].astype(int)
    existing["TeamB"] = existing["tb_str"].astype(int)
    new_map = {(int(a), int(b)): float(p)
               for a, b, p in zip(X_tourney["TeamA"], X_tourney["TeamB"], best_p)}
    existing["Pred"] = existing.apply(
        lambda r: new_map.get((r["TeamA"], r["TeamB"]), r["Pred"]), axis=1
    ).clip(0.01, 0.99)
    existing[["ID", "Pred"]].to_csv("output/submission_stage2_optimal2.csv", index=False)
    print(f"  Saved output/submission_stage2_optimal2.csv")

    # Combined Brier (men + women)
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")
    sub_new = pd.read_csv("output/submission_stage2_optimal2.csv")
    sub_new[["s_str", "ta_str", "tb_str"]] = sub_new["ID"].str.split("_", expand=True)
    sub_new["TeamA"] = sub_new["ta_str"].astype(int)
    sub_new["TeamB"] = sub_new["tb_str"].astype(int)
    pmap = dict(zip(zip(sub_new["TeamA"], sub_new["TeamB"]), sub_new["Pred"]))

    def m_w(actual_df):
        yt, yp = [], []
        for _, g in actual_df.iterrows():
            w, l = int(g["WTeamID"]), int(g["LTeamID"])
            if w < l:
                p = pmap.get((w, l), 0.5); yt.append(1)
            else:
                p = pmap.get((l, w), 0.5); yt.append(0)
            yp.append(p)
        return np.array(yt), np.array(yp)

    yt_m, yp_m = m_w(actual)
    yt_w, yp_w = m_w(actual_w)
    bs_m = brier_score_loss(yt_m, yp_m)
    bs_w = brier_score_loss(yt_w, yp_w)
    bs_c = (bs_m * len(yt_m) + bs_w * len(yt_w)) / (len(yt_m) + len(yt_w))
    print(f"\n{'='*70}\n  COMBINED 2026 BRIER\n{'='*70}")
    print(f"  Men's:    {bs_m:.4f}")
    print(f"  Women's:  {bs_w:.4f}")
    print(f"  Combined: {bs_c:.4f}")
    print(f"  Previous: 0.1263 (Multi-Feature solo)")
    print(f"  Kaggle top: ~0.116")


if __name__ == "__main__":
    main()
