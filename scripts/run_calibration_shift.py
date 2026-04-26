"""
Calibration under year-level distribution shift.

Quantifies the gap between LOTO calibration improvement and out-of-year
calibration degradation, then evaluates whether shift-aware calibrators close
the gap.

Procedure:
  1. Generate LOTO OOF predictions for 2014-2025 from Multi-Feature Logistic
  2. Train final logistic on 2014-2025; predict 2026 actual matchups
  3. For each calibrator (isotonic, Platt, temperature, robust variants):
       - LOTO inner-CV: fit on K-1 seasons, evaluate on Kth season; report mean Brier
       - Hold-out: fit on full LOTO OOF, evaluate on 2026; report Brier
       - Transfer gap = LOTO improvement - Hold-out improvement
  4. Plot reliability diagrams for each calibrator on hold-out year

Outputs:
  output/shift_calibration_results.csv
  output/shift_calibration_reliability.png
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import make_build_features_fn
from src.models import MultiFeatureLogistic
from scripts.generate_kaggle_submission import build_submission_features
from src.shift_calibration import (
    IsotonicCalibrator, PlattCalibrator, TemperatureCalibrator,
    LOSOIsotonicEnsemble, ShrinkageTemperature, HierarchicalIsotonic,
    evaluate_calibrator,
)


plt.style.use("seaborn-v0_8-whitegrid")


def reliability_curve(p, y, n_bins=10):
    """Return (bin_centers, observed_freq, n_per_bin)."""
    bins = np.linspace(0, 1, n_bins + 1)
    centers, obs, ns = [], [], []
    for i in range(n_bins):
        mask = (p >= bins[i]) & (p < bins[i + 1] if i < n_bins - 1 else p <= bins[i + 1])
        if mask.sum() > 0:
            centers.append((bins[i] + bins[i + 1]) / 2)
            obs.append(y[mask].mean())
            ns.append(mask.sum())
        else:
            centers.append((bins[i] + bins[i + 1]) / 2)
            obs.append(np.nan)
            ns.append(0)
    return np.array(centers), np.array(obs), np.array(ns)


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    seasons = [s for s in range(2014, 2026) if s != 2020]
    X_all, y_all = build_fn(seasons)
    season_arr = X_all["Season"].values

    # === Generate LOTO OOF predictions ===
    print(f"\n{'='*70}\n  Generating LOTO OOF predictions\n{'='*70}")
    p_oof = np.zeros(len(X_all))
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        m = MultiFeatureLogistic(C=0.5)
        m.fit(X_all.loc[tr], y_all[tr])
        p_oof[te] = m.predict_proba(X_all.loc[te])[:, 1]
    print(f"  LOTO OOF Brier (uncal) = {brier_score_loss(y_all, p_oof):.4f}")

    # === Generate 2026 hold-out predictions ===
    print(f"\n{'='*70}\n  Hold-out: 2026 actual\n{'='*70}")
    final = MultiFeatureLogistic(C=0.5)
    final.fit(X_all, y_all)

    sub_path = str(DATA_DIR / "SampleSubmissionStage2.csv")
    _, X_2026, _ = build_submission_features(data, 2026, sub_path)
    p_2026_all = final.predict_proba(X_2026)[:, 1]

    actual = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    pair_idx = {(int(r["TeamA"]), int(r["TeamB"])): i
                for i, r in X_2026.reset_index(drop=True).iterrows()}
    p_holdout, y_holdout = [], []
    for _, g in actual.iterrows():
        w, l = int(g["WTeamID"]), int(g["LTeamID"])
        if w < l:
            idx = pair_idx.get((w, l))
            if idx is None: continue
            p_holdout.append(p_2026_all[idx]); y_holdout.append(1)
        else:
            idx = pair_idx.get((l, w))
            if idx is None: continue
            p_holdout.append(p_2026_all[idx]); y_holdout.append(0)
    p_holdout = np.array(p_holdout)
    y_holdout = np.array(y_holdout)
    print(f"  Hold-out Brier (uncal) = {brier_score_loss(y_holdout, p_holdout):.4f}  "
          f"(N={len(y_holdout)})")

    # === Evaluate calibrators ===
    print(f"\n{'='*70}\n  Evaluating calibrators\n{'='*70}")
    factories = [
        ("isotonic",            lambda: IsotonicCalibrator()),
        ("platt",               lambda: PlattCalibrator()),
        ("temperature",         lambda: TemperatureCalibrator()),
        ("loso_isotonic_ens",   lambda: LOSOIsotonicEnsemble()),
        ("shrink_temperature",  lambda: ShrinkageTemperature()),
        ("hier_isotonic",       lambda: HierarchicalIsotonic()),
    ]
    results = []
    cal_preds_holdout = {}
    for name, factory in factories:
        out = evaluate_calibrator(factory, p_oof, y_all, season_arr, p_holdout, y_holdout)
        results.append(out)
        # Also store calibrated hold-out preds for reliability plot
        cal = factory()
        cal.fit(p_oof, y_all, season_arr)
        cal_preds_holdout[name] = cal.predict(p_holdout)
        print(f"\n  {name}:")
        print(f"    LOTO uncal -> cal:    {out['brier_oof_uncal']:.4f} -> "
              f"{out['brier_oof_cal']:.4f}  (improve {out['loto_improvement']:+.4f})")
        print(f"    Holdout uncal -> cal: {out['brier_holdout_uncal']:.4f} -> "
              f"{out['brier_holdout_cal']:.4f}  (change {out['holdout_change']:+.4f})")
        print(f"    Transfer gap:         {out['transfer_gap']:+.4f}")

    df = pd.DataFrame(results)
    df.to_csv("output/shift_calibration_results.csv", index=False)

    # === Reliability plots ===
    print(f"\n{'='*70}\n  Reliability diagrams (hold-out 2026)\n{'='*70}")
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()
    for ax, (name, _) in zip(axes, factories):
        c, o, n = reliability_curve(cal_preds_holdout[name], y_holdout, n_bins=8)
        valid = ~np.isnan(o)
        ax.plot([0, 1], [0, 1], "k--", alpha=0.4)
        ax.plot(c[valid], o[valid], "o-", color="#2563eb", label="calibrated")
        c0, o0, _ = reliability_curve(p_holdout, y_holdout, n_bins=8)
        valid0 = ~np.isnan(o0)
        ax.plot(c0[valid0], o0[valid0], "x-", color="#dc2626", alpha=0.6, label="uncal")
        bs_cal = brier_score_loss(y_holdout, cal_preds_holdout[name])
        ax.set_title(f"{name}\nholdout Brier {bs_cal:.4f}")
        ax.set_xlabel("Predicted P")
        ax.set_ylabel("Observed freq")
        ax.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig("output/shift_calibration_reliability.png", dpi=150, bbox_inches="tight")
    print("  Saved output/shift_calibration_reliability.png")

    # === Summary table ===
    print(f"\n{'='*70}\n  SUMMARY: Transfer gap by calibrator\n{'='*70}")
    df_sorted = df.sort_values("transfer_gap")
    print(df_sorted[["calibrator", "loto_improvement", "holdout_change",
                     "transfer_gap"]].to_string(index=False))


if __name__ == "__main__":
    main()
