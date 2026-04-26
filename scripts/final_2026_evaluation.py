"""
Final 2026 evaluation: men's + women's, leak-free, with Bayesian market calibration.

Generates the comprehensive comparison table for the final paper.
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import brier_score_loss

from src.data_collection import DATA_DIR
from src.pipeline import _parse_seed_num
from src.models import MarketCalibratedEnsemble


def load_predictions_and_actuals():
    """Load men's + women's predictions and actuals."""
    sub = pd.read_csv("output/submission_stage2.csv")
    sub[["s", "ta", "tb"]] = sub["ID"].str.split("_", expand=True)
    sub["ta"] = sub["ta"].astype(int)
    sub["tb"] = sub["tb"].astype(int)
    pred_map = dict(zip(zip(sub["ta"], sub["tb"]), sub["Pred"]))

    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")
    return pred_map, actual_m, actual_w


def collect_y_true_pred(actual_df, pred_map):
    y_true, y_pred = [], []
    for _, g in actual_df.iterrows():
        w, l = g["WTeamID"], g["LTeamID"]
        if w < l:
            p = pred_map.get((w, l), 0.5)
            y_true.append(1)
        else:
            p = pred_map.get((l, w), 0.5)
            y_true.append(0)
        y_pred.append(p)
    return np.array(y_true), np.array(y_pred)


def seed_implied_prob(diff: int) -> float:
    """Crude logistic mapping from seed difference to win probability."""
    return 1 / (1 + np.exp(0.13 * diff))


def main():
    pred_map, actual_m, actual_w = load_predictions_and_actuals()
    print(f"Men's actual games: {len(actual_m)}")
    print(f"Women's actual games: {len(actual_w)}")

    # Load 2026 seeds
    seeds_m = pd.read_csv(DATA_DIR / "MNCAATourneySeeds.csv")
    seeds_w = pd.read_csv(DATA_DIR / "WNCAATourneySeeds.csv")
    sd_m = dict(zip(seeds_m[seeds_m["Season"] == 2026]["TeamID"],
                    seeds_m[seeds_m["Season"] == 2026]["Seed"].apply(_parse_seed_num)))
    sd_w = dict(zip(seeds_w[seeds_w["Season"] == 2026]["TeamID"],
                    seeds_w[seeds_w["Season"] == 2026]["Seed"].apply(_parse_seed_num)))

    # === Pure model predictions ===
    yt_m, yp_m = collect_y_true_pred(actual_m, pred_map)
    yt_w, yp_w = collect_y_true_pred(actual_w, pred_map)

    bs_m = brier_score_loss(yt_m, yp_m)
    bs_w = brier_score_loss(yt_w, yp_w)
    bs_combined = (bs_m * len(yt_m) + bs_w * len(yt_w)) / (len(yt_m) + len(yt_w))

    print(f"\n{'='*60}\n  PURE MODEL PREDICTIONS (LEAK-FREE)\n{'='*60}")
    print(f"  Men's Brier:     {bs_m:.4f} ({len(yt_m)} games)")
    print(f"  Women's Brier:   {bs_w:.4f} ({len(yt_w)} games)")
    print(f"  Combined Brier:  {bs_combined:.4f} ({len(yt_m)+len(yt_w)} games)")
    print(f"  Vegas/Markets benchmark (men's): 0.1536")

    # === Seed baseline ===
    yp_seed_m = []
    for _, g in actual_m.iterrows():
        w, l = g["WTeamID"], g["LTeamID"]
        sw, sl = sd_m.get(w, 16), sd_m.get(l, 16)
        diff = sw - sl if w < l else sl - sw
        yp_seed_m.append(seed_implied_prob(diff))
    bs_seed_m = brier_score_loss(yt_m, np.array(yp_seed_m))

    yp_seed_w = []
    for _, g in actual_w.iterrows():
        w, l = g["WTeamID"], g["LTeamID"]
        sw, sl = sd_w.get(w, 16), sd_w.get(l, 16)
        diff = sw - sl if w < l else sl - sw
        yp_seed_w.append(seed_implied_prob(diff))
    bs_seed_w = brier_score_loss(yt_w, np.array(yp_seed_w))

    print(f"\n{'='*60}\n  SEED-ONLY BASELINE\n{'='*60}")
    print(f"  Men's seed Brier:   {bs_seed_m:.4f}")
    print(f"  Women's seed Brier: {bs_seed_w:.4f}")

    # === Bayesian market calibration with seed-implied prior ===
    print(f"\n{'='*60}\n  BAYESIAN CALIBRATION WITH SEED PRIOR\n{'='*60}")
    print(f"  alpha = trust in model (1.0 = pure model, 0.0 = pure market)")

    rows = []
    for alpha in [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        cal = MarketCalibratedEnsemble(alpha=alpha)
        # Men's
        p_cal_m = cal.calibrate(yp_m, np.array(yp_seed_m))
        bs_cal_m = brier_score_loss(yt_m, p_cal_m)
        # Women's
        p_cal_w = cal.calibrate(yp_w, np.array(yp_seed_w))
        bs_cal_w = brier_score_loss(yt_w, p_cal_w)
        bs_cal_c = (bs_cal_m * len(yt_m) + bs_cal_w * len(yt_w)) / (len(yt_m) + len(yt_w))

        rows.append({"alpha": alpha,
                     "men_brier": bs_cal_m,
                     "women_brier": bs_cal_w,
                     "combined": bs_cal_c})
        print(f"  alpha={alpha:.1f}  men={bs_cal_m:.4f}  women={bs_cal_w:.4f}  combined={bs_cal_c:.4f}")

    # Summary table
    summary_rows = [
        {"strategy": "Seed only", "men": bs_seed_m, "women": bs_seed_w,
         "combined": (bs_seed_m * len(yt_m) + bs_seed_w * len(yt_w)) / (len(yt_m) + len(yt_w))},
        {"strategy": "Multi-Feature (pure)", "men": bs_m, "women": bs_w, "combined": bs_combined},
    ]
    for r in rows:
        if r["alpha"] in [0.7, 0.8]:
            summary_rows.append({"strategy": f"Bayesian fusion alpha={r['alpha']:.1f}",
                                 "men": r["men_brier"], "women": r["women_brier"],
                                 "combined": r["combined"]})

    print(f"\n{'='*60}\n  FINAL SUMMARY (2026 ACTUAL RESULTS)\n{'='*60}")
    sf = pd.DataFrame(summary_rows)
    print(sf.to_string(index=False))
    sf.to_csv("output/eval_2026_final.csv", index=False)
    pd.DataFrame(rows).to_csv("output/calibration_alpha_sweep.csv", index=False)

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    plt.style.use("seaborn-v0_8-whitegrid")

    # Bar chart: model vs seed vs vegas
    ax = axes[0]
    methods = ["Seed only", "Multi-Feature\nLogistic", "Bayesian\nfusion (α=0.7)"]
    men_vals = [bs_seed_m, bs_m, rows[2]["men_brier"]]
    women_vals = [bs_seed_w, bs_w, rows[2]["women_brier"]]
    x = np.arange(len(methods))
    w = 0.35
    ax.bar(x - w/2, men_vals, w, label="Men's", color="#2563eb", alpha=0.85)
    ax.bar(x + w/2, women_vals, w, label="Women's", color="#16a34a", alpha=0.85)
    ax.axhline(y=0.1536, color="red", linestyle="--", alpha=0.6, label="Vegas (men's)")
    ax.set_xticks(x)
    ax.set_xticklabels(methods)
    ax.set_ylabel("Brier Score (lower is better)")
    ax.set_title("2026 Actual Tournament Brier Scores")
    ax.legend()

    # Alpha sweep
    ax = axes[1]
    alphas = [r["alpha"] for r in rows]
    ax.plot(alphas, [r["men_brier"] for r in rows], "o-", label="Men's", color="#2563eb")
    ax.plot(alphas, [r["women_brier"] for r in rows], "s-", label="Women's", color="#16a34a")
    ax.plot(alphas, [r["combined"] for r in rows], "^-", label="Combined", color="#7c3aed")
    ax.set_xlabel("alpha (trust in model)")
    ax.set_ylabel("Brier on 2026 actual")
    ax.set_title("Bayesian Calibration Alpha Sweep")
    ax.legend()

    plt.tight_layout()
    plt.savefig("output/eval_2026_final.png", dpi=150, bbox_inches="tight")
    print("\nSaved eval_2026_final.png")


if __name__ == "__main__":
    main()
