"""
Combine ceiling + calibration findings into a paper-ready summary figure.

Panels:
  (a) Information ceiling convergence: men's vs women's
      bar chart of estimators (KMeans/RF/GBM/kNN/Logistic) showing
      linear logistic sits inside the model-class range.
  (b) Calibration evaluation artifact:
      uncalibrated vs in-sample isotonic vs honest LOSO isotonic.
      In-sample appears to improve; honest LOSO does not.
  (c) Futility regime: scatter of C(f̂) vs R_N for men's and women's,
      with futility line at C = R_N.
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.style.use("seaborn-v0_8-whitegrid")


def main():
    # === Panel A data ===
    mens = {
        "KMeans (K=80)":       0.1775,
        "Random Forest":       0.1929,
        "Gradient Boost":      0.2102,
        "k-NN":                0.1947,
        "Multi-Feat Logistic": 0.1893,
    }
    womens = {
        "KMeans (K=80)":       0.1465,
        "Random Forest":       0.1533,
        "Gradient Boost":      0.1613,
        "k-NN":                0.1661,
        "Multi-Feat Logistic": 0.1446,
    }

    # === Panel B data (men's + women's) ===
    cal_data = {
        "Men's": {"uncal": 0.1893, "in_sample": 0.1833, "loso": 0.1942},
        "Women's": {"uncal": 0.1446, "in_sample": 0.1370, "loso": 0.1455},
    }

    # === Panel C data ===
    futility = pd.read_csv("output/futility_test.csv")

    # === Plot ===
    fig = plt.figure(figsize=(15, 5))

    # Panel A
    ax = fig.add_subplot(1, 3, 1)
    methods = list(mens.keys())
    x = np.arange(len(methods))
    w = 0.38
    ax.bar(x - w/2, [mens[m] for m in methods], w, color="#2563eb", alpha=0.85, label="Men's")
    ax.bar(x + w/2, [womens[m] for m in methods], w, color="#16a34a", alpha=0.85, label="Women's")
    # Highlight logistic as "ceiling-touching"
    li = methods.index("Multi-Feat Logistic")
    ax.axvspan(li - 0.5, li + 0.5, alpha=0.08, color="red")
    ax.set_xticks(x)
    ax.set_xticklabels(methods, rotation=30, ha="right")
    ax.set_ylabel("Brier (LOTO / IID-CV)")
    ax.set_title("(a) Information ceiling: linear logistic\nmatches flexible models")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0.13, 0.22)

    # Panel B
    ax = fig.add_subplot(1, 3, 2)
    settings = ["Uncalibrated", "In-sample\nisotonic", "Honest LOSO\nisotonic"]
    x = np.arange(len(settings))
    w = 0.38
    mens_vals = [cal_data["Men's"]["uncal"], cal_data["Men's"]["in_sample"], cal_data["Men's"]["loso"]]
    womens_vals = [cal_data["Women's"]["uncal"], cal_data["Women's"]["in_sample"], cal_data["Women's"]["loso"]]
    ax.bar(x - w/2, mens_vals, w, color="#2563eb", alpha=0.85, label="Men's")
    ax.bar(x + w/2, womens_vals, w, color="#16a34a", alpha=0.85, label="Women's")
    ax.axhline(cal_data["Men's"]["uncal"], color="#2563eb", linestyle="--", alpha=0.4)
    ax.axhline(cal_data["Women's"]["uncal"], color="#16a34a", linestyle="--", alpha=0.4)
    # Annotation arrows
    ax.annotate("artifact", xy=(1, 0.165), xytext=(2.3, 0.15),
                fontsize=8, color="#7c2d12",
                arrowprops=dict(arrowstyle="->", color="#7c2d12"))
    ax.set_xticks(x)
    ax.set_xticklabels(settings)
    ax.set_ylabel("Brier")
    ax.set_title("(b) In-sample isotonic looks great,\nhonest LOSO does not improve")
    ax.legend(loc="upper right", fontsize=9)
    ax.set_ylim(0.12, 0.20)

    # Panel C
    ax = fig.add_subplot(1, 3, 3)
    Cf = futility["C_f"].values
    RN = futility["R_N"].values
    domains = futility["domain"].values
    colors = ["#2563eb", "#16a34a"]
    for i, dom in enumerate(domains):
        ax.errorbar(Cf[i], RN[i], yerr=futility["R_N_std"].values[i],
                    fmt="o", color=colors[i], markersize=14, capsize=4,
                    label=dom.capitalize())
        ax.annotate(f"  {dom}\n  ratio={futility['ratio'].values[i]:.2f}",
                    (Cf[i], RN[i]), fontsize=9, color=colors[i])
    lim = max(Cf.max(), RN.max()) * 1.4
    ax.plot([0, lim], [0, lim], "k--", alpha=0.5, label="$R_N = C(f)$")
    ax.fill_between([0, lim], [0, lim], [lim, lim], alpha=0.10, color="red",
                     label="Futility regime")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Calibration error $C(\\hat f)$")
    ax.set_ylabel("Isotonic estimation error $R_N$")
    ax.set_title("(c) Calibration futility test:\nmen's in futility regime")
    ax.legend(loc="lower right", fontsize=9)

    plt.tight_layout()
    plt.savefig("output/ceiling_calibration_summary.png", dpi=180, bbox_inches="tight")
    plt.savefig("output/ceiling_calibration_summary.pdf", bbox_inches="tight")
    print("Saved output/ceiling_calibration_summary.{png,pdf}")


if __name__ == "__main__":
    main()
