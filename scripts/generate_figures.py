"""
Generate key EDA figures for the project update paper.

Outputs (saved to output/):
  - seed_win_rate.png: Bar chart of tournament win rate by seed (1-16), 2003-2026
  - feature_target_correlation.png: Horizontal bar chart of diff feature correlations with outcome
  - rating_correlation.png: Heatmap of inter-correlation between rating systems (2025 season)
"""

import sys
from pathlib import Path

# Ensure project root is on the path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from src.data_collection import load_all_mens_data
from src.pipeline import build_feature_matrix, build_rating_features_for_season, _parse_seed_num

OUTPUT_DIR = PROJECT_ROOT / "output"
OUTPUT_DIR.mkdir(exist_ok=True)

plt.style.use("seaborn-v0_8-whitegrid")


def fig_seed_win_rate(data: dict):
    """Bar chart of tournament win rate by seed (1-16), 2003-2026."""
    tourney = data["tourney_compact"]
    seeds = data["seeds"]

    # Filter to 2003-2026
    tourney = tourney[(tourney["Season"] >= 2003) & (tourney["Season"] <= 2026)].copy()
    seeds = seeds[(seeds["Season"] >= 2003) & (seeds["Season"] <= 2026)].copy()
    seeds["SeedNum"] = seeds["Seed"].apply(_parse_seed_num)

    # Map each game's winner and loser to their seed
    seed_map = {}
    for _, row in seeds.iterrows():
        seed_map[(row["Season"], row["TeamID"])] = row["SeedNum"]

    records = {s: {"wins": 0, "losses": 0} for s in range(1, 17)}

    for _, g in tourney.iterrows():
        season = g["Season"]
        w_seed = seed_map.get((season, g["WTeamID"]))
        l_seed = seed_map.get((season, g["LTeamID"]))
        if w_seed is not None and 1 <= w_seed <= 16:
            records[w_seed]["wins"] += 1
        if l_seed is not None and 1 <= l_seed <= 16:
            records[l_seed]["losses"] += 1

    seed_nums = list(range(1, 17))
    win_rates = []
    for s in seed_nums:
        total = records[s]["wins"] + records[s]["losses"]
        win_rates.append(records[s]["wins"] / total if total > 0 else 0)

    fig, ax = plt.subplots(figsize=(10, 5))
    colors = sns.color_palette("RdYlGn_r", n_colors=16)
    bars = ax.bar(seed_nums, win_rates, color=colors, edgecolor="white", linewidth=0.5)

    ax.set_xlabel("Tournament Seed", fontsize=12)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_title("NCAA Tournament Win Rate by Seed (2003-2026)", fontsize=14, fontweight="bold")
    ax.set_xticks(seed_nums)
    ax.set_ylim(0, 1.0)

    # Add value labels on bars
    for bar, wr in zip(bars, win_rates):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f"{wr:.2f}", ha="center", va="bottom", fontsize=8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "seed_win_rate.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved seed_win_rate.png")


def fig_feature_target_correlation(data: dict):
    """Horizontal bar chart of diff feature correlations with binary outcome."""
    seasons = list(range(2014, 2027))
    X, y = build_feature_matrix(data, seasons)

    # Select only "diff" features
    diff_cols = [c for c in X.columns if "diff" in c.lower()]
    if not diff_cols:
        print("WARNING: no diff columns found, skipping feature_target_correlation")
        return

    correlations = {}
    for col in diff_cols:
        valid = X[col].notna()
        if valid.sum() > 30:
            correlations[col] = np.corrcoef(X.loc[valid, col], y[valid])[0, 1]

    corr_series = pd.Series(correlations).sort_values()

    fig, ax = plt.subplots(figsize=(8, 6))
    colors = ["#2563EB" if v < 0 else "#DC2626" for v in corr_series.values]
    ax.barh(range(len(corr_series)), corr_series.values, color=colors, edgecolor="white", linewidth=0.5)
    ax.set_yticks(range(len(corr_series)))
    ax.set_yticklabels(corr_series.index, fontsize=9)
    ax.set_xlabel("Pearson Correlation with Outcome", fontsize=12)
    ax.set_title("Feature-Outcome Correlation (2014-2026)", fontsize=14, fontweight="bold")
    ax.axvline(0, color="black", linewidth=0.8)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "feature_target_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved feature_target_correlation.png")


def fig_rating_correlation(data: dict):
    """Heatmap of inter-correlation between rating systems for 2025 season."""
    ratings = build_rating_features_for_season(data, season=2025)

    if ratings.empty:
        print("WARNING: no rating data for 2025, skipping rating_correlation")
        return

    # Drop columns with too many NaNs
    ratings = ratings.dropna(axis=1, thresh=int(0.5 * len(ratings)))
    ratings = ratings.dropna()

    corr_matrix = ratings.corr()

    fig, ax = plt.subplots(figsize=(8, 7))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(
        corr_matrix,
        mask=mask,
        annot=True,
        fmt=".2f",
        cmap="RdYlBu_r",
        vmin=0.5,
        vmax=1.0,
        square=True,
        linewidths=0.5,
        ax=ax,
        cbar_kws={"label": "Pearson Correlation"},
    )
    ax.set_title("Rating System Inter-correlation (2025 Season)", fontsize=14, fontweight="bold")

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "rating_correlation.png", dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("Saved rating_correlation.png")


if __name__ == "__main__":
    print("Loading data...")
    data = load_all_mens_data()

    print("\n--- Generating seed_win_rate.png ---")
    fig_seed_win_rate(data)

    print("\n--- Generating feature_target_correlation.png ---")
    fig_feature_target_correlation(data)

    print("\n--- Generating rating_correlation.png ---")
    fig_rating_correlation(data)

    print("\nDone! All figures saved to output/")
