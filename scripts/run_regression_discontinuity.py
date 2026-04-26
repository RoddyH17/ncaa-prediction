"""
Regression Discontinuity Analysis: Causal Effect of Seed Assignment

Research question: holding underlying team quality (Barttorvik AdjEM) constant,
does the seed assignment itself causally affect tournament outcomes?

Design:
  - Selection committee assigns seeds based on multiple inputs (NET, AdjEM, etc.)
  - At seed boundaries (e.g., 4 vs 5), teams of similar quality get different seeds
  - This is quasi-random near the boundary
  - Compare team performance just above vs just below the boundary

Approach:
  1. Use Barttorvik NetRtg as continuous quality proxy
  2. Estimate "expected seed" via regression: seed ~ f(NetRtg)
  3. "Surprise seed" = actual seed - expected seed
  4. Test: does surprise seed predict tournament outcome controlling for NetRtg?

Findings will quantify:
  - Whether the committee's seed assignments contain information beyond AdjEM
  - Or conversely, whether seed assignments introduce bias
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import _parse_seed_num

plt.style.use("seaborn-v0_8-whitegrid")


def get_barttorvik(season: int) -> pd.DataFrame:
    bart = pd.read_csv(DATA_DIR / "external" / f"barttorvik_{season}.csv")
    bart = bart.drop_duplicates(subset="TeamID").set_index("TeamID")
    return bart


def main():
    print("Loading data...")
    data = load_all_mens_data()
    seasons = [s for s in range(2014, 2026) if s != 2020]
    tourney = data["tourney_compact"]
    actual_2026 = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")

    # === Build dataset of (team, season, seed, AdjEM, tournament_wins, conference) ===
    rows = []
    for season in seasons:
        try:
            bart = get_barttorvik(season)
        except FileNotFoundError:
            continue
        seeds = data["seeds"][data["seeds"]["Season"] == season].copy()
        seeds["SeedNum"] = seeds["Seed"].apply(_parse_seed_num)

        # Tournament results
        if season < 2026:
            t_games = tourney[tourney["Season"] == season]
        else:
            t_games = actual_2026

        wins_per_team = {}
        played_per_team = {}
        for _, g in t_games.iterrows():
            wins_per_team[g["WTeamID"]] = wins_per_team.get(g["WTeamID"], 0) + 1
            played_per_team[g["WTeamID"]] = played_per_team.get(g["WTeamID"], 0) + 1
            played_per_team[g["LTeamID"]] = played_per_team.get(g["LTeamID"], 0) + 1

        # Conference info
        conf_df = data.get("conferences")
        team_conf = {}
        if conf_df is not None:
            sub = conf_df[conf_df["Season"] == season]
            team_conf = dict(zip(sub["TeamID"], sub["ConfAbbrev"]))

        for _, srow in seeds.iterrows():
            tid = srow["TeamID"]
            seed = srow["SeedNum"]
            try:
                netrtg = float(bart.loc[tid, "NetRtg"]) if tid in bart.index else np.nan
                if hasattr(netrtg, "iloc"): netrtg = netrtg.iloc[0]
            except (KeyError, ValueError, TypeError):
                netrtg = np.nan
            rows.append({
                "season": season,
                "team_id": tid,
                "seed": seed,
                "netrtg": netrtg,
                "tournament_wins": wins_per_team.get(tid, 0),
                "tournament_games_played": played_per_team.get(tid, 1),
                "conference": team_conf.get(tid, "UNK"),
            })

    df = pd.DataFrame(rows).dropna(subset=["netrtg"])
    print(f"Tournament teams with full data: {len(df)} across {df['season'].nunique()} seasons")

    # === Step 1: Predict seed from NetRtg ===
    # Better teams (higher NetRtg) get lower seed numbers
    print(f"\n{'='*70}\n  STEP 1: Predict seed from NetRtg\n{'='*70}")
    X_seed = df[["netrtg"]].values
    y_seed = df["seed"].values
    seed_model = LinearRegression().fit(X_seed, y_seed)
    df["expected_seed"] = seed_model.predict(X_seed)
    df["surprise_seed"] = df["seed"] - df["expected_seed"]
    print(f"  Slope of seed on NetRtg: {seed_model.coef_[0]:.4f}")
    print(f"  Intercept: {seed_model.intercept_:.2f}")
    print(f"  R²: {seed_model.score(X_seed, y_seed):.3f}")
    print(f"  Surprise seed: mean={df['surprise_seed'].mean():.3f}, "
          f"std={df['surprise_seed'].std():.3f}")

    # === Step 2: Test if surprise_seed predicts tournament wins controlling for NetRtg ===
    print(f"\n{'='*70}\n  STEP 2: Does surprise seed predict tournament wins?\n{'='*70}")
    # Outcome: number of tournament wins
    # If positive: teams that received WORSE seed than deserved still won more (committee under-seeded them)
    # If negative: teams that received BETTER seed than deserved won less (committee over-seeded them)
    # If null: surprise seed has no effect (committee adds no info beyond NetRtg)

    from sklearn.linear_model import PoissonRegressor

    X = df[["netrtg", "surprise_seed"]].values
    y_wins = df["tournament_wins"].values

    # Poisson regression
    poisson = PoissonRegressor(alpha=0.1, max_iter=1000)
    poisson.fit(X, y_wins)

    # OLS for interpretability
    from sklearn.preprocessing import StandardScaler
    Xs = StandardScaler().fit_transform(X)
    ols = LinearRegression().fit(Xs, y_wins)

    print(f"  OLS coefficients (standardized features):")
    print(f"    NetRtg:        {ols.coef_[0]:+.4f} wins per 1 std")
    print(f"    Surprise seed: {ols.coef_[1]:+.4f} wins per 1 std")

    # Statistical test: is surprise_seed coefficient significantly different from 0?
    # Use bootstrap
    n_boot = 1000
    boot_coefs = []
    rng = np.random.default_rng(42)
    for _ in range(n_boot):
        idx = rng.integers(0, len(X), len(X))
        Xs_b = StandardScaler().fit_transform(X[idx])
        ols_b = LinearRegression().fit(Xs_b, y_wins[idx])
        boot_coefs.append(ols_b.coef_)
    boot_coefs = np.array(boot_coefs)
    ci_low = np.quantile(boot_coefs, 0.025, axis=0)
    ci_high = np.quantile(boot_coefs, 0.975, axis=0)
    print(f"\n  Bootstrap 95% CI:")
    print(f"    NetRtg:        [{ci_low[0]:+.4f}, {ci_high[0]:+.4f}]")
    print(f"    Surprise seed: [{ci_low[1]:+.4f}, {ci_high[1]:+.4f}]")
    surprise_significant = (ci_low[1] > 0) or (ci_high[1] < 0)
    print(f"    Surprise seed effect statistically significant: {surprise_significant}")

    # === Step 3: Conference-level bias ===
    print(f"\n{'='*70}\n  STEP 3: Per-conference seeding bias\n{'='*70}")
    print("  Power-conference teams: are they systematically over/under-seeded?")

    POWER_CONFS = ["acc", "big_ten", "big_twelve", "sec", "big_east", "pac_twelve", "pac_ten"]
    # Map conference codes (lowercase, with underscores)
    df["conf_lower"] = df["conference"].str.lower()
    df["is_power"] = df["conf_lower"].isin(POWER_CONFS)
    print(f"  Power conference teams: {df['is_power'].sum()} ({df['is_power'].mean():.1%})")
    print(f"  Mean surprise_seed (power conf):     {df[df['is_power']]['surprise_seed'].mean():+.3f}")
    print(f"  Mean surprise_seed (non-power conf): {df[~df['is_power']]['surprise_seed'].mean():+.3f}")

    # t-test for difference
    power_ss = df[df["is_power"]]["surprise_seed"].values
    nonpower_ss = df[~df["is_power"]]["surprise_seed"].values
    t_stat, p_val = stats.ttest_ind(power_ss, nonpower_ss, equal_var=False)
    print(f"  t-test (power vs non-power): t={t_stat:.2f}, p={p_val:.4f}")

    # By specific conference
    print(f"\n  Mean surprise_seed by conference (top 10):")
    conf_stats = df.groupby("conference").agg(
        n=("seed", "count"),
        mean_surprise=("surprise_seed", "mean"),
        mean_seed=("seed", "mean"),
        mean_netrtg=("netrtg", "mean"),
    ).sort_values("mean_surprise")
    print(conf_stats.head(10).to_string())
    print(f"\n  Conferences with WORST seeding (most under-seeded — high surprise):")
    print(conf_stats.tail(5).to_string())

    df.to_csv("output/seed_rd_data.csv", index=False)
    conf_stats.to_csv("output/seed_rd_conf_stats.csv")

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    ax = axes[0]
    ax.scatter(df["netrtg"], df["seed"], alpha=0.5, s=15, color="#2563eb")
    netrtg_range = np.linspace(df["netrtg"].min(), df["netrtg"].max(), 100)
    ax.plot(netrtg_range, seed_model.predict(netrtg_range.reshape(-1, 1)), "r-", linewidth=2)
    ax.set_xlabel("Barttorvik NetRtg")
    ax.set_ylabel("Tournament seed (1-16)")
    ax.set_title(f"Seed vs NetRtg ($R^2$ = {seed_model.score(X_seed, y_seed):.2f})")
    ax.invert_yaxis()

    ax = axes[1]
    # Surprise seed vs tournament wins
    ax.scatter(df["surprise_seed"], df["tournament_wins"],
               alpha=0.5, s=15, c=df["netrtg"], cmap="coolwarm")
    ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
    ax.set_xlabel("Surprise seed (actual - expected)")
    ax.set_ylabel("Tournament wins")
    ax.set_title(f"Surprise seed → wins (controlling for NetRtg)")

    plt.tight_layout()
    plt.savefig("output/regression_discontinuity.png", dpi=150, bbox_inches="tight")
    print("\nSaved regression_discontinuity.png")


if __name__ == "__main__":
    main()
