"""
Generate Kaggle submission: P(TeamA beats TeamB) for all possible matchups.
Uses the best model: Ridge spread -> Normal CDF, trained on all available data.
"""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Ridge
from src.data_collection import load_all_mens_data
from src.pipeline import _parse_seed_num

print("Loading data...")
data = load_all_mens_data()
tourney = data["tourney_compact"]
seeds_raw = data["seeds"].copy()
seeds_raw["SeedNum"] = seeds_raw["Seed"].apply(_parse_seed_num)
eff = pd.read_csv("data/external/adjusted_efficiency.csv")

CLIP_LO, CLIP_HI = 0.025, 0.975


# --- Build training data from ALL available tournament games ---
def build_training_data():
    rows, ms = [], []
    for season in range(2014, 2026):
        games = tourney[tourney["Season"] == season]
        if games.empty:
            continue
        ss = seeds_raw[seeds_raw["Season"] == season]
        seed_map = dict(zip(ss["TeamID"], ss["SeedNum"]))
        se = eff[eff["Season"] == season].set_index("TeamID")
        for _, g in games.iterrows():
            w, l = g["WTeamID"], g["LTeamID"]
            margin = g["WScore"] - g["LScore"]
            if w < l:
                a, b, m = w, l, margin
            else:
                a, b, m = l, w, -margin
            sa, sb = seed_map.get(a, 16), seed_map.get(b, 16)
            ea = se.loc[a, "AdjEM"] if a in se.index else 0
            eb = se.loc[b, "AdjEM"] if b in se.index else 0
            rows.append({"seed_diff": sa - sb, "adjEM_diff": ea - eb})
            ms.append(m)

    X = pd.DataFrame(rows)
    m = np.array(ms, dtype=float)
    # Flip-and-double
    Xf = X.copy()
    Xf["seed_diff"] = -X["seed_diff"]
    Xf["adjEM_diff"] = -X["adjEM_diff"]
    X_full = pd.concat([X, Xf], ignore_index=True)
    m_full = np.concatenate([m, -m])
    return X_full, m_full


print("Building training data (2014-2025, flip-doubled)...")
X_train, m_train = build_training_data()
print(f"Training: {len(X_train)} rows")

# Train final model
feat = ["adjEM_diff", "seed_diff"]
model = Ridge(alpha=1.0)
model.fit(X_train[feat], m_train)
sigma = (m_train - model.predict(X_train[feat])).std()
print(f"Model sigma: {sigma:.2f}")
print(f"Model coefs: adjEM_diff={model.coef_[0]:.4f}, seed_diff={model.coef_[1]:.4f}, intercept={model.intercept_:.4f}")

# --- Generate predictions for 2026 ---
print("\nGenerating 2026 predictions...")

# Load 2026 efficiency and seeds
eff_2026 = eff[eff["Season"] == 2026].set_index("TeamID")
seeds_2026 = seeds_raw[seeds_raw["Season"] == 2026]
seed_map_2026 = dict(zip(seeds_2026["TeamID"], seeds_2026["SeedNum"]))

# Load sample submission
sub = pd.read_csv("data/SampleSubmissionStage2.csv")
print(f"Submission rows: {len(sub)}")

# Parse IDs and generate predictions
preds = []
for _, row in sub.iterrows():
    parts = row["ID"].split("_")
    season = int(parts[0])
    team_a = int(parts[1])
    team_b = int(parts[2])

    if season != 2026:
        preds.append(0.5)
        continue

    # Seeds (default 16 if not in tournament)
    sa = seed_map_2026.get(team_a, 16)
    sb = seed_map_2026.get(team_b, 16)

    # Efficiency (default 0 if not available)
    ea = eff_2026.loc[team_a, "AdjEM"] if team_a in eff_2026.index else 0
    eb = eff_2026.loc[team_b, "AdjEM"] if team_b in eff_2026.index else 0

    # Predict
    x = np.array([[ea - eb, sa - sb]])
    spread = model.predict(x)[0]
    prob = norm.cdf(spread / sigma)
    prob = np.clip(prob, CLIP_LO, CLIP_HI)
    preds.append(prob)

sub["Pred"] = preds
sub.to_csv("output/submission.csv", index=False)
print(f"\nSubmission saved to output/submission.csv")

# Sanity checks
tourney_teams = set(seeds_2026["TeamID"])
tourney_rows = sub[sub["ID"].apply(lambda x: int(x.split("_")[1]) in tourney_teams and int(x.split("_")[2]) in tourney_teams)]
print(f"Tournament team matchup rows: {len(tourney_rows)}")
print(f"Pred stats: mean={sub['Pred'].mean():.4f}, std={sub['Pred'].std():.4f}")
print(f"Min={sub['Pred'].min():.4f}, Max={sub['Pred'].max():.4f}")

# Show some example predictions
teams = pd.read_csv("data/MTeams.csv")
team_names = dict(zip(teams["TeamID"], teams["TeamName"]))
print("\nSample predictions (top seeds):")
top_seeds = seeds_2026.nsmallest(8, "SeedNum")
for _, s1 in top_seeds.head(4).iterrows():
    for _, s2 in top_seeds.tail(4).iterrows():
        a, b = min(s1["TeamID"], s2["TeamID"]), max(s1["TeamID"], s2["TeamID"])
        row = sub[sub["ID"] == f"2026_{a}_{b}"]
        if not row.empty:
            p = row.iloc[0]["Pred"]
            na, nb = team_names.get(a, "?"), team_names.get(b, "?")
            print(f"  {na} vs {nb}: P({na} wins) = {p:.3f}")
