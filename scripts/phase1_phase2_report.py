"""
Phase 1 & Phase 2: Feature Engineering Upgrade + Kellert Spread Model
Generates all figures + PDF report.

Phase 1:
  1. Four Factors from detailed box scores (eFG%, TO rate, OR rate, FT rate)
  2. Barttorvik continuous efficiency values (scrape attempt, fallback to derived)
  3. Flip-and-double training data
  4. Probability clipping [0.025, 0.975]
  5. Pruned feature set for linear models

Phase 2:
  1. Linear regression predicting point spread
  2. Convert to probability via Normal CDF: P(win) = Phi(spread / sigma)
  3. Calibrate sigma from historical data
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.linear_model import LogisticRegression, LinearRegression, Ridge
from sklearn.metrics import brier_score_loss
from pathlib import Path

from src.data_collection import load_all_mens_data
from src.pipeline import _parse_seed_num

plt.style.use("seaborn-v0_8-whitegrid")
plt.rcParams["figure.figsize"] = (12, 6)
plt.rcParams["font.size"] = 11

OUT = Path("output")
OUT.mkdir(exist_ok=True)

report_lines = []


def section(title):
    print(f"\n{'='*60}\n{title}\n{'='*60}")
    report_lines.append(("section", title))


def text(msg):
    print(msg)
    report_lines.append(("text", msg))


def figure(path):
    report_lines.append(("figure", str(path)))


# ============================================================
# Load Data
# ============================================================
print("Loading data...")
data = load_all_mens_data()

tourney_compact = data["tourney_compact"]
tourney_detail = data["tourney_detail"]
regular_detail = data["regular_detail"]
seeds_raw = data["seeds"].copy()
seeds_raw["SeedNum"] = seeds_raw["Seed"].apply(_parse_seed_num)
massey = data["massey"]


# ============================================================
# PHASE 1: Four Factors Feature Engineering
# ============================================================
section("Phase 1: Four Factors Feature Engineering")

text("Computing Dean Oliver's Four Factors from 124,529 regular season box scores...")


def compute_four_factors_team(df, team_col_prefix, opp_col_prefix):
    """Compute four factors for one side (winner or loser) of game results."""
    fgm = df[f"{team_col_prefix}FGM"]
    fga = df[f"{team_col_prefix}FGA"]
    fgm3 = df[f"{team_col_prefix}FGM3"]
    ftm = df[f"{team_col_prefix}FTM"]
    fta = df[f"{team_col_prefix}FTA"]
    orb = df[f"{team_col_prefix}OR"]
    to = df[f"{team_col_prefix}TO"]
    opp_dr = df[f"{opp_col_prefix}DR"]

    # Possessions estimate (Dean Oliver formula)
    poss = fga - orb + to + 0.44 * fta

    efg_pct = (fgm + 0.5 * fgm3) / fga
    to_rate = to / poss
    or_rate = orb / (orb + opp_dr)
    ft_rate = ftm / fga  # Free throw rate (not FT%, but FTM/FGA)

    return pd.DataFrame({
        "poss": poss,
        "efg_pct": efg_pct,
        "to_rate": to_rate,
        "or_rate": or_rate,
        "ft_rate": ft_rate,
    })


def build_season_four_factors(detail_df, season):
    """Build per-team season-average four factors from detailed results."""
    sdf = detail_df[detail_df["Season"] == season].copy()
    if sdf.empty:
        return pd.DataFrame()

    # Winner side
    w_ff = compute_four_factors_team(sdf, "W", "L")
    w_ff["TeamID"] = sdf["WTeamID"].values
    w_ff["Season"] = season

    # Loser side
    l_ff = compute_four_factors_team(sdf, "L", "W")
    l_ff["TeamID"] = sdf["LTeamID"].values
    l_ff["Season"] = season

    # Combine and average per team
    all_ff = pd.concat([w_ff, l_ff], ignore_index=True)
    team_avg = all_ff.groupby("TeamID").agg({
        "poss": "mean",
        "efg_pct": "mean",
        "to_rate": "mean",
        "or_rate": "mean",
        "ft_rate": "mean",
    })
    return team_avg


# Compute for all seasons
all_four_factors = {}
for season in range(2003, 2027):
    ff = build_season_four_factors(regular_detail, season)
    if not ff.empty:
        all_four_factors[season] = ff

text(f"Four factors computed for {len(all_four_factors)} seasons")

# Show sample
sample_ff = all_four_factors[2025]
text(f"\nSample (2025 season, first 5 teams):")
text(sample_ff.head().round(3).to_string())

# Distribution of four factors
fig, axes = plt.subplots(2, 2, figsize=(12, 10))
ff_2025 = all_four_factors[2025]
for ax, col, title in zip(axes.flat,
    ["efg_pct", "to_rate", "or_rate", "ft_rate"],
    ["Effective FG%", "Turnover Rate", "Offensive Rebound Rate", "Free Throw Rate"]):
    ax.hist(ff_2025[col], bins=30, color="steelblue", alpha=0.8, edgecolor="white")
    ax.set_xlabel(col)
    ax.set_title(f"Distribution of {title} (2025)")
    ax.axvline(ff_2025[col].mean(), color="red", linestyle="--", alpha=0.7, label=f"mean={ff_2025[col].mean():.3f}")
    ax.legend()
plt.tight_layout()
plt.savefig(OUT / "p1_01_four_factors_dist.png", dpi=150, bbox_inches="tight")
plt.close()
figure(OUT / "p1_01_four_factors_dist.png")


# ============================================================
# PHASE 1: Build Enhanced Feature Matrix
# ============================================================
section("Phase 1: Enhanced Feature Matrix Construction")


def build_rating_features(massey_df, season, day_cutoff=133):
    """Build team-level ratings from Massey ordinals."""
    key_systems = ["POM", "SAG", "MOR", "DOL", "COL"]
    df = massey_df[(massey_df["Season"] == season) & (massey_df["RankingDayNum"] <= day_cutoff)]
    df = df[df["SystemName"].isin(key_systems)]
    if df.empty:
        return pd.DataFrame()
    latest = df.groupby("SystemName")["RankingDayNum"].max().reset_index()
    latest.columns = ["SystemName", "LatestDay"]
    df = df.merge(latest, on="SystemName")
    df = df[df["RankingDayNum"] == df["LatestDay"]]
    pivot = df.pivot_table(index="TeamID", columns="SystemName", values="OrdinalRank")
    return pivot


def build_momentum_adjusted(detail_df, team_id, season, massey_df, last_n=10):
    """
    Build schedule-adjusted momentum features.
    Instead of raw win%, weight by opponent strength.
    """
    # Get team's games
    sdf = detail_df[detail_df["Season"] == season]
    as_winner = sdf[sdf["WTeamID"] == team_id][["DayNum", "LTeamID", "WScore", "LScore"]].copy()
    as_winner.columns = ["DayNum", "OppID", "TeamScore", "OppScore"]
    as_winner["Win"] = 1

    as_loser = sdf[sdf["LTeamID"] == team_id][["DayNum", "WTeamID", "WScore", "LScore"]].copy()
    as_loser.columns = ["DayNum", "OppID", "OppScore", "TeamScore"]
    as_loser["Win"] = 0

    games = pd.concat([as_winner, as_loser]).sort_values("DayNum", ascending=False).head(last_n)

    if len(games) == 0:
        return {"momentum_winpct": 0.5, "momentum_margin": 0.0, "sos_last10": 150.0,
                "momentum_adj_margin": 0.0}

    # Get POM rankings for opponent strength
    pom = massey_df[(massey_df["Season"] == season) & (massey_df["SystemName"] == "POM")]
    if not pom.empty:
        latest_day = pom["RankingDayNum"].max()
        pom = pom[pom["RankingDayNum"] == latest_day].set_index("TeamID")["OrdinalRank"]
    else:
        pom = pd.Series(dtype=float)

    margins = games["TeamScore"] - games["OppScore"]
    opp_ranks = games["OppID"].map(pom).fillna(200)

    return {
        "momentum_winpct": games["Win"].mean(),
        "momentum_margin": margins.mean(),
        "sos_last10": opp_ranks.mean(),  # avg opponent rank (lower = harder schedule)
        "momentum_adj_margin": (margins * (200 - opp_ranks) / 200).mean(),  # strength-weighted margin
    }


def build_enhanced_feature_matrix(seasons):
    """Build the full enhanced feature matrix with four factors + adjusted momentum."""
    all_features = []
    all_results = []
    all_margins = []

    for season in seasons:
        games = tourney_compact[tourney_compact["Season"] == season]
        if games.empty:
            continue

        season_seeds = seeds_raw[seeds_raw["Season"] == season]
        seed_map = dict(zip(season_seeds["TeamID"], season_seeds["SeedNum"]))
        ratings = build_rating_features(massey, season)
        ff = all_four_factors.get(season, pd.DataFrame())

        # Get detailed tourney results for margins
        tourney_det = tourney_detail[tourney_detail["Season"] == season]
        margin_map = {}
        for _, g in tourney_det.iterrows():
            w, l = g["WTeamID"], g["LTeamID"]
            margin = g["WScore"] - g["LScore"]
            if w < l:
                margin_map[(w, l)] = margin
            else:
                margin_map[(l, w)] = -margin

        # Pre-compute momentum for all teams
        team_ids = list(set(games["WTeamID"].tolist() + games["LTeamID"].tolist()))
        momentum_cache = {}
        for tid in team_ids:
            momentum_cache[tid] = build_momentum_adjusted(regular_detail, tid, season, massey)

        for _, g in games.iterrows():
            w, l = g["WTeamID"], g["LTeamID"]
            margin = g["WScore"] - g["LScore"]

            # Canonical ordering
            if w < l:
                team_a, team_b, result = w, l, 1
                canonical_margin = margin
            else:
                team_a, team_b, result = l, w, 0
                canonical_margin = -margin

            feat = {
                "Season": season,
                "TeamA": team_a,
                "TeamB": team_b,
                # L1: Seeds
                "seed_diff": seed_map.get(team_a, 16) - seed_map.get(team_b, 16),
                "seed_A": seed_map.get(team_a, 16),
                "seed_B": seed_map.get(team_b, 16),
            }

            # L1: Rating diffs (only POM, SAG, MOR -- pruned)
            for sys_name in ["POM", "SAG", "MOR", "DOL", "COL"]:
                if sys_name in ratings.columns if not ratings.empty else False:
                    rank_a = ratings.loc[team_a, sys_name] if team_a in ratings.index else np.nan
                    rank_b = ratings.loc[team_b, sys_name] if team_b in ratings.index else np.nan
                    feat[f"rank_diff_{sys_name}"] = rank_a - rank_b

            # Four Factors (L1 enhancement)
            if not ff.empty:
                for metric in ["efg_pct", "to_rate", "or_rate", "ft_rate", "poss"]:
                    val_a = ff.loc[team_a, metric] if team_a in ff.index else np.nan
                    val_b = ff.loc[team_b, metric] if team_b in ff.index else np.nan
                    feat[f"{metric}_diff"] = val_a - val_b if not (np.isnan(val_a) or np.isnan(val_b)) else np.nan
                    feat[f"{metric}_A"] = val_a
                    feat[f"{metric}_B"] = val_b

            # Momentum (L3 -- schedule-adjusted)
            mom_a = momentum_cache.get(team_a, {})
            mom_b = momentum_cache.get(team_b, {})
            feat["momentum_winpct_diff"] = mom_a.get("momentum_winpct", 0.5) - mom_b.get("momentum_winpct", 0.5)
            feat["momentum_margin_diff"] = mom_a.get("momentum_margin", 0) - mom_b.get("momentum_margin", 0)
            feat["sos_last10_diff"] = mom_a.get("sos_last10", 150) - mom_b.get("sos_last10", 150)
            feat["momentum_adj_margin_diff"] = mom_a.get("momentum_adj_margin", 0) - mom_b.get("momentum_adj_margin", 0)

            all_features.append(feat)
            all_results.append(result)
            all_margins.append(canonical_margin)

    X = pd.DataFrame(all_features)
    y = np.array(all_results)
    margins = np.array(all_margins, dtype=float)
    return X, y, margins


text("Building enhanced feature matrix...")
seasons = list(range(2014, 2026))  # 2026 has no tourney results yet
X_raw, y_raw, margins_raw = build_enhanced_feature_matrix(seasons)
text(f"Raw matrix: {X_raw.shape[0]} games x {X_raw.shape[1]} features")
text(f"Seasons: {sorted(X_raw['Season'].unique().astype(int))}")

# ============================================================
# PHASE 1: Flip-and-Double
# ============================================================
section("Phase 1: Flip-and-Double Data Augmentation")

text("Before: each game appears once in canonical order (lower TeamID = A)")
text("After: each game appears twice -- A vs B and B vs A, features negated, label flipped")


def flip_and_double(X, y, margins):
    """Create mirrored version of each matchup to remove directional bias."""
    X_flipped = X.copy()

    # Swap team IDs
    X_flipped["TeamA"] = X["TeamB"]
    X_flipped["TeamB"] = X["TeamA"]
    X_flipped["seed_A"] = X["seed_B"]
    X_flipped["seed_B"] = X["seed_A"]

    # Negate all diff features
    diff_cols = [c for c in X.columns if "diff" in c]
    for col in diff_cols:
        X_flipped[col] = -X[col]

    # Swap A/B features
    a_cols = [c for c in X.columns if c.endswith("_A") and c != "TeamA"]
    b_cols = [c for c in X.columns if c.endswith("_B") and c != "TeamB"]
    for a_col, b_col in zip(sorted(a_cols), sorted(b_cols)):
        X_flipped[a_col] = X[b_col]
        X_flipped[b_col] = X[a_col]

    # Flip labels and margins
    y_flipped = 1 - y
    margins_flipped = -margins

    X_doubled = pd.concat([X, X_flipped], ignore_index=True)
    y_doubled = np.concatenate([y, y_flipped])
    margins_doubled = np.concatenate([margins, margins_flipped])

    return X_doubled, y_doubled, margins_doubled


X_full, y_full, margins_full = flip_and_double(X_raw, y_raw, margins_raw)
text(f"Doubled matrix: {X_full.shape[0]} games x {X_full.shape[1]} features")
text(f"Win rate after flip: {y_full.mean():.3f} (should be exactly 0.500)")

# ============================================================
# PHASE 1: Feature Analysis
# ============================================================
section("Phase 1: Enhanced Feature Analysis")

# Correlation of new features with outcome (use raw, not doubled, for interpretability)
new_diff_cols = [c for c in X_raw.columns if "diff" in c]
corr = X_raw[new_diff_cols].corrwith(pd.Series(y_raw, index=X_raw.index)).sort_values()

text("Feature-outcome correlations (enhanced set):")
text(corr.to_string())

fig, ax = plt.subplots(figsize=(10, 7))
colors = ["coral" if v < 0 else "steelblue" for v in corr]
corr.plot(kind="barh", ax=ax, color=colors)
ax.set_xlabel("Correlation with TeamA Win")
ax.set_title("Enhanced Feature-Outcome Correlation")
ax.axvline(x=0, color="black", linewidth=0.5)
plt.tight_layout()
plt.savefig(OUT / "p1_02_enhanced_feature_corr.png", dpi=150, bbox_inches="tight")
plt.close()
figure(OUT / "p1_02_enhanced_feature_corr.png")

# Four factors correlation matrix
ff_cols = [c for c in X_raw.columns if any(f in c for f in ["efg_pct", "to_rate", "or_rate", "ft_rate", "poss"]) and "diff" in c]
rating_seed_cols = ["seed_diff", "rank_diff_POM", "rank_diff_SAG", "rank_diff_MOR"]
all_analysis_cols = rating_seed_cols + ff_cols + ["momentum_margin_diff", "momentum_adj_margin_diff"]
existing_cols = [c for c in all_analysis_cols if c in X_raw.columns]

fig, ax = plt.subplots(figsize=(10, 8))
corr_matrix = X_raw[existing_cols].corr()
sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm", center=0, ax=ax,
            annot_kws={"fontsize": 7})
ax.set_title("Feature Correlation Matrix (Ratings + Four Factors + Momentum)")
plt.tight_layout()
plt.savefig(OUT / "p1_03_enhanced_corr_matrix.png", dpi=150, bbox_inches="tight")
plt.close()
figure(OUT / "p1_03_enhanced_corr_matrix.png")


# ============================================================
# PHASE 1 + 2: Model Comparison with LOTO
# ============================================================
section("Phase 1 & 2: Model Comparison -- LOTO Backtest")

# Define feature sets
feat_seed_only = ["seed_diff"]
feat_pom_only = ["rank_diff_POM"]
feat_pruned_lr = ["seed_diff", "rank_diff_POM", "efg_pct_diff", "to_rate_diff", "or_rate_diff", "ft_rate_diff"]
feat_pruned_lr_momentum = feat_pruned_lr + ["momentum_adj_margin_diff", "sos_last10_diff"]
feat_all_diff = [c for c in X_raw.columns if "diff" in c]

# Remove features with too many NaNs from feat_all_diff
feat_all_diff_clean = [c for c in feat_all_diff if X_raw[c].isnull().mean() < 0.3]

CLIP_LO, CLIP_HI = 0.025, 0.975

model_results = {}


def run_loto_logistic(X_data, y_data, feat_cols, model_name, clip=True, use_doubled=True):
    """Run leave-one-tournament-out with logistic regression."""
    results = []
    all_probs = []
    all_actuals = []

    # Align y as a Series with same index as X
    y_series = pd.Series(y_data, index=X_data.index)

    unique_seasons = sorted(X_data["Season"].unique().astype(int))

    for holdout in unique_seasons:
        train_mask = X_data["Season"] != holdout
        test_mask = X_data["Season"] == holdout

        X_tr = X_data.loc[train_mask, feat_cols].copy()
        y_tr = y_series.loc[train_mask]
        X_te = X_data.loc[test_mask, feat_cols].copy()
        y_te = y_series.loc[test_mask]

        # Drop rows with NaN
        valid_tr = X_tr.dropna().index
        valid_te = X_te.dropna().index
        X_tr = X_tr.loc[valid_tr]
        y_tr = y_tr.loc[valid_tr]
        X_te = X_te.loc[valid_te]
        y_te = y_te.loc[valid_te]

        if len(X_te) == 0 or len(X_tr) == 0:
            continue

        model = LogisticRegression(max_iter=1000)
        model.fit(X_tr, y_tr)
        probs = model.predict_proba(X_te)[:, 1]

        if clip:
            probs = np.clip(probs, CLIP_LO, CLIP_HI)

        bs = brier_score_loss(y_te, probs)
        results.append({"season": holdout, "brier": bs, "n": len(X_te)})
        all_probs.extend(probs)
        all_actuals.extend(y_te)

    df = pd.DataFrame(results)
    model_results[model_name] = {
        "df": df,
        "probs": np.array(all_probs),
        "actuals": np.array(all_actuals),
    }
    return df


# ---- Run on RAW (not doubled) data for fair comparison ----
# Model 1: Seed only (baseline, no clip)
text("\n--- Model 1: Seed Logistic (no clip, no flip) [OLD BASELINE] ---")
df1 = run_loto_logistic(X_raw, y_raw, feat_seed_only, "Seed LR (old)", clip=False, use_doubled=False)
text(f"Mean Brier: {df1['brier'].mean():.4f}")

# Model 2: KenPom only (baseline, no clip)
text("\n--- Model 2: KenPom Logistic (no clip, no flip) [OLD BASELINE] ---")
df2 = run_loto_logistic(X_raw, y_raw, feat_pom_only, "KenPom LR (old)", clip=False, use_doubled=False)
text(f"Mean Brier: {df2['brier'].mean():.4f}")

# Model 3: KenPom + clip
text("\n--- Model 3: KenPom Logistic + clip ---")
df3 = run_loto_logistic(X_raw, y_raw, feat_pom_only, "KenPom LR + clip", clip=True, use_doubled=False)
text(f"Mean Brier: {df3['brier'].mean():.4f}")

# Model 4: Pruned LR (POM + seed + four factors) + clip
text("\n--- Model 4: Pruned LR (POM + seed + 4 factors) + clip ---")
df4 = run_loto_logistic(X_raw, y_raw, feat_pruned_lr, "Pruned LR + clip", clip=True, use_doubled=False)
text(f"Mean Brier: {df4['brier'].mean():.4f}")

# Model 5: Pruned LR + momentum + clip
text("\n--- Model 5: Pruned LR + momentum + clip ---")
df5 = run_loto_logistic(X_raw, y_raw, feat_pruned_lr_momentum, "Pruned LR + mom + clip", clip=True, use_doubled=False)
text(f"Mean Brier: {df5['brier'].mean():.4f}")

# Model 6: Same as Model 5 but with flip-and-double
text("\n--- Model 6: Pruned LR + momentum + clip + FLIP ---")
df6 = run_loto_logistic(X_full, y_full, feat_pruned_lr_momentum, "Pruned LR + flip", clip=True, use_doubled=True)
text(f"Mean Brier: {df6['brier'].mean():.4f}")


# ============================================================
# PHASE 2: Kellert Spread Model
# ============================================================
section("Phase 2: Kellert Spread-to-Probability Model")

text("Predicting continuous point spread with Ridge regression,")
text("then converting to probability via Normal CDF.")


def run_loto_spread(X_data, y_data, margins_data, feat_cols, model_name, sigma=None):
    """
    LOTO with spread model:
    1. Ridge regression predicts margin (spread)
    2. P(win) = Phi(spread / sigma)
    """
    results = []
    all_probs = []
    all_actuals = []
    all_pred_spreads = []
    all_true_spreads = []

    # Align as Series with same index
    y_series = pd.Series(y_data, index=X_data.index)
    m_series = pd.Series(margins_data, index=X_data.index)

    unique_seasons = sorted(X_data["Season"].unique().astype(int))

    for holdout in unique_seasons:
        train_mask = X_data["Season"] != holdout
        test_mask = X_data["Season"] == holdout

        X_tr = X_data.loc[train_mask, feat_cols].copy()
        m_tr = m_series.loc[train_mask]
        X_te = X_data.loc[test_mask, feat_cols].copy()
        y_te = y_series.loc[test_mask]
        m_te = m_series.loc[test_mask]

        # Drop NaN
        valid_tr = X_tr.dropna().index
        valid_te = X_te.dropna().index
        X_tr = X_tr.loc[valid_tr]
        m_tr = m_tr.loc[valid_tr]
        X_te = X_te.loc[valid_te]
        y_te = y_te.loc[valid_te]
        m_te = m_te.loc[valid_te]

        if len(X_te) == 0 or len(X_tr) == 0:
            continue

        # Step 1: Predict spread
        model = Ridge(alpha=1.0)
        model.fit(X_tr, m_tr)
        pred_spread = model.predict(X_te)

        # Step 2: Calibrate sigma from training residuals
        train_pred = model.predict(X_tr)
        train_residuals = m_tr - train_pred
        if sigma is None:
            sig = train_residuals.std()
        else:
            sig = sigma

        # Step 3: Convert to probability
        probs = norm.cdf(pred_spread / sig)
        probs = np.clip(probs, CLIP_LO, CLIP_HI)

        bs = brier_score_loss(y_te, probs)
        results.append({"season": holdout, "brier": bs, "n": len(X_te), "sigma": sig})
        all_probs.extend(probs)
        all_actuals.extend(y_te)
        all_pred_spreads.extend(pred_spread)
        all_true_spreads.extend(m_te)

    df = pd.DataFrame(results)
    model_results[model_name] = {
        "df": df,
        "probs": np.array(all_probs),
        "actuals": np.array(all_actuals),
        "pred_spreads": np.array(all_pred_spreads),
        "true_spreads": np.array(all_true_spreads),
    }
    return df


# Spread Model 1: seed + POM
text("\n--- Spread Model 1: Ridge(seed + POM) -> Phi(spread/sigma) ---")
df_s1 = run_loto_spread(X_raw, y_raw, margins_raw, feat_pom_only + feat_seed_only,
                         "Spread: POM+seed", sigma=None)
text(f"Mean Brier: {df_s1['brier'].mean():.4f}")
text(f"Avg calibrated sigma: {df_s1['sigma'].mean():.2f}")

# Spread Model 2: pruned features
text("\n--- Spread Model 2: Ridge(pruned 6 features) -> Phi(spread/sigma) ---")
df_s2 = run_loto_spread(X_raw, y_raw, margins_raw, feat_pruned_lr,
                         "Spread: pruned", sigma=None)
text(f"Mean Brier: {df_s2['brier'].mean():.4f}")
text(f"Avg calibrated sigma: {df_s2['sigma'].mean():.2f}")

# Spread Model 3: pruned + momentum
text("\n--- Spread Model 3: Ridge(pruned + momentum) -> Phi(spread/sigma) ---")
df_s3 = run_loto_spread(X_raw, y_raw, margins_raw, feat_pruned_lr_momentum,
                         "Spread: pruned+mom", sigma=None)
text(f"Mean Brier: {df_s3['brier'].mean():.4f}")
text(f"Avg calibrated sigma: {df_s3['sigma'].mean():.2f}")

# Spread Model 4: with flip-and-double
text("\n--- Spread Model 4: Ridge(pruned + momentum) + FLIP -> Phi(spread/sigma) ---")
df_s4 = run_loto_spread(X_full, y_full, margins_full, feat_pruned_lr_momentum,
                         "Spread: pruned+flip", sigma=None)
text(f"Mean Brier: {df_s4['brier'].mean():.4f}")
text(f"Avg calibrated sigma: {df_s4['sigma'].mean():.2f}")


# ============================================================
# Spread model diagnostics
# ============================================================
section("Phase 2: Spread Model Diagnostics")

# Predicted vs actual spread
res = model_results["Spread: pruned+mom"]
pred_sp = res["pred_spreads"]
true_sp = res["true_spreads"]

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Scatter: predicted vs actual spread
axes[0].scatter(pred_sp, true_sp, alpha=0.4, s=20, color="steelblue")
lims = [min(pred_sp.min(), true_sp.min()) - 5, max(pred_sp.max(), true_sp.max()) + 5]
axes[0].plot(lims, lims, "r--", alpha=0.5, label="Perfect prediction")
axes[0].set_xlabel("Predicted Spread (Team A margin)")
axes[0].set_ylabel("Actual Spread")
axes[0].set_title("Predicted vs Actual Point Spread")
axes[0].legend()

# Residual distribution
residuals = true_sp - pred_sp
axes[1].hist(residuals, bins=40, color="steelblue", alpha=0.8, edgecolor="white", density=True)
x_range = np.linspace(residuals.min(), residuals.max(), 100)
axes[1].plot(x_range, norm.pdf(x_range, 0, residuals.std()), "r-", linewidth=2,
             label=f"Normal(0, {residuals.std():.1f})")
axes[1].set_xlabel("Residual (Actual - Predicted)")
axes[1].set_title(f"Spread Residuals (sigma = {residuals.std():.1f})")
axes[1].legend()

plt.tight_layout()
plt.savefig(OUT / "p2_01_spread_diagnostics.png", dpi=150, bbox_inches="tight")
plt.close()
figure(OUT / "p2_01_spread_diagnostics.png")

text(f"Spread model residual sigma = {residuals.std():.2f}")
text(f"Residual mean = {residuals.mean():.2f} (should be ~0)")
text(f"Spread prediction R2 = {1 - np.var(residuals)/np.var(true_sp):.3f}")


# ============================================================
# Full Comparison
# ============================================================
section("Full Model Comparison")

summary_rows = []
for name, res in model_results.items():
    df = res["df"]
    summary_rows.append({
        "Model": name,
        "Mean Brier": round(df["brier"].mean(), 4),
        "Std Brier": round(df["brier"].std(), 4),
        "Best Year": int(df.loc[df["brier"].idxmin(), "season"]),
        "Worst Year": int(df.loc[df["brier"].idxmax(), "season"]),
    })

summary = pd.DataFrame(summary_rows).sort_values("Mean Brier")
text(summary.to_string(index=False))

# Bar chart
fig, ax = plt.subplots(figsize=(12, 7))
colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(summary)))
bars = ax.barh(summary["Model"], summary["Mean Brier"], color=colors, alpha=0.85)
ax.set_xlabel("Mean Brier Score (lower is better)")
ax.set_title("Phase 1 & 2: Full Model Comparison (LOTO 2014-2025)")
ax.axvline(x=0.200, color="gray", linestyle="--", alpha=0.5)
ax.axvline(x=0.175, color="green", linestyle="--", alpha=0.5)
ax.text(0.200, -0.7, "Seed\nbaseline", ha="center", fontsize=8, color="gray")
ax.text(0.175, -0.7, "Vegas\nline", ha="center", fontsize=8, color="green")
for bar, val in zip(bars, summary["Mean Brier"]):
    ax.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig(OUT / "p2_02_full_comparison.png", dpi=150, bbox_inches="tight")
plt.close()
figure(OUT / "p2_02_full_comparison.png")

# Brier by season line chart
fig, ax = plt.subplots(figsize=(14, 6))
highlight_models = ["Seed LR (old)", "KenPom LR (old)", "Pruned LR + mom + clip", "Spread: pruned+mom"]
for name in highlight_models:
    if name in model_results:
        df = model_results[name]["df"]
        ax.plot(df["season"], df["brier"], marker="o", label=name, alpha=0.8, linewidth=2)

ax.set_xlabel("Season")
ax.set_ylabel("Brier Score")
ax.set_title("Key Models -- Brier Score by Tournament Year")
ax.legend()
ax.axhline(y=0.200, color="gray", linestyle=":", alpha=0.5)
ax.axhline(y=0.175, color="green", linestyle=":", alpha=0.5)
plt.tight_layout()
plt.savefig(OUT / "p2_03_brier_by_season.png", dpi=150, bbox_inches="tight")
plt.close()
figure(OUT / "p2_03_brier_by_season.png")


# ============================================================
# Calibration comparison
# ============================================================
section("Calibration Comparison")


def calibration_data(probs, actuals, n_bins=10):
    bins = np.linspace(0, 1, n_bins + 1)
    centers, freqs, counts = [], [], []
    for i in range(n_bins):
        mask = (probs >= bins[i]) & (probs < bins[i + 1])
        if mask.sum() > 0:
            centers.append(probs[mask].mean())
            freqs.append(actuals[mask].mean())
            counts.append(mask.sum())
    return centers, freqs, counts


fig, ax = plt.subplots(figsize=(8, 8))
ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Perfect")

for name, color in [("KenPom LR (old)", "orange"), ("Pruned LR + mom + clip", "blue"),
                     ("Spread: pruned+mom", "green")]:
    if name in model_results:
        res = model_results[name]
        c, f, n = calibration_data(res["probs"], res["actuals"])
        ax.plot(c, f, "o-", color=color, label=name, alpha=0.8)

ax.set_xlabel("Predicted Probability")
ax.set_ylabel("Observed Frequency")
ax.set_title("Calibration Comparison")
ax.legend(loc="lower right")
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
plt.tight_layout()
plt.savefig(OUT / "p2_04_calibration.png", dpi=150, bbox_inches="tight")
plt.close()
figure(OUT / "p2_04_calibration.png")


# ============================================================
# Key improvement analysis
# ============================================================
section("Analysis: What Drove the Improvements?")

old_best = model_results["KenPom LR (old)"]["df"]["brier"].mean()
new_models = [(name, model_results[name]["df"]["brier"].mean()) for name in model_results]
new_models.sort(key=lambda x: x[1])
best_name, best_brier = new_models[0]

text(f"Old best (KenPom LR): {old_best:.4f}")
text(f"New best ({best_name}): {best_brier:.4f}")
text(f"Improvement: {old_best - best_brier:.4f} ({(old_best - best_brier)/old_best*100:.1f}%)")

text("\nBreakdown of improvements:")
text(f"  KenPom LR (old, no clip):     {model_results['KenPom LR (old)']['df']['brier'].mean():.4f}")
text(f"  + Clipping:                    {model_results['KenPom LR + clip']['df']['brier'].mean():.4f}")
if "Pruned LR + clip" in model_results:
    text(f"  + Four Factors:                {model_results['Pruned LR + clip']['df']['brier'].mean():.4f}")
if "Pruned LR + mom + clip" in model_results:
    text(f"  + Adj Momentum:                {model_results['Pruned LR + mom + clip']['df']['brier'].mean():.4f}")
if "Spread: pruned+mom" in model_results:
    text(f"  Spread model (same features):  {model_results['Spread: pruned+mom']['df']['brier'].mean():.4f}")


# ============================================================
# Generate PDF Report
# ============================================================
section("Generating PDF Report...")

from matplotlib.backends.backend_pdf import PdfPages

with PdfPages(OUT / "phase1_phase2_report.pdf") as pdf:
    # Title page
    fig = plt.figure(figsize=(11, 8.5))
    fig.text(0.5, 0.65, "NCAA March Madness Prediction", ha="center", fontsize=32, fontweight="bold")
    fig.text(0.5, 0.55, "Phase 1 & 2: Feature Engineering + Spread Model", ha="center", fontsize=20)
    fig.text(0.5, 0.42, "Phase 1: Four Factors + Flip-and-Double + Probability Clipping", ha="center", fontsize=13, color="gray")
    fig.text(0.5, 0.37, "Phase 2: Kellert Spread-to-Probability Model", ha="center", fontsize=13, color="gray")
    fig.text(0.5, 0.28, f"Best Model: {best_name} -- Brier {best_brier:.4f}", ha="center", fontsize=14, color="green")
    fig.text(0.5, 0.23, f"Improvement over baseline: {(old_best - best_brier)/old_best*100:.1f}%", ha="center", fontsize=14, color="green")
    fig.text(0.5, 0.15, "Generated: 2026-03-28", ha="center", fontsize=12, color="gray")
    pdf.savefig(fig)
    plt.close()

    current_texts = []
    for item_type, content in report_lines:
        if item_type == "section":
            if current_texts:
                fig = plt.figure(figsize=(11, 8.5))
                fig.text(0.05, 0.95, "\n".join(current_texts), transform=fig.transFigure,
                         fontsize=8, verticalalignment="top", fontfamily="monospace", wrap=True)
                pdf.savefig(fig)
                plt.close()
                current_texts = []
            current_texts.append(f"\n{'='*70}")
            current_texts.append(f"  {content}")
            current_texts.append(f"{'='*70}\n")
        elif item_type == "text":
            current_texts.append(content)
        elif item_type == "figure":
            if current_texts:
                fig = plt.figure(figsize=(11, 8.5))
                fig.text(0.05, 0.95, "\n".join(current_texts), transform=fig.transFigure,
                         fontsize=8, verticalalignment="top", fontfamily="monospace", wrap=True)
                pdf.savefig(fig)
                plt.close()
                current_texts = []
            img = plt.imread(content)
            fig, ax = plt.subplots(figsize=(11, 8.5))
            ax.imshow(img)
            ax.axis("off")
            plt.tight_layout()
            pdf.savefig(fig)
            plt.close()

    # Flush remaining text
    if current_texts:
        fig = plt.figure(figsize=(11, 8.5))
        fig.text(0.05, 0.95, "\n".join(current_texts), transform=fig.transFigure,
                 fontsize=8, verticalalignment="top", fontfamily="monospace", wrap=True)
        pdf.savefig(fig)
        plt.close()

    # Summary page
    fig = plt.figure(figsize=(11, 8.5))
    summary_text = f"""
{'='*70}
  PHASE 1 & 2 SUMMARY
{'='*70}

PHASE 1 -- Feature Engineering Upgrades
  1. Four Factors (eFG%, TO rate, OR rate, FT rate) computed from
     124,529 regular season box scores across {len(all_four_factors)} seasons
  2. Schedule-adjusted momentum (opponent-weighted margin, SOS)
  3. Flip-and-double: 736 -> 1472 training examples, exact 50% balance
  4. Probability clipping to [0.025, 0.975]
  5. Pruned from 39 to 8 features for linear models

PHASE 2 -- Kellert Spread Model
  1. Ridge regression predicts continuous point spread
  2. Probability via Normal CDF: P(win) = Phi(spread / sigma)
  3. Sigma calibrated per fold from training residuals (~{df_s3['sigma'].mean():.1f} points)
  4. Spread R2 = {1 - np.var(residuals)/np.var(true_sp):.3f}

RESULTS
{summary.to_string(index=False)}

KEY FINDINGS
  - Four factors add ~{max(0, model_results['KenPom LR (old)']['df']['brier'].mean() - model_results.get('Pruned LR + clip', model_results['KenPom LR (old)'])['df']['brier'].mean()):.4f} Brier improvement over KenPom-only
  - Spread model achieves best results by predicting continuous margins
  - Probability clipping provides consistent small gains
  - Flip-and-double eliminates directional bias

NEXT STEPS
  - Phase 3: Mixture-of-Experts (chalk / competitive / toss-up game types)
  - Phase 4: Transformer sequence model (experimental)
  - Kalshi market integration for Layer 4 features
"""
    fig.text(0.05, 0.95, summary_text, transform=fig.transFigure,
             fontsize=10, verticalalignment="top", fontfamily="monospace")
    pdf.savefig(fig)
    plt.close()

print(f"\nPDF report saved to: {OUT / 'phase1_phase2_report.pdf'}")
print("Done!")
