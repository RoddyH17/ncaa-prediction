"""Step 2: MoE by game type — LOTO backtest comparison."""
import sys
sys.path.insert(0, ".")

import pandas as pd
import numpy as np
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.metrics import brier_score_loss
from src.data_collection import load_all_mens_data
from src.pipeline import _parse_seed_num

data = load_all_mens_data()
tourney = data["tourney_compact"]
seeds_raw = data["seeds"].copy()
seeds_raw["SeedNum"] = seeds_raw["Seed"].apply(_parse_seed_num)
eff = pd.read_csv("data/external/adjusted_efficiency.csv")

CLIP_LO, CLIP_HI = 0.025, 0.975


def build_matrix(seasons):
    rows, ys, ms = [], [], []
    for season in seasons:
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
                a, b, result, m = w, l, 1, margin
            else:
                a, b, result, m = l, w, 0, -margin
            sa, sb = seed_map.get(a, 16), seed_map.get(b, 16)
            ea = se.loc[a, "AdjEM"] if a in se.index else 0
            eb = se.loc[b, "AdjEM"] if b in se.index else 0
            rows.append({
                "Season": season, "seed_diff": sa - sb,
                "adjEM_diff": ea - eb, "seed_gap": abs(sa - sb),
            })
            ys.append(result)
            ms.append(m)
    return pd.DataFrame(rows), np.array(ys), np.array(ms, dtype=float)


def game_type(gap):
    if gap >= 5:
        return "chalk"
    elif gap >= 2:
        return "competitive"
    else:
        return "toss_up"


def flip(X, y, m):
    Xf = X.copy()
    Xf["seed_diff"] = -X["seed_diff"]
    Xf["adjEM_diff"] = -X["adjEM_diff"]
    return pd.concat([X, Xf], ignore_index=True), np.concatenate([y, 1 - y]), np.concatenate([m, -m])


seasons = list(range(2014, 2026))
X_raw, y_raw, m_raw = build_matrix(seasons)
X_raw["type"] = X_raw["seed_gap"].apply(game_type)
X_full, y_full, m_full = flip(X_raw, y_raw, m_raw)
X_full["type"] = X_full["seed_gap"].apply(game_type)

feat_cols = ["adjEM_diff", "seed_diff"]


# --- Global Spread ---
def loto_global(X_data, y_data, m_data):
    results = []
    y_s = pd.Series(y_data, index=X_data.index)
    m_s = pd.Series(m_data, index=X_data.index)
    all_probs, all_y = [], []
    for holdout in sorted(X_data["Season"].unique()):
        tr = X_data["Season"] != holdout
        te = X_data["Season"] == holdout
        model = Ridge(alpha=1.0)
        model.fit(X_data.loc[tr, feat_cols], m_s.loc[tr])
        sigma = (m_s.loc[tr] - model.predict(X_data.loc[tr, feat_cols])).std()
        probs = np.clip(norm.cdf(model.predict(X_data.loc[te, feat_cols]) / sigma), CLIP_LO, CLIP_HI)
        bs = brier_score_loss(y_s.loc[te], probs)
        results.append({"season": int(holdout), "brier": bs})
        all_probs.extend(probs)
        all_y.extend(y_s.loc[te])
    return pd.DataFrame(results), np.array(all_probs), np.array(all_y)


# --- MoE: Separate model per game type ---
def loto_moe_separate(X_data, y_data, m_data):
    results = []
    y_s = pd.Series(y_data, index=X_data.index)
    m_s = pd.Series(m_data, index=X_data.index)
    all_probs, all_y = [], []
    for holdout in sorted(X_data["Season"].unique()):
        tr = X_data["Season"] != holdout
        te = X_data["Season"] == holdout
        te_probs = pd.Series(0.5, index=X_data.loc[te].index)

        for gtype in ["chalk", "competitive", "toss_up"]:
            tr_g = tr & (X_data["type"] == gtype)
            te_g = te & (X_data["type"] == gtype)
            if tr_g.sum() < 10 or te_g.sum() == 0:
                # Fallback to global
                model = Ridge(alpha=1.0)
                model.fit(X_data.loc[tr, feat_cols], m_s.loc[tr])
                sigma = (m_s.loc[tr] - model.predict(X_data.loc[tr, feat_cols])).std()
            else:
                model = Ridge(alpha=1.0)
                model.fit(X_data.loc[tr_g, feat_cols], m_s.loc[tr_g])
                sigma = (m_s.loc[tr_g] - model.predict(X_data.loc[tr_g, feat_cols])).std()
            if te_g.sum() > 0:
                pred = model.predict(X_data.loc[te_g, feat_cols])
                te_probs.loc[X_data.loc[te_g].index] = norm.cdf(pred / sigma)

        probs = np.clip(te_probs.values, CLIP_LO, CLIP_HI)
        bs = brier_score_loss(y_s.loc[te], probs)
        results.append({"season": int(holdout), "brier": bs})
        all_probs.extend(probs)
        all_y.extend(y_s.loc[te])
    return pd.DataFrame(results), np.array(all_probs), np.array(all_y)


# --- MoE: Global model, group-specific sigma ---
def loto_moe_sigma(X_data, y_data, m_data):
    results = []
    y_s = pd.Series(y_data, index=X_data.index)
    m_s = pd.Series(m_data, index=X_data.index)
    all_probs, all_y = [], []
    for holdout in sorted(X_data["Season"].unique()):
        tr = X_data["Season"] != holdout
        te = X_data["Season"] == holdout
        model = Ridge(alpha=1.0)
        model.fit(X_data.loc[tr, feat_cols], m_s.loc[tr])
        train_resid = m_s.loc[tr] - model.predict(X_data.loc[tr, feat_cols])
        global_sigma = train_resid.std()

        group_sigma = {}
        for gtype in ["chalk", "competitive", "toss_up"]:
            mask = tr & (X_data["type"] == gtype)
            if mask.sum() > 20:
                group_sigma[gtype] = train_resid.loc[mask].std()
            else:
                group_sigma[gtype] = global_sigma

        pred_spread = model.predict(X_data.loc[te, feat_cols])
        sigmas = X_data.loc[te, "type"].map(group_sigma).values
        probs = np.clip(norm.cdf(pred_spread / sigmas), CLIP_LO, CLIP_HI)
        bs = brier_score_loss(y_s.loc[te], probs)
        results.append({"season": int(holdout), "brier": bs})
        all_probs.extend(probs)
        all_y.extend(y_s.loc[te])
    return pd.DataFrame(results), np.array(all_probs), np.array(all_y)


# --- Ensemble: weighted average of global + MoE ---
def ensemble_probs(probs_list, weights):
    stacked = np.column_stack(probs_list)
    return np.clip(np.average(stacked, axis=1, weights=weights), CLIP_LO, CLIP_HI)


print("Running LOTO backtests...\n")

r_global, p_global, y_global = loto_global(X_full, y_full, m_full)
print(f"Global Spread (AdjEM+seed+flip):          {r_global['brier'].mean():.4f}")

r_moe, p_moe, y_moe = loto_moe_separate(X_full, y_full, m_full)
print(f"MoE Spread (separate per group):          {r_moe['brier'].mean():.4f}")

r_moe_sig, p_moe_sig, y_moe_sig = loto_moe_sigma(X_full, y_full, m_full)
print(f"MoE Sigma-only (global + group sigma):    {r_moe_sig['brier'].mean():.4f}")

# Ensemble: search for best weight
print("\n--- Ensemble Search ---")
best_brier = 1.0
best_w = None
for w1 in np.arange(0.0, 1.05, 0.1):
    for w2 in np.arange(0.0, 1.05 - w1, 0.1):
        w3 = 1.0 - w1 - w2
        if w3 < -0.01:
            continue
        ens_p = ensemble_probs([p_global, p_moe, p_moe_sig], [w1, w2, w3])
        bs = brier_score_loss(y_global, ens_p)
        if bs < best_brier:
            best_brier = bs
            best_w = (w1, w2, w3)

print(f"Best ensemble: Global={best_w[0]:.1f}, MoE-Sep={best_w[1]:.1f}, MoE-Sig={best_w[2]:.1f}")
print(f"Best ensemble Brier: {best_brier:.4f}")

# Also try simple 2-model ensemble
print("\n--- 2-Model Ensemble ---")
for w in np.arange(0.0, 1.05, 0.1):
    ens_p = ensemble_probs([p_global, p_moe], [w, 1 - w])
    bs = brier_score_loss(y_global, ens_p)
    if abs(w - 0.5) < 0.01 or abs(w - 0.7) < 0.01 or abs(w - 1.0) < 0.01 or abs(w - 0.0) < 0.01 or bs < 0.1876:
        print(f"  w_global={w:.1f}: Brier={bs:.4f}")

# Per-season comparison
print("\n--- Per Season ---")
comp = r_global.rename(columns={"brier": "global"}).merge(
    r_moe.rename(columns={"brier": "moe_sep"}), on="season").merge(
    r_moe_sig.rename(columns={"brier": "moe_sig"}), on="season")
for _, row in comp.iterrows():
    vals = [row["global"], row["moe_sep"], row["moe_sig"]]
    best = min(vals)
    marks = ["*" if v == best else " " for v in vals]
    print(f"  {int(row['season'])}: Global={row['global']:.4f}{marks[0]}  MoE-Sep={row['moe_sep']:.4f}{marks[1]}  MoE-Sig={row['moe_sig']:.4f}{marks[2]}")

print("\nDone.")
