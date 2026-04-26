"""
Upset Prediction: a classifier specifically trained to identify tournament
upsets (lower-seeded team wins).

Key questions:
  1. Can we predict upsets better than seed difference alone?
  2. How many actual 2026 upsets did we identify?
  3. Are the upsets we predict positively-correlated with high-value picks
     (i.e., upsets others won't see)?

Approach:
  - Define upset: any game where lower seed (higher number) beats higher seed
  - Class imbalance: ~27% upset rate historically
  - Model: XGBoost with class_weight balanced
  - Metrics: AUC, precision@k, recall, F1
  - Compare to: seed-only baseline, P(win) model thresholded
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import xgboost as xgb

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import make_build_features_fn, _parse_seed_num
from src.models import BarttovikLogistic, MultiFeatureLogistic

plt.style.use("seaborn-v0_8-whitegrid")


def upset_target(X: pd.DataFrame, y: np.ndarray) -> np.ndarray:
    """Define upset = lower-seeded team won the game.
    In our canonical ordering (lower TeamID = TeamA), the outcome y=1 means
    TeamA won. The upset is when the team with WORSE seed wins.
    """
    # seed_diff = seed_A - seed_B. Negative = TeamA is better seed.
    # Upset: better-seeded team lost.
    # If TeamA better seed (seed_diff < 0): upset if y=0 (TeamB won)
    # If TeamB better seed (seed_diff > 0): upset if y=1 (TeamA won)
    seed_diff = X["seed_diff"].values
    y = np.asarray(y)
    upset = np.where(seed_diff < 0, 1 - y, y)
    # No upset if seeds equal
    upset[seed_diff == 0] = 0
    return upset


def evaluate_upset_predictions(p_upset: np.ndarray, true_upset: np.ndarray, name: str):
    """Compute upset detection metrics."""
    auc = roc_auc_score(true_upset, p_upset) if len(set(true_upset)) > 1 else np.nan
    ap = average_precision_score(true_upset, p_upset) if len(set(true_upset)) > 1 else np.nan
    n_actual = int(true_upset.sum())
    base_rate = float(true_upset.mean())

    # Top-k recall: among top-k highest predicted-upset games, how many were upsets?
    k_vals = [5, 10, 16]
    topk_metrics = {}
    for k in k_vals:
        if k > len(p_upset): continue
        top_k_idx = np.argsort(-p_upset)[:k]
        precision_at_k = float(true_upset[top_k_idx].mean())
        recall_at_k = float(true_upset[top_k_idx].sum() / max(n_actual, 1))
        topk_metrics[f"prec@{k}"] = precision_at_k
        topk_metrics[f"recall@{k}"] = recall_at_k

    return {
        "model": name,
        "AUC": auc,
        "AP": ap,
        "n_actual_upsets": n_actual,
        "n_total": len(true_upset),
        "base_rate": base_rate,
        **topk_metrics,
    }


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    seasons = [s for s in range(2014, 2026) if s != 2020]

    # Build full feature matrix
    X, y = build_fn(seasons)
    upsets_all = upset_target(X, y)
    print(f"Historical 2014-2025: {upsets_all.sum()}/{len(upsets_all)} upsets ({upsets_all.mean():.1%})")

    # Feature columns: only diff features (avoid per-team raw values that may
    # not be present in the 2026 submission feature build)
    DROP = ["Season", "TeamA", "TeamB"]
    feature_cols = [c for c in X.columns if c not in DROP and
                    not c.startswith("rank_A_") and not c.startswith("rank_B_") and
                    not c.startswith("momentum_winpct_A") and
                    not c.startswith("momentum_winpct_B") and
                    not c.startswith("momentum_margin_A") and
                    not c.startswith("momentum_margin_B") and
                    not c.startswith("net_eff_A") and not c.startswith("net_eff_B") and
                    not c.startswith("seed_A") and not c.startswith("seed_B")]

    # === LOTO upset prediction ===
    print(f"\n{'='*70}\n  UPSET PREDICTION LOTO\n{'='*70}")

    all_results = []
    season_results = []

    for holdout in seasons:
        train_mask = X["Season"] != holdout
        test_mask = X["Season"] == holdout
        if test_mask.sum() == 0: continue

        X_train_df = X[train_mask][feature_cols]
        y_train = y[train_mask.values]
        X_test_df = X[test_mask][feature_cols]
        y_test = y[test_mask.values]

        upset_train = upset_target(X[train_mask], y_train)
        upset_test = upset_target(X[test_mask], y_test)

        # Convert to numeric
        X_train_n = X_train_df.apply(pd.to_numeric, errors="coerce").values
        X_test_n = X_test_df.apply(pd.to_numeric, errors="coerce").values

        # Impute and scale
        imp = SimpleImputer(strategy="median")
        X_train_imp = imp.fit_transform(X_train_n)
        X_test_imp = imp.transform(X_test_n)

        # Train upset-specific classifier (XGBoost with class weight)
        scale_pos = (1 - upset_train.mean()) / upset_train.mean()  # for imbalance
        clf = xgb.XGBClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos, eval_metric="logloss", verbosity=0,
        )
        clf.fit(X_train_imp, upset_train)
        p_upset_xgb = clf.predict_proba(X_test_imp)[:, 1]

        # Baseline 1: seed difference (larger |seed_diff| → more chalky → less upset)
        seed_diffs = X[test_mask]["seed_diff"].values
        # Upset prob proxy: closer seeds = more likely upset (high P(close-game upset))
        p_upset_seed = 1.0 / (1.0 + np.abs(seed_diffs))
        # Direction: which TEAM upsets? Upset is when worse seed wins.
        # We don't predict direction here, just probability.

        # Baseline 2: invert win prob from Multi-Feature
        # P(upset) = P(worse-seeded team wins)
        mf_model = MultiFeatureLogistic(C=0.5)
        mf_model.fit(X[train_mask], y_train)
        p_winA = mf_model.predict_proba(X[test_mask])[:, 1]
        # Upset prob: probability that the worse-seeded team wins
        # If TeamA better seed (seed_diff < 0): P(upset) = 1 - p_winA (P(B wins))
        # If TeamB better seed (seed_diff > 0): P(upset) = p_winA (P(A wins))
        p_upset_mf = np.where(seed_diffs < 0, 1 - p_winA, p_winA)
        p_upset_mf[seed_diffs == 0] = 0.5

        # Same for Barttorvik logistic
        bart_model = BarttovikLogistic()
        bart_model.fit(X[train_mask], y_train)
        p_winA_bart = bart_model.predict_proba(X[test_mask])[:, 1]
        p_upset_bart = np.where(seed_diffs < 0, 1 - p_winA_bart, p_winA_bart)
        p_upset_bart[seed_diffs == 0] = 0.5

        # Evaluate each
        season_summary = {"season": holdout, "n_games": len(y_test),
                          "n_upsets": int(upset_test.sum())}
        for name, p in [("UpsetXGBoost", p_upset_xgb),
                         ("Seed_proximity", p_upset_seed),
                         ("MultiFeature_invertedP", p_upset_mf),
                         ("Barttorvik_invertedP", p_upset_bart)]:
            metrics = evaluate_upset_predictions(p, upset_test, name)
            metrics["season"] = holdout
            all_results.append(metrics)

        if upset_test.sum() > 0:
            print(f"  {holdout}: {upset_test.sum()}/{len(upset_test)} upsets, "
                  f"XGB AUC={evaluate_upset_predictions(p_upset_xgb, upset_test, '')['AUC']:.3f}")

    df = pd.DataFrame(all_results)

    # Aggregate by model
    print(f"\n{'='*70}\n  AGGREGATE UPSET PREDICTION (mean across seasons)\n{'='*70}")
    agg = df.groupby("model").agg({
        "AUC": "mean", "AP": "mean",
        "prec@5": "mean", "prec@10": "mean",
        "recall@5": "mean", "recall@10": "mean",
    }).round(3)
    print(agg)
    df.to_csv("output/upset_loto.csv", index=False)
    agg.to_csv("output/upset_aggregate.csv")

    # === 2026 actual evaluation ===
    print(f"\n{'='*70}\n  UPSETS ON ACTUAL 2026 RESULTS\n{'='*70}")

    actual_2026 = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    seeds_2026 = data["seeds"][data["seeds"]["Season"] == 2026].copy()
    seeds_2026["SeedNum"] = seeds_2026["Seed"].apply(_parse_seed_num)
    seed_map = dict(zip(seeds_2026["TeamID"], seeds_2026["SeedNum"]))

    # Identify upsets in actual 2026
    actual_2026["WSeed"] = actual_2026["WTeamID"].map(seed_map)
    actual_2026["LSeed"] = actual_2026["LTeamID"].map(seed_map)
    actual_2026["upset"] = actual_2026["WSeed"] > actual_2026["LSeed"]
    n_upsets_2026 = int(actual_2026["upset"].sum())
    print(f"\n2026 actual: {n_upsets_2026} upsets out of {len(actual_2026)} games "
          f"({n_upsets_2026/len(actual_2026):.1%})")
    print("\nUpsets in 2026:")
    upset_games = actual_2026[actual_2026["upset"]]
    teams = data["teams"]
    name_map = dict(zip(teams["TeamID"], teams["TeamName"]))
    for _, g in upset_games.iterrows():
        wname = name_map.get(g["WTeamID"], str(g["WTeamID"]))
        lname = name_map.get(g["LTeamID"], str(g["LTeamID"]))
        print(f"  ({g['WSeed']:>2}) {wname} beat ({g['LSeed']:>2}) {lname} "
              f"{g['WScore']}-{g['LScore']}")

    # Train upset model on 2014-2025, predict 2026 upsets
    train_mask = X["Season"].isin(seasons)
    X_train_n = X[train_mask][feature_cols].apply(pd.to_numeric, errors="coerce").values
    upset_train = upset_target(X[train_mask], y[train_mask.values])

    imp = SimpleImputer(strategy="median")
    X_train_imp = imp.fit_transform(X_train_n)
    scale_pos = (1 - upset_train.mean()) / upset_train.mean()
    clf = xgb.XGBClassifier(
        n_estimators=200, max_depth=4, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        scale_pos_weight=scale_pos, eval_metric="logloss", verbosity=0,
    )
    clf.fit(X_train_imp, upset_train)

    # Build 2026 features for actual played games
    from scripts.generate_kaggle_submission import build_submission_features
    sub_path = str(DATA_DIR / "SampleSubmissionStage2.csv")
    _, X_2026, _ = build_submission_features(data, 2026, sub_path)

    X_2026_n = X_2026[feature_cols].apply(pd.to_numeric, errors="coerce").values
    X_2026_imp = imp.transform(X_2026_n)
    p_upset_2026 = clf.predict_proba(X_2026_imp)[:, 1]

    # Lookup table: (team_a, team_b) -> upset prob
    upset_lookup = {}
    for i, (_, row) in enumerate(X_2026.iterrows()):
        ta, tb = int(row["TeamA"]), int(row["TeamB"])
        upset_lookup[(min(ta, tb), max(ta, tb))] = p_upset_2026[i]

    # For each ACTUAL 2026 game, what was our predicted upset prob?
    print("\n=== Predicted upset probability for actual 2026 games (sorted) ===")
    actual_2026["pair"] = actual_2026.apply(
        lambda r: (min(r["WTeamID"], r["LTeamID"]), max(r["WTeamID"], r["LTeamID"])), axis=1)
    actual_2026["p_upset_pred"] = actual_2026["pair"].map(upset_lookup)
    sorted_games = actual_2026.sort_values("p_upset_pred", ascending=False).head(20)
    print(f"{'pred':<8}{'actual':<10}{'matchup'}")
    for _, g in sorted_games.iterrows():
        wname = name_map.get(g["WTeamID"], str(g["WTeamID"]))
        lname = name_map.get(g["LTeamID"], str(g["LTeamID"]))
        actual_str = "UPSET" if g["upset"] else "chalk"
        print(f"  {g['p_upset_pred']:.3f}  {actual_str:<8}  "
              f"({g['WSeed']}) {wname} d. ({g['LSeed']}) {lname}")

    # How many of our top-N predicted upsets were actually upsets?
    print("\n=== Top-N upset prediction performance on 2026 ===")
    sorted_actual = actual_2026.sort_values("p_upset_pred", ascending=False)
    for top_n in [5, 10, 16, 20]:
        top_pred = sorted_actual.head(top_n)
        hits = int(top_pred["upset"].sum())
        print(f"  Top-{top_n} predicted-upset games: {hits}/{top_n} actual upsets "
              f"(precision {hits/top_n:.1%})")

    # Compute model's AUC on 2026 upsets
    valid = actual_2026.dropna(subset=["p_upset_pred"])
    if len(valid) > 0 and valid["upset"].nunique() > 1:
        auc_2026 = roc_auc_score(valid["upset"].astype(int), valid["p_upset_pred"])
        print(f"\n  2026 Upset AUC: {auc_2026:.3f}")

    actual_2026.to_csv("output/upset_2026_predictions.csv", index=False)


if __name__ == "__main__":
    main()
