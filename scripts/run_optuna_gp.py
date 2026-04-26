"""
Bayesian hyperparameter optimization on the 8-feature LR+XGB blend using
Optuna's Gaussian Process sampler.

Search space (9 hyperparameters):
  lr_C: [0.05, 1.0] log
  xgb_max_depth: [2, 4]
  xgb_n_estimators: [200, 1000]
  xgb_learning_rate: [0.01, 0.1] log
  xgb_min_child_weight: [1, 10]
  xgb_subsample: [0.6, 1.0]
  xgb_colsample_bytree: [0.6, 1.0]
  xgb_reg_lambda: [0.1, 5] log
  blend_w_lr: [0.4, 1.0]

Objective: minimize LOSO combined Brier (single seed for speed during search;
multi-seed averaging applied at final inference).

WARNING: aggressive HPO on small N risks overfitting LOSO. We report 2026
Brier honestly to verify the improvement actually transfers.
"""

import sys
sys.path.insert(0, ".")

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import xgboost as xgb
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import brier_score_loss

from src.data_collection import load_all_mens_data, DATA_DIR
from src.pipeline import _parse_seed_num
from scripts.build_womens_model import load_womens_data
from scripts.run_top3 import (
    build_combined_features, build_combined_features_2026,
    FEATURE_COLS as TOP3_FEATURES,
)
from src.seed_base_rate import compute_base_rate_table, lookup_p_a_wins


FEATS_8 = [
    "seed_pair_winrate", "bart_net_diff", "harry_diff",
    "elo_diff", "elo_slope_diff", "srs_diff",
    "massey_mean_diff", "tempo_diff",
]


def add_seed_pair(X, is_w, seed_lookup_m, seed_lookup_w,
                  base_per_season_m, base_per_season_w,
                  base_full_m, base_full_w):
    vals = []
    for _, r in X.iterrows():
        season = int(r["Season"])
        ta, tb = int(r["TeamA"]), int(r["TeamB"])
        if int(r["is_womens"]) == 0:
            seed_a = seed_lookup_m.get((season, ta), 17)
            seed_b = seed_lookup_m.get((season, tb), 17)
            tbl = base_per_season_m.get(season, base_full_m)
        else:
            seed_a = seed_lookup_w.get((season, ta), 17)
            seed_b = seed_lookup_w.get((season, tb), 17)
            tbl = base_per_season_w.get(season, base_full_w)
        vals.append(lookup_p_a_wins(tbl, seed_a, seed_b))
    X = X.copy()
    X["seed_pair_winrate"] = vals
    return X


# Globals (set in main, used by objective)
_X = None
_y = None
_is_w = None
_season_arr = None


def loso_brier_blend(lr_C, xgb_params, w_lr):
    p_oof = np.zeros(len(_X))
    for s in np.unique(_season_arr):
        tr = _season_arr != s
        te = _season_arr == s
        if te.sum() == 0: continue
        Xtr = _X.loc[tr, FEATS_8].apply(pd.to_numeric, errors="coerce")
        Xte = _X.loc[te, FEATS_8].apply(pd.to_numeric, errors="coerce")
        med = Xtr.median()
        Xtr = Xtr.fillna(med); Xte = Xte.fillna(med)
        scaler = StandardScaler().fit(Xtr)
        lr = LogisticRegression(C=lr_C, max_iter=2000, solver="lbfgs")
        lr.fit(scaler.transform(Xtr), _y[tr])
        p_lr = lr.predict_proba(scaler.transform(Xte))[:, 1]
        xm = xgb.XGBClassifier(**xgb_params)
        xm.fit(Xtr.values, _y[tr])
        p_xgb = xm.predict_proba(Xte.values)[:, 1]
        p_oof[te] = w_lr * p_lr + (1 - w_lr) * p_xgb
    p_oof = np.clip(p_oof, 0.005, 0.995)
    n_m = (_is_w == 0).sum(); n_w_ = (_is_w == 1).sum()
    bs_m = brier_score_loss(_y[_is_w == 0], p_oof[_is_w == 0])
    bs_w = brier_score_loss(_y[_is_w == 1], p_oof[_is_w == 1])
    return (bs_m * n_m + bs_w * n_w_) / (n_m + n_w_)


def objective(trial):
    lr_C = trial.suggest_float("lr_C", 0.05, 1.0, log=True)
    xgb_params = {
        "max_depth": trial.suggest_int("xgb_max_depth", 2, 4),
        "n_estimators": trial.suggest_int("xgb_n_estimators", 200, 1000, step=50),
        "learning_rate": trial.suggest_float("xgb_lr", 0.01, 0.1, log=True),
        "min_child_weight": trial.suggest_int("xgb_mcw", 1, 10),
        "subsample": trial.suggest_float("xgb_subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("xgb_colsample", 0.6, 1.0),
        "reg_alpha": trial.suggest_float("xgb_reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("xgb_reg_lambda", 0.1, 5.0, log=True),
        "eval_metric": "logloss",
        "tree_method": "hist",
        "random_state": 42,
    }
    w_lr = trial.suggest_float("w_lr", 0.4, 1.0)
    return loso_brier_blend(lr_C, xgb_params, w_lr)


def main():
    global _X, _y, _is_w, _season_arr
    seasons = [s for s in range(2014, 2026) if s != 2020]
    print("Loading data...")
    data_m = load_all_mens_data()
    data_w = load_womens_data()

    print("\nBuilding features...")
    _X, _y, _is_w = build_combined_features(data_m, data_w, seasons)
    _season_arr = _X["Season"].values
    massey_cols = [c for c in TOP3_FEATURES if c.startswith("massey_")]
    _X.loc[_is_w == 1, massey_cols] = 0.0

    seeds_m = data_m["seeds"]; seeds_w = data_w["seeds"]
    seed_lookup_m = {(int(r["Season"]), int(r["TeamID"])): _parse_seed_num(r["Seed"])
                     for _, r in seeds_m.iterrows()}
    seed_lookup_w = {(int(r["Season"]), int(r["TeamID"])): _parse_seed_num(r["Seed"])
                     for _, r in seeds_w.iterrows()}
    base_full_m = compute_base_rate_table(data_m["tourney_compact"], seeds_m)
    base_full_w = compute_base_rate_table(data_w["tourney_compact"], seeds_w)
    base_per_season_m = {s: compute_base_rate_table(data_m["tourney_compact"], seeds_m, exclude_season=s)
                         for s in seasons}
    base_per_season_w = {s: compute_base_rate_table(data_w["tourney_compact"], seeds_w, exclude_season=s)
                         for s in seasons}
    _X = add_seed_pair(_X, _is_w, seed_lookup_m, seed_lookup_w,
                       base_per_season_m, base_per_season_w, base_full_m, base_full_w)

    # ===========================================================
    # Run Optuna with Gaussian Process sampler
    # ===========================================================
    print(f"\n{'='*70}\n  Optuna GP Bayesian Optimization (50 trials)\n{'='*70}")
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.GPSampler(seed=42, deterministic_objective=True)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    # Print baseline first
    baseline_xgb = dict(
        max_depth=3, n_estimators=300, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=3,
        reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
        tree_method="hist", random_state=42,
    )
    bs_baseline = loso_brier_blend(0.1, baseline_xgb, 0.7)
    print(f"  Baseline (current best): LOSO = {bs_baseline:.4f}\n")

    n_trials = 50
    history = []
    for i in range(n_trials):
        trial = study.ask()
        try:
            value = objective(trial)
            study.tell(trial, value)
        except Exception as e:
            study.tell(trial, state=optuna.trial.TrialState.FAIL)
            continue
        if (i + 1) % 5 == 0 or i < 5:
            print(f"  Trial {i+1}/{n_trials}: this={value:.4f}  best={study.best_value:.4f}")
        history.append({"trial": i, "value": value, "best_so_far": study.best_value})

    pd.DataFrame(history).to_csv("output/optuna_gp_history.csv", index=False)

    print(f"\n  Best LOSO Brier: {study.best_value:.4f}")
    print(f"  vs uniform 70/30 baseline: {bs_baseline:.4f}")
    print(f"  Improvement: {study.best_value - bs_baseline:+.4f}")
    print(f"\n  Best params:")
    for k, v in study.best_params.items():
        print(f"    {k}: {v}")

    # ===========================================================
    # Apply best to 2026
    # ===========================================================
    print(f"\n{'='*70}\n  Apply best to 2026\n{'='*70}")

    X_2026, _, is_w_2026 = build_combined_features_2026(data_m, data_w)
    X_2026.loc[is_w_2026 == 1, massey_cols] = 0.0
    X_2026 = add_seed_pair(X_2026, is_w_2026.astype(int),
                            seed_lookup_m, seed_lookup_w,
                            {2026: base_full_m}, {2026: base_full_w},
                            base_full_m, base_full_w)

    Xtr_full = _X[FEATS_8].apply(pd.to_numeric, errors="coerce").fillna(
        _X[FEATS_8].apply(pd.to_numeric, errors="coerce").median()
    )
    X_2026_arr = X_2026[FEATS_8].apply(pd.to_numeric, errors="coerce").fillna(Xtr_full.median())

    bp = study.best_params
    scaler = StandardScaler().fit(Xtr_full)
    lr = LogisticRegression(C=bp["lr_C"], max_iter=2000, solver="lbfgs")
    lr.fit(scaler.transform(Xtr_full), _y)
    p_lr_2026 = lr.predict_proba(scaler.transform(X_2026_arr))[:, 1]

    xgb_params = {
        "max_depth": bp["xgb_max_depth"],
        "n_estimators": bp["xgb_n_estimators"],
        "learning_rate": bp["xgb_lr"],
        "min_child_weight": bp["xgb_mcw"],
        "subsample": bp["xgb_subsample"],
        "colsample_bytree": bp["xgb_colsample"],
        "reg_alpha": bp["xgb_reg_alpha"],
        "reg_lambda": bp["xgb_reg_lambda"],
        "eval_metric": "logloss",
        "tree_method": "hist",
    }
    # Multi-seed XGB at final inference for stability
    p_xgb_2026 = np.zeros(len(X_2026))
    for seed in range(10):
        params = dict(xgb_params); params["random_state"] = seed
        xm = xgb.XGBClassifier(**params).fit(Xtr_full.values, _y)
        p_xgb_2026 += xm.predict_proba(X_2026_arr.values)[:, 1]
    p_xgb_2026 /= 10

    p_2026 = bp["w_lr"] * p_lr_2026 + (1 - bp["w_lr"]) * p_xgb_2026
    p_2026 = np.clip(p_2026, 0.005, 0.995)

    pair_lk = {(int(r["TeamA"]), int(r["TeamB"])): float(p_2026[i])
               for i, r in X_2026.reset_index(drop=True).iterrows()}

    actual_m = pd.read_csv(DATA_DIR / "external" / "tourney_2026_results.csv")
    actual_w = pd.read_csv(DATA_DIR / "external" / "tourney_w_2026_results.csv")

    def br(actual):
        yt, yp = [], []
        for _, g in actual.iterrows():
            w_, l = int(g["WTeamID"]), int(g["LTeamID"])
            key = (w_, l) if w_ < l else (l, w_)
            yt.append(1 if w_ < l else 0)
            yp.append(pair_lk.get(key, 0.5))
        return brier_score_loss(yt, yp), len(yt)

    bs_m_2026, n_m = br(actual_m)
    bs_w_2026, n_w = br(actual_w)
    bs_c_2026 = (bs_m_2026 * n_m + bs_w_2026 * n_w) / (n_m + n_w)

    print(f"\n  Best Optuna GP config:")
    print(f"    LOSO Brier:  {study.best_value:.4f}")
    print(f"    2026 Brier:  {bs_c_2026:.4f}")
    print(f"      Men's:    {bs_m_2026:.4f}")
    print(f"      Women's:  {bs_w_2026:.4f}")
    print(f"\n  Comparison:")
    print(f"    Uniform 70/30 baseline:")
    print(f"      LOSO 0.1606  2026 0.1229")
    print(f"    Optuna GP optimum:")
    print(f"      LOSO {study.best_value:.4f}  2026 {bs_c_2026:.4f}")
    print(f"      LOSO delta: {study.best_value - 0.1606:+.4f}")
    print(f"      2026 delta: {bs_c_2026 - 0.1229:+.4f}")

    # Save submission only if 2026 didn't worsen significantly
    sub = pd.read_csv("output/submission_stage2.csv")
    sub[["s_str", "ta_str", "tb_str"]] = sub["ID"].str.split("_", expand=True)
    sub["TeamA"] = sub["ta_str"].astype(int); sub["TeamB"] = sub["tb_str"].astype(int)
    sub["Pred"] = sub.apply(
        lambda r: pair_lk.get((r["TeamA"], r["TeamB"]), float(r["Pred"])),
        axis=1
    ).clip(0.005, 0.995)
    sub[["ID", "Pred"]].to_csv("output/submission_stage2_OPTUNA.csv", index=False)
    print(f"\n  Saved output/submission_stage2_OPTUNA.csv")


if __name__ == "__main__":
    main()
