"""
Empirically verify the calibration futility regime.

For both men's and women's tournaments, compute:
  C(f_hat) = calibration error of Multi-Feature Logistic on LOTO OOF
  R_N  = bootstrap-estimated isotonic estimation error
  ratio = R_N / C(f_hat)

If ratio >= 1, isotonic's variance dominates its bias-correction benefit, so
calibration is provably futile in expectation. This explains the empirical
observation that LOSO isotonic does not improve LOTO Brier.

Outputs:
  output/futility_test.csv
"""

import sys
sys.path.insert(0, ".")

import numpy as np
import pandas as pd

from src.data_collection import load_all_mens_data
from src.pipeline import make_build_features_fn
from src.models import MultiFeatureLogistic
from scripts.build_womens_model import (
    load_womens_data, build_womens_features, WomensLogistic,
)
from src.calibration_regret import futility_test


def loto_oof(X_all, y_all, season_arr, model_cls, **kwargs):
    p_oof = np.zeros(len(X_all))
    for s in np.unique(season_arr):
        tr = season_arr != s
        te = season_arr == s
        m = model_cls(**kwargs)
        m.fit(X_all.loc[tr], y_all[tr])
        p_oof[te] = m.predict_proba(X_all.loc[te])[:, 1]
    return p_oof


def main():
    rows = []

    # ===== Men's =====
    print("Men's: building LOTO OOF...")
    data_m = load_all_mens_data()
    build_fn = make_build_features_fn(data_m)
    seasons = [s for s in range(2014, 2026) if s != 2020]
    X_m, y_m = build_fn(seasons)
    p_m = loto_oof(X_m, y_m, X_m["Season"].values, MultiFeatureLogistic, C=0.5)
    res_m = futility_test(p_m, y_m.astype(int), n_bins=15, n_boot=200, seed=42)
    print(f"  C(f_hat)  = {res_m['C_f']:.5f}")
    print(f"  R_N   = {res_m['R_N']:.5f} (+/- {res_m['R_N_std']:.5f})")
    print(f"  ratio = {res_m['ratio']:.2f}")
    print(f"  futility = {res_m['futility']}")
    rows.append({"domain": "mens", **res_m})

    # ===== Women's =====
    print("\nWomen's: building LOTO OOF...")
    data_w = load_womens_data()
    X_w, y_w = build_womens_features(data_w, seasons)
    p_w = loto_oof(X_w, y_w, X_w["Season"].values, WomensLogistic, C=0.5)
    res_w = futility_test(p_w, y_w.astype(int), n_bins=15, n_boot=200, seed=42)
    print(f"  C(f_hat)  = {res_w['C_f']:.5f}")
    print(f"  R_N   = {res_w['R_N']:.5f} (+/- {res_w['R_N_std']:.5f})")
    print(f"  ratio = {res_w['ratio']:.2f}")
    print(f"  futility = {res_w['futility']}")
    rows.append({"domain": "womens", **res_w})

    df = pd.DataFrame(rows)
    df.to_csv("output/futility_test.csv", index=False)
    print(f"\nSaved output/futility_test.csv")

    print(f"\n{'='*70}\n  FUTILITY TEST RESULTS\n{'='*70}")
    print(df.to_string(index=False))
    print(f"\n  Interpretation: ratio >= 1 means isotonic estimation error")
    print(f"  exceeds the calibration error it would correct, so calibration")
    print(f"  is provably futile in expectation.")


if __name__ == "__main__":
    main()
