"""
Run baseline models through LOTO backtest and produce comparison table.

Usage:
    python scripts/run_baselines.py
"""

import sys
sys.path.insert(0, ".")

import time
import pandas as pd
from src.data_collection import load_all_mens_data
from src.pipeline import make_build_features_fn
from src.evaluation import leave_one_tournament_out, compare_models
from src.models import (
    SeedLogistic, KenPomLogistic, EloSklearnWrapper,
    GradientBoostingModel, MixtureOfExperts,
)
from src.transformer_model import TransformerSklearnWrapper


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    # No tournament in 2020 (COVID) or 2026 (not yet played)
    seasons = [s for s in range(2014, 2026) if s != 2020]

    models = {
        "SeedLogistic": lambda: SeedLogistic(),
        "KenPomLogistic": lambda: KenPomLogistic(),
        "Elo": lambda: EloSklearnWrapper(data),
        "XGBoost": lambda: GradientBoostingModel("xgboost"),
        "MoE (K=4)": lambda: MixtureOfExperts(n_experts=4),
        "Transformer": lambda: TransformerSklearnWrapper(
            data, d_model=64, n_layers=4, n_heads=4, finetune_epochs=30
        ),
    }

    results = {}
    for name, factory in models.items():
        print(f"\n{'='*50}")
        print(f"  {name}")
        print(f"{'='*50}")
        t0 = time.time()
        results[name] = leave_one_tournament_out(build_fn, factory, seasons)
        elapsed = time.time() - t0
        mean_bs = results[name]["brier_score"].mean()
        print(f"  Mean Brier: {mean_bs:.4f} ({elapsed:.1f}s)")

    print(f"\n{'='*60}")
    print("  MODEL COMPARISON")
    print(f"{'='*60}")
    comparison = compare_models(results)
    print(comparison.to_string(index=False))

    # Save results
    comparison.to_csv("output/baseline_comparison.csv", index=False)
    print("\nSaved to output/baseline_comparison.csv")

    # Save per-season results for each model
    for name, df in results.items():
        safe_name = name.replace(" ", "_").replace("(", "").replace(")", "")
        df.to_csv(f"output/loto_{safe_name}.csv", index=False)


if __name__ == "__main__":
    main()
