"""
LOTO backtest for Hybrid Transformer.

Usage:
    python scripts/run_hybrid_transformer.py
"""

import sys
sys.path.insert(0, ".")

import time
from src.data_collection import load_all_mens_data
from src.pipeline import make_build_features_fn
from src.evaluation import leave_one_tournament_out
from src.hybrid_transformer import HybridTransformerWrapper


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    seasons = [s for s in range(2014, 2026) if s != 2020]

    t0 = time.time()
    print(f"\n{'='*60}\n  Hybrid Transformer LOTO\n{'='*60}")
    results = leave_one_tournament_out(
        build_fn,
        lambda: HybridTransformerWrapper(
            data, d_model=32, n_layers=2, n_heads=2,
            pretrain_epochs=10, finetune_epochs=30,
        ),
        seasons,
    )
    elapsed = time.time() - t0
    mean_bs = results["brier_score"].mean()
    std_bs = results["brier_score"].std()
    print(f"\nMean Brier: {mean_bs:.4f} ± {std_bs:.4f} ({elapsed:.1f}s)")
    print(results.to_string(index=False))

    results.to_csv("output/loto_hybrid_transformer.csv", index=False)
    print("\nSaved to output/loto_hybrid_transformer.csv")


if __name__ == "__main__":
    main()
