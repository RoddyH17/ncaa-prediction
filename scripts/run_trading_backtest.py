"""
Run trading strategy backtest over historical tournaments.

Usage:
    python scripts/run_trading_backtest.py
"""

import sys
sys.path.insert(0, ".")

from src.data_collection import load_all_mens_data
from src.pipeline import make_build_features_fn
from src.models import MultiFeatureLogistic
from src.kalshi_trader import SyntheticMarketPrices, TradingStrategy, TradingBacktester


def main():
    print("Loading data...")
    data = load_all_mens_data()
    build_fn = make_build_features_fn(data)
    seasons = [s for s in range(2014, 2026) if s != 2020]

    print("Building synthetic market prices from seed win rates...")
    market_gen = SyntheticMarketPrices(data, noise_std=0.04, flb_strength=0.02)

    # --- Threshold tuning ---
    print("\n" + "=" * 60)
    print("  THRESHOLD TUNING (quarter-Kelly, $10K bankroll)")
    print("=" * 60)

    strategy = TradingStrategy(threshold=0.05, kelly_fraction=0.25,
                               max_per_game=0.05, max_total_exposure=0.20,
                               max_contracts=100, spread=0.03)
    backtester = TradingBacktester(strategy, market_gen, initial_bankroll=10000)

    tune_results = backtester.tune_threshold(
        data, build_fn,
        lambda: MultiFeatureLogistic(C=0.5),
        seasons,
        thresholds=[0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.10, 0.15],
    )
    print("\nThreshold tuning results:")
    print(tune_results.to_string(index=False))

    # --- Run with best threshold ---
    best_tau = tune_results.loc[tune_results["total_pnl"].idxmax(), "threshold"]
    print(f"\nBest threshold: tau = {best_tau:.2f}")
    print("\n" + "=" * 60)
    print(f"  FULL BACKTEST (tau={best_tau:.2f})")
    print("=" * 60)

    strategy.threshold = best_tau
    backtester_final = TradingBacktester(strategy, market_gen, initial_bankroll=10000)
    results = backtester_final.run_backtest(
        data, build_fn,
        lambda: MultiFeatureLogistic(C=0.5),
        seasons,
    )

    print("\nPer-season results:")
    print(results.to_string(index=False))

    total_pnl = results["gross_pnl"].sum()
    total_trades = results["n_trades"].sum()
    final_bankroll = results["bankroll"].iloc[-1]
    avg_win = results["win_rate"].mean()

    sharpe = results.attrs.get("sharpe", 0)
    max_dd = results.attrs.get("max_drawdown", 0)

    print(f"\n--- Summary ---")
    print(f"Total trades:    {int(total_trades)}")
    print(f"Total P&L:       ${total_pnl:+,.2f}")
    print(f"ROI:             {total_pnl/10000*100:+.1f}%")
    print(f"Final bankroll:  ${final_bankroll:,.2f}")
    print(f"Avg win rate:    {avg_win:.1%}")
    print(f"Sharpe ratio:    {sharpe:.2f}")
    print(f"Max drawdown:    ${max_dd:+,.2f}")

    # Save
    results.to_csv("output/trading_backtest.csv", index=False)
    tune_results.to_csv("output/threshold_tuning.csv", index=False)
    print("\nSaved to output/trading_backtest.csv and output/threshold_tuning.csv")


if __name__ == "__main__":
    main()
