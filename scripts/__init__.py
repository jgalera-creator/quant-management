from pathlib import Path

import numpy as np
import pandas as pd

from quantmgmt.data.loader import load_prices

from quantmgmt.backtesting.comparison import (
    default_market_weights,
    run_strategy_comparison,
)

from quantmgmt.backtesting.metrics import (
    evaluate_backtest,
)


TICKERS = [
    "SPY", "IWM", "EFA", "EEM",
    "TLT", "IEF", "LQD", "HYG",
    "GLD", "DBC", "VNQ", "TIP",
]

START_DATE = "2012-01-01"
END_DATE = "2026-11-03"

OUTPUT_DIR = Path("data/p3_results")


def main():

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Download adjusted prices using the existing loader.
    prices = load_prices(
        tickers=TICKERS,
        start=START_DATE,
        end=END_DATE,
    )

    prices = prices.reindex(
        columns=TICKERS
    ).sort_index()

    print("\n=== DATA QUALITY ===")

    print("Price observations:", len(prices))
    print("First date:", prices.index.min())
    print("Last date:", prices.index.max())

    missing = prices.isna().sum()

    print("\nMissing values:")
    print(missing)

    if missing.any():
        raise ValueError(
            "Missing prices detected. "
            "Investigate before running the backtest."
        )

    # 2. Calculate simple daily returns.
    returns = prices.pct_change(
        fill_method=None
    ).dropna()

    if not np.isfinite(
        returns.to_numpy()
    ).all():
        raise ValueError(
            "Non-finite returns detected"
        )

    # 3. Strategic Black-Litterman benchmark.
    market_weights = default_market_weights()

    # 4. Run the five strategies.
    print("\n=== RUNNING WALK-FORWARD BACKTEST ===")

    comparison = run_strategy_comparison(
        returns=returns,
        market_weights=market_weights,
        risk_free_rate=0.0,
        risk_aversion=2.5,
        tau=0.05,
        momentum_months=12,
        expected_spread=0.03,
        confidence=0.60,
        lookback_months=36,
        initial_capital=100_000.0,
        transaction_cost_bps=0.0,
    )

    # 5. Evaluate performance.
    metrics = evaluate_backtest(
        comparison_result=comparison,
        risk_free_rate=0.0,
        periods_per_year=252,
    )

    print("\n=== PERFORMANCE METRICS ===")
    print(metrics.round(4).to_string())

    print("\n=== FINAL PORTFOLIO VALUES ===")
    print(
        comparison["nav"]
        .iloc[-1]
        .round(2)
        .to_string()
    )

    print("\n=== REBALANCE COUNTS ===")

    for strategy, weights in comparison["weights"].items():
        print(
            strategy,
            len(weights),
        )

    print("\n=== OPTIMIZATION INCIDENTS ===")

    for strategy, incidents in comparison["incidents"].items():
        print(
            strategy,
            len(incidents),
        )

        if not incidents.empty:
            print(incidents.to_string(index=False))

    # 6. Save reproducible outputs.
    metrics.to_csv(
        OUTPUT_DIR / "baseline_metrics.csv"
    )

    comparison["nav"].to_csv(
        OUTPUT_DIR / "baseline_nav.csv"
    )

    comparison["portfolio_returns"].to_csv(
        OUTPUT_DIR / "baseline_returns.csv"
    )

    comparison["turnover"].to_csv(
        OUTPUT_DIR / "baseline_turnover.csv"
    )

    for strategy, weights in comparison["weights"].items():
        weights.to_csv(
            OUTPUT_DIR / f"weights_{strategy}.csv"
        )

    print("\nResults saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()