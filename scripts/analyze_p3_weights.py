from pathlib import Path

import pandas as pd

from quantmgmt.backtesting.diagnostics import (
    weight_diagnostics,
    weight_stability,
)


RESULTS_DIR = Path("data/p3_results")

STRATEGIES = [
    "equal_weight",
    "min_variance",
    "max_sharpe",
    "black_litterman",
    "hrp",
]


def main():

    summary = []

    for strategy in STRATEGIES:

        file = RESULTS_DIR / f"weights_{strategy}.csv"

        weights = pd.read_csv(
            file,
            index_col=0,
            parse_dates=True,
        )

        diagnostics = weight_diagnostics(weights)

        stability = weight_stability(weights)

        print(f"\n{'=' * 60}")
        print(f"STRATEGY: {strategy.upper()}")
        print("=" * 60)

        print("\nTOP 5 ASSETS BY AVERAGE WEIGHT")

        print(
            diagnostics
            .sort_values("mean_weight", ascending=False)
            .head(5)
            .round(4)
            .to_string()
        )

        print("\nLATEST PORTFOLIO WEIGHTS")

        print(
            diagnostics["latest_weight"]
            .sort_values(ascending=False)
            .round(4)
            .to_string()
        )

        average_change = stability.iloc[1:].mean()
        maximum_change = stability.max()

        print("\nWEIGHT STABILITY")

        print(f"Average change: {average_change:.4f}")
        print(f"Maximum change: {maximum_change:.4f}")

        summary.append({
            "strategy": strategy,
            "average_target_change": average_change,
            "maximum_target_change": maximum_change,
            "largest_average_weight": diagnostics[
                "mean_weight"
            ].max(),
            "largest_historical_weight": diagnostics[
                "max_weight"
            ].max(),
        })

        diagnostics.to_csv(
            RESULTS_DIR / f"diagnostics_{strategy}.csv"
        )

    summary = pd.DataFrame(summary).set_index("strategy")

    print("\n=== STRATEGY STABILITY COMPARISON ===")
    print(summary.round(4).to_string())

    summary.to_csv(
        RESULTS_DIR / "weight_stability_summary.csv"
    )


if __name__ == "__main__":
    main()