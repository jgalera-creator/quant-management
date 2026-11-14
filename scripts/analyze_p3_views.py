from pathlib import Path

import pandas as pd

from quantmgmt.backtesting.comparison import (
    default_market_weights,
    run_strategy_comparison,
)

from quantmgmt.backtesting.metrics import (
    evaluate_backtest,
)


RESULTS_DIR = Path("data/p3_results")

CONFIGURATIONS = {
    "BL-1": {
        "include_gld_view": False,
        "include_spy_view": False,
    },
    "BL-2A": {
        "include_gld_view": True,
        "include_spy_view": False,
    },
    "BL-2B": {
        "include_gld_view": False,
        "include_spy_view": True,
    },
    "BL-3": {
        "include_gld_view": True,
        "include_spy_view": True,
    },
}


def main():

    returns = pd.read_csv(
        RESULTS_DIR / "asset_returns.csv",
        index_col=0,
        parse_dates=True,
    )

    benchmark = default_market_weights()

    rows = []

    for name, config in CONFIGURATIONS.items():

        print(f"\nRunning {name}...")

        comparison = run_strategy_comparison(
            returns=returns,
            market_weights=benchmark,
            risk_free_rate=0.0,
            risk_aversion=2.5,
            tau=0.05,
            momentum_months=12,
            expected_spread=0.03,
            confidence=0.60,
            lookback_months=36,
            initial_capital=100_000.0,
            transaction_cost_bps=0.0,
            bl_view_mode="three",
            **config,
        )

        metrics = evaluate_backtest(
            comparison_result=comparison,
            risk_free_rate=0.0,
            periods_per_year=252,
        )

        bl_metrics = metrics.loc["black_litterman"]

        rows.append({
            "configuration": name,
            "cagr": bl_metrics["cagr"],
            "volatility": bl_metrics["volatility"],
            "sharpe": bl_metrics["sharpe"],
            "max_drawdown": bl_metrics["max_drawdown"],
            "annual_turnover": bl_metrics["annual_turnover"],
            "effective_assets": bl_metrics[
                "average_effective_assets"
            ],
            "final_nav": comparison["nav"][
                "black_litterman"
            ].iloc[-1],
        })

    result = pd.DataFrame(rows).set_index(
        "configuration"
    )

    print("\n=== BLACK-LITTERMAN VIEW ATTRIBUTION ===")
    print(result.round(4).to_string())

    result.to_csv(
        RESULTS_DIR / "bl_view_attribution.csv"
    )


if __name__ == "__main__":
    main()