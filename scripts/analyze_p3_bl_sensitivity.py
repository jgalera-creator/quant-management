from itertools import product
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

RISK_AVERSIONS = [1.5, 2.5, 4.0]
CONFIDENCES = [0.40, 0.60, 0.80]


def main():

    returns = pd.read_csv(
        RESULTS_DIR / "asset_returns.csv",
        index_col=0,
        parse_dates=True,
    )

    benchmark = default_market_weights()

    rows = []

    for delta, confidence in product(
        RISK_AVERSIONS,
        CONFIDENCES,
    ):

        print(
            f"Running delta={delta}, "
            f"momentum confidence={confidence:.0%}"
        )

        comparison = run_strategy_comparison(
            returns=returns,
            market_weights=benchmark,
            risk_free_rate=0.0,
            risk_aversion=delta,
            tau=0.05,
            momentum_months=12,
            expected_spread=0.03,
            confidence=confidence,
            lookback_months=36,
            initial_capital=100_000.0,
            transaction_cost_bps=0.0,
            bl_view_mode="three",
            include_gld_view=True,
            include_spy_view=True,
        )

        metrics = evaluate_backtest(
            comparison_result=comparison,
            risk_free_rate=0.0,
            periods_per_year=252,
        )

        bl = metrics.loc["black_litterman"]

        rows.append({
            "risk_aversion": delta,
            "momentum_confidence": confidence,
            "cagr": bl["cagr"],
            "volatility": bl["volatility"],
            "sharpe": bl["sharpe"],
            "max_drawdown": bl["max_drawdown"],
            "annual_turnover": bl["annual_turnover"],
            "effective_assets":
                bl["average_effective_assets"],
        })

    results = pd.DataFrame(rows)

    print("\n=== CAGR SENSITIVITY ===")

    print(
        results.pivot(
            index="risk_aversion",
            columns="momentum_confidence",
            values="cagr",
        ).round(4).to_string()
    )

    print("\n=== SHARPE SENSITIVITY ===")

    print(
        results.pivot(
            index="risk_aversion",
            columns="momentum_confidence",
            values="sharpe",
        ).round(4).to_string()
    )

    print("\n=== MAX DRAWDOWN SENSITIVITY ===")

    print(
        results.pivot(
            index="risk_aversion",
            columns="momentum_confidence",
            values="max_drawdown",
        ).round(4).to_string()
    )

    results.to_csv(
        RESULTS_DIR / "bl_parameter_sensitivity.csv",
        index=False,
    )


if __name__ == "__main__":
    main()