from pathlib import Path

import numpy as np
import pandas as pd

from quantmgmt.optimizers.hrp import (
    hrp_weights,
    inverse_volatility_weights,
    risk_contribution,
)

from quantmgmt.backtesting.walk_forward import (
    generate_walk_forward_weights,
)

from quantmgmt.backtesting.engine import (
    simulate_portfolio,
)

from quantmgmt.backtesting.metrics import (
    concentration_metrics,
)


RESULTS_DIR = Path("data/p3_results")


def main():

    returns = pd.read_csv(
        RESULTS_DIR / "asset_returns.csv",
        index_col=0,
        parse_dates=True,
    )

    def hrp_strategy(window, date):
        covariance = window.cov() * 252
        return hrp_weights(covariance)

    def ivp_strategy(window, date):
        covariance = window.cov() * 252
        return inverse_volatility_weights(covariance)

    strategies = {
        "HRP": hrp_strategy,
        "IVP": ivp_strategy,
    }

    summary = []

    for name, strategy in strategies.items():

        print(f"\nRunning {name}...")

        weights = generate_walk_forward_weights(
            returns=returns,
            weight_function=strategy,
            lookback_months=36,
        )

        result = simulate_portfolio(
            returns=returns,
            target_weights=weights,
            initial_capital=100_000.0,
            transaction_cost_bps=0.0,
        )

        portfolio_returns = result["portfolio_returns"]

        # Risk metrics
        from quantmgmt.risk.returns import cagr
        from quantmgmt.risk.volatility import annualized_volatility
        from quantmgmt.risk.ratios import sharpe_ratio
        from quantmgmt.risk.drawdown import max_drawdown

        metrics = {
            "strategy": name,
            "cagr": float(cagr(portfolio_returns)),
            "volatility": float(
                annualized_volatility(portfolio_returns)
            ),
            "sharpe": float(
                sharpe_ratio(portfolio_returns)
            ),
            "max_drawdown": float(
                max_drawdown(portfolio_returns)
            ),
            "final_nav": float(result["nav"].iloc[-1]),
        }

        concentration = concentration_metrics(weights)

        metrics["average_effective_assets"] = (
            concentration["effective_assets"].mean()
        )

        # Risk contributions at the latest rebalance.
        last_date = weights.index[-1]

        start = last_date - pd.DateOffset(months=36)

        window = returns.loc[
            (returns.index > start)
            & (returns.index <= last_date)
        ]

        covariance = window.cov() * 252

        latest_weights = weights.iloc[-1]

        rc = risk_contribution(
            latest_weights,
            covariance,
        )

        print("\nLatest weights:")
        print(latest_weights.sort_values(ascending=False).round(4))

        print("\nRisk contributions:")
        print(rc)

        summary.append(metrics)

        weights.to_csv(
            RESULTS_DIR / f"weights_{name.lower()}.csv"
        )

        result["nav"].to_csv(
            RESULTS_DIR / f"nav_{name.lower()}.csv"
        )

    summary = pd.DataFrame(summary).set_index("strategy")

    print("\n=== HRP VS IVP COMPARISON ===")
    print(summary.round(4).to_string())

    summary.to_csv(
        RESULTS_DIR / "hrp_vs_ivp.csv"
    )


if __name__ == "__main__":
    main()