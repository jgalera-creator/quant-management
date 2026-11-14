from pathlib import Path

import numpy as np
import pandas as pd


RESULTS_DIR = Path("data/p3_results")

STRATEGIES = [
    "equal_weight",
    "min_variance",
    "max_sharpe",
    "black_litterman",
    "hrp",
]


def calculate_risk_contributions(
    weights: pd.Series,
    covariance: pd.DataFrame,
) -> pd.Series:
    """
    Percentage contribution of each asset
    to total portfolio volatility.
    """
    assets = covariance.index

    w = weights.reindex(assets).to_numpy(dtype=float)

    sigma = covariance.loc[
        assets, assets
    ].to_numpy(dtype=float)

    portfolio_variance = float(w @ sigma @ w)

    if not np.isfinite(portfolio_variance):
        raise ValueError("Non-finite portfolio variance")

    if portfolio_variance <= 0:
        raise ValueError("Portfolio variance must be positive")

    marginal = sigma @ w

    contributions = (
        w * marginal / portfolio_variance
    )

    return pd.Series(
        contributions,
        index=assets,
        name="risk_contribution_pct",
    )


def main():

    returns = pd.read_csv(
        RESULTS_DIR / "asset_returns.csv",
        index_col=0,
        parse_dates=True,
    )

    summary = []

    for strategy in STRATEGIES:

        weights = pd.read_csv(
            RESULTS_DIR / f"weights_{strategy}.csv",
            index_col=0,
            parse_dates=True,
        )

        contributions_history = {}

        for date, portfolio_weights in weights.iterrows():

            window_start = (
                date - pd.DateOffset(months=36)
            )

            historical_returns = returns.loc[
                (returns.index > window_start)
                & (returns.index <= date)
            ]

            covariance = historical_returns.cov() * 252

            contributions = calculate_risk_contributions(
                portfolio_weights,
                covariance,
            )

            contributions_history[date] = contributions

        risk_history = pd.DataFrame.from_dict(
            contributions_history,
            orient="index",
        )

        np.testing.assert_allclose(
            risk_history.sum(axis=1).to_numpy(),
            np.ones(len(risk_history)),
            atol=1e-8,
        )

        average_risk = risk_history.mean()

        latest_risk = risk_history.iloc[-1]

        print(f"\n{'=' * 60}")
        print(f"STRATEGY: {strategy.upper()}")
        print("=" * 60)

        print("\nAVERAGE RISK CONTRIBUTIONS")

        print(
            (average_risk * 100)
            .sort_values(ascending=False)
            .round(2)
            .to_string()
        )

        print("\nLATEST RISK CONTRIBUTIONS")

        print(
            (latest_risk * 100)
            .sort_values(ascending=False)
            .round(2)
            .to_string()
        )

        largest_average_contribution = (
            average_risk.max()
        )

        summary.append({
            "strategy": strategy,
            "largest_average_risk_contribution":
                largest_average_contribution,
            "largest_latest_risk_contribution":
                latest_risk.max(),
        })

        risk_history.to_csv(
            RESULTS_DIR
            / f"risk_contributions_{strategy}.csv"
        )

    summary = pd.DataFrame(summary).set_index(
        "strategy"
    )

    print("\n=== RISK CONCENTRATION COMPARISON ===")

    print(
        (summary * 100)
        .round(2)
        .to_string()
    )

    summary.to_csv(
        RESULTS_DIR / "risk_concentration_summary.csv"
    )


if __name__ == "__main__":
    main()