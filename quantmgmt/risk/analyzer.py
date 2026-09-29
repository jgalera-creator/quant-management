import pandas as pd

from quantmgmt.risk.custom import capital_resilience_ratio
from quantmgmt.risk.drawdown import (
    drawdown_series,
    max_drawdown,
    recovery_time,
    time_under_water,
)
from quantmgmt.risk.ratios import (
    calmar_ratio,
    sharpe_ratio,
    sortino_ratio,
    tracking_error,
)
from quantmgmt.risk.returns import (
    annualized_return,
    cagr,
    cumulative_returns,
)
from quantmgmt.risk.tail import (
    cvar_historical,
    cvar_parametric,
    kurtosis,
    skewness,
    var_historical,
    var_parametric,
)
from quantmgmt.risk.volatility import (
    annualized_volatility,
    downside_deviation,
    rolling_volatility,
)


class RiskAnalyzer:
    """
    High-level interface for portfolio and asset risk analysis.

    The class stores a return series and delegates all mathematical
    calculations to the functional risk API.
    """

    def __init__(
        self,
        returns: pd.Series | pd.DataFrame,
        periods_per_year: int = 252,
    ):
        if not isinstance(returns, (pd.Series, pd.DataFrame)):
            raise TypeError(
                "returns must be a pandas Series or DataFrame"
            )

        if periods_per_year <= 0:
            raise ValueError(
                "periods_per_year must be positive"
            )

        self.returns = returns
        self.periods_per_year = periods_per_year


    def cumulative_returns(self):
        return cumulative_returns(self.returns)

    def annualized_return(self):
        return annualized_return(
            self.returns,
            periods_per_year=self.periods_per_year,
        )

    def cagr(self):
        return cagr(
            self.returns,
            periods_per_year=self.periods_per_year,
        )

    def volatility(self):
        return annualized_volatility(
            self.returns,
            periods_per_year=self.periods_per_year,
        )

    def downside_deviation(
        self,
        target_return: float = 0.0,
    ):
        return downside_deviation(
            self.returns,
            target_return=target_return,
            periods_per_year=self.periods_per_year,
        )

    def rolling_volatility(
        self,
        window: int = 20,
    ):
        return rolling_volatility(
            self.returns,
            window=window,
            periods_per_year=self.periods_per_year,
        )



    def drawdown_series(self):
        return drawdown_series(self.returns)

    def max_drawdown(self):
        return max_drawdown(self.returns)

    def time_under_water(self):
        return time_under_water(self.returns)

    def recovery_time(self):
        return recovery_time(self.returns)



    def sharpe(
        self,
        risk_free_rate: float = 0.0,
    ):
        return sharpe_ratio(
            self.returns,
            risk_free_rate=risk_free_rate,
            periods_per_year=self.periods_per_year,
        )

    def sortino(
        self,
        target_return: float = 0.0,
    ):
        return sortino_ratio(
            self.returns,
            target_return=target_return,
            periods_per_year=self.periods_per_year,
        )

    def calmar(self):
        return calmar_ratio(
            self.returns,
            periods_per_year=self.periods_per_year,
        )

    def tracking_error(
        self,
        benchmark: pd.Series | pd.DataFrame,
    ):
        return tracking_error(
            self.returns,
            benchmark,
            periods_per_year=self.periods_per_year,
        )


    def var(
        self,
        confidence_level: float = 0.95,
        method: str = "historical",
    ):
        if method == "historical":
            return var_historical(
                self.returns,
                confidence_level=confidence_level,
            )

        if method in {
            "gaussian",
            "cornish-fisher",
        }:
            return var_parametric(
                self.returns,
                confidence_level=confidence_level,
                method=method,
            )

        raise ValueError(
            "method must be 'historical', "
            "'gaussian' or 'cornish-fisher'"
        )

    def cvar(
        self,
        confidence_level: float = 0.95,
        method: str = "historical",
    ):
        if method == "historical":
            return cvar_historical(
                self.returns,
                confidence_level=confidence_level,
            )

        if method == "gaussian":
            return cvar_parametric(
                self.returns,
                confidence_level=confidence_level,
            )

        raise ValueError(
            "method must be 'historical' or 'gaussian'"
        )

    def skewness(self):
        return skewness(self.returns)

    def kurtosis(self):
        return kurtosis(self.returns)


    def capital_resilience(
        self,
        confidence_level: float = 0.95,
    ):
        return capital_resilience_ratio(
            self.returns,
            confidence_level=confidence_level,
            periods_per_year=self.periods_per_year,
        )



    def summary(
        self,
        risk_free_rate: float = 0.0,
        target_return: float = 0.0,
        confidence_level: float = 0.95,
    ) -> pd.DataFrame:
        """
        Build a summary table with the main return and risk metrics.

        Parameters
        ----------
        risk_free_rate : float, default=0.0
            Annual risk-free rate used for Sharpe ratio.
        target_return : float, default=0.0
            Annual minimum acceptable return used for Sortino ratio.
        confidence_level : float, default=0.95
            Confidence level used for VaR, CVaR and CRR.

        Returns
        -------
        pd.DataFrame
            Summary table with one row per asset or portfolio.
        """

        metrics = {
            "Annualized Return": self.annualized_return(),
            "CAGR": self.cagr(),
            "Volatility": self.volatility(),
            "Downside Deviation": self.downside_deviation(
                target_return=0.0,
            ),
            "Sharpe": self.sharpe(
                risk_free_rate=risk_free_rate,
            ),
            "Sortino": self.sortino(
                target_return=target_return,
            ),
            "Max Drawdown": self.max_drawdown(),
            "Recovery Time": self.recovery_time(),
            "Skewness": self.skewness(),
            "Kurtosis": self.kurtosis(),
            "VaR Historical": self.var(
                confidence_level=confidence_level,
                method="historical",
            ),
            "VaR Gaussian": self.var(
                confidence_level=confidence_level,
                method="gaussian",
            ),
            "VaR Cornish-Fisher": self.var(
                confidence_level=confidence_level,
                method="cornish-fisher",
            ),
            "CVaR Historical": self.cvar(
                confidence_level=confidence_level,
                method="historical",
            ),
            "CVaR Gaussian": self.cvar(
                confidence_level=confidence_level,
                method="gaussian",
            ),
            "Calmar": self.calmar(),
            "Capital Resilience": self.capital_resilience(
                confidence_level=confidence_level,
            ),
        }

        if isinstance(self.returns, pd.Series):
            name = (
                self.returns.name
                if self.returns.name is not None
                else "Portfolio"
            )

            return pd.DataFrame(
                {
                    metric: [value]
                    for metric, value in metrics.items()
                },
                index=[name],
            )

        return pd.DataFrame(metrics)