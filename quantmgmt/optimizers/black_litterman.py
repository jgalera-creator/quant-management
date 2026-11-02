
import numpy as np
import pandas as pd


def implied_equilibrium_returns(
    covariance_matrix: pd.DataFrame,
    market_weights: pd.Series,
    risk_aversion: float,
) -> pd.Series:
    """
    Compute Black-Litterman implied equilibrium returns.

    pi = delta * Sigma @ w_market

    All inputs must use compatible annualization.
    """
    if risk_aversion <= 0 or not np.isfinite(risk_aversion):
        raise ValueError("risk_aversion must be positive and finite")

    if not covariance_matrix.index.equals(covariance_matrix.columns):
        raise ValueError("covariance labels must match")

    if not covariance_matrix.index.is_unique:
        raise ValueError("asset labels must be unique")

    if not market_weights.index.is_unique:
        raise ValueError("market weight labels must be unique")

    if set(market_weights.index) != set(covariance_matrix.index):
        raise ValueError("market weights and covariance assets differ")

    weights = market_weights.reindex(covariance_matrix.index)

    if not np.isfinite(covariance_matrix.to_numpy()).all():
        raise ValueError("covariance must be finite")

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("weights must be finite")

    if (weights < 0).any():
        raise ValueError("market weights must be non-negative")

    if not np.isclose(weights.sum(), 1.0):
        raise ValueError("market weights must sum to one")

    pi = (
        risk_aversion
        * covariance_matrix.to_numpy()
        @ weights.to_numpy()
    )

    return pd.Series(
        pi,
        index=covariance_matrix.index,
        name="implied_return",
    )



def posterior_returns(
    equilibrium_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    P: pd.DataFrame,
    Q: pd.Series,
    omega: pd.DataFrame,
    tau: float = 0.05,
) -> pd.Series:
    """
    Compute Black-Litterman posterior expected returns.

    All returns and covariance inputs must share
    the same annualization convention.

    P rows identify views and P columns identify assets.
    Q and omega use the same view labels.
    """
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive and finite")

    assets = covariance_matrix.index
    views = P.index

    if not all(isinstance(x, (pd.Series, pd.DataFrame))
               for x in [equilibrium_returns,
                         covariance_matrix, P, Q, omega]):
        raise TypeError("inputs must be pandas objects")

    if any(not idx.is_unique for idx in [
        assets, covariance_matrix.columns,
        equilibrium_returns.index,
        P.columns, views,
        Q.index, omega.index, omega.columns,
    ]):
        raise ValueError("asset and view labels must be unique")

    if set(assets) != set(covariance_matrix.columns):
        raise ValueError("covariance labels do not match")

    if set(assets) != set(equilibrium_returns.index):
        raise ValueError("equilibrium return labels do not match")

    if set(assets) != set(P.columns):
        raise ValueError("P asset labels do not match")

    if len(views) == 0 or set(views) != set(Q.index):
        raise ValueError("view labels do not match Q")

    if set(views) != set(omega.index) or set(views) != set(omega.columns):
        raise ValueError("omega view labels do not match")

    sigma = covariance_matrix.loc[assets, assets].to_numpy(dtype=float)
    pi = equilibrium_returns.reindex(assets).to_numpy(dtype=float)
    p = P.reindex(columns=assets).to_numpy(dtype=float)
    q = Q.reindex(views).to_numpy(dtype=float)
    om = omega.loc[views, views].to_numpy(dtype=float)

    if not all(np.isfinite(a).all() for a in [sigma, pi, p, q, om]):
        raise ValueError("all inputs must be finite")

    if not np.allclose(sigma, sigma.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")

    if not np.allclose(om, om.T, atol=1e-10):
        raise ValueError("omega must be symmetric")

    if np.linalg.eigvalsh(sigma).min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")

    if np.linalg.eigvalsh(om).min() < -1e-10:
        raise ValueError("omega must be positive semidefinite")

    prior_uncertainty = tau * sigma
    view_uncertainty = p @ prior_uncertainty @ p.T + om

    try:
        correction = (
            prior_uncertainty
            @ p.T
            @ np.linalg.solve(
                view_uncertainty,
                q - p @ pi,
            )
        )
    except np.linalg.LinAlgError as exc:
        raise ValueError(
            "view uncertainty system is singular"
        ) from exc

    return pd.Series(
        pi + correction,
        index=assets,
        name="posterior_return",
    )



def build_omega(
    covariance_matrix: pd.DataFrame,
    P: pd.DataFrame,
    tau: float = 0.05,
    confidences: pd.Series | None = None,
) -> pd.DataFrame:
    """
    Construct a diagonal Black-Litterman view uncertainty matrix.

    If confidences is None:
        omega_i = tau * p_i @ Sigma @ p_i.T

    Otherwise, for 0 < confidence_i <= 1:
        omega_i = ((1 - confidence_i) / confidence_i)
                  * tau * p_i @ Sigma @ p_i.T

    Confidence 1 means zero view uncertainty.
    """
    if not np.isfinite(tau) or tau <= 0:
        raise ValueError("tau must be positive and finite")

    assets = covariance_matrix.index

    if (
        not covariance_matrix.index.is_unique
        or not covariance_matrix.columns.is_unique
        or not P.index.is_unique
        or not P.columns.is_unique
    ):
        raise ValueError("asset and view labels must be unique")

    if (
        set(assets) != set(covariance_matrix.columns)
        or set(assets) != set(P.columns)
    ):
        raise ValueError("asset labels do not match")

    if P.empty:
        raise ValueError("at least one view is required")

    sigma = covariance_matrix.loc[
        assets, assets
    ].to_numpy(dtype=float)

    p = P.reindex(
        columns=assets
    ).to_numpy(dtype=float)

    if not np.isfinite(sigma).all() or not np.isfinite(p).all():
        raise ValueError("inputs must be finite")

    if not np.allclose(sigma, sigma.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")

    if np.linalg.eigvalsh(sigma).min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")

    view_variances = np.diag(
        p @ (tau * sigma) @ p.T
    )

    if (view_variances <= 0).any():
        raise ValueError("view variances must be positive")

    if confidences is None:
        diagonal = view_variances
    else:
        if (
            not confidences.index.is_unique
            or set(confidences.index) != set(P.index)
        ):
            raise ValueError("confidence labels do not match views")

        c = confidences.reindex(P.index).to_numpy(dtype=float)

        if not np.isfinite(c).all() or ((c <= 0) | (c > 1)).any():
            raise ValueError("confidences must be in (0, 1]")

        diagonal = ((1 - c) / c) * view_variances

    return pd.DataFrame(
        np.diag(diagonal),
        index=P.index,
        columns=P.index,
    )




def estimate_risk_aversion(
    market_returns: pd.Series,
    risk_free_rate: float = 0.0,
    periods_per_year: int = 252,
) -> float:
    """
    Estimate market-implied risk aversion:

        delta = (E[Rm] - Rf) / Var(Rm)

    Uses annualized arithmetic mean and sample variance.
    risk_free_rate is an annualized rate.
    """
    if not isinstance(market_returns, pd.Series):
        raise TypeError("market_returns must be a Series")

    if periods_per_year <= 0:
        raise ValueError("periods_per_year must be positive")

    if not np.isfinite(risk_free_rate):
        raise ValueError("risk_free_rate must be finite")

    returns = market_returns.dropna().astype(float)

    if len(returns) < 2 or not np.isfinite(returns).all():
        raise ValueError("at least two finite observations required")

    annual_mean = returns.mean() * periods_per_year
    annual_variance = returns.var(ddof=1) * periods_per_year

    if annual_variance <= 0:
        raise ValueError("market variance must be positive")

    delta = (
        annual_mean - risk_free_rate
    ) / annual_variance

    if not np.isfinite(delta) or delta <= 0:
        raise ValueError(
            "estimated risk aversion is not positive; "
            "review the estimation window or use an "
            "explicit, justified calibration"
        )

    return float(delta)
