
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform
from scipy.cluster.hierarchy import leaves_list


def correlation_distance(
    correlation_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert an asset correlation matrix into
    a hierarchical-clustering distance matrix.

    d_ij = sqrt((1 - rho_ij) / 2)
    """
    if not isinstance(correlation_matrix, pd.DataFrame):
        raise TypeError("correlation_matrix must be a DataFrame")

    if correlation_matrix.empty:
        raise ValueError("correlation matrix cannot be empty")

    if (
        not correlation_matrix.index.is_unique
        or not correlation_matrix.columns.is_unique
        or set(correlation_matrix.index)
        != set(correlation_matrix.columns)
    ):
        raise ValueError("invalid asset labels")

    corr = correlation_matrix.loc[
        correlation_matrix.index,
        correlation_matrix.index,
    ].to_numpy(dtype=float)

    if not np.isfinite(corr).all():
        raise ValueError("correlations must be finite")

    if not np.allclose(corr, corr.T, atol=1e-10):
        raise ValueError("correlation matrix must be symmetric")

    if (
        (corr < -1 - 1e-10).any()
        or (corr > 1 + 1e-10).any()
    ):
        raise ValueError("correlations must be in [-1, 1]")

    if not np.allclose(np.diag(corr), 1.0, atol=1e-10):
        raise ValueError("correlation diagonal must equal one")

    if np.linalg.eigvalsh(corr).min() < -1e-8:
        raise ValueError("correlation matrix must be positive semidefinite")

    distance = np.sqrt(
        np.maximum((1.0 - np.clip(corr, -1, 1)) / 2.0, 0.0)
    )
    np.fill_diagonal(distance, 0.0)

    return pd.DataFrame(
        distance,
        index=correlation_matrix.index,
        columns=correlation_matrix.index,
    )


def hierarchical_clustering(
    distance_matrix: pd.DataFrame,
    method: str = "single",
) -> np.ndarray:
    """
    Compute hierarchical clustering linkage.

    Supported: single, complete, average.
    """
    if method not in {"single", "complete", "average"}:
        raise ValueError("unsupported linkage method")

    if not isinstance(distance_matrix, pd.DataFrame):
        raise TypeError("distance_matrix must be a DataFrame")

    if len(distance_matrix) < 2:
        raise ValueError("at least two assets are required")

    if (
        not distance_matrix.index.equals(distance_matrix.columns)
        or not distance_matrix.index.is_unique
    ):
        raise ValueError("distance labels must match")

    distances = distance_matrix.to_numpy(dtype=float)

    if not np.isfinite(distances).all():
        raise ValueError("distances must be finite")

    if not np.allclose(distances, distances.T, atol=1e-10):
        raise ValueError("distance matrix must be symmetric")

    if (distances < -1e-10).any():
        raise ValueError("distances cannot be negative")

    if not np.allclose(np.diag(distances), 0.0, atol=1e-10):
        raise ValueError("distance diagonal must be zero")

    condensed = squareform(
        distances,
        checks=True,
    )

    return linkage(
        condensed,
        method=method,
        optimal_ordering=True,
    )


def quasi_diagonalize(
    linkage_matrix: np.ndarray,
    assets: pd.Index,
) -> pd.Index:
    """
    Return assets ordered according to the leaves
    of the hierarchical clustering tree.
    """
    if not isinstance(assets, pd.Index):
        raise TypeError("assets must be a pandas Index")

    if not assets.is_unique:
        raise ValueError("asset labels must be unique")

    if len(assets) < 2:
        raise ValueError("at least two assets are required")

    linkage_matrix = np.asarray(
        linkage_matrix, dtype=float
    )

    if linkage_matrix.shape != (len(assets) - 1, 4):
        raise ValueError("invalid linkage matrix shape")

    if not np.isfinite(linkage_matrix).all():
        raise ValueError("linkage must contain finite values")

    ordered_indices = leaves_list(linkage_matrix)

    return assets.take(ordered_indices)




def cluster_variance(
    covariance_matrix: pd.DataFrame,
    cluster_assets: list[str],
) -> float:
    """
    Estimate cluster variance using inverse-variance
    weights inside the cluster.

    These internal weights are used to evaluate
    cluster risk, not necessarily as final HRP weights.
    """
    if not isinstance(covariance_matrix, pd.DataFrame):
        raise TypeError("covariance_matrix must be a DataFrame")

    if not cluster_assets:
        raise ValueError("cluster cannot be empty")

    if len(cluster_assets) != len(set(cluster_assets)):
        raise ValueError("cluster assets must be unique")

    if (
        not covariance_matrix.index.is_unique
        or not covariance_matrix.columns.is_unique
        or set(covariance_matrix.index)
        != set(covariance_matrix.columns)
    ):
        raise ValueError("invalid covariance labels")

    if not set(cluster_assets).issubset(covariance_matrix.index):
        raise ValueError("unknown cluster asset")

    cov = covariance_matrix.loc[
        cluster_assets, cluster_assets
    ].to_numpy(dtype=float)

    if not np.isfinite(cov).all():
        raise ValueError("covariance must be finite")

    if not np.allclose(cov, cov.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")

    if np.linalg.eigvalsh(cov).min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")

    variances = np.diag(cov)

    if (variances <= 0).any():
        raise ValueError("asset variances must be positive")

    inverse_variances = 1.0 / variances
    weights = inverse_variances / inverse_variances.sum()

    variance = float(weights @ cov @ weights)

    if variance < -1e-10:
        raise ValueError("computed cluster variance is negative")

    return max(variance, 0.0)




def recursive_bisection(
    covariance_matrix: pd.DataFrame,
    ordered_assets: pd.Index,
) -> pd.Series:
    """
    Allocate portfolio weights using HRP recursive bisection.

    The asset ordering must come from quasi_diagonalize().
    Cluster risks are estimated using inverse-variance
    internal weights.
    """
    if not isinstance(ordered_assets, pd.Index):
        raise TypeError("ordered_assets must be a pandas Index")

    if len(ordered_assets) == 0:
        raise ValueError("ordered_assets cannot be empty")

    if not ordered_assets.is_unique:
        raise ValueError("ordered_assets must be unique")

    if (
        not covariance_matrix.index.is_unique
        or not covariance_matrix.columns.is_unique
        or set(covariance_matrix.index) != set(covariance_matrix.columns)
    ):
        raise ValueError("invalid covariance labels")

    if set(ordered_assets) != set(covariance_matrix.index):
        raise ValueError("ordered_assets must contain every asset")

    weights = pd.Series(
        1.0,
        index=ordered_assets,
        dtype=float,
    )

    clusters = [ordered_assets.tolist()]

    while clusters:
        next_clusters = []

        for cluster in clusters:
            if len(cluster) <= 1:
                continue

            midpoint = len(cluster) // 2

            left = cluster[:midpoint]
            right = cluster[midpoint:]

            left_variance = cluster_variance(
                covariance_matrix, left
            )
            right_variance = cluster_variance(
                covariance_matrix, right
            )

            total_variance = left_variance + right_variance

            if total_variance <= 0:
                raise ValueError(
                    "cluster variances must have a positive sum"
                )

            alpha = right_variance / total_variance

            weights.loc[left] *= alpha
            weights.loc[right] *= 1.0 - alpha

            if len(left) > 1:
                next_clusters.append(left)

            if len(right) > 1:
                next_clusters.append(right)

        clusters = next_clusters

    return weights.rename("weight")



def hrp_weights(
    covariance_matrix: pd.DataFrame,
    linkage_method: str = "single",
) -> pd.Series:
    """
    Compute Hierarchical Risk Parity portfolio weights.

    Parameters
    ----------
    covariance_matrix : pd.DataFrame
        Asset covariance matrix.
    linkage_method : str
        Hierarchical clustering method:
        single, complete or average.

    Returns
    -------
    pd.Series
        Long-only HRP weights, indexed by original asset order.
    """
    if not isinstance(covariance_matrix, pd.DataFrame):
        raise TypeError("covariance_matrix must be a DataFrame")

    if covariance_matrix.empty:
        raise ValueError("covariance matrix cannot be empty")

    assets = covariance_matrix.index

    if (
        not assets.is_unique
        or not covariance_matrix.columns.is_unique
        or set(assets) != set(covariance_matrix.columns)
    ):
        raise ValueError("invalid covariance labels")

    covariance_matrix = covariance_matrix.loc[assets, assets]

    sigma = covariance_matrix.to_numpy(dtype=float)

    if not np.isfinite(sigma).all():
        raise ValueError("covariance must be finite")

    if not np.allclose(sigma, sigma.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")

    if np.linalg.eigvalsh(sigma).min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")

    variances = np.diag(sigma)

    if (variances <= 0).any():
        raise ValueError("asset variances must be positive")

    if len(assets) == 1:
        return pd.Series(
            [1.0], index=assets, name="weight"
        )

    # 1. Covariance -> correlation
    std = np.sqrt(variances)
    corr_values = sigma / np.outer(std, std)

    corr_values = np.clip(corr_values, -1.0, 1.0)
    np.fill_diagonal(corr_values, 1.0)

    correlation = pd.DataFrame(
        corr_values,
        index=assets,
        columns=assets,
    )

    # 2. Correlation -> distances
    distance = correlation_distance(correlation)

    # 3. Hierarchical clustering
    linkage_matrix = hierarchical_clustering(
        distance,
        method=linkage_method,
    )

    # 4. Quasi-diagonalization
    ordered_assets = quasi_diagonalize(
        linkage_matrix,
        assets,
    )

    # 5. Recursive bisection
    weights = recursive_bisection(
        covariance_matrix,
        ordered_assets,
    )

    # Restore original asset ordering
    weights = weights.reindex(assets)

    if not np.isfinite(weights.to_numpy()).all():
        raise ValueError("non-finite portfolio weights")

    if (weights < -1e-10).any():
        raise ValueError("negative HRP weights")

    if not np.isclose(weights.sum(), 1.0, atol=1e-8):
        raise ValueError("HRP weights do not sum to one")

    return weights.rename("weight")




def risk_contribution(
    weights: pd.Series,
    covariance_matrix: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compute Euler contributions to portfolio volatility.

    Returns:
        weight: capital allocation
        marginal_risk: marginal contribution to volatility
        risk_contribution: absolute volatility contribution
        risk_contribution_pct: fraction of total volatility
    """
    if not isinstance(weights, pd.Series):
        raise TypeError("weights must be a pandas Series")

    if not isinstance(covariance_matrix, pd.DataFrame):
        raise TypeError("covariance_matrix must be a DataFrame")

    assets = covariance_matrix.index

    if (
        not assets.is_unique
        or not covariance_matrix.columns.is_unique
        or not weights.index.is_unique
        or set(assets) != set(covariance_matrix.columns)
        or set(assets) != set(weights.index)
    ):
        raise ValueError("asset labels do not match")

    if len(assets) == 0:
        raise ValueError("portfolio cannot be empty")

    sigma = covariance_matrix.loc[
        assets, assets
    ].to_numpy(dtype=float)

    w = weights.reindex(assets).to_numpy(dtype=float)

    if not np.isfinite(sigma).all() or not np.isfinite(w).all():
        raise ValueError("inputs must be finite")

    if not np.allclose(sigma, sigma.T, atol=1e-10):
        raise ValueError("covariance must be symmetric")

    if np.linalg.eigvalsh(sigma).min() < -1e-10:
        raise ValueError("covariance must be positive semidefinite")

    if not np.isclose(w.sum(), 1.0, atol=1e-8):
        raise ValueError("weights must sum to one")

    portfolio_variance = float(w @ sigma @ w)

    if portfolio_variance <= 0:
        raise ValueError("portfolio variance must be positive")

    portfolio_volatility = np.sqrt(portfolio_variance)

    marginal = (sigma @ w) / portfolio_volatility
    absolute = w * marginal
    relative = absolute / portfolio_volatility

    return pd.DataFrame(
        {
            "weight": w,
            "marginal_risk": marginal,
            "risk_contribution": absolute,
            "risk_contribution_pct": relative,
        },
        index=assets,
    )



def inverse_volatility_weights(
    covariance_matrix: pd.DataFrame,
) -> pd.Series:
    """
    Naive inverse-volatility portfolio.

    Weights are proportional to the inverse
    of individual asset volatility.
    """
    if not isinstance(covariance_matrix, pd.DataFrame):
        raise TypeError(
            "covariance_matrix must be a DataFrame"
        )

    if covariance_matrix.empty:
        raise ValueError(
            "covariance_matrix cannot be empty"
        )

    if (
        not covariance_matrix.index.is_unique
        or not covariance_matrix.columns.is_unique
        or not covariance_matrix.index.equals(
            covariance_matrix.columns
        )
    ):
        raise ValueError(
            "covariance labels must be aligned and unique"
        )

    diagonal = np.diag(
        covariance_matrix.to_numpy(dtype=float)
    )

    if not np.isfinite(diagonal).all():
        raise ValueError(
            "invalid covariance diagonal"
        )

    if (diagonal <= 0).any():
        raise ValueError(
            "asset variances must be positive"
        )

    volatilities = np.sqrt(diagonal)

    inverse_vol = 1.0 / volatilities

    weights = inverse_vol / inverse_vol.sum()

    return pd.Series(
        weights,
        index=covariance_matrix.index,
        name="weight",
    )