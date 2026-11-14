import numpy as np
import pandas as pd


def weight_diagnostics(
    weights: pd.DataFrame,
) -> pd.DataFrame:
    """
    Analyze portfolio target-weight stability.

    Returns asset-level statistics across rebalance dates.
    """
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a DataFrame")

    if weights.empty:
        raise ValueError("weights cannot be empty")

    values = weights.to_numpy(dtype=float)

    if not np.isfinite(values).all():
        raise ValueError("weights must be finite")

    if (values < -1e-8).any():
        raise ValueError("negative weights are not allowed")

    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-8):
        raise ValueError("weights must sum to one")

    return pd.DataFrame(
        {
            "mean_weight": weights.mean(),
            "median_weight": weights.median(),
            "max_weight": weights.max(),
            "min_weight": weights.min(),
            "weight_std": weights.std(ddof=0),
            "latest_weight": weights.iloc[-1],
            "zero_weight_pct": (
                weights.abs() < 1e-8
            ).mean(),
        }
    )


def weight_stability(
    weights: pd.DataFrame,
) -> pd.Series:
    """
    Compute target-weight changes between rebalances.

    Uses L1 distance between consecutive target allocations.

    This is a diagnostic measure, not realized turnover.
    """
    if not isinstance(weights, pd.DataFrame):
        raise TypeError("weights must be a DataFrame")

    if weights.empty:
        raise ValueError("weights cannot be empty")

    if not weights.index.is_monotonic_increasing:
        raise ValueError("dates must be sorted")

    if not weights.index.is_unique:
        raise ValueError("dates must be unique")

    changes = weights.diff().abs().sum(axis=1)

    changes.iloc[0] = 0.0

    return changes.rename("target_weight_change")