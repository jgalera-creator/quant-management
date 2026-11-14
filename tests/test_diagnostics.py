import numpy as np
import pandas as pd
import pytest

from quantmgmt.backtesting.diagnostics import (
    weight_diagnostics,
    weight_stability,
)


def test_weight_diagnostics():

    dates = pd.bdate_range(
        "2024-01-01",
        periods=3,
    )

    weights = pd.DataFrame(
        {
            "A": [0.5, 0.6, 0.7],
            "B": [0.5, 0.4, 0.3],
        },
        index=dates,
    )

    result = weight_diagnostics(weights)

    assert result.loc["A", "mean_weight"] == pytest.approx(
        0.6
    )

    assert result.loc["B", "latest_weight"] == pytest.approx(
        0.3
    )


def test_weight_stability():

    dates = pd.bdate_range(
        "2024-01-01",
        periods=3,
    )

    weights = pd.DataFrame(
        {
            "A": [0.5, 0.6, 0.7],
            "B": [0.5, 0.4, 0.3],
        },
        index=dates,
    )

    changes = weight_stability(weights)

    np.testing.assert_allclose(
        changes.to_numpy(),
        [0.0, 0.2, 0.2],
        atol=1e-12,
    )


def test_weight_diagnostics_rejects_invalid_sum():

    weights = pd.DataFrame(
        {
            "A": [0.8],
            "B": [0.8],
        }
    )

    with pytest.raises(ValueError, match="sum to one"):
        weight_diagnostics(weights)