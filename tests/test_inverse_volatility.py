import numpy as np
import pandas as pd
import pytest

from quantmgmt.optimizers.hrp import (
    inverse_volatility_weights,
)


def test_inverse_volatility_weights():

    covariance = pd.DataFrame(
        [
            [0.04, 0.00],
            [0.00, 0.01],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    weights = inverse_volatility_weights(covariance)

    assert weights["A"] == pytest.approx(1 / 3)
    assert weights["B"] == pytest.approx(2 / 3)

    assert weights.sum() == pytest.approx(1.0)


def test_inverse_volatility_rejects_zero_variance():

    covariance = pd.DataFrame(
        [
            [0.04, 0.00],
            [0.00, 0.00],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(ValueError, match="positive"):
        inverse_volatility_weights(covariance)


def test_inverse_volatility_order_invariance():

    covariance = pd.DataFrame(
        [
            [0.04, 0.005],
            [0.005, 0.01],
        ],
        index=["A", "B"],
        columns=["A", "B"],
    )

    original = inverse_volatility_weights(covariance)

    reordered = covariance.loc[
        ["B", "A"],
        ["B", "A"],
    ]

    result = inverse_volatility_weights(reordered)

    pd.testing.assert_series_equal(
        original.sort_index(),
        result.sort_index(),
    )