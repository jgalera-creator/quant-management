import numpy as np
import pandas as pd
import pytest

from quantmgmt.optimizers.mean_variance import estimate_inputs


def test_estimate_inputs_expected_returns():
    returns = pd.DataFrame(
        {
            "Asset_A": [0.01, 0.02, 0.03],
            "Asset_B": [-0.01, 0.01, 0.02],
        }
    )

    expected_returns, _ = estimate_inputs(
        returns,
        periods_per_year=12,
    )

    expected = returns.mean() * 12

    pd.testing.assert_series_equal(
        expected_returns,
        expected,
    )


def test_estimate_inputs_covariance():
    returns = pd.DataFrame(
        {
            "Asset_A": [0.01, 0.02, 0.03],
            "Asset_B": [-0.01, 0.01, 0.02],
        }
    )

    _, covariance = estimate_inputs(
        returns,
        periods_per_year=12,
    )

    expected = returns.cov() * 12

    pd.testing.assert_frame_equal(
        covariance,
        expected,
    )


def test_covariance_matrix_is_symmetric():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, -0.01, 0.03],
            "B": [0.02, -0.01, 0.01, 0.04],
            "C": [-0.01, 0.00, 0.02, 0.01],
        }
    )

    _, covariance = estimate_inputs(returns)

    assert np.allclose(
        covariance.values,
        covariance.values.T,
    )


def test_estimate_inputs_uses_common_sample():
    returns = pd.DataFrame(
        {
            "A": [0.01, 0.02, 0.03, 0.04],
            "B": [0.01, np.nan, 0.02, 0.03],
        }
    )

    expected_returns, covariance = estimate_inputs(
        returns,
        periods_per_year=12,
    )

    clean = returns.dropna()

    pd.testing.assert_series_equal(
        expected_returns,
        clean.mean() * 12,
    )

    pd.testing.assert_frame_equal(
        covariance,
        clean.cov() * 12,
    )



def test_estimate_inputs_invalid_input():
    with pytest.raises(TypeError):
        estimate_inputs(
            [0.01, 0.02]
        )


def test_estimate_inputs_invalid_periods():
    returns = pd.DataFrame({
        "A": [0.01, 0.02],
    })

    with pytest.raises(ValueError):
        estimate_inputs(
            returns,
            periods_per_year=0,
        )


def test_estimate_inputs_requires_two_observations():
    returns = pd.DataFrame({
        "A": [0.01],
        "B": [0.02],
    })

    with pytest.raises(ValueError):
        estimate_inputs(returns)