import numpy as np
import pandas as pd
import pytest

from statistics import NormalDist

from quantmgmt.risk.tail import (
    cvar_historical,
    cvar_parametric,
    kurtosis,
    skewness,
    var_historical,
    var_parametric,
)


def test_skewness_series():
    returns = pd.Series([
        -0.08,
        -0.02,
        -0.01,
        0.00,
        0.01,
        0.02,
    ])

    result = skewness(returns)

    expected = returns.skew()

    assert np.isclose(result, expected)


def test_skewness_detects_negative_asymmetry():
    returns = pd.Series([
        -0.20,
        -0.03,
        -0.02,
        0.01,
        0.02,
        0.03,
        0.04,
    ])

    result = skewness(returns)

    assert result < 0


def test_skewness_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [
                -0.08,
                -0.02,
                0.01,
                0.02,
                0.03,
            ],
            "Asset_B": [
                -0.02,
                -0.01,
                0.00,
                0.01,
                0.02,
            ],
        }
    )

    result = skewness(returns)

    expected = returns.skew()

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_kurtosis_series():
    returns = pd.Series([
        -0.10,
        -0.02,
        -0.01,
        0.00,
        0.01,
        0.02,
        0.10,
    ])

    result = kurtosis(returns)

    expected = returns.kurt()

    assert np.isclose(result, expected)



def test_kurtosis_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [
                -0.10,
                -0.02,
                0.00,
                0.02,
                0.10,
            ],
            "Asset_B": [
                -0.02,
                -0.01,
                0.00,
                0.01,
                0.02,
            ],
        }
    )

    result = kurtosis(returns)

    expected = returns.kurt()

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_skewness_invalid_input():
    with pytest.raises(TypeError):
        skewness([0.01, -0.01])


def test_kurtosis_invalid_input():
    with pytest.raises(TypeError):
        kurtosis([0.01, -0.01])



def test_var_historical_series():
    returns = pd.Series([
        -0.10,
        -0.05,
        -0.02,
        0.01,
        0.03,
        0.04,
    ])

    confidence_level = 0.95

    result = var_historical(
        returns,
        confidence_level=confidence_level,
    )

    expected = -returns.quantile(
        1 - confidence_level
    )

    assert np.isclose(result, expected)



def test_var_historical_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [
                -0.10,
                -0.05,
                -0.02,
                0.01,
                0.03,
            ],
            "Asset_B": [
                -0.04,
                -0.02,
                -0.01,
                0.02,
                0.04,
            ],
        }
    )

    result = var_historical(
        returns,
        confidence_level=0.95,
    )

    expected = -returns.quantile(0.05)

    pd.testing.assert_series_equal(
        result,
        expected,
    )


def test_var_historical_invalid_confidence_level():
    returns = pd.Series([0.01, -0.01])

    with pytest.raises(ValueError):
        var_historical(
            returns,
            confidence_level=1.0,
        )


def test_var_historical_invalid_input():
    with pytest.raises(TypeError):
        var_historical([0.01, -0.01])



def test_var_parametric_series():
    returns = pd.Series([
        -0.03,
        -0.01,
        0.00,
        0.01,
        0.02,
        0.03,
    ])

    confidence_level = 0.95

    alpha = 1 - confidence_level
    z_score = NormalDist().inv_cdf(alpha)

    expected_quantile = (
        returns.mean()
        + z_score * returns.std()
    )

    expected = max(
        0.0,
        -expected_quantile,
    )

    result = var_parametric(
        returns,
        confidence_level=confidence_level,
    )

    assert np.isclose(result, expected)


def test_var_parametric_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [
                -0.04,
                -0.02,
                0.00,
                0.01,
                0.03,
            ],
            "Asset_B": [
                -0.02,
                -0.01,
                0.00,
                0.01,
                0.02,
            ],
        }
    )

    confidence_level = 0.95

    z_score = NormalDist().inv_cdf(
        1 - confidence_level
    )

    quantile = (
        returns.mean()
        + z_score * returns.std()
    )

    expected = (-quantile).clip(lower=0.0)

    result = var_parametric(
        returns,
        confidence_level=confidence_level,
    )

    pd.testing.assert_series_equal(
        result,
        expected,
    )



def test_var_parametric_invalid_confidence_level():
    returns = pd.Series([0.01, -0.01])

    with pytest.raises(ValueError):
        var_parametric(
            returns,
            confidence_level=0.0,
        )


def test_var_parametric_invalid_input():
    with pytest.raises(TypeError):
        var_parametric([0.01, -0.01])


def test_var_parametric_cannot_be_negative():
    returns = pd.Series([
        0.10,
        0.11,
        0.12,
        0.13,
    ])

    result = var_parametric(
        returns,
        confidence_level=0.95,
    )

    assert result >= 0





def test_var_parametric_cornish_fisher():
    returns = pd.Series([
        -0.12,
        -0.04,
        -0.02,
        -0.01,
        0.00,
        0.01,
        0.02,
        0.03,
        0.04,
    ])

    confidence_level = 0.95

    z = NormalDist().inv_cdf(
        1 - confidence_level
    )

    skew = returns.skew()
    kurt = returns.kurt()

    z_cf = (
        z
        + ((z ** 2 - 1) * skew / 6)
        + ((z ** 3 - 3 * z) * kurt / 24)
        - (
            (2 * z ** 3 - 5 * z)
            * (skew ** 2)
            / 36
        )
    )

    expected_quantile = (
        returns.mean()
        + z_cf * returns.std()
    )

    expected = max(
        0.0,
        -expected_quantile,
    )

    result = var_parametric(
        returns,
        confidence_level=confidence_level,
        method="cornish-fisher",
    )

    assert np.isclose(result, expected)




def test_var_parametric_cornish_fisher_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [
                -0.12,
                -0.04,
                -0.02,
                -0.01,
                0.01,
                0.02,
                0.03,
            ],
            "Asset_B": [
                -0.03,
                -0.02,
                -0.01,
                0.00,
                0.01,
                0.02,
                0.03,
            ],
        }
    )

    result = var_parametric(
        returns,
        confidence_level=0.95,
        method="cornish-fisher",
    )

    assert isinstance(result, pd.Series)

    assert "Asset_A" in result.index
    assert "Asset_B" in result.index

    assert (result >= 0).all()



def test_var_parametric_invalid_method():
    returns = pd.Series([
        0.01,
        -0.01,
        0.02,
    ])

    with pytest.raises(ValueError):
        var_parametric(
            returns,
            method="invalid",
        )


def test_cvar_historical_series():
    returns = pd.Series([
        -0.10,
        -0.05,
        -0.02,
        0.01,
        0.03,
        0.04,
    ])

    result = cvar_historical(
        returns,
        confidence_level=0.80,
    )

    expected = 0.075

    assert np.isclose(result, expected)
   


def test_cvar_historical_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [
                -0.10,
                -0.05,
                -0.02,
                0.01,
                0.03,
                0.04,
            ],
            "Asset_B": [
                -0.04,
                -0.03,
                -0.01,
                0.01,
                0.02,
                0.03,
            ],
        }
    )

    confidence_level = 0.80
    alpha = round(1 - confidence_level, 12)

    result = cvar_historical(
        returns,
        confidence_level=confidence_level,
    )

    assert isinstance(result, pd.Series)

    for column in returns.columns:
        threshold = returns[column].quantile(alpha)

        tail = returns[column][
            returns[column] <= threshold
        ]

        expected = max(
            0.0,
            -tail.mean(),
        )

        assert np.isclose(
            result[column],
            expected,
        )
    returns = pd.DataFrame(
        {
            "Asset_A": [
                -0.10,
                -0.05,
                -0.02,
                0.01,
                0.03,
                0.04,
            ],
            "Asset_B": [
                -0.04,
                -0.03,
                -0.01,
                0.01,
                0.02,
                0.03,
            ],
        }
    )

    confidence_level = 0.80

    result = cvar_historical(
        returns,
        confidence_level=confidence_level,
    )

    assert isinstance(result, pd.Series)

    for column in returns.columns:
        alpha = round(1 - confidence_level, 12)
        threshold = returns[column].quantile(alpha)

        tail = returns[column][
            returns[column] <= threshold
        ]

        expected = max(
            0.0,
            -tail.mean(),
        )

        assert np.isclose(
            result[column],
            expected,
        )


def test_cvar_is_at_least_as_large_as_var():
    returns = pd.Series([
        -0.15,
        -0.08,
        -0.04,
        -0.02,
        0.00,
        0.01,
        0.02,
        0.03,
        0.04,
        0.05,
    ])

    var = var_historical(
        returns,
        confidence_level=0.80,
    )

    cvar = cvar_historical(
        returns,
        confidence_level=0.80,
    )

    assert cvar >= var



def test_cvar_historical_invalid_confidence_level():
    returns = pd.Series([0.01, -0.01])

    with pytest.raises(ValueError):
        cvar_historical(
            returns,
            confidence_level=1.0,
        )


def test_cvar_historical_invalid_input():
    with pytest.raises(TypeError):
        cvar_historical([0.01, -0.01])



def test_cvar_parametric_series():
    returns = pd.Series([
        -0.03,
        -0.01,
        0.00,
        0.01,
        0.02,
        0.03,
    ])

    confidence_level = 0.95
    alpha = 1 - confidence_level

    normal = NormalDist()

    z_score = normal.inv_cdf(alpha)
    tail_density = normal.pdf(z_score)

    expected = (
        -returns.mean()
        + returns.std() * tail_density / alpha
    )

    expected = max(0.0, expected)

    result = cvar_parametric(
        returns,
        confidence_level=confidence_level,
    )

    assert np.isclose(result, expected)





def test_cvar_parametric_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [
                -0.04,
                -0.02,
                0.00,
                0.01,
                0.03,
            ],
            "Asset_B": [
                -0.02,
                -0.01,
                0.00,
                0.01,
                0.02,
            ],
        }
    )

    result = cvar_parametric(
        returns,
        confidence_level=0.95,
    )

    assert isinstance(result, pd.Series)

    assert "Asset_A" in result.index
    assert "Asset_B" in result.index

    assert (result >= 0).all()



def test_parametric_cvar_is_at_least_as_large_as_var():
    returns = pd.Series([
        -0.04,
        -0.02,
        -0.01,
        0.00,
        0.01,
        0.02,
        0.03,
    ])

    var = var_parametric(
        returns,
        confidence_level=0.95,
        method="gaussian",
    )

    cvar = cvar_parametric(
        returns,
        confidence_level=0.95,
    )

    assert cvar >= var



def test_cvar_parametric_invalid_confidence_level():
    returns = pd.Series([0.01, -0.01])

    with pytest.raises(ValueError):
        cvar_parametric(
            returns,
            confidence_level=1.0,
        )


def test_cvar_parametric_invalid_input():
    with pytest.raises(TypeError):
        cvar_parametric([0.01, -0.01])



def test_var_historical_cannot_be_negative():
    returns = pd.Series([
        0.01,
        0.02,
        0.03,
        0.04,
    ])

    result = var_historical(
        returns,
        confidence_level=0.95,
    )

    assert np.isclose(result, 0.0)



def test_var_historical_dataframe_cannot_be_negative():
    returns = pd.DataFrame(
        {
            "Positive": [
                0.01,
                0.02,
                0.03,
                0.04,
            ],
            "Mixed": [
                -0.04,
                -0.02,
                0.01,
                0.03,
            ],
        }
    )

    result = var_historical(
        returns,
        confidence_level=0.95,
    )

    assert result["Positive"] == 0.0
    assert result["Mixed"] > 0.0