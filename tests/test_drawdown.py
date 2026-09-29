import numpy as np
import pandas as pd
import pytest

from quantmgmt.risk.drawdown import (
    drawdown_series,
    max_drawdown,
    recovery_time,
    time_under_water,
)

def test_drawdown_series():
    returns = pd.Series([0.10, -0.05, -0.10, 0.08])

    result = drawdown_series(returns)

    wealth = (1 + returns).cumprod()
    running_peak = wealth.cummax()

    expected = wealth / running_peak - 1

    pd.testing.assert_series_equal(result, expected)


def test_drawdown_is_zero_at_new_high():
    returns = pd.Series([0.10, 0.05, 0.02])

    result = drawdown_series(returns)

    expected = pd.Series([0.0, 0.0, 0.0])

    pd.testing.assert_series_equal(result, expected)


def test_drawdown_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [0.10, -0.10, 0.05],
            "Asset_B": [0.02, 0.03, 0.04],
        }
    )

    result = drawdown_series(returns)

    assert isinstance(result, pd.DataFrame)

    assert result["Asset_A"].iloc[1] < 0
    assert np.isclose(result["Asset_B"].iloc[-1], 0.0)


def test_drawdown_invalid_input():
    with pytest.raises(TypeError):
        drawdown_series([0.01, -0.02])


def test_max_drawdown_series():
    returns = pd.Series([0.10, -0.05, -0.10, 0.08])

    result = max_drawdown(returns)

    expected = drawdown_series(returns).min()

    assert np.isclose(result, expected)


def test_max_drawdown_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [0.10, -0.10, 0.05],
            "Asset_B": [0.02, 0.03, 0.04],
        }
    )

    result = max_drawdown(returns)

    assert isinstance(result, pd.Series)

    assert result["Asset_A"] < 0
    assert np.isclose(result["Asset_B"], 0.0)


def test_max_drawdown_no_losses():
    returns = pd.Series([0.02, 0.03, 0.01])

    result = max_drawdown(returns)

    assert np.isclose(result, 0.0)


def test_max_drawdown_invalid_input():
    with pytest.raises(TypeError):
        max_drawdown([0.01, -0.02])


def test_drawdown_detects_loss_from_initial_wealth():
    returns = pd.Series([-0.10, 0.05])

    result = drawdown_series(returns)

    assert np.isclose(result.iloc[0], -0.10)
    assert np.isclose(result.iloc[1], -0.055)



def test_time_under_water_series():
    returns = pd.Series([
        0.10,
        -0.05,
        -0.05,
        0.20,
        -0.10,
    ])

    result = time_under_water(returns)

    expected = pd.Series(
        [0, 1, 2, 0, 1],
        dtype=int,
    )

    pd.testing.assert_series_equal(result, expected)


def test_time_under_water_from_initial_loss():
    returns = pd.Series([
        -0.10,
        0.05,
        0.10,
    ])

    result = time_under_water(returns)

    expected = pd.Series(
        [1, 2, 0],
        dtype=int,
    )

    pd.testing.assert_series_equal(result, expected)



def test_time_under_water_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [0.10, -0.05, -0.05, 0.20],
            "Asset_B": [0.02, 0.03, 0.01, 0.04],
        }
    )

    result = time_under_water(returns)

    expected_a = pd.Series(
        [0, 1, 2, 0],
        dtype=int,
        name="Asset_A",
    )

    expected_b = pd.Series(
        [0, 0, 0, 0],
        dtype=int,
        name="Asset_B",
    )

    pd.testing.assert_series_equal(
        result["Asset_A"],
        expected_a,
    )

    pd.testing.assert_series_equal(
        result["Asset_B"],
        expected_b,
    )


def test_time_under_water_invalid_input():
    with pytest.raises(TypeError):
        time_under_water([0.01, -0.02])



def test_recovery_time_series():
    returns = pd.Series([
        0.10,
        -0.20,
        0.10,
        0.15,
    ])

    result = recovery_time(returns)

    assert np.isclose(result, 2.0)



def test_recovery_time_unrecovered_drawdown():
    returns = pd.Series([
        0.10,
        -0.20,
        0.05,
        0.05,
    ])

    result = recovery_time(returns)

    assert np.isnan(result)



def test_recovery_time_no_drawdown():
    returns = pd.Series([
        0.02,
        0.03,
        0.01,
    ])

    result = recovery_time(returns)

    assert np.isclose(result, 0.0)



def test_recovery_time_dataframe():
    returns = pd.DataFrame(
        {
            "Asset_A": [0.10, -0.20, 0.10, 0.15],
            "Asset_B": [0.02, 0.03, 0.01, 0.04],
        }
    )

    result = recovery_time(returns)

    assert isinstance(result, pd.Series)

    assert np.isclose(result["Asset_A"], 2.0)
    assert np.isclose(result["Asset_B"], 0.0)




def test_recovery_time_invalid_input():
    with pytest.raises(TypeError):
        recovery_time([0.01, -0.02])