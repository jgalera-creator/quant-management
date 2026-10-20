import numpy as np
import pandas as pd
import pytest

from quantmgmt.analytics.factors import (
    rolling_factor_betas,
    run_factor_regression,
)


def test_capm_regression_recovers_known_beta():
    index = pd.date_range(
        "2020-01-01",
        periods=60,
        freq="MS",
    )

    market = pd.Series(
        np.linspace(
            -0.05,
            0.05,
            len(index),
        ),
        index=index,
    )

    rf = pd.Series(
        0.001,
        index=index,
    )

    alpha_monthly = 0.002
    beta = 1.5

    returns = (
        rf
        + alpha_monthly
        + beta * market
    )

    factors = pd.DataFrame(
        {
            "Mkt-RF": market,
            "RF": rf,
        },
        index=index,
    )

    result = run_factor_regression(
        returns,
        factors,
        model="capm",
    )

    assert np.isclose(
        result["alpha_annual"],
        alpha_monthly * 12,
    )

    assert np.isclose(
        result["beta_Mkt-RF"],
        beta,
    )

    assert np.isclose(
        result["adj_r_squared"],
        1.0,
    )


def test_ff3_regression_recovers_known_betas():
    rng = np.random.default_rng(42)

    index = pd.date_range(
        "2015-01-01",
        periods=120,
        freq="MS",
    )

    market = pd.Series(
        rng.normal(0.006, 0.04, 120),
        index=index,
    )

    smb = pd.Series(
        rng.normal(0.002, 0.025, 120),
        index=index,
    )

    hml = pd.Series(
        rng.normal(0.002, 0.025, 120),
        index=index,
    )

    rf = pd.Series(
        0.001,
        index=index,
    )

    returns = (
        rf
        + 0.001
        + 1.2 * market
        + 0.4 * smb
        - 0.3 * hml
    )

    factors = pd.DataFrame(
        {
            "Mkt-RF": market,
            "SMB": smb,
            "HML": hml,
            "RF": rf,
        },
        index=index,
    )

    result = run_factor_regression(
        returns,
        factors,
        model="ff3",
    )

    assert np.isclose(
        result["beta_Mkt-RF"],
        1.2,
    )

    assert np.isclose(
        result["beta_SMB"],
        0.4,
    )

    assert np.isclose(
        result["beta_HML"],
        -0.3,
    )


def test_factor_regression_rejects_unknown_model():
    returns = pd.Series(
        [0.01, 0.02, 0.03]
    )

    factors = pd.DataFrame(
        {
            "RF": [
                0.001,
                0.001,
                0.001,
            ]
        }
    )

    with pytest.raises(ValueError):
        run_factor_regression(
            returns,
            factors,
            model="unknown",
        )


def test_factor_regression_requires_model_factors():
    returns = pd.Series(
        [0.01] * 12
    )

    factors = pd.DataFrame(
        {
            "Mkt-RF": [0.01] * 12,
            "RF": [0.001] * 12,
        }
    )

    with pytest.raises(ValueError):
        run_factor_regression(
            returns,
            factors,
            model="ff5",
        )


def test_rolling_factor_betas_recovers_known_beta():
    index = pd.date_range(
        "2010-01-01",
        periods=60,
        freq="MS",
    )

    market = pd.Series(
        np.linspace(
            -0.04,
            0.05,
            60,
        ),
        index=index,
    )

    rf = pd.Series(
        0.001,
        index=index,
    )

    returns = (
        rf
        + 1.5 * market
    )

    factors = pd.DataFrame(
        {
            "Mkt-RF": market,
            "RF": rf,
        },
        index=index,
    )

    rolling = rolling_factor_betas(
        returns,
        factors,
        model="capm",
        window=36,
    )

    assert len(rolling) == 25

    assert np.allclose(
        rolling["beta_Mkt-RF"],
        1.5,
    )


def test_rolling_factor_betas_rejects_short_sample():
    index = pd.date_range(
        "2020-01-01",
        periods=12,
        freq="MS",
    )

    returns = pd.Series(
        0.01,
        index=index,
    )

    factors = pd.DataFrame(
        {
            "Mkt-RF": 0.01,
            "RF": 0.001,
        },
        index=index,
    )

    with pytest.raises(ValueError):
        rolling_factor_betas(
            returns,
            factors,
            model="capm",
            window=36,
        )