import numpy as np
import pandas as pd
import pytest

from quantmgmt.analytics.capm import (
    capm_beta,
    estimate_sector_betas,
    estimate_sml,
)


def test_capm_beta_equals_known_relationship():
    market = pd.Series(
        [0.01, -0.02, 0.03, 0.04, -0.01]
    )

    asset = 2.0 * market

    beta = capm_beta(
        asset,
        market,
    )

    assert np.isclose(
        beta,
        2.0,
    )


def test_capm_beta_aligns_missing_observations():
    market = pd.Series(
        [0.01, -0.02, 0.03, 0.04],
        index=[
            "2020-01",
            "2020-02",
            "2020-03",
            "2020-04",
        ],
    )

    asset = pd.Series(
        [0.02, 0.06, 0.08],
        index=[
            "2020-01",
            "2020-03",
            "2020-04",
        ],
    )

    beta = capm_beta(
        asset,
        market,
    )

    aligned_market = market.loc[
        asset.index
    ]

    expected_beta = (
        asset.cov(aligned_market)
        / aligned_market.var()
    )

    assert np.isclose(
        beta,
        expected_beta,
    )


def test_capm_beta_rejects_zero_market_variance():
    market = pd.Series(
        [0.01, 0.01, 0.01]
    )

    asset = pd.Series(
        [0.02, 0.03, 0.04]
    )

    with pytest.raises(ValueError):
        capm_beta(
            asset,
            market,
        )


def test_estimate_sector_betas_returns_all_assets():
    market = pd.Series(
        [0.01, -0.02, 0.03, 0.04]
    )

    returns = pd.DataFrame(
        {
            "A": market,
            "B": 1.5 * market,
            "C": 0.5 * market,
        }
    )

    betas = estimate_sector_betas(
        returns,
        market,
    )

    assert set(betas.index) == {
        "A",
        "B",
        "C",
    }

    assert np.isclose(
        betas["A"],
        1.0,
    )

    assert np.isclose(
        betas["B"],
        1.5,
    )

    assert np.isclose(
        betas["C"],
        0.5,
    )


def test_estimate_sml_recovers_linear_relation():
    market = pd.Series(
        [
            -0.02,
            0.01,
            0.03,
            -0.01,
            0.04,
            0.02,
        ]
    )

    returns = pd.DataFrame(
        {
            "Low": (
                0.002
                + 0.5 * market
            ),
            "Medium": (
                0.004
                + 1.0 * market
            ),
            "High": (
                0.006
                + 1.5 * market
            ),
        }
    )

    sml = estimate_sml(
        returns,
        market,
        periods_per_year=12,
    )

    assert list(sml.columns) == [
        "beta",
        "annual_return",
        "sml_fitted_return",
    ]

    assert len(sml) == 3
    assert np.isfinite(
        sml.attrs["intercept"]
    )
    assert np.isfinite(
        sml.attrs["slope"]
    )