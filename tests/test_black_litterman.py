
import numpy as np
import pandas as pd
import pytest

from quantmgmt.optimizers.black_litterman import (
    implied_equilibrium_returns,
)
from quantmgmt.optimizers.mean_variance import max_sharpe

def test_implied_equilibrium_returns():
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    # Deliberately reversed to verify label alignment.
    weights = pd.Series({"B": 0.4, "A": 0.6})

    result = implied_equilibrium_returns(
        covariance_matrix=covariance,
        market_weights=weights,
        risk_aversion=2.5,
    )

    expected = np.array([0.07, 0.105])

    np.testing.assert_allclose(
        result.to_numpy(),
        expected,
        rtol=1e-12,
    )


@pytest.mark.parametrize(
    "risk_aversion",
    [0.0, -1.0, np.nan],
)
def test_invalid_risk_aversion(risk_aversion):
    covariance = pd.DataFrame(
        [[0.04]],
        index=["A"],
        columns=["A"],
    )

    weights = pd.Series({"A": 1.0})

    with pytest.raises(ValueError):
        implied_equilibrium_returns(
            covariance,
            weights,
            risk_aversion,
        )



from quantmgmt.optimizers.black_litterman import posterior_returns


def test_posterior_single_view():
    assets = ["A", "B"]

    sigma = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]],
        index=assets,
        columns=assets,
    )
    pi = pd.Series({"A": 0.06, "B": 0.08})

    P = pd.DataFrame(
        [[1.0, 0.0]],
        index=["view_1"],
        columns=assets,
    )
    Q = pd.Series({"view_1": 0.10})
    omega = pd.DataFrame(
        [[0.002]],
        index=["view_1"],
        columns=["view_1"],
    )

    result = posterior_returns(
        pi, sigma, P, Q, omega, tau=0.05
    )

    # tau * sigma_AA = 0.002, equal to omega.
    # Posterior A moves halfway from 6% to 10%.
    assert result["A"] == pytest.approx(0.08)
    assert result["B"] == pytest.approx(0.08)


def test_lower_omega_increases_view_influence():
    sigma = pd.DataFrame(
        [[0.04]], index=["A"], columns=["A"]
    )
    pi = pd.Series({"A": 0.06})
    P = pd.DataFrame(
        [[1.0]], index=["v"], columns=["A"]
    )
    Q = pd.Series({"v": 0.12})

    def solve(uncertainty):
        omega = pd.DataFrame(
            [[uncertainty]], index=["v"], columns=["v"]
        )
        return posterior_returns(
            pi, sigma, P, Q, omega
        )["A"]

    high_confidence = solve(0.0001)
    low_confidence = solve(0.01)

    assert high_confidence > low_confidence
    assert high_confidence < 0.12
    assert low_confidence > 0.06


def test_posterior_aligns_asset_labels():
    sigma = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    pi = pd.Series({"B": 0.08, "A": 0.06})

    P = pd.DataFrame(
        [[0.0, 1.0]],
        index=["v"],
        columns=["B", "A"],
    )
    Q = pd.Series({"v": 0.10})
    omega = pd.DataFrame(
        [[0.002]], index=["v"], columns=["v"]
    )

    result = posterior_returns(
        pi, sigma, P, Q, omega
    )

    assert result.index.tolist() == ["A", "B"]
    assert result["A"] > 0.06


def test_posterior_rejects_invalid_tau():
    sigma = pd.DataFrame(
        [[0.04]], index=["A"], columns=["A"]
    )
    pi = pd.Series({"A": 0.06})
    P = pd.DataFrame(
        [[1.0]], index=["v"], columns=["A"]
    )
    Q = pd.Series({"v": 0.10})
    omega = pd.DataFrame(
        [[0.002]], index=["v"], columns=["v"]
    )

    with pytest.raises(ValueError):
        posterior_returns(
            pi, sigma, P, Q, omega, tau=0.0
        )


def test_posterior_rejects_nonsymmetric_covariance():
    sigma = pd.DataFrame(
        [[0.04, 0.02], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )
    pi = pd.Series({"A": 0.06, "B": 0.08})
    P = pd.DataFrame(
        [[1.0, 0.0]],
        index=["v"],
        columns=["A", "B"],
    )
    Q = pd.Series({"v": 0.10})
    omega = pd.DataFrame(
        [[0.002]], index=["v"], columns=["v"]
    )

    with pytest.raises(ValueError, match="symmetric"):
        posterior_returns(pi, sigma, P, Q, omega)




from quantmgmt.optimizers.black_litterman import build_omega


def test_build_omega_default():
    sigma = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    P = pd.DataFrame(
        [[1.0, -1.0]],
        index=["A_minus_B"],
        columns=["A", "B"],
    )

    omega = build_omega(sigma, P, tau=0.05)

    # Variance of A-B = 0.04 + 0.09 - 2*0.01 = 0.11
    assert omega.iloc[0, 0] == pytest.approx(0.0055)


def test_build_omega_confidence():
    sigma = pd.DataFrame(
        [[0.04]],
        index=["A"],
        columns=["A"],
    )
    P = pd.DataFrame(
        [[1.0]],
        index=["view"],
        columns=["A"],
    )

    low = build_omega(
        sigma,
        P,
        confidences=pd.Series({"view": 0.25}),
    )

    high = build_omega(
        sigma,
        P,
        confidences=pd.Series({"view": 0.75}),
    )

    assert high.iloc[0, 0] < low.iloc[0, 0]


def test_build_omega_full_confidence():
    sigma = pd.DataFrame(
        [[0.04]],
        index=["A"],
        columns=["A"],
    )
    P = pd.DataFrame(
        [[1.0]],
        index=["view"],
        columns=["A"],
    )

    omega = build_omega(
        sigma,
        P,
        confidences=pd.Series({"view": 1.0}),
    )

    assert omega.iloc[0, 0] == pytest.approx(0.0)


def test_build_omega_invalid_confidence():
    sigma = pd.DataFrame(
        [[0.04]],
        index=["A"],
        columns=["A"],
    )
    P = pd.DataFrame(
        [[1.0]],
        index=["view"],
        columns=["A"],
    )

    with pytest.raises(ValueError):
        build_omega(
            sigma,
            P,
            confidences=pd.Series({"view": 1.5}),
        )




from quantmgmt.optimizers.black_litterman import (
    estimate_risk_aversion,
)


def test_estimate_risk_aversion():
    returns = pd.Series([0.01, 0.03, 0.02, 0.04])

    result = estimate_risk_aversion(
        returns,
        risk_free_rate=0.05,
        periods_per_year=12,
    )

    annual_mean = returns.mean() * 12
    annual_variance = returns.var(ddof=1) * 12

    expected = (annual_mean - 0.05) / annual_variance

    assert result == pytest.approx(expected)


def test_risk_aversion_rejects_zero_variance():
    returns = pd.Series([0.01, 0.01, 0.01])

    with pytest.raises(ValueError, match="variance"):
        estimate_risk_aversion(returns)


def test_risk_aversion_rejects_negative_estimate():
    returns = pd.Series([-0.03, 0.01, -0.02, 0.02])

    with pytest.raises(ValueError, match="not positive"):
        estimate_risk_aversion(
            returns,
            risk_free_rate=0.05,
            periods_per_year=12,
        )




def test_black_litterman_end_to_end():
    assets = ["A", "B", "C"]

    sigma = pd.DataFrame(
        [
            [0.04, 0.005, 0.002],
            [0.005, 0.06, 0.004],
            [0.002, 0.004, 0.09],
        ],
        index=assets,
        columns=assets,
    )

    market_weights = pd.Series(
        {"A": 0.5, "B": 0.3, "C": 0.2}
    )

    pi = implied_equilibrium_returns(
        covariance_matrix=sigma,
        market_weights=market_weights,
        risk_aversion=2.5,
    )

    P = pd.DataFrame(
        [[1.0, -1.0, 0.0]],
        index=["A_outperforms_B"],
        columns=assets,
    )

    Q = pd.Series(
        {"A_outperforms_B": 0.05}
    )

    omega = build_omega(
        covariance_matrix=sigma,
        P=P,
        tau=0.05,
    )

    posterior = posterior_returns(
        equilibrium_returns=pi,
        covariance_matrix=sigma,
        P=P,
        Q=Q,
        omega=omega,
        tau=0.05,
    )

    weights = max_sharpe(
        expected_returns=posterior,
        covariance_matrix=sigma,
        risk_free_rate=0.0,
        long_only=True,
    )

    assert np.isfinite(weights).all()
    assert weights.sum() == pytest.approx(1.0, abs=1e-6)
    assert (weights >= -1e-8).all()
    assert (weights <= 1 + 1e-8).all()

    # A positive relative view should raise the expected
    # return spread between A and B.
    assert (
        posterior["A"] - posterior["B"]
        > pi["A"] - pi["B"]
    )
