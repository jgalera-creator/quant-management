
import numpy as np
import pandas as pd
import pytest

from quantmgmt.optimizers.hrp import (
    correlation_distance,
    hierarchical_clustering,
)

from quantmgmt.optimizers.hrp import quasi_diagonalize
from quantmgmt.optimizers.hrp import cluster_variance
from quantmgmt.optimizers.hrp import recursive_bisection
from quantmgmt.optimizers.hrp import hrp_weights
from quantmgmt.optimizers.hrp import risk_contribution


def test_correlation_distance_known_values():
    corr = pd.DataFrame(
        [
            [1.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
            [-1.0, 0.0, 1.0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    distance = correlation_distance(corr)

    assert distance.loc["A", "A"] == pytest.approx(0.0)
    assert distance.loc["A", "B"] == pytest.approx(
        np.sqrt(0.5)
    )
    assert distance.loc["A", "C"] == pytest.approx(1.0)


def test_correlation_distance_symmetry():
    corr = pd.DataFrame(
        [[1.0, 0.5], [0.5, 1.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    distance = correlation_distance(corr)

    np.testing.assert_allclose(
        distance.to_numpy(),
        distance.to_numpy().T,
    )


def test_correlation_distance_rejects_invalid_corr():
    corr = pd.DataFrame(
        [[1.0, 1.5], [1.5, 1.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(ValueError):
        correlation_distance(corr)


def test_hierarchical_clustering():
    corr = pd.DataFrame(
        [
            [1.0, 0.8, 0.1],
            [0.8, 1.0, 0.2],
            [0.1, 0.2, 1.0],
        ],
        index=["A", "B", "C"],
        columns=["A", "B", "C"],
    )

    distance = correlation_distance(corr)

    result = hierarchical_clustering(
        distance,
        method="single",
    )

    assert result.shape == (2, 4)

    # Assets A and B must be merged first.
    assert set(result[0, :2].astype(int)) == {0, 1}


def test_clustering_rejects_invalid_method():
    distance = pd.DataFrame(
        [[0.0, 0.5], [0.5, 0.0]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(ValueError):
        hierarchical_clustering(
            distance,
            method="invalid",
        )





def test_quasi_diagonalize_preserves_assets():
    assets = pd.Index(["A", "B", "C", "D"])

    corr = pd.DataFrame(
        [
            [1.0, 0.8, 0.1, 0.1],
            [0.8, 1.0, 0.1, 0.1],
            [0.1, 0.1, 1.0, 0.8],
            [0.1, 0.1, 0.8, 1.0],
        ],
        index=assets,
        columns=assets,
    )

    distance = correlation_distance(corr)
    linkage_matrix = hierarchical_clustering(distance)

    ordered = quasi_diagonalize(linkage_matrix, assets)

    assert len(ordered) == 4
    assert set(ordered) == set(assets)


def test_quasi_diagonalize_keeps_clusters_adjacent():
    assets = pd.Index(["A", "C", "B", "D"])

    corr = pd.DataFrame(
        [
            [1.0, 0.1, 0.9, 0.1],
            [0.1, 1.0, 0.1, 0.9],
            [0.9, 0.1, 1.0, 0.1],
            [0.1, 0.9, 0.1, 1.0],
        ],
        index=assets,
        columns=assets,
    )

    distance = correlation_distance(corr)
    linkage_matrix = hierarchical_clustering(distance)

    ordered = quasi_diagonalize(linkage_matrix, assets)

    positions = {asset: i for i, asset in enumerate(ordered)}

    assert abs(positions["A"] - positions["B"]) == 1
    assert abs(positions["C"] - positions["D"]) == 1


def test_quasi_diagonalize_rejects_wrong_shape():
    assets = pd.Index(["A", "B", "C"])

    invalid_linkage = np.zeros((1, 4))

    with pytest.raises(ValueError, match="shape"):
        quasi_diagonalize(invalid_linkage, assets)



def test_cluster_variance_independent_assets():
    cov = pd.DataFrame(
        [[0.01, 0.0], [0.0, 0.04]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    result = cluster_variance(cov, ["A", "B"])

    assert result == pytest.approx(0.008)


def test_cluster_variance_single_asset():
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    result = cluster_variance(cov, ["B"])

    assert result == pytest.approx(0.09)


def test_cluster_variance_asset_order():
    cov = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    first = cluster_variance(cov, ["A", "B"])
    second = cluster_variance(cov, ["B", "A"])

    assert first == pytest.approx(second)


def test_cluster_variance_invalid_asset():
    cov = pd.DataFrame(
        [[0.04]],
        index=["A"],
        columns=["A"],
    )

    with pytest.raises(ValueError, match="unknown"):
        cluster_variance(cov, ["B"])


def test_cluster_variance_rejects_zero_variance():
    cov = pd.DataFrame(
        [[0.0]],
        index=["A"],
        columns=["A"],
    )

    with pytest.raises(ValueError, match="positive"):
        cluster_variance(cov, ["A"])







def test_recursive_bisection_two_assets():
    covariance = pd.DataFrame(
        [[0.01, 0.0], [0.0, 0.04]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    weights = recursive_bisection(
        covariance,
        pd.Index(["A", "B"]),
    )

    assert weights["A"] == pytest.approx(0.8)
    assert weights["B"] == pytest.approx(0.2)
    assert weights.sum() == pytest.approx(1.0)


def test_recursive_bisection_equal_variances():
    assets = pd.Index(["A", "B", "C", "D"])

    covariance = pd.DataFrame(
        np.eye(4) * 0.04,
        index=assets,
        columns=assets,
    )

    weights = recursive_bisection(covariance, assets)

    np.testing.assert_allclose(
        weights.to_numpy(),
        np.full(4, 0.25),
        atol=1e-12,
    )


def test_recursive_bisection_three_assets():
    assets = pd.Index(["A", "B", "C"])

    covariance = pd.DataFrame(
        np.diag([0.01, 0.04, 0.09]),
        index=assets,
        columns=assets,
    )

    weights = recursive_bisection(covariance, assets)

    assert len(weights) == 3
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()


def test_recursive_bisection_single_asset():
    covariance = pd.DataFrame(
        [[0.04]],
        index=["A"],
        columns=["A"],
    )

    weights = recursive_bisection(
        covariance,
        pd.Index(["A"]),
    )

    assert weights["A"] == pytest.approx(1.0)


def test_recursive_bisection_invalid_assets():
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    with pytest.raises(ValueError, match="every asset"):
        recursive_bisection(
            covariance,
            pd.Index(["A"]),
        )







def test_hrp_weights_equal_independent_assets():
    assets = ["A", "B", "C", "D"]

    covariance = pd.DataFrame(
        np.eye(4) * 0.04,
        index=assets,
        columns=assets,
    )

    weights = hrp_weights(covariance)

    np.testing.assert_allclose(
        weights.to_numpy(),
        np.full(4, 0.25),
        atol=1e-10,
    )


def test_hrp_weights_two_assets():
    covariance = pd.DataFrame(
        [[0.01, 0.0], [0.0, 0.04]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    weights = hrp_weights(covariance)

    assert weights["A"] == pytest.approx(0.8)
    assert weights["B"] == pytest.approx(0.2)


def test_hrp_weights_preserves_labels():
    assets = ["GLD", "SPY", "TLT"]

    covariance = pd.DataFrame(
        [
            [0.025, 0.001, 0.002],
            [0.001, 0.04, 0.005],
            [0.002, 0.005, 0.015],
        ],
        index=assets,
        columns=assets,
    )

    weights = hrp_weights(covariance)

    assert weights.index.tolist() == assets
    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()


@pytest.mark.parametrize(
    "method",
    ["single", "complete", "average"],
)
def test_hrp_weights_linkage_methods(method):
    assets = ["A", "B", "C"]

    covariance = pd.DataFrame(
        [
            [0.04, 0.005, 0.001],
            [0.005, 0.09, 0.003],
            [0.001, 0.003, 0.025],
        ],
        index=assets,
        columns=assets,
    )

    weights = hrp_weights(
        covariance,
        linkage_method=method,
    )

    assert weights.sum() == pytest.approx(1.0)
    assert (weights >= 0).all()


def test_hrp_weights_single_asset():
    covariance = pd.DataFrame(
        [[0.04]],
        index=["A"],
        columns=["A"],
    )

    weights = hrp_weights(covariance)

    assert weights["A"] == pytest.approx(1.0)


def test_hrp_weights_rejects_negative_variance():
    covariance = pd.DataFrame(
        [[-0.04]],
        index=["A"],
        columns=["A"],
    )

    with pytest.raises(ValueError):
        hrp_weights(covariance)







def test_risk_contribution_equal_independent_assets():
    assets = ["A", "B"]

    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.04]],
        index=assets,
        columns=assets,
    )

    weights = pd.Series({"A": 0.5, "B": 0.5})

    result = risk_contribution(weights, covariance)

    assert result.loc["A", "risk_contribution_pct"] == pytest.approx(0.5)
    assert result.loc["B", "risk_contribution_pct"] == pytest.approx(0.5)


def test_risk_contributions_sum_to_volatility():
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    weights = pd.Series({"A": 0.6, "B": 0.4})

    result = risk_contribution(weights, covariance)

    w = weights.to_numpy()
    expected_vol = np.sqrt(
        w @ covariance.to_numpy() @ w
    )

    assert result["risk_contribution"].sum() == pytest.approx(
        expected_vol
    )
    assert result["risk_contribution_pct"].sum() == pytest.approx(
        1.0
    )


def test_risk_contribution_aligns_labels():
    covariance = pd.DataFrame(
        [[0.04, 0.01], [0.01, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    weights = pd.Series({"B": 0.4, "A": 0.6})

    result = risk_contribution(weights, covariance)

    assert result.index.tolist() == ["A", "B"]
    assert result.loc["A", "weight"] == pytest.approx(0.6)


def test_risk_contribution_can_be_negative():
    covariance = pd.DataFrame(
        [[0.04, -0.025], [-0.025, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    weights = pd.Series({"A": 0.9, "B": 0.1})

    result = risk_contribution(weights, covariance)

    assert result.loc["B", "risk_contribution"] < 0


def test_risk_contribution_rejects_invalid_weights():
    covariance = pd.DataFrame(
        [[0.04, 0.0], [0.0, 0.09]],
        index=["A", "B"],
        columns=["A", "B"],
    )

    weights = pd.Series({"A": 0.8, "B": 0.8})

    with pytest.raises(ValueError, match="sum to one"):
        risk_contribution(weights, covariance)
