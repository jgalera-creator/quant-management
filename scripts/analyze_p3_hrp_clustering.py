from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.cluster.hierarchy import (
    linkage,
    leaves_list,
    dendrogram,
)

from scipy.spatial.distance import squareform


RESULTS_DIR = Path("data/p3_results")
FIGURES_DIR = Path("data/p3_results/figures")


def main():

    FIGURES_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 1. Historical asset returns.
    returns = pd.read_csv(
        RESULTS_DIR / "asset_returns.csv",
        index_col=0,
        parse_dates=True,
    )

    # 2. Actual HRP rebalance dates.
    weights = pd.read_csv(
        RESULTS_DIR / "weights_hrp.csv",
        index_col=0,
        parse_dates=True,
    )

    last_date = weights.index[-1]

    start_date = (
        last_date - pd.DateOffset(months=36)
    )

    window = returns.loc[
        (returns.index > start_date)
        & (returns.index <= last_date)
    ]

    # 3. Correlation matrix.
    correlation = window.corr()

    # 4. Correlation distance.
    distance = np.sqrt(
        np.maximum(
            (1.0 - correlation) / 2.0,
            0.0,
        )
    )

    # Numerical symmetry and exact zero diagonal.
    distance_values = distance.to_numpy()

    distance_values = (
        distance_values + distance_values.T
    ) / 2.0

    np.fill_diagonal(distance_values, 0.0)

    # 5. Hierarchical clustering.
    condensed_distance = squareform(
        distance_values,
        checks=True,
    )

    linkage_matrix = linkage(
        condensed_distance,
        method="single",
        optimal_ordering=True,
    )

    # 6. Dendrogram.
    fig, ax = plt.subplots(
        figsize=(12, 6),
    )

    dendrogram(
        linkage_matrix,
        labels=correlation.index.tolist(),
        ax=ax,
    )

    ax.set_title(
        "HRP: hierarchical clustering of ETF returns"
    )

    ax.set_ylabel("Correlation distance")

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "hrp_dendrogram.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # 7. Quasi-diagonalization.
    ordered_indices = leaves_list(
        linkage_matrix
    )

    ordered_assets = correlation.index[
        ordered_indices
    ].tolist()

    reordered_correlation = correlation.loc[
        ordered_assets,
        ordered_assets,
    ]

    # 8. Reordered correlation heatmap.
    fig, ax = plt.subplots(
        figsize=(10, 8),
    )

    image = ax.imshow(
        reordered_correlation.to_numpy(),
        vmin=-1,
        vmax=1,
        cmap="RdBu_r",
    )

    ax.set_xticks(
        range(len(ordered_assets)),
        labels=ordered_assets,
        rotation=45,
        ha="right",
    )

    ax.set_yticks(
        range(len(ordered_assets)),
        labels=ordered_assets,
    )

    ax.set_title(
        "HRP: reordered correlation matrix"
    )

    fig.colorbar(
        image,
        ax=ax,
        label="Correlation",
    )

    fig.tight_layout()

    fig.savefig(
        FIGURES_DIR / "hrp_reordered_correlation.png",
        dpi=200,
        bbox_inches="tight",
    )

    plt.close(fig)

    # 9. Save reordered matrix.
    reordered_correlation.to_csv(
        RESULTS_DIR / "hrp_reordered_correlation.csv"
    )

    print("\n=== HRP CLUSTERING ===")

    print("Rebalance date:", last_date)
    print("Training window start:", start_date)
    print("Observations:", len(window))

    print("\nQuasi-diagonalized asset order:")

    print(" -> ".join(ordered_assets))

    print("\nFigures saved to:", FIGURES_DIR)


if __name__ == "__main__":
    main()