from __future__ import annotations

import pandas as pd
from pandas_datareader import data as web


def _normalize_factor_frame(
    frame: pd.DataFrame,
) -> pd.DataFrame:
    """
    Normalize Fama-French factor data.

    Converts:
    - percentage values to decimal returns;
    - PeriodIndex to month-start DatetimeIndex.
    """
    result = frame.copy()

    result = result.astype(float) / 100.0

    if isinstance(result.index, pd.PeriodIndex):
        result.index = result.index.to_timestamp(
            how="start"
        )

    result.index.name = "Date"

    return result.sort_index()


def load_fama_french_factors(
    start: str = "2010-01-01",
) -> pd.DataFrame:
    """
    Load monthly Fama-French and momentum factors.

    Returns
    -------
    pd.DataFrame
        Columns:
        Mkt-RF, SMB, HML, RMW, CMA, Mom, RF

        Values are decimal monthly returns.
    """
    ff5_raw = web.DataReader(
        "F-F_Research_Data_5_Factors_2x3",
        "famafrench",
        start=start,
    )[0]

    momentum_raw = web.DataReader(
        "F-F_Momentum_Factor",
        "famafrench",
        start=start,
    )[0]

    ff5 = _normalize_factor_frame(
        ff5_raw
    )

    momentum = _normalize_factor_frame(
        momentum_raw
    )

    factors = ff5.join(
        momentum[["Mom"]],
        how="inner",
    )

    factors = factors[
        [
            "Mkt-RF",
            "SMB",
            "HML",
            "RMW",
            "CMA",
            "Mom",
            "RF",
        ]
    ]

    return factors