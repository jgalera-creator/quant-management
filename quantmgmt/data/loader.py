from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Sequence

import pandas as pd
import yfinance as yf


DEFAULT_CACHE_DIR = Path("data/cache")


def _cache_path(
    tickers: Sequence[str],
    start: str,
    end: str | None,
    interval: str,
    cache_dir: Path,
) -> Path:
    """
    Build a deterministic cache path for a market-data request.
    """
    normalized_tickers = sorted(
        ticker.upper() for ticker in tickers
    )

    key = "|".join(
        [
            ",".join(normalized_tickers),
            start,
            end or "latest",
            interval,
        ]
    )

    digest = hashlib.sha256(
        key.encode("utf-8")
    ).hexdigest()[:12]

    return cache_dir / f"prices_{digest}.csv"


def load_prices(
    tickers: Sequence[str],
    start: str,
    end: str | None = None,
    interval: str = "1d",
    cache_dir: str | Path = DEFAULT_CACHE_DIR,
    force_refresh: bool = False,
) -> pd.DataFrame:
    """
    Load adjusted closing prices from Yahoo Finance.

    Results are cached locally to make subsequent analyses
    reproducible and avoid unnecessary network requests.

    Parameters
    ----------
    tickers : sequence of str
        Yahoo Finance ticker symbols.
    start : str
        Start date in YYYY-MM-DD format. Inclusive.
    end : str or None, default=None
        End date in YYYY-MM-DD format. Exclusive.
        If None, data is downloaded through the latest available date.
    interval : str, default="1d"
        Data frequency accepted by yfinance.
    cache_dir : str or Path, default="data/cache"
        Directory used to store cached price data.
    force_refresh : bool, default=False
        If True, ignore an existing cache file and download again.

    Returns
    -------
    pd.DataFrame
        Adjusted closing prices with dates as index and tickers
        as columns.

    Raises
    ------
    TypeError
        If tickers is not a sequence of strings.
    ValueError
        If no tickers are provided or Yahoo returns no data.
    """
    if isinstance(tickers, str):
        raise TypeError(
            "tickers must be a sequence of ticker symbols"
        )

    tickers = [
        ticker.upper()
        for ticker in tickers
    ]

    if not tickers:
        raise ValueError(
            "at least one ticker must be provided"
        )

    if not all(
        isinstance(ticker, str) and ticker
        for ticker in tickers
    ):
        raise TypeError(
            "all tickers must be non-empty strings"
        )

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    cache_file = _cache_path(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        cache_dir=cache_dir,
    )

    if cache_file.exists() and not force_refresh:
        prices = pd.read_csv(
            cache_file,
            index_col=0,
            parse_dates=True,
        )

        prices.index.name = "Date"

        return prices

    data = yf.download(
        tickers=tickers,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=True,
        progress=False,
        group_by="column",
        multi_level_index=True,
    )

    if data is None or data.empty:
        raise ValueError(
            "no market data returned by Yahoo Finance"
        )

    if isinstance(data.columns, pd.MultiIndex):
        if "Close" not in data.columns.get_level_values(0):
            raise ValueError(
                "Yahoo Finance response does not contain Close prices"
            )

        prices = data["Close"].copy()

    else:
        if "Close" not in data.columns:
            raise ValueError(
                "Yahoo Finance response does not contain Close prices"
            )

        prices = data[["Close"]].copy()

        if len(tickers) == 1:
            prices.columns = [tickers[0]]

    prices = prices.sort_index()

    prices = prices.dropna(
        how="all",
    )

    prices.index.name = "Date"

    prices.to_csv(cache_file)

    return prices