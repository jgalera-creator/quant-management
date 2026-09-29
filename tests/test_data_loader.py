import pandas as pd
import pytest

from quantmgmt.data.loader import load_prices


def test_load_prices_downloads_and_extracts_close(
    tmp_path,
    monkeypatch,
):
    dates = pd.date_range(
        "2026-01-01",
        periods=3,
        freq="D",
    )

    columns = pd.MultiIndex.from_tuples(
        [
            ("Close", "SPY"),
            ("Close", "TLT"),
            ("Volume", "SPY"),
            ("Volume", "TLT"),
        ]
    )

    fake_data = pd.DataFrame(
        [
            [100.0, 90.0, 1_000, 2_000],
            [101.0, 91.0, 1_100, 2_100],
            [102.0, 92.0, 1_200, 2_200],
        ],
        index=dates,
        columns=columns,
    )

    def fake_download(**kwargs):
        return fake_data

    monkeypatch.setattr(
        "quantmgmt.data.loader.yf.download",
        fake_download,
    )

    result = load_prices(
        tickers=["SPY", "TLT"],
        start="2026-01-01",
        end="2026-01-04",
        cache_dir=tmp_path,
    )

    expected = pd.DataFrame(
        {
            "SPY": [100.0, 101.0, 102.0],
            "TLT": [90.0, 91.0, 92.0],
        },
        index=dates,
    )

    expected.index.name = "Date"

    pd.testing.assert_frame_equal(
        result,
        expected,
    )



def test_load_prices_uses_cache(
    tmp_path,
    monkeypatch,
):
    dates = pd.date_range(
        "2026-01-01",
        periods=3,
        freq="D",
    )

    columns = pd.MultiIndex.from_tuples(
        [
            ("Close", "SPY"),
        ]
    )

    fake_data = pd.DataFrame(
        [
            [100.0],
            [101.0],
            [102.0],
        ],
        index=dates,
        columns=columns,
    )

    call_count = 0

    def fake_download(**kwargs):
        nonlocal call_count
        call_count += 1
        return fake_data

    monkeypatch.setattr(
        "quantmgmt.data.loader.yf.download",
        fake_download,
    )

    first = load_prices(
        tickers=["SPY"],
        start="2026-01-01",
        end="2026-01-04",
        cache_dir=tmp_path,
    )

    second = load_prices(
        tickers=["SPY"],
        start="2026-01-01",
        end="2026-01-04",
        cache_dir=tmp_path,
    )

    assert call_count == 1

    pd.testing.assert_frame_equal(
        first,
        second,
        check_freq=False,
    )


def test_load_prices_force_refresh(
    tmp_path,
    monkeypatch,
):
    dates = pd.date_range(
        "2026-01-01",
        periods=2,
        freq="D",
    )

    columns = pd.MultiIndex.from_tuples(
        [
            ("Close", "SPY"),
        ]
    )

    fake_data = pd.DataFrame(
        [
            [100.0],
            [101.0],
        ],
        index=dates,
        columns=columns,
    )

    call_count = 0

    def fake_download(**kwargs):
        nonlocal call_count
        call_count += 1
        return fake_data

    monkeypatch.setattr(
        "quantmgmt.data.loader.yf.download",
        fake_download,
    )

    load_prices(
        tickers=["SPY"],
        start="2026-01-01",
        cache_dir=tmp_path,
    )

    load_prices(
        tickers=["SPY"],
        start="2026-01-01",
        cache_dir=tmp_path,
        force_refresh=True,
    )

    assert call_count == 2



def test_load_prices_raises_on_empty_data(
    tmp_path,
    monkeypatch,
):
    def fake_download(**kwargs):
        return pd.DataFrame()

    monkeypatch.setattr(
        "quantmgmt.data.loader.yf.download",
        fake_download,
    )

    with pytest.raises(
        ValueError,
        match="no market data",
    ):
        load_prices(
            tickers=["INVALID"],
            start="2026-01-01",
            cache_dir=tmp_path,
        )


def test_load_prices_rejects_string_as_tickers():
    with pytest.raises(TypeError):
        load_prices(
            tickers="SPY",
            start="2026-01-01",
        )


def test_load_prices_requires_at_least_one_ticker():
    with pytest.raises(ValueError):
        load_prices(
            tickers=[],
            start="2026-01-01",
        )