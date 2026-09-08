"""Tests for CSVFetcher."""

import pytest
from pathlib import Path
from litreview.fetchers.csv_fetcher import CSVFetcher


def test_csv_fetcher_loads_mock_data():
    fixture_path = Path(__file__).parent.parent / "data" / "fixtures" / "mock_papers.csv"
    assert fixture_path.exists()

    fetcher = CSVFetcher(fixture_path)
    df = fetcher.fetch()

    assert len(df) == 12
    assert "Title" in df.columns
    assert "Abstract Note" in df.columns
    assert "Publication Year" in df.columns
    assert "Source" in df.columns
    assert "Item Type" in df.columns
    assert df["Publication Year"].iloc[0] == 2023


def test_csv_fetcher_missing_file():
    with pytest.raises(FileNotFoundError):
        CSVFetcher("non_existent_file.csv")
