"""Tests for ParquetFetcher."""

from pathlib import Path
import pandas as pd
import pytest

from litreview.fetchers.parquet_fetcher import ParquetFetcher


def test_parquet_fetcher_loads_data(tmp_path: Path):
    df_raw = pd.DataFrame({
        "paper_title": ["Paper A", "Paper B", "Paper C"],
        "abstract": ["Abstract for paper A.", "Abstract for paper B.", "   "],
        "pub_year": ["2024", "2023", ""],
        "doi": ["10.1000/1", None, "10.1000/3"],
    })
    parquet_file = tmp_path / "papers.parquet"
    df_raw.to_parquet(parquet_file, engine="pyarrow")

    fetcher = ParquetFetcher(parquet_file)
    df = fetcher.fetch()

    # Paper C has empty abstract, should be filtered out
    assert len(df) == 2
    assert list(df.columns) == ["Title", "Abstract Note", "Publication Year", "Source", "Item Type", "DOI"]
    assert df["Title"].iloc[0] == "Paper A"
    assert df["Abstract Note"].iloc[0] == "Abstract for paper A."
    assert df["Publication Year"].iloc[0] == 2024
    assert df["Publication Year"].iloc[1] == 2023
    assert df["Source"].iloc[0] == "Parquet"
    assert df["Item Type"].iloc[0] == "journalArticle"
    assert df["DOI"].iloc[0] == "10.1000/1"


def test_parquet_fetcher_missing_file():
    with pytest.raises(FileNotFoundError):
        ParquetFetcher("non_existent_file.parquet")


def test_parquet_fetcher_missing_required_columns(tmp_path: Path):
    df_invalid = pd.DataFrame({"author": ["Alice", "Bob"]})
    parquet_file = tmp_path / "invalid.parquet"
    df_invalid.to_parquet(parquet_file, engine="pyarrow")

    fetcher = ParquetFetcher(parquet_file)
    with pytest.raises(ValueError, match="must contain a 'Title'"):
        fetcher.fetch()


def test_pipeline_with_parquet_fetcher(tmp_path: Path, sample_config):
    from unittest.mock import MagicMock
    from litreview.pipeline import ReviewPipeline

    df_raw = pd.DataFrame({
        "Title": ["Fluid Dynamics", "Numerical Optimization"],
        "Abstract Note": ["CFD simulation with deep learning.", "Numerical methods in fluid mechanics."],
        "Publication Year": [2023, 2022],
    })
    parquet_file = tmp_path / "test_corpus.parquet"
    df_raw.to_parquet(parquet_file, engine="pyarrow")

    fetcher = ParquetFetcher(parquet_file)
    pipeline = ReviewPipeline(sample_config, skip_bertopic=True, fetcher=fetcher)
    pipeline.zeroshot_analyzer = MagicMock()
    mock_val_df = pd.DataFrame(
        {"label": ["HPC", "HPC"], "score": [0.9, 0.85], "classified": [True, True]},
        index=range(2),
    )
    pipeline.zeroshot_analyzer.transform.return_value = mock_val_df
    pipeline.zeroshot_analyzer.results = {"label_counts": {"HPC": 2}, "threshold": 0.5}

    report = pipeline.run()

    assert len(report.df) == 2
    assert report.corpus_stats["total_papers"] == 2

