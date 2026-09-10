"""Unit tests for columnar Parquet storage and statistical uncertainty metrics."""

import os
import tempfile
import pandas as pd
import pytest

from litreview.pipeline import Report
from litreview.statistics import compute_topic_coverage


def test_report_export_parquet():
    df = pd.DataFrame({
        "Title": ["Attention Is All You Need", "Deep Residual Learning"],
        "Publication Year": [2017, 2016],
        "Abstract Note": ["Transformer architecture.", "Deeper neural networks."],
        "Score": [0.95, 0.89],
    })

    with tempfile.TemporaryDirectory() as tmpdir:
        parquet_path = os.path.join(tmpdir, "classified.parquet")
        report = Report(
            df=df,
            corpus_stats={},
            topic_coverage={},
            gap_analysis={},
            cross_analysis={},
            bertopic_results={},
            zeroshot_results={},
            config=None,
        )
        report.export_parquet(parquet_path)

        assert os.path.exists(parquet_path)
        loaded_df = pd.read_parquet(parquet_path)
        assert len(loaded_df) == 2
        assert list(loaded_df["Title"]) == ["Attention Is All You Need", "Deep Residual Learning"]
        assert loaded_df["Score"].iloc[0] == 0.95


def test_compute_topic_coverage_uncertainty():
    topic_df = pd.DataFrame({"topic": [0, 0, 1, 1, -1]})
    validation_df = pd.DataFrame({
        "score": [0.90, 0.85, 0.60, 0.55, 0.40],
        "classified": [True, True, True, True, False],
        "label": ["A", "A", "B", "B", "A"],
    })

    stats = compute_topic_coverage(topic_df, validation_df)

    assert "mean_confidence" in stats
    assert "median_confidence" in stats
    assert "std_confidence" in stats
    assert "variance_confidence" in stats
    assert "confidence_ci_95" in stats

    assert stats["mean_confidence"] == 0.66
    assert stats["std_confidence"] > 0.0
    assert len(stats["confidence_ci_95"]) == 2
    # Lower bound <= mean <= Upper bound
    assert stats["confidence_ci_95"][0] <= stats["mean_confidence"] <= stats["confidence_ci_95"][1]
