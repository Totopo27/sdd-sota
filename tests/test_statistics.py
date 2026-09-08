"""Tests for statistics module."""

import pandas as pd
import pytest

from litreview.statistics import (
    compute_corpus_stats,
    compute_topic_coverage,
    compute_gap_analysis,
    compute_cross_analysis,
)


@pytest.fixture
def topic_df():
    return pd.DataFrame({"topic": [0, 0, 1, 1, -1]})


@pytest.fixture
def validation_df():
    return pd.DataFrame({
        "label": ["A", "A", "B", "unknown", "C"],
        "score": [0.9, 0.8, 0.7, 0.2, 0.6],
        "classified": [True, True, True, False, True],
    })


class TestComputeCorpusStats:
    def test_basic(self, sample_df):
        stats = compute_corpus_stats(sample_df)
        assert "total_papers" in stats
        assert "papers_with_abstracts" in stats
        assert stats["total_papers"] == len(sample_df)

    def test_empty_df(self):
        df = pd.DataFrame({
            "Publication Year": [],
            "Abstract Note": [],
            "Source": [],
            "Item Type": [],
        })
        stats = compute_corpus_stats(df)
        assert stats["total_papers"] == 0


class TestComputeTopicCoverage:
    def test_basic(self, topic_df, validation_df):
        coverage = compute_topic_coverage(topic_df, validation_df)
        assert "num_topics" in coverage
        assert "topic_sizes" in coverage
        assert "outlier_count" in coverage
        assert coverage["num_topics"] == 2
        assert coverage["outlier_count"] == 1


class TestComputeGapAnalysis:
    def test_basic(self, topic_df, validation_df):
        gaps = compute_gap_analysis(topic_df, validation_df, seed_topics=[["A"], ["MISSING_SEED"]])
        assert "gaps" in gaps
        assert "num_gaps" in gaps
        assert gaps["num_gaps"] >= 1
        gap_types = [g["type"] for g in gaps["gaps"]]
        assert "seed_no_matches" in gap_types


class TestComputeCrossAnalysis:
    def test_basic(self, topic_df, validation_df):
        cross = compute_cross_analysis(topic_df, validation_df)
        assert "bertopic_classified" in cross
        assert "zeroshot_classified" in cross
        assert "both_classified" in cross
        assert "neither_classified" in cross
        assert cross["bertopic_classified"] == 4
