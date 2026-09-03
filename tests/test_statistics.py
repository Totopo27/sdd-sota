"""Tests for statistics module."""

from litreview.statistics import (
    compute_corpus_stats,
    compute_topic_coverage,
    compute_gap_analysis,
    compute_cross_analysis,
)


class TestComputeCorpusStats:
    def test_basic(self, sample_df):
        stats = compute_corpus_stats(sample_df)
        assert "total_papers" in stats
        assert "papers_with_abstract" in stats
        assert "avg_abstract_length" in stats
        assert stats["total_papers"] == len(sample_df)

    def test_empty_df(self):
        import pandas as pd
        df = pd.DataFrame({"title": [], "abstract": []})
        stats = compute_corpus_stats(df)
        assert stats["total_papers"] == 0


class TestComputeTopicCoverage:
    def test_basic(self, sample_df):
        topic_results = {
            "topic_assignments": [0, 0, 1, 1, -1],
            "topic_sizes": {0: 2, 1: 2},
            "topic_words": {0: ["test", "paper"], 1: ["study", "data"]},
        }
        coverage = compute_topic_coverage(topic_results, sample_df)
        assert "num_topics" in coverage
        assert "topic_sizes" in coverage
        assert "coverage_ratio" in coverage
        assert coverage["num_topics"] == 2


class TestComputeGapAnalysis:
    def test_basic(self, sample_df):
        topic_results = {
            "topic_assignments": [0, 0, 1, 1, -1],
            "topic_sizes": {0: 2, 1: 2},
            "topic_words": {0: ["test", "paper"], 1: ["study", "data"]},
        }
        gaps = compute_gap_analysis(topic_results, sample_df)
        assert "seed_no_matches" in gaps
        assert "seed_few_matches" in gaps
        assert "large_outliers" in gaps


class TestComputeCrossAnalysis:
    def test_basic(self, sample_df):
        topic_results = {
            "topic_assignments": [0, 0, 1, 1, -1],
            "topic_sizes": {0: 2, 1: 2},
        }
        validation_results = {
            "classifications": sample_df[["title", "abstract"]].head(5),
            "label_counts": {"A": 3, "B": 2},
        }
        cross = compute_cross_analysis(topic_results, validation_results, sample_df)
        assert "agreement_matrix" in cross
        assert "disagreements" in cross
