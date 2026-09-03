"""Tests for visualization module."""

import os
from litreview.visualization import (
    plot_year_distribution,
    plot_source_distribution,
    plot_topic_coverage,
    plot_gap_analysis,
    plot_confidence_distribution,
)


class TestPlotYearDistribution:
    def test_basic(self, sample_df, temp_output_dir):
        path = plot_year_distribution(sample_df, output_dir=temp_output_dir)
        assert os.path.exists(path)
        assert path.endswith(".png")


class TestPlotSourceDistribution:
    def test_basic(self, sample_df, temp_output_dir):
        path = plot_source_distribution(sample_df, output_dir=temp_output_dir)
        assert os.path.exists(path)
        assert path.endswith(".png")


class TestPlotTopicCoverage:
    def test_basic(self, temp_output_dir):
        topic_results = {
            "topic_sizes": {0: 10, 1: 5, 2: 3},
            "topic_words": {0: ["test", "paper"], 1: ["study", "data"], 2: ["method", "analysis"]},
        }
        path = plot_topic_coverage(topic_results, output_dir=temp_output_dir)
        assert os.path.exists(path)
        assert path.endswith(".png")


class TestPlotGapAnalysis:
    def test_basic(self, temp_output_dir):
        gaps = {
            "seed_no_matches": ["seed1"],
            "seed_few_matches": ["seed2"],
            "large_outliers": [10],
        }
        path = plot_gap_analysis(gaps, output_dir=temp_output_dir)
        assert os.path.exists(path)
        assert path.endswith(".png")


class TestPlotConfidenceDistribution:
    def test_basic(self, sample_df, temp_output_dir):
        validation_results = {
            "classifications": sample_df[["title", "abstract"]].head(5),
            "label_counts": {"A": 3, "B": 2},
        }
        path = plot_confidence_distribution(validation_results, output_dir=temp_output_dir)
        assert os.path.exists(path)
        assert path.endswith(".png")
