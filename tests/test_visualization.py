"""Tests for visualization module."""

import os
import pandas as pd
from litreview.visualization import (
    plot_year_distribution,
    plot_source_distribution,
    plot_topic_coverage,
    plot_gap_analysis,
    plot_confidence_distribution,
)


class TestPlotYearDistribution:
    def test_basic(self, sample_df, temp_output_dir):
        path = os.path.join(temp_output_dir, "year_dist.png")
        plot_year_distribution(sample_df, path)
        assert os.path.exists(path)
        assert path.endswith(".png")


class TestPlotSourceDistribution:
    def test_basic(self, sample_df, temp_output_dir):
        path = os.path.join(temp_output_dir, "source_dist.png")
        plot_source_distribution(sample_df, path)
        assert os.path.exists(path)
        assert path.endswith(".png")


class TestPlotTopicCoverage:
    def test_basic(self, temp_output_dir):
        coverage = {
            "topic_sizes": {0: 10, 1: 5, 2: 3},
            "num_topics": 3,
        }
        path = os.path.join(temp_output_dir, "topic_coverage.png")
        plot_topic_coverage(coverage, path)
        assert os.path.exists(path)
        assert path.endswith(".png")


class TestPlotGapAnalysis:
    def test_basic(self, temp_output_dir):
        gaps = {
            "gaps": [
                {"type": "seed_no_matches", "seed": "seed1", "severity": "high"},
                {"type": "seed_few_matches", "seed": "seed2", "severity": "medium"},
            ],
            "num_gaps": 2,
        }
        path = os.path.join(temp_output_dir, "gap_analysis.png")
        plot_gap_analysis(gaps, path)
        assert os.path.exists(path)
        assert path.endswith(".png")


class TestPlotConfidenceDistribution:
    def test_basic(self, temp_output_dir):
        validation_results = {
            "classifications": pd.DataFrame({
                "score": [0.95, 0.82, 0.73, 0.65, 0.40],
                "label": ["A", "B", "A", "C", "unknown"],
                "classified": [True, True, True, True, False],
            }),
        }
        path = os.path.join(temp_output_dir, "confidence.png")
        plot_confidence_distribution(validation_results, path)
        assert os.path.exists(path)
        assert path.endswith(".png")
