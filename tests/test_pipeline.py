"""Tests for pipeline module."""

import os
import pandas as pd
from litreview.pipeline import ReviewPipeline, Report


class TestReport:
    def test_summary(self, sample_df, sample_config):
        report = Report(
            df=sample_df,
            corpus_stats={"total_papers": 5, "year_range": "2021-2023"},
            topic_coverage={},
            gap_analysis={},
            cross_analysis={},
            bertopic_results={"num_topics": 2, "outlier_count": 0, "topic_sizes": {0: 2, 1: 2}},
            zeroshot_results={"label_counts": {"A": 3, "B": 2}},
            config=sample_config,
        )
        summary = report.summary()
        assert isinstance(summary, str)
        assert "5" in summary

    def test_export_csv(self, sample_df, sample_config, temp_output_dir):
        report = Report(
            df=sample_df,
            corpus_stats={"total_papers": 5, "year_range": "2021-2023"},
            topic_coverage={},
            gap_analysis={},
            cross_analysis={},
            bertopic_results={"num_topics": 2, "outlier_count": 0, "topic_sizes": {0: 2, 1: 2}},
            zeroshot_results={"label_counts": {"A": 3, "B": 2}},
            config=sample_config,
        )
        out_path = os.path.join(temp_output_dir, "test.csv")
        report.export_csv(out_path)
        assert os.path.exists(out_path)

    def test_export_json(self, sample_df, sample_config, temp_output_dir):
        report = Report(
            df=sample_df,
            corpus_stats={"total_papers": 5, "year_range": "2021-2023"},
            topic_coverage={},
            gap_analysis={},
            cross_analysis={},
            bertopic_results={"num_topics": 2, "outlier_count": 0, "topic_sizes": {0: 2, 1: 2}},
            zeroshot_results={"label_counts": {"A": 3, "B": 2}},
            config=sample_config,
        )
        out_path = os.path.join(temp_output_dir, "test.json")
        report.export_json(out_path)
        assert os.path.exists(out_path)


class TestReviewPipeline:
    def test_init(self, sample_config):
        pipeline = ReviewPipeline(sample_config)
        assert pipeline.config == sample_config

    def test_run(self, sample_config, sample_df, temp_output_dir):
        """Test that run returns a Report object using a mock fetcher."""
        from unittest.mock import MagicMock
        mock_fetcher = MagicMock()
        mock_fetcher.fetch.return_value = sample_df.copy()

        pipeline = ReviewPipeline(sample_config, skip_bertopic=True, fetcher=mock_fetcher)
        # Mock zeroshot to avoid downloading heavy models in unit test
        pipeline.zeroshot_analyzer = MagicMock()
        mock_val_df = pd.DataFrame(
            {"label": ["BLT"] * len(sample_df), "score": [0.9] * len(sample_df), "classified": [True] * len(sample_df)},
            index=sample_df.index,
        )
        pipeline.zeroshot_analyzer.transform.return_value = mock_val_df
        pipeline.zeroshot_analyzer.results = {"label_counts": {"BLT": 5}, "threshold": 0.5}

        report = pipeline.run()
        assert isinstance(report, Report)
        assert len(report.df) == len(sample_df)
