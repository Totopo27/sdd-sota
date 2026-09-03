"""Tests for pipeline module."""

import os
from litreview.pipeline import ReviewPipeline, Report


class TestReport:
    def test_summary(self, sample_df):
        report = Report(
            papers=sample_df,
            topic_results={"topic_sizes": {0: 2, 1: 2}},
            validation_results={"label_counts": {"A": 3, "B": 2}},
            corpus_stats={"total_papers": 5},
        )
        summary = report.summary()
        assert isinstance(summary, str)
        assert "5" in summary

    def test_export_csv(self, sample_df, temp_output_dir):
        report = Report(
            papers=sample_df,
            topic_results={"topic_sizes": {0: 2, 1: 2}},
            validation_results={"label_counts": {"A": 3, "B": 2}},
            corpus_stats={"total_papers": 5},
        )
        path = report.export_csv(output_dir=temp_output_dir)
        assert os.path.exists(path)
        assert path.endswith(".csv")

    def test_export_json(self, sample_df, temp_output_dir):
        report = Report(
            papers=sample_df,
            topic_results={"topic_sizes": {0: 2, 1: 2}},
            validation_results={"label_counts": {"A": 3, "B": 2}},
            corpus_stats={"total_papers": 5},
        )
        path = report.export_json(output_dir=temp_output_dir)
        assert os.path.exists(path)
        assert path.endswith(".json")

    def test_generate_plots(self, sample_df, temp_output_dir):
        report = Report(
            papers=sample_df,
            topic_results={"topic_sizes": {0: 2, 1: 2}},
            validation_results={"label_counts": {"A": 3, "B": 2}},
            corpus_stats={"total_papers": 5},
        )
        paths = report.generate_plots(output_dir=temp_output_dir)
        assert isinstance(paths, list)
        assert len(paths) > 0


class TestReviewPipeline:
    def test_init(self, sample_config):
        pipeline = ReviewPipeline(sample_config)
        assert pipeline.config == sample_config

    def test_run(self, sample_config, temp_output_dir):
        """Test that run returns a Report object."""
        # This will try to fetch from Zotero, which requires credentials.
        # We'll skip this test if no credentials are available.
        import os
        if not os.environ.get("ZOTERO_LIBRARY_ID"):
            return  # Skip if no credentials

        pipeline = ReviewPipeline(sample_config)
        report = pipeline.run()
        assert isinstance(report, Report)
