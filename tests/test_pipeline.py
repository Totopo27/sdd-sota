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

    def test_pipeline_clean_dedup_and_oa(self, sample_config):
        from unittest.mock import patch, MagicMock

        dirty_df = pd.DataFrame([
            {
                "Title": "Transformer Models for Code",
                "Abstract Note": "First abstract.",
                "DOI": "10.1234/code.1",
                "Publication Year": 2023,
            },
            {
                "Title": "Transformer Models for Code.",
                "Abstract Note": "First abstract with more details.",
                "DOI": "10.1234/code.1",
                "Publication Year": 2024,
            },
            {
                "Title": "Second Unique Paper",
                "Abstract Note": "",
                "DOI": "10.5678/oa.2",
                "Publication Year": 2024,
            },
        ])

        with patch("litreview.pipeline.OAResolver") as mock_oa_cls:
            mock_resolver = MagicMock()
            mock_resolver.enrich_dataframe.side_effect = lambda df, **kwargs: df.assign(
                **{"Abstract Note": df["Abstract Note"].replace("", "Recovered OA Abstract"), "is_oa": True}
            )
            mock_oa_cls.return_value = mock_resolver

            cleaned = ReviewPipeline._clean(dirty_df, resolve_oa=True, dedup=True)

            assert len(cleaned) == 2  # The duplicate was merged
            assert "Recovered OA Abstract" in cleaned["Abstract Note"].values

