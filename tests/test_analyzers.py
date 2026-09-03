"""Tests for analyzers module."""

from litreview.analyzers.base import Analyzer
from litreview.analyzers.bertopic import BERTopicAnalyzer
from litreview.analyzers.zeroshot import ZeroShotAnalyzer


class TestAnalyzer:
    def test_is_abstract(self):
        try:
            Analyzer()
        except TypeError:
            pass  # Expected
        else:
            assert False, "Expected TypeError for abstract class"


class TestBERTopicAnalyzer:
    def test_init(self):
        analyzer = BERTopicAnalyzer()
        assert analyzer.config.embedding_model == "all-MiniLM-L6-v2"
        assert analyzer.config.min_topic_size == 2

    def test_init_with_params(self):
        analyzer = BERTopicAnalyzer(
            embedding_model="all-mpnet-base-v2",
            min_topic_size=5,
        )
        assert analyzer.config.embedding_model == "all-mpnet-base-v2"
        assert analyzer.config.min_topic_size == 5

    def test_init_with_candidate_labels(self):
        analyzer = BERTopicAnalyzer(
            candidate_labels={"BLT": "bedload transport", "GPU": "gpu accelerated"},
        )
        assert analyzer.config.candidate_labels == {"BLT": "bedload transport", "GPU": "gpu accelerated"}

    def test_build_enriched_texts(self, sample_df):
        """Test that enriched texts prepend candidate labels."""
        analyzer = BERTopicAnalyzer(
            candidate_labels={"A": "label a", "B": "label b"},
        )
        texts = sample_df["Abstract Note"].head(2)
        enriched = analyzer._build_enriched_texts(texts)
        for e in enriched:
            assert e.startswith("label a | label b: ")

    def test_build_enriched_texts_no_labels(self, sample_df):
        """Test that texts pass through unchanged when no labels."""
        analyzer = BERTopicAnalyzer()
        analyzer.config.candidate_labels = {}
        texts = sample_df["Abstract Note"].head(2)
        enriched = analyzer._build_enriched_texts(texts)
        assert enriched == texts.tolist()

    def test_fit_transform(self, sample_df):
        """Test that fit_transform returns a DataFrame with topic assignments."""
        analyzer = BERTopicAnalyzer(min_topic_size=1)
        abstracts = sample_df["Abstract Note"].dropna()
        results = analyzer.fit_transform(abstracts)
        assert isinstance(results, dict)
        assert "topic_assignments" in results
        assert "topic_sizes" in results
        assert "topic_words" in results
        assert "topic_representatives" in results
        assert "num_topics" in results
        assert "outlier_count" in results
        assert results["num_topics"] >= 0

    def test_results(self, sample_df):
        """Test that results property returns the same as fit_transform."""
        analyzer = BERTopicAnalyzer(min_topic_size=1)
        abstracts = sample_df["Abstract Note"].dropna()
        analyzer.fit_transform(abstracts)
        results = analyzer.results
        assert "topic_assignments" in results


class TestZeroShotAnalyzer:
    def test_init(self):
        analyzer = ZeroShotAnalyzer()
        assert len(analyzer.models) >= 1
        assert analyzer.threshold == 0.5

    def test_init_with_params(self):
        analyzer = ZeroShotAnalyzer(
            models=["facebook/bart-large-mnli"],
            threshold=0.7,
        )
        assert analyzer.models == ["facebook/bart-large-mnli"]
        assert analyzer.threshold == 0.7

    def test_fit_transform(self, sample_df):
        """Test that fit_transform returns a DataFrame with classifications."""
        analyzer = ZeroShotAnalyzer(
            models=["facebook/bart-large-mnli"],
            threshold=0.5,
            candidate_labels={"A": "test label"},
        )
        abstracts = sample_df["abstract"].dropna().head(5)
        results = analyzer.fit_transform(abstracts)
        assert isinstance(results, dict)
        assert "classifications" in results
        assert "label_counts" in results
        assert "threshold" in results
        assert len(results["classifications"]) == len(abstracts)

    def test_results(self, sample_df):
        """Test that results property returns the same as fit_transform."""
        analyzer = ZeroShotAnalyzer(
            models=["facebook/bart-large-mnli"],
            threshold=0.5,
            candidate_labels={"A": "test label"},
        )
        abstracts = sample_df["abstract"].dropna().head(5)
        analyzer.fit_transform(abstracts)
        results = analyzer.results
        assert "classifications" in results
