"""BERTopic-based topic discovery analyzer.

DEPRECATED: This class is a backward-compatibility wrapper around the new
composable modules in litreview.analyzers.bertopic.

New code should use:
    from litreview.analyzers.bertopic import BERTopicFitter, TopicDistributionAnalyzer

    fitter = BERTopicFitter(config)
    model = fitter.fit(texts)

    dist = TopicDistributionAnalyzer(model, config)
    dist_df = dist.fit_transform(texts)

This class remains for backward compatibility and internally delegates
to the new modules.
"""

import warnings

import pandas as pd
import numpy as np

from litreview.analyzers.base import Analyzer
from litreview.analyzers.bertopic.fitter import BERTopicFitter
from litreview.analyzers.bertopic.distribution import TopicDistributionAnalyzer
from litreview.analyzers.bertopic.words import TopicWordExtractor
from litreview.analyzers.bertopic.representatives import TopicRepresentativeDocs
from litreview.config import BERTopicConfig


class BERTopicAnalyzer(Analyzer):
    """Discover topics in text corpus using BERTopic with seed guidance.

    DEPRECATED: Use BERTopicFitter + TopicDistributionAnalyzer etc. instead.
    This class wraps the new modules for backward compatibility.

    Uses sentence-transformers for embeddings and BERTopic for clustering.
    Supports seed topics to guide discovery toward user-defined themes.
    """

    def __init__(self, config: BERTopicConfig | None = None, **kwargs):
        warnings.warn(
            "BERTopicAnalyzer is deprecated. "
            "Use BERTopicFitter + TopicDistributionAnalyzer from "
            "litreview.analyzers.bertopic instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        if config is None:
            config = BERTopicConfig(**kwargs)
        self.config = config
        self._fitter = BERTopicFitter(config)
        self._dist_analyzer = None
        self._words_extractor = None
        self._reps_extractor = None
        self._topics: np.ndarray | None = None
        self._topic_probs: np.ndarray | None = None

    def _build_enriched_texts(self, texts: pd.Series) -> list[str]:
        """Prepend candidate labels to texts for domain-specific context."""
        labels = list(self.config.candidate_labels.values())
        if not labels:
            return texts.tolist()
        label_prefix = " | ".join(labels)
        return [f"{label_prefix}: {text}" for text in texts.tolist()]

    def fit(self, texts: pd.Series) -> "BERTopicAnalyzer":
        """Fit BERTopic on texts.

        Args:
            texts: Series of text strings to discover topics in.

        Returns:
            self for chaining.
        """
        self._fitter.fit(texts)
        self._topics = self._fitter.topic_assignments
        self._topic_probs = self._fitter._topic_probs
        return self

    def transform(self, texts: pd.Series) -> pd.DataFrame:
        """Transform texts into topic assignments.

        Must call fit() first.

        Args:
            texts: Series of text strings to assign topics to.

        Returns:
            DataFrame with 'topic' and 'topic_probability' columns.
        """
        if self._topics is None:
            raise RuntimeError("Must call fit() before transform()")

        topics, probs = self._fitter.topic_model.transform(texts.tolist())
        return pd.DataFrame({
            "topic": topics,
            "topic_probability": probs.max(axis=1) if probs.ndim > 1 else 0.0,
        })

    @property
    def results(self) -> dict:
        """Return analysis results as dict.

        Returns:
            Dict with topic_assignments, topic_info, topic_sizes,
            num_topics, outlier_count, topic_representatives.
        """
        if self._fitter.topic_model is None or self._topics is None:
            return {}

        topic_sizes = self._fitter.topic_model.get_topic_freq()
        outlier_count = topic_sizes.get(-1, 0) if -1 in topic_sizes else 0

        # Get representative documents per topic
        topic_representatives = {}
        for topic_id in sorted(set(self._topics)):
            if topic_id == -1:
                continue
            try:
                reps = self._fitter.topic_model.representative_documents_per_topic(
                    topic_id, documents=[], n=3
                )
                topic_representatives[topic_id] = reps
            except Exception:
                topic_representatives[topic_id] = []

        # Get topic words
        topic_words = {}
        for topic_id, words in self._fitter.topic_model.get_topics().items():
            if topic_id == -1:
                continue
            topic_words[topic_id] = [w for w, _ in words[:10]]

        return {
            "topic_assignments": self._topics,
            "topic_probabilities": self._topic_probs,
            "topic_sizes": topic_sizes,
            "topic_words": topic_words,
            "topic_representatives": topic_representatives,
            "num_topics": len([t for t in set(self._topics) if t != -1]),
            "outlier_count": outlier_count,
        }
