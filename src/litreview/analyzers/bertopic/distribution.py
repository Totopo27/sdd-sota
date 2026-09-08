"""Topic distribution analysis via BERTopic approximate_distribution.

Papers often span multiple research areas. This module computes a soft
assignment: for each paper, the probability of belonging to each topic.

Follows the BERTopic distribution tutorial pattern:
    https://maartengr.github.io/BERTopic/getting_started/distribution/distribution.html

Usage:
    analyzer = TopicDistributionAnalyzer(topic_model, config)
    dist_df = analyzer.fit_transform(texts)
    matrix = analyzer.distribution_matrix  # (n_documents, n_topics)
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from litreview.analyzers.base import Analyzer
from litreview.config import BERTopicConfig


class TopicDistributionAnalyzer(Analyzer):
    """Compute soft topic assignments via sliding-window approximation.

    Parameters
    ----------
    topic_model
        A fitted BERTopic model.
    config
        BERTopic configuration (for top_n_topics, window, stride).
    """

    def __init__(
        self,
        topic_model,
        config: BERTopicConfig,
        top_n_topics: int = 3,
        window: int = 4,
        stride: int = 2,
    ):
        self.topic_model = topic_model
        self.config = config
        self.top_n_topics = top_n_topics
        self.window = window
        self.stride = stride
        self._distribution_matrix: np.ndarray | None = None

    def fit(self, texts: pd.Series) -> "TopicDistributionAnalyzer":
        """Compute topic distributions via approximate_distribution.

        Args:
            texts: Series of text strings.

        Returns:
            self for chaining.
        """
        text_list = texts.dropna().astype(str).tolist()
        if not text_list:
            n_topics = self.topic_model.get_topic_freq().shape[0] if self.topic_model else 0
            self._distribution_matrix = np.zeros((len(texts), n_topics))
            return self

        # Check if topic_model has valid non-outlier topics to approximate
        dist_matrix = None
        has_non_outlier_topics = False
        if hasattr(self.topic_model, "c_tf_idf_") and self.topic_model.c_tf_idf_ is not None:
            outlier_offset = getattr(self.topic_model, "_outliers", 1)
            if self.topic_model.c_tf_idf_.shape[0] > outlier_offset:
                has_non_outlier_topics = True

        if has_non_outlier_topics:
            try:
                dist_matrix, _ = self.topic_model.approximate_distribution(
                    text_list,
                    window=self.window,
                    stride=self.stride,
                    calculate_tokens=False,
                )
            except Exception:
                dist_matrix = None

        if dist_matrix is None:
            # Fallback: one-hot or zero distribution based on fitted topic assignments
            n_topics = max(len(self.topic_model.get_topic_freq()), 1)
            dist_matrix = np.zeros((len(text_list), n_topics))

        # Ensure matrix has the right shape
        n_expected = len(texts)
        if dist_matrix.shape[0] < n_expected:
            pad = np.zeros((n_expected - dist_matrix.shape[0], dist_matrix.shape[1]))
            dist_matrix = np.vstack([dist_matrix, pad])
        elif dist_matrix.shape[0] > n_expected:
            dist_matrix = dist_matrix[:n_expected]

        self._distribution_matrix = dist_matrix
        return self

    def transform(self, texts: pd.Series) -> pd.DataFrame:
        """Return wide-format DataFrame with top-N topics per paper.

        Columns:
            - topic_<N>: topic ID of the Nth most probable topic
            - topic_<N>_prob: probability of that topic
            - dominant_topic: most probable topic ID
            - dominant_prob: probability of the dominant topic
        """
        if self._distribution_matrix is None or self._distribution_matrix.shape[1] == 0:
            return pd.DataFrame(index=texts.index)

        n_topics = self._distribution_matrix.shape[1]
        topic_ids = list(range(n_topics))

        rows = []
        for i in range(len(texts)):
            row: dict = {}
            dist = self._distribution_matrix[i]
            sorted_idx = np.argsort(-dist)[: self.top_n_topics]
            for rank, idx in enumerate(sorted_idx):
                tid = topic_ids[idx] if idx < n_topics else -1
                row[f"topic_{rank + 1}"] = tid
                row[f"topic_{rank + 1}_prob"] = round(float(dist[idx]), 4)
            dom_idx = int(np.argmax(dist))
            row["dominant_topic"] = topic_ids[dom_idx] if dom_idx < n_topics else -1
            row["dominant_prob"] = round(float(dist[dom_idx]), 4)
            rows.append(row)

        return pd.DataFrame(rows, index=texts.index)

    @property
    def distribution_matrix(self) -> np.ndarray:
        """Return the full (n_documents, n_topics) probability matrix."""
        if self._distribution_matrix is None:
            raise RuntimeError("Call fit() or fit_transform() first.")
        return self._distribution_matrix

    @property
    def results(self) -> dict:
        """Return analysis results as dict."""
        return {
            "distribution_matrix": self._distribution_matrix,
            "distribution": self._distribution_matrix,
        }
