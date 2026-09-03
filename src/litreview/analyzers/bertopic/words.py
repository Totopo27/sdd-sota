"""Topic word extraction from a fitted BERTopic model.

Extracts the top N words per topic from the c-TF-IDF representation.

Usage:
    extractor = TopicWordExtractor(topic_model)
    results = extractor.results  # {topic_id: [word1, word2, ...]}
"""

from __future__ import annotations


class TopicWordExtractor:
    """Extract top words per topic from a fitted BERTopic model.

    Parameters
    ----------
    topic_model
        A fitted BERTopic model.
    top_n_words
        Number of top words to extract per topic.
    """

    def __init__(self, topic_model, top_n_words: int = 10):
        self.topic_model = topic_model
        self.top_n_words = top_n_words

    def fit(self, texts=None):
        """No-op — words are extracted directly from the fitted model.

        Args:
            texts: Ignored.

        Returns:
            self for chaining.
        """
        return self

    def transform(self, texts=None):
        """No-op — words are not a per-document transformation.

        Args:
            texts: Ignored.

        Returns:
            Empty DataFrame (for API compatibility).
        """
        import pandas as pd
        return pd.DataFrame()

    def fit_transform(self, texts=None):
        """Fit and transform in one call.

        Args:
            texts: Ignored.

        Returns:
            Empty DataFrame (for API compatibility).
        """
        self.fit(texts)
        return self.transform(texts)

    @property
    def results(self) -> dict:
        """Return topic words as dict.

        Returns:
            Dict mapping topic_id -> list of top N word strings.
        """
        topic_words = {}
        for topic_id, words in self.topic_model.get_topics().items():
            if topic_id == -1:
                continue
            topic_words[topic_id] = [w for w, _ in words[: self.top_n_words]]
        return {"topic_words": topic_words}
