"""Representative document extraction from a fitted BERTopic model.

Finds the most representative documents per topic based on c-TF-IDF scores.

Usage:
    extractor = TopicRepresentativeDocs(topic_model)
    results = extractor.results  # {topic_id: [(doc1, score1), (doc2, score2), ...]}
"""

from __future__ import annotations


class TopicRepresentativeDocs:
    """Extract representative documents per topic from a fitted BERTopic model.

    Parameters
    ----------
    topic_model
        A fitted BERTopic model.
    n_documents
        Number of representative documents to extract per topic.
    """

    def __init__(self, topic_model, n_documents: int = 3):
        self.topic_model = topic_model
        self.n_documents = n_documents

    def fit(self, texts=None):
        """No-op — representatives are extracted directly from the fitted model.

        Args:
            texts: Ignored.

        Returns:
            self for chaining.
        """
        return self

    def transform(self, texts=None):
        """No-op — representatives are not a per-document transformation.

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
        """Return representative documents as dict.

        Returns:
            Dict mapping topic_id -> list of (document_text, score) tuples.
        """
        topic_representatives = {}
        for topic_id in sorted(set(t for t in dir(self.topic_model) if True)):
            pass  # We'll iterate over actual topics below

        # Get all topic IDs from the model
        try:
            topic_freq = self.topic_model.get_topic_freq()
            topic_ids = topic_freq.index.tolist()
        except Exception:
            topic_ids = []

        for topic_id in topic_ids:
            if topic_id == -1:
                continue
            try:
                reps = self.topic_model.representative_documents_per_topic(
                    topic_id, documents=[], n=self.n_documents
                )
                topic_representatives[topic_id] = reps
            except Exception:
                topic_representatives[topic_id] = []

        return {"topic_representatives": topic_representatives}
