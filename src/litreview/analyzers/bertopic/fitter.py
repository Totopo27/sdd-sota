"""BERTopic model creation and fitting.

Creates a BERTopic model from text data, handling:
- Embedding model initialization
- Text enrichment (stop word removal + label prefix)
- UMAP/HDBSCAN parameter defaults to avoid KDTree errors
- Model fitting and topic extraction

This is the foundation module — all other BERTopic analyses
receive the fitted model from this class.
"""

from __future__ import annotations

import re

import nltk
import numpy as np
import pandas as pd
import torch

from litreview.config import BERTopicConfig

# Download NLTK stopwords on first import
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# English stop words — NLTK list minus domain-relevant prepositions
_STOP_WORDS = set(nltk.corpus.stopwords.words("english"))
_STOP_WORDS -= {
    "across", "along", "around", "above", "among", "behind", "beneath",
    "beside", "between", "beyond", "down", "during", "into", "near",
    "onto", "through", "throughout", "toward", "under", "underneath",
    "upon", "within", "without",
}


class BERTopicFitter:
    """Create and fit a BERTopic model on text corpus.

    Handles text enrichment (stop word removal + candidate label prefix),
    embedding model initialization, and BERTopic fitting with safe defaults
    for UMAP/HDBSCAN to avoid KDTree errors on small clusters.

    Parameters
    ----------
    config
        BERTopic configuration dataclass.

    Attributes
    ----------
    topic_model
        The fitted BERTopic model (set after fit()).
    topic_assignments
        Hard topic assignments per document (set after fit()).
    """

    def __init__(self, config: BERTopicConfig):
        self.config = config
        self.topic_model = None
        self.topic_assignments: np.ndarray | None = None
        self._topic_probs: np.ndarray | None = None

    @staticmethod
    def _remove_stopwords(text: str) -> str:
        """Remove common English stop words, keeping meaningful prepositions.

        Preserves words that look like prepositions but are often domain-relevant
        (e.g. 'across', 'through', 'within').

        Args:
            text: Raw text string.

        Returns:
            Text with stop words removed, words separated by spaces.
        """
        words = re.findall(r"\b\w+\b", text.lower())
        return " ".join(w for w in words if w not in _STOP_WORDS)

    def _enrich_texts(self, texts: pd.Series) -> list[str]:
        """Prepend candidate labels and remove stop words for better embedding context.

        This gives the embedding model domain-specific context so that
        documents are embedded in a space that reflects the research
        themes we care about, leading to more refined topic discovery.

        Args:
            texts: Series of raw text strings.

        Returns:
            List of enriched text strings with label prefixes and stop words removed.
        """
        labels = list(self.config.candidate_labels.values())
        if not labels:
            return [self._remove_stopwords(t) for t in texts.tolist()]

        label_prefix = " | ".join(labels)
        return [
            f"{label_prefix}: {self._remove_stopwords(text)}"
            for text in texts.tolist()
        ]

    def fit(self, texts: pd.Series) -> "BERTopicFitter":
        """Create and fit a BERTopic model.

        Args:
            texts: Series of text strings to discover topics in.

        Returns:
            self for chaining.
        """
        from bertopic import BERTopic
        from bertopic.vectorizers import ClassTfidfTransformer
        from hdbscan import HDBSCAN
        from sentence_transformers import SentenceTransformer
        import torch
        from umap import UMAP

        # Robust GPU detection
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.cuda.device_count() > 0:
            device = "cuda"
        else:
            device = "cpu"

        embedding_model = SentenceTransformer(
            self.config.embedding_model, device=device
        )

        # Enrich texts with label context and stop word removal
        enriched_texts = self._enrich_texts(texts)

        # Filter out empty texts before fitting — empty strings cause
        # UMAP/HDBSCAN to produce 0-row embeddings → crashes.
        valid_mask = [bool(t.strip()) for t in enriched_texts]
        if not any(valid_mask):
            raise ValueError(
                "All enriched texts are empty after stop-word removal. "
                "Check that your corpus has non-empty abstracts."
            )
        valid_texts = [t for t, v in zip(enriched_texts, valid_mask) if v]
        original_indices = [i for i, v in enumerate(valid_mask) if v]

        # Embed the full valid set ONCE, then validate. Generative models
        # (e.g. 'google/embeddinggemma-300m') can silently produce NaN / all-zero
        # vectors that BERTopic 0.17.x filters out before clustering — leaving
        # HDBSCAN with 0 samples and a cryptic "Invalid shape in axis 0: 0" error.
        # By embedding first we catch this early AND avoid double-encoding.
        embeddings = embedding_model.encode(
            valid_texts, show_progress_bar=False, normalize_embeddings=True
        )
        if not np.isfinite(embeddings).all():
            nan_count = int(~np.isfinite(embeddings)).sum()
            raise ValueError(
                f"Embedding model '{self.config.embedding_model}' produced {nan_count} "
                "NaN/Inf values across the corpus. This often happens when a generative "
                "model is used instead of a dedicated embedding model. Try "
                "'sentence-transformers/all-MiniLM-L6-v2'."
            )
        if np.all(embeddings == 0):
            raise ValueError(
                f"Embedding model '{self.config.embedding_model}' produced all-zero "
                "embeddings for all {} valid texts. Use a dedicated embedding model "
                "such as 'sentence-transformers/all-MiniLM-L6-v2'.".format(len(valid_texts))
            )

        num_samples = len(valid_texts)

        if num_samples < 10:
            from sklearn.cluster import KMeans
            from sklearn.decomposition import PCA

            pca_components = min(2, max(1, num_samples - 1))
            umap_model = PCA(n_components=pca_components)
            n_clusters = min(num_samples, max(2, num_samples // 2)) if num_samples >= 3 else 1
            hdbscan_model = KMeans(n_clusters=n_clusters, n_init="auto", random_state=42)
            min_topic_size = 1
        else:
            # Prevent UMAP n_neighbors and n_components from exceeding sample size
            n_neighbors = min(5, max(2, num_samples - 1))
            n_components = min(5, max(1, num_samples - 2)) if num_samples > 3 else 1
            init_mode = "random" if num_samples < 15 else "spectral"

            # Safe defaults for UMAP
            umap_kwargs = dict(
                n_neighbors=n_neighbors,
                n_components=n_components,
                init=init_mode,
                min_dist=0.0,
                metric="cosine",
                **self.config.umap_kwargs,
            )

            # Set safe defaults for HDBSCAN (prediction_data=True is required for .transform())
            hdbscan_kwargs = dict(
                min_samples=min(1, max(1, num_samples - 1)),
                prediction_data=True,
                **self.config.hdbscan_kwargs,
            )

            umap_model = UMAP(**umap_kwargs)
            hdbscan_model = HDBSCAN(**hdbscan_kwargs)
            min_topic_size = self.config.min_topic_size

        seed_topics = self.config.seed_topics if self.config.seed_topics else None
        seed_words = self.config.seed_words

        # Build c-TF-IDF transformer with seed words (if any)
        if seed_words:
            ctfidf_model = ClassTfidfTransformer(
                seed_words=seed_words, seed_multiplier=2.0
            )
        else:
            ctfidf_model = None

        representation_model = None
        if getattr(self.config, "use_keybert", True):
            try:
                from bertopic.representation import KeyBERTInspired
                representation_model = KeyBERTInspired()
            except Exception:
                representation_model = None

        self.topic_model = BERTopic(
            embedding_model=embedding_model,
            min_topic_size=min_topic_size,
            seed_topic_list=seed_topics,
            ctfidf_model=ctfidf_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            representation_model=representation_model,
        )

        labels, topic_probs = self.topic_model.fit_transform(
            documents=valid_texts, embeddings=embeddings
        )
        self.topic_assignments = np.asarray(labels)
        self._topic_probs = topic_probs

        # Map assignments back to the original index order (-1 = filtered out)
        full_assignments = np.full(len(enriched_texts), -1, dtype=self.topic_assignments.dtype)
        full_assignments[original_indices] = self.topic_assignments
        self.topic_assignments = full_assignments

        return self


    @property
    def results(self) -> dict:
        """Return analysis results as dict.

        Returns:
            Dict with topic_model, topic_assignments, topic_sizes,
            num_topics, outlier_count.
        """
        if self.topic_model is None or self.topic_assignments is None:
            return {}

        topic_sizes = self.topic_model.get_topic_freq()
        outlier_count = topic_sizes.get(-1, 0) if -1 in topic_sizes else 0

        return {
            "topic_model": self.topic_model,
            "topic_assignments": self.topic_assignments,
            "topic_probabilities": self._topic_probs,
            "topic_sizes": topic_sizes,
            "num_topics": len([t for t in set(self.topic_assignments) if t != -1]),
            "outlier_count": outlier_count,
        }
