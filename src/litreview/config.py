"""Configuration dataclasses and loader."""

from dataclasses import dataclass, field
from pathlib import Path
import yaml
import os
from dotenv import load_dotenv

load_dotenv()


@dataclass
class ZoteroConfig:
    library_id: str
    api_key: str
    library_type: str = "user"
    collection_names: list[str] | None = None

    @property
    def collection_name(self) -> str | None:
        """Legacy single-collection accessor for backwards compatibility."""
        if self.collection_names is None:
            return None
        return self.collection_names[0] if self.collection_names else None

    @classmethod
    def from_config(cls, cfg: dict) -> "ZoteroConfig":
        collections = cfg.get("collection_names")
        # Support legacy single collection_name for backwards compatibility
        if collections is None and "collection_name" in cfg:
            val = cfg["collection_name"]
            if val is not None:
                collections = [val] if isinstance(val, str) else val
        return cls(
            library_id=os.environ.get("ZOTERO_LIBRARY_ID", cfg.get("library_id", "")),
            api_key=os.environ.get("ZOTERO_API_KEY", cfg.get("api_key", "")),
            library_type=cfg.get("library_type", "user"),
            collection_names=collections,
        )


@dataclass
class BERTopicConfig:
    """Configuration for BERTopic topic discovery and analyses.

    Parameters
    ----------
    compute : bool
        Enable/disable the entire BERTopic pipeline.
    analyses : list[str]
        Which analyses to run after fitting. Options:
        "distribution" (soft topic assignments),
        "words" (top words per topic),
        "representatives" (representative documents per topic).
        The fitter always runs when compute=True; these are extras.
    embedding_model : str
        Sentence-transformers model for document embeddings.
    min_topic_size : int
        Minimum number of documents per topic.
    seed_topics : list[list[str]] | None
        Seed topics to guide discovery.
    seed_words : list[str] | None
        Seed words to guide topic extraction.
    umap_kwargs : dict
        Keyword arguments passed to UMAP.
    hdbscan_kwargs : dict
        Keyword arguments passed to HDBSCAN.
    candidate_labels : dict[str, str]
        Domain labels used for text enrichment and seed words.
    top_n_topics : int
        Number of top topics to keep per document in distribution output.
    distribution_window : int
        Token window size for approximate_distribution sliding window.
    distribution_stride : int
        Step size for approximate_distribution window shifts.
    """

    compute: bool = True
    analyses: list[str] = field(
        default_factory=lambda: ["distribution", "words", "representatives"]
    )
    embedding_model: str = "all-MiniLM-L6-v2"
    min_topic_size: int = 2
    seed_topics: list[list[str]] | None = None
    seed_words: list[str] | None = None
    umap_kwargs: dict = field(default_factory=dict)
    hdbscan_kwargs: dict = field(default_factory=dict)
    candidate_labels: dict[str, str] = field(default_factory=dict)
    top_n_topics: int = 3
    distribution_window: int = 4
    distribution_stride: int = 2
    use_keybert: bool = True

    @classmethod
    def from_config(cls, cfg: dict) -> "BERTopicConfig":
        candidate_labels = {k: v for k, v in cfg.get("candidate_labels", {}).items()}
        # Auto-extract seed words from candidate label descriptions
        seed_words = cfg.get("seed_words")
        if seed_words is None and candidate_labels:
            seed_words = list(candidate_labels.values())
        return cls(
            compute=cfg.get("compute", True),
            analyses=cfg.get("analyses", ["distribution", "words", "representatives"]),
            embedding_model=cfg.get("embedding_model", "all-MiniLM-L6-v2"),
            min_topic_size=cfg.get("min_topic_size", 2),
            seed_topics=cfg.get("seed_topics"),
            seed_words=seed_words,
            candidate_labels=candidate_labels,
            top_n_topics=cfg.get("top_n_topics", 3),
            distribution_window=cfg.get("distribution_window", 4),
            distribution_stride=cfg.get("distribution_stride", 2),
            use_keybert=cfg.get("use_keybert", True),
        )


@dataclass
class ZeroShotConfig:
    models: list[str] = field(
        default_factory=lambda: [
            # "facebook/bart-large-mnli",  # too slow — use deberta-v3-base instead
            "MoritzLaurer/deberta-v3-base-zeroshot-v2.0",
            "MoritzLaurer/deberta-v3-large-zeroshot-v2.0",
        ]
    )
    threshold: float = 0.5
    batch_size: int = 64
    candidate_labels: dict[str, str] = field(default_factory=dict)

    @classmethod
    def from_config(cls, cfg: dict) -> "ZeroShotConfig":
        return cls(
            models=cfg.get(
                "models",
                ["facebook/bart-large-mnli", "MoritzLaurer/deberta-v3-large-zeroshot-v2.0"],
            ),
            threshold=cfg.get("threshold", 0.5),
            batch_size=cfg.get("batch_size", 16),
            candidate_labels={k: v for k, v in cfg.get("candidate_labels", {}).items()},
        )


@dataclass
class PipelineConfig:
    zotero: ZoteroConfig
    bertopic: BERTopicConfig
    zeroshot: ZeroShotConfig
    paths: dict = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: str | Path = "config.yaml") -> "PipelineConfig":
        with open(path) as f:
            cfg = yaml.safe_load(f) or {}

        top_labels = cfg.get("labels", {})
        top_models = cfg.get("models", [])

        bertopic_dict = dict(cfg.get("bertopic", {}))
        if "candidate_labels" not in bertopic_dict and top_labels:
            bertopic_dict["candidate_labels"] = top_labels

        zeroshot_dict = dict(cfg.get("zeroshot", {}))
        if "candidate_labels" not in zeroshot_dict and top_labels:
            zeroshot_dict["candidate_labels"] = top_labels
        if "models" not in zeroshot_dict and top_models:
            zeroshot_dict["models"] = top_models

        return cls(
            zotero=ZoteroConfig.from_config(cfg.get("zotero", {})),
            bertopic=BERTopicConfig.from_config(bertopic_dict),
            zeroshot=ZeroShotConfig.from_config(zeroshot_dict),
            paths=cfg.get("paths", {}),
        )


def load_config(path: str | Path = "config.yaml") -> PipelineConfig:
    """Load configuration from a YAML file. Returns a PipelineConfig dataclass."""
    return PipelineConfig.from_file(path)
