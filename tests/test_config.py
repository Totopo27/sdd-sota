"""Tests for config module."""

import os
from litreview.config import (
    PipelineConfig,
    ZoteroConfig,
    BERTopicConfig,
    ZeroShotConfig,
    load_config,
)


class TestZoteroConfig:
    def test_from_config(self):
        cfg = ZoteroConfig.from_config({
            "library_type": "user",
            "collection_names": ["my_collection"],
        })
        assert cfg.library_type == "user"
        assert cfg.collection_names == ["my_collection"]

    def test_from_config_legacy_collection_name(self):
        """Backwards compatibility: single collection_name key."""
        cfg = ZoteroConfig.from_config({
            "library_type": "user",
            "collection_name": "my_collection",
        })
        assert cfg.library_type == "user"
        assert cfg.collection_names == ["my_collection"]

    def test_from_config_missing_optional(self):
        cfg = ZoteroConfig.from_config({})
        assert cfg.library_type == "user"
        assert cfg.collection_names is None


class TestBERTopicConfig:
    def test_from_config(self):
        cfg = BERTopicConfig.from_config({
            "embedding_model": "all-mpnet-base-v2",
            "min_topic_size": 5,
            "seed_topics": [["deep learning"], ["optimization"]],
        })
        assert cfg.embedding_model == "all-mpnet-base-v2"
        assert cfg.min_topic_size == 5
        assert cfg.seed_topics == [["deep learning"], ["optimization"]]

    def test_from_config_defaults(self):
        cfg = BERTopicConfig.from_config({})
        assert cfg.embedding_model == "all-MiniLM-L6-v2"
        assert cfg.min_topic_size == 2
        assert cfg.seed_topics is None
        assert cfg.candidate_labels == {}
        assert cfg.compute is True
        assert cfg.analyses == ["distribution", "words", "representatives"]
        assert cfg.top_n_topics == 3
        assert cfg.distribution_window == 4
        assert cfg.distribution_stride == 2

    def test_from_config_with_candidate_labels(self):
        cfg = BERTopicConfig.from_config({
            "candidate_labels": {"BLT": "bedload transport", "GPU": "gpu accelerated"},
        })
        assert cfg.candidate_labels == {"BLT": "bedload transport", "GPU": "gpu accelerated"}

    def test_from_config_compute_false(self):
        cfg = BERTopicConfig.from_config({"compute": False})
        assert cfg.compute is False

    def test_from_config_analyses_subset(self):
        cfg = BERTopicConfig.from_config({"analyses": ["distribution", "words"]})
        assert cfg.analyses == ["distribution", "words"]

    def test_from_config_auto_seed_words(self):
        cfg = BERTopicConfig.from_config({
            "candidate_labels": {"A": "label a", "B": "label b"},
        })
        assert cfg.seed_words == ["label a", "label b"]

    def test_from_config_explicit_seed_words(self):
        cfg = BERTopicConfig.from_config({
            "candidate_labels": {"A": "label a"},
            "seed_words": ["custom", "words"],
        })
        assert cfg.seed_words == ["custom", "words"]


class TestZeroShotConfig:
    def test_from_config(self):
        cfg = ZeroShotConfig.from_config({
            "threshold": 0.7,
            "candidate_labels": {"A": "Label A", "B": "Label B"},
        })
        assert cfg.threshold == 0.7
        assert cfg.candidate_labels == {"A": "Label A", "B": "Label B"}

    def test_from_config_defaults(self):
        cfg = ZeroShotConfig.from_config({})
        assert len(cfg.models) == 2
        assert cfg.threshold == 0.5


class TestPipelineConfig:
    def test_from_file(self, sample_config_yaml):
        cfg = PipelineConfig.from_file(sample_config_yaml)
        assert cfg.zotero.library_type == "user"
        assert cfg.zotero.collection_name == "test_collection"
        assert cfg.bertopic.embedding_model == "all-MiniLM-L6-v2"
        assert cfg.zeroshot.threshold == 0.5
        assert "BLT" in cfg.zeroshot.candidate_labels

    def test_load_config(self, sample_config_yaml):
        cfg = load_config(sample_config_yaml)
        assert isinstance(cfg, PipelineConfig)
        assert cfg.paths["plots"] == "results/plots"


class TestLoadConfig:
    def test_load_config_nonexistent(self):
        try:
            load_config("nonexistent.yaml")
        except FileNotFoundError:
            pass  # Expected
        else:
            assert False, "Expected FileNotFoundError"
