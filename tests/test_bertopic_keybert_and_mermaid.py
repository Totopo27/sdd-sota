import pandas as pd
import pytest

from litreview.config import BERTopicConfig
from litreview.analyzers.bertopic.fitter import BERTopicFitter
from litreview.reporting.executive_report import generate_topic_mermaid, generate_executive_markdown


def test_bertopic_config_use_keybert():
    cfg_default = BERTopicConfig()
    assert cfg_default.use_keybert is True

    cfg_custom = BERTopicConfig.from_config({"use_keybert": False})
    assert cfg_custom.use_keybert is False


def test_small_corpus_bertopic_with_kmeans_and_keybert():
    texts = pd.Series([
        "SWE-agent provides automated software engineering workflows with language agents.",
        "AutoCodeRover autonomous program improvement, fault localization and bug repair.",
        "Benchmarking large language models and software engineering autonomous agents.",
    ])
    config = BERTopicConfig(use_keybert=True, min_topic_size=1)
    fitter = BERTopicFitter(config)
    fitter.fit(texts)

    assert fitter.topic_model is not None
    assert fitter.topic_assignments is not None
    assert len(fitter.topic_assignments) == 3
    results = fitter.results
    assert "topic_model" in results
    assert "num_topics" in results
    assert results["num_topics"] >= 1

    # Verify KeyBERT representation produced keywords
    topics = fitter.topic_model.get_topics()
    assert len(topics) > 0
    for topic_id, words in topics.items():
        if topic_id != -1:
            assert len(words) > 0
            # Words should have non-empty string and float score
            assert isinstance(words[0][0], str)
            assert len(words[0][0]) > 0


def test_generate_topic_mermaid():
    topics_data = {
        "num_topics": 2,
        "topic_sizes": {0: 2, 1: 1},
        "top_words": {
            "0": [["agents", 0.45], ["software", 0.40], ["engineering", 0.35]],
            "1": [["bugs", 0.42], ["repairs", 0.38], ["programs", 0.31]],
        },
        "topic_papers": {
            0: ["SWE-agent: Agent-Computer Interfaces", "Demystifying LLM SE Agents"],
            1: ["AutoCodeRover: Autonomous Program Improvement"],
        },
    }

    mermaid_code = generate_topic_mermaid(topics_data)
    assert "```mermaid" in mermaid_code
    assert "graph TD" in mermaid_code
    assert "Topic 0" in mermaid_code
    assert "Topic 1" in mermaid_code
    assert "SWE-agent" in mermaid_code
    assert "AutoCodeRover" in mermaid_code


def test_executive_markdown_includes_topic_mermaid_and_table():
    summary = {
        "corpus": {
            "total_papers": 3,
            "non_empty_abstracts": 3,
            "min_year": 2024,
            "max_year": 2025,
        },
        "topics": {
            "num_topics": 2,
            "topic_sizes": {0: 2, 1: 1},
            "top_words": {
                "0": [["agents", 0.45], ["software", 0.40]],
                "1": [["repairs", 0.38], ["bugs", 0.35]],
            },
            "topic_papers": {
                0: ["Paper Alpha", "Paper Beta"],
                1: ["Paper Gamma"],
            },
        },
        "taxonomy_validation": {"label_counts": {}},
        "graph_ranking": {
            "read_first_queue": [],
            "top_authorities": [],
            "top_hubs": [],
        },
    }

    md = generate_executive_markdown(summary)
    assert "## 🗺️ Semantic Topic Topology & Clusters (BERTopic + KeyBERT)" in md
    assert "```mermaid" in md
    assert "Paper Alpha" in md
    assert "Paper Gamma" in md
    assert "| Topic | Size | KeyBERT Semantic Descriptors |" in md
