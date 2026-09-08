"""Shared test fixtures."""

import os
import tempfile
from pathlib import Path

import pandas as pd
import pytest

from litreview.config import (
    PipelineConfig,
    ZoteroConfig,
    BERTopicConfig,
    ZeroShotConfig,
)


@pytest.fixture
def sample_df():
    """Sample DataFrame mimicking Zotero fetch output."""
    return pd.DataFrame({
        "Title": [
            "Deep Learning for Fluid Dynamics",
            "Optimization of Numerical Methods",
            "GPU Accelerated CFD Simulations",
            "Stochastic Models in Hydrology",
            "Finite Volume Methods for Shallow Water",
        ],
        "Abstract Note": [
            "This paper presents a deep learning approach to fluid dynamics simulations using neural networks.",
            "We propose an optimization framework for numerical methods in computational fluid dynamics.",
            "GPU accelerated computational fluid dynamics simulations achieve 10x speedup over CPU.",
            "Stochastic models are used to predict hydrological events under uncertainty.",
            "Finite volume methods are applied to shallow water equations for flood modeling.",
        ],
        "Publication Year": [2023, 2022, 2023, 2021, 2023],
        "Source": ["Zotero"] * 5,
        "Item Type": ["journalArticle"] * 5,
    })


@pytest.fixture
def sample_config():
    """Minimal PipelineConfig for testing."""
    return PipelineConfig(
        zotero=ZoteroConfig(
            library_id="test_lib",
            api_key="test_key",
            library_type="user",
            collection_names=["test_collection"],
        ),
        bertopic=BERTopicConfig(
            embedding_model="all-MiniLM-L6-v2",
            min_topic_size=2,
            candidate_labels={
                "BLT": "bedload transport",
                "HPC": "high performance computing",
                "GPU": "gpu accelerated",
            },
        ),
        zeroshot=ZeroShotConfig(
            models=["facebook/bart-large-mnli"],
            threshold=0.5,
            candidate_labels={
                "BLT": "bedload transport",
                "HPC": "high performance computing",
                "GPU": "gpu accelerated",
            },
        ),
        paths={
            "raw_data": "data/raw",
            "processed_data": "data/processed",
            "plots": "results/plots",
        },
    )


@pytest.fixture
def sample_config_yaml():
    """Create a temporary config.yaml file for testing."""
    content = """
zotero:
  library_type: "user"
  collection_name: "test_collection"

bertopic:
  embedding_model: "all-MiniLM-L6-v2"
  min_topic_size: 2
  seed_topics: []

zeroshot:
  models:
    - "facebook/bart-large-mnli"
  threshold: 0.5
  candidate_labels:
    BLT: "bedload transport"
    HPC: "high performance computing"

paths:
  raw_data: "data/raw"
  processed_data: "data/processed"
  plots: "results/plots"
"""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write(content)
        f.flush()
        temp_name = f.name
    try:
        yield temp_name
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)


@pytest.fixture
def temp_output_dir():
    """Create a temporary directory for output files."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield tmpdir
