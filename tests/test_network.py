"""Tests for citation network and Semantic Scholar module."""

import json
import os
import tempfile
import networkx as nx
import pandas as pd
import pytest

from litreview.network.semantic_scholar import SemanticScholarClient
from litreview.network.citation_graph import CitationGraphBuilder
from litreview.network.visualization import plot_citation_network


class MockResponse:
    def __init__(self, json_data, status_code=200):
        self._json_data = json_data
        self.status_code = status_code

    def json(self):
        return self._json_data


class TestSemanticScholarClient:
    def test_cache_hit_and_save(self, tmp_path):
        cache_file = tmp_path / "s2_cache.json"
        client = SemanticScholarClient(cache_path=cache_file)
        assert client.cache == {}

        # Save mock paper in cache
        client._update_cache("paper_123", {"title": "Test Paper", "citationCount": 50})
        assert cache_file.exists()

        # Reload client from same cache file
        client2 = SemanticScholarClient(cache_path=cache_file)
        assert client2.cache.get("paper_123")["title"] == "Test Paper"

    def test_batch_context_manager_defers_save(self, tmp_path):
        cache_file = tmp_path / "s2_batch_cache.json"
        client = SemanticScholarClient(cache_path=cache_file)

        with client:
            client._update_cache("p1", {"title": "Paper 1"})
            # File should not be written to disk yet during batch mode
            assert not cache_file.exists()
            client._update_cache("p2", {"title": "Paper 2"})
            assert not cache_file.exists()

        # Flushed to disk after exiting context manager
        assert cache_file.exists()
        with open(cache_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        assert "p1" in data and "p2" in data

    def test_get_paper_for_record_mocked(self, monkeypatch, tmp_path):
        cache_file = tmp_path / "s2_cache.json"
        client = SemanticScholarClient(cache_path=cache_file)

        mock_data = {
            "paperId": "p_001",
            "title": "Attention Is All You Need",
            "year": 2017,
            "citationCount": 100000,
            "influentialCitationCount": 15000,
            "references": [
                {"paperId": "ref_1", "title": "Neural Machine Translation", "year": 2014}
            ],
            "citations": [
                {"paperId": "cit_1", "title": "BERT: Pre-training", "year": 2018}
            ],
        }

        def mock_get(url, headers=None, timeout=None):
            return MockResponse(mock_data)

        monkeypatch.setattr("requests.get", mock_get)

        paper = client.get_paper_for_record(title="Attention Is All You Need", doi="10.5555/3295222.3295349")
        assert paper is not None
        assert paper["title"] == "Attention Is All You Need"
        assert paper["citationCount"] == 100000
        # Check that it was saved in cache
        assert "DOI:10.5555/3295222.3295349" in client.cache


class TestCitationGraphBuilder:
    @pytest.fixture
    def mock_papers_df(self):
        return pd.DataFrame({
            "Title": ["Paper A", "Paper B", "Paper C"],
            "Abstract Note": ["Abstract A", "Abstract B", "Abstract C"],
            "Publication Year": [2023, 2022, 2021],
            "Source": ["Zotero", "Zotero", "Zotero"],
            "Item Type": ["journalArticle", "journalArticle", "journalArticle"],
            "DOI": ["doi/a", "doi/b", "doi/c"],
        })

    def test_build_graph_synthetic(self):
        builder = CitationGraphBuilder()
        # Synthetic network:
        # Paper A cites Paper B and Paper C
        # Paper B cites Paper C
        # Seminal paper (Paper C) should have the highest in-degree / PageRank
        builder.add_paper_record("Paper A", year=2023, doi="doi/a", in_corpus=True)
        builder.add_paper_record("Paper B", year=2022, doi="doi/b", in_corpus=True)
        builder.add_paper_record("Paper C", year=2021, doi="doi/c", in_corpus=True)

        builder.add_citation("Paper A", "Paper B")
        builder.add_citation("Paper A", "Paper C")
        builder.add_citation("Paper B", "Paper C")

        summary = builder.summary()
        assert summary["total_nodes"] == 3
        assert summary["total_edges"] == 3

        foundational = builder.get_foundational_papers(top_k=1)
        assert len(foundational) == 1
        assert foundational[0]["title"] == "Paper C"

        derivatives = builder.get_derivative_works(top_k=1)
        assert len(derivatives) == 1
        assert derivatives[0]["title"] == "Paper A"

    def test_bibliographic_coupling(self):
        builder = CitationGraphBuilder()
        builder.add_paper_record("Paper A", year=2023, in_corpus=True)
        builder.add_paper_record("Paper B", year=2023, in_corpus=True)
        builder.add_paper_record("External Ref 1", year=2015, in_corpus=False)
        builder.add_paper_record("External Ref 2", year=2016, in_corpus=False)

        # Both Paper A and Paper B cite External Ref 1 and 2
        builder.add_citation("Paper A", "External Ref 1")
        builder.add_citation("Paper A", "External Ref 2")
        builder.add_citation("Paper B", "External Ref 1")
        builder.add_citation("Paper B", "External Ref 2")

        coupling = builder.compute_bibliographic_coupling()
        assert len(coupling) > 0
        pair = coupling[0]
        assert set([pair["paper_1"], pair["paper_2"]]) == {"Paper A", "Paper B"}
        assert pair["shared_references_count"] == 2


class TestNetworkVisualization:
    def test_plot_citation_network(self, tmp_path):
        builder = CitationGraphBuilder()
        builder.add_paper_record("Paper A", year=2023, in_corpus=True)
        builder.add_paper_record("Paper B", year=2022, in_corpus=True)
        builder.add_citation("Paper A", "Paper B")

        out_path = str(tmp_path / "network.png")
        plot_citation_network(builder, out_path)
        assert os.path.exists(out_path)
        assert os.path.getsize(out_path) > 0
