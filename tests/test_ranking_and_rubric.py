"""Unit tests for Feynman scientometric intelligence (Read-First scoring, Graph roles, NeurIPS rubric)."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from litreview.network.citation_graph import CitationGraphBuilder
from litreview.analyzers.rubric import NeurIPSRubricAnalyzer, NEURIPS_RUBRIC_CRITERIA


def test_graph_roles_computation():
    builder = CitationGraphBuilder()
    
    # 1. Foundation node (external cited paper, older, high in-degree)
    builder.add_paper_record("Vaswani et al. Attention is All You Need", year=2017, in_corpus=False, citation_count=100000)
    
    # 2. Frontier node (in corpus, recent 2025, high citation velocity)
    builder.add_paper_record("DeepSeek-R1 Technical Report", year=2025, in_corpus=True, citation_count=500)
    
    # 3. Standard corpus paper
    builder.add_paper_record("Survey on LLM Agents", year=2024, in_corpus=True, citation_count=50)

    # Add citations
    builder.add_citation("DeepSeek-R1 Technical Report", "Vaswani et al. Attention is All You Need")
    builder.add_citation("Survey on LLM Agents", "Vaswani et al. Attention is All You Need")
    builder.add_citation("Survey on LLM Agents", "DeepSeek-R1 Technical Report")

    roles = builder.compute_graph_roles()
    assert roles["Vaswani et al. Attention is All You Need"] == "foundation"
    assert roles["DeepSeek-R1 Technical Report"] == "frontier"
    assert roles["Survey on LLM Agents"] in ("corpus_work", "bridge", "methodology_anchor")


def test_read_first_score_calculation():
    builder = CitationGraphBuilder()
    
    p1 = "Attention is All You Need"
    p2 = "Modern Fast LLM"
    
    builder.add_paper_record(p1, year=2017, in_corpus=True, citation_count=10000)
    builder.add_paper_record(p2, year=2025, in_corpus=True, citation_count=200)
    builder.add_citation(p2, p1)

    topical_map = {p1: 0.9, p2: 0.95}
    methodology_map = {p1: 0.8, p2: 0.85}

    recommendations = builder.compute_read_first_scores(
        topical_relevance_map=topical_map,
        methodology_scores_map=methodology_map,
        top_k=2,
    )

    assert len(recommendations) == 2
    # Verify required keys
    for r in recommendations:
        assert "read_first_score" in r
        assert "role" in r
        assert "score_components" in r
        assert 0.0 <= r["read_first_score"] <= 1.0


@patch("litreview.analyzers.rubric.pipeline")
def test_neurips_rubric_analyzer(mock_pipeline_fn):
    mock_pipe = MagicMock()
    mock_pipeline_fn.return_value = mock_pipe

    # Mock zero-shot classification output
    criteria_labels = list(NEURIPS_RUBRIC_CRITERIA.values())
    mock_pipe.return_value = [
        {
            "labels": criteria_labels,
            "scores": [0.8, 0.75, 0.9, 0.4, 0.6],
        }
    ]

    analyzer = NeurIPSRubricAnalyzer()
    analyzer.fit(pd.Series(["We release our code at github.com/test and benchmark against baselines."]))
    df = analyzer.transform(pd.Series(["We release our code at github.com/test and benchmark against baselines."]))

    assert len(df) == 1
    assert "rubric_score" in df.columns
    assert "rubric_reproducibility" in df.columns
    assert df["rubric_reproducibility"].iloc[0] == 0.75
    assert df["rubric_score"].iloc[0] > 0.0

    summary = analyzer.results
    assert "overall_mean_rigor" in summary
    assert summary["overall_mean_rigor"] > 0.0
