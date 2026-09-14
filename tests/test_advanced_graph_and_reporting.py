"""Unit tests for advanced graph scientometrics (HITS Hubs/Authorities) and executive markdown report generation."""

import pandas as pd
import pytest

from litreview.network.citation_graph import CitationGraphBuilder
from litreview.reporting.executive_report import (
    generate_executive_markdown,
    export_markdown_report,
)


def test_compute_hits_authorities_and_hubs():
    builder = CitationGraphBuilder()

    survey = "Comprehensive Survey on LLM Agents"
    p1 = "Foundational Tool-Augmented Agent"
    p2 = "Autonomous Code Synthesis Architecture"
    p3 = "Self-Reflective Planning for Agents"

    builder.add_paper_record(survey, year=2024, in_corpus=True, citation_count=80)
    builder.add_paper_record(p1, year=2023, in_corpus=True, citation_count=500)
    builder.add_paper_record(p2, year=2024, in_corpus=True, citation_count=350)
    builder.add_paper_record(p3, year=2023, in_corpus=False, citation_count=200)

    # Survey cites all three foundational works
    builder.add_citation(survey, p1)
    builder.add_citation(survey, p2)
    builder.add_citation(survey, p3)

    # p2 also cites p1
    builder.add_citation(p2, p1)

    hubs, authorities = builder.compute_hits()
    assert len(hubs) > 0
    assert len(authorities) > 0
    assert hubs[survey] > 0.0
    assert authorities[p1] > 0.0

    top_auth = builder.get_top_authorities(top_k=2)
    assert len(top_auth) == 2
    assert "authority_score" in top_auth[0]
    assert top_auth[0]["title"] == p1

    top_hubs = builder.get_top_hubs(top_k=2)
    assert len(top_hubs) >= 1
    assert "hub_score" in top_hubs[0]
    assert top_hubs[0]["title"] == survey

    summary = builder.summary()
    assert "top_authorities" in summary
    assert "top_hubs" in summary


def test_generate_executive_markdown():
    summary_data = {
        "status": "success",
        "corpus": {
            "total_papers": 2,
            "year_range": [2024, 2025],
            "sources": {"Zotero": 2},
        },
        "open_access": {
            "oa_count": 2,
            "total_papers": 2,
            "oa_ratio": 1.0,
        },
        "citation_network": {
            "total_nodes": 45,
            "total_edges": 52,
            "corpus_papers_modeled": 2,
            "role_distribution": {"frontier": 1, "bridge": 1},
            "read_first_recommendations": [
                {
                    "title": "Autonomous Code Agent",
                    "year": 2024,
                    "read_first_score": 0.892,
                    "role": "frontier",
                    "citations": 420,
                    "doi": "10.1000/aca",
                    "score_components": {
                        "topical_relevance": 0.95,
                        "graph_prestige": 0.85,
                        "citation_impact": 0.80,
                        "citation_velocity": 0.90,
                        "methodology": 0.88,
                    },
                }
            ],
            "top_authorities": [
                {"title": "Foundational LLM Agent", "year": 2023, "authority_score": 0.62, "total_citations": 500}
            ],
            "top_hubs": [
                {"title": "Comprehensive Survey", "year": 2024, "hub_score": 0.85, "references_cited_count": 45}
            ],
            "bibliographic_coupling": [
                {
                    "paper_1": "Autonomous Code Agent",
                    "paper_2": "Second Agent Paper",
                    "shared_references_count": 5,
                    "shared_references": ["Foundational LLM Agent", "Ref B"],
                }
            ],
        },
        "neurips_rubric": {
            "criteria_means": {
                "rubric_limitations": 0.70,
                "rubric_reproducibility": 0.85,
                "rubric_experimental_rigor": 0.90,
                "rubric_statistical_significance": 0.40,
                "rubric_compute_resources": 0.60,
                "rubric_score": 0.69,
            },
            "overall_mean_rigor": 0.69,
            "full_text_evaluated_count": 2,
            "corpus_size": 2,
        },
        "taxonomy_validation": {
            "label_counts": {
                "software engineering agents": 2,
            }
        },
    }

    scored_df = pd.DataFrame([
        {
            "Title": "Autonomous Code Agent",
            "Publication Year": 2024,
            "DOI": "10.1000/aca",
            "rigor_source": "full_text",
            "rubric_limitations": 0.70,
            "rubric_reproducibility": 0.85,
            "rubric_experimental_rigor": 0.90,
            "rubric_compute_resources": 0.60,
            "rubric_score": 0.69,
        }
    ])

    md = generate_executive_markdown(summary_data, scored_df=scored_df)
    assert isinstance(md, str)
    assert "# SDD-SOTA Executive Research Briefing" in md
    assert "Read-First Priority Queue" in md
    assert "NeurIPS Rigor & Methodology Audit" in md
    assert "Citation Network & SOTA Anchors" in md
    assert "Autonomous Code Agent" in md
    assert "full_text" in md


def test_export_markdown_report(tmp_path):
    summary_data = {
        "corpus": {"total_papers": 1, "year_range": [2024, 2024]},
        "neurips_rubric": {"overall_mean_rigor": 0.75},
        "citation_network": {},
    }
    out_file = tmp_path / "sdd_briefing.md"
    written_path = export_markdown_report(str(out_file), summary_data)
    assert out_file.exists()
    assert written_path == str(out_file)
    content = out_file.read_text(encoding="utf-8")
    assert "# SDD-SOTA Executive Research Briefing" in content


def test_report_export_markdown(tmp_path):
    from unittest.mock import MagicMock
    from litreview.pipeline import Report

    df = pd.DataFrame([
        {"Title": "Paper A", "Publication Year": 2024, "is_oa": True, "DOI": "10.1000/a"}
    ])
    mock_config = MagicMock()
    report = Report(
        df=df,
        corpus_stats={"total_papers": 1, "year_range": [2024, 2024]},
        topic_coverage={},
        gap_analysis={},
        cross_analysis={},
        bertopic_results={},
        zeroshot_results={"label_counts": {"Agents": 1}},
        config=mock_config,
    )

    out_file = tmp_path / "report_out.md"
    report.export_markdown(str(out_file))
    assert out_file.exists()
    text = out_file.read_text(encoding="utf-8")
    assert "# SDD-SOTA Executive Research Briefing" in text
    assert "Paper A" in text

