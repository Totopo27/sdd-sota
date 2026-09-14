"""Unit tests for academic PDF section parser."""

import io
from unittest.mock import MagicMock, patch
import pytest

from litreview.discovery.pdf_parser import (
    PDFSectionParser,
    clean_academic_text,
    segment_academic_sections,
)


def test_clean_academic_text():
    raw_text = "This is a trans-\nformer architecture that runs on mul-\ntiple GPUs.\n\nPage 1 of 12\n\nNext sentence."
    cleaned = clean_academic_text(raw_text)
    assert "transformer architecture" in cleaned
    assert "multiple GPUs." in cleaned
    # Hyphens removed across wraps
    assert "trans-" not in cleaned


def test_segment_academic_sections_all_present():
    synthetic_paper = """
Abstract
This paper introduces an agentic workflow for automated program synthesis.

1. Introduction
Modern software engineering requires rigorous evaluation.

2. Methodology
We propose a dual-agent architecture with planning and verification lanes.

3. Experiments and Results
We evaluate on HumanEval and SWE-bench, achieving 84.5% pass@1.

4. Compute and Hardware Resources
All models were trained on 8x NVIDIA H100 SXM5 80GB GPUs for 140 GPU-hours using PyTorch with DeepSpeed ZeRO-3.

5. Limitations and Ethical Considerations
Our model may produce insecure code if prompt constraints are omitted. The study is limited to Python.

6. Reproducibility Statement
The source code, training logs, and benchmark datasets are publicly available at https://github.com/example/repro.
"""
    sections = segment_academic_sections(synthetic_paper)

    assert "agentic workflow for automated program synthesis" in sections["abstract"]
    assert "dual-agent architecture" in sections["methodology"]
    assert "SWE-bench" in sections["experiments"]
    assert "8x NVIDIA H100" in sections["compute"]
    assert "limited to Python" in sections["limitations"]
    assert "https://github.com/example/repro" in sections["reproducibility"]
    assert sections["has_limitations_section"] is True
    assert sections["has_compute_section"] is True
    assert sections["has_reproducibility_section"] is True


def test_segment_academic_sections_missing_limitations():
    short_paper = """
Abstract
A basic study without detailed sections.

Introduction
Some introductory context.

Methodology
Our proposed algorithm.

Conclusion
We conclude the paper.
"""
    sections = segment_academic_sections(short_paper)
    assert sections["has_limitations_section"] is False
    assert sections["has_compute_section"] is False
    assert sections["has_reproducibility_section"] is False
    assert sections["limitations"] == ""
    assert "proposed algorithm" in sections["methodology"]


@patch("litreview.discovery.pdf_parser.pypdf.PdfReader")
def test_pdf_section_parser_from_bytes(mock_reader_cls):
    mock_reader = MagicMock()
    mock_reader_cls.return_value = mock_reader
    mock_page1 = MagicMock()
    mock_page1.extract_text.return_value = "Abstract\nNovel self-healing agents.\n\n1. Introduction\nBackground."
    mock_page2 = MagicMock()
    mock_page2.extract_text.return_value = "5. Limitations\nLimited context window.\n\nCompute\nTrained on 4x RTX 4090."
    mock_reader.pages = [mock_page1, mock_page2]

    parser = PDFSectionParser()
    result = parser.parse_pdf(b"%PDF-1.4 dummy bytes")

    assert result["page_count"] == 2
    assert "Novel self-healing agents" in result["abstract"]
    assert "Limited context window" in result["limitations"]
    assert "4x RTX 4090" in result["compute"]
    assert result["has_limitations_section"] is True
    assert result["has_compute_section"] is True


@patch("litreview.cli.run_agent_summary.requests.get")
@patch("litreview.discovery.pdf_parser.pypdf.PdfReader")
def test_resolve_and_parse_pdfs(mock_reader_cls, mock_get, tmp_path):
    import pandas as pd
    from litreview.cli.run_agent_summary import resolve_and_parse_pdfs

    mock_reader = MagicMock()
    mock_reader_cls.return_value = mock_reader
    mock_page = MagicMock()
    mock_page.extract_text.return_value = "Abstract\nTesting.\n\n4. Compute\n1x GPU."
    mock_reader.pages = [mock_page]

    # Mock requests download
    mock_resp = MagicMock()
    mock_resp.status_code = 200
    mock_resp.content = b"%PDF-1.4 mock content"
    mock_get.return_value = mock_resp

    df = pd.DataFrame([
        {
            "Title": "Paper OA",
            "DOI": "10.1000/test_oa",
            "oa_url": "https://example.com/paper.pdf",
        }
    ])

    parsed = resolve_and_parse_pdfs(df, pdf_dir=str(tmp_path))
    assert "10.1000/test_oa" in parsed
    assert parsed["10.1000/test_oa"]["has_compute_section"] is True
    assert "1x GPU" in parsed["10.1000/test_oa"]["compute"]

