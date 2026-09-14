"""Unit tests for section-targeted NeurIPS rubric evaluation."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest

from litreview.analyzers.rubric import NeurIPSRubricAnalyzer


@patch("litreview.analyzers.rubric.pipeline")
def test_rubric_transform_sections(mock_pipeline_fn):
    # Mock zero-shot pipeline
    mock_pipe = MagicMock()
    mock_pipeline_fn.return_value = mock_pipe

    # Return high scores for actual limitations/compute text, lower for generic abstracts
    def mock_classify(texts, candidate_labels, **kwargs):
        results = []
        for t in texts:
            text_lower = t.lower()
            scores = []
            for label in candidate_labels:
                if "hardware" in label or "compute" in label:
                    scores.append(0.95 if "nvidia" in text_lower or "gpu" in text_lower else 0.25)
                elif "limitations" in label:
                    scores.append(0.92 if "limited to" in text_lower or "failure mode" in text_lower else 0.20)
                elif "reproduction" in label or "repository" in label:
                    scores.append(0.88 if "github.com" in text_lower else 0.30)
                else:
                    scores.append(0.70)
            results.append({"labels": candidate_labels, "scores": scores})
        return results

    mock_pipe.side_effect = mock_classify

    analyzer = NeurIPSRubricAnalyzer()
    analyzer.fit(pd.Series(["dummy text"]))

    df = pd.DataFrame([
        {
            "Title": "Paper With Full PDF",
            "DOI": "10.1000/full_pdf",
            "Abstract Note": "Abstract mentioning agentic workflows.",
        },
        {
            "Title": "Paper With Abstract Only",
            "DOI": "10.2000/abstract_only",
            "Abstract Note": "Generic abstract without limitations or compute details.",
        },
    ])

    parsed_sections = {
        "10.1000/full_pdf": {
            "abstract": "Abstract mentioning agentic workflows.",
            "methodology": "We describe our multi-agent architecture.",
            "experiments": "Experiments show high pass rates on benchmarks.",
            "limitations": "The study is limited to Python and has potential failure modes.",
            "compute": "Trained on 8x NVIDIA H100 GPUs for 120 hours.",
            "reproducibility": "Code available at https://github.com/org/repo.",
            "has_limitations_section": True,
            "has_compute_section": True,
            "has_reproducibility_section": True,
        }
    }

    scored_df = analyzer.transform_sections(df, parsed_sections=parsed_sections)

    assert len(scored_df) == 2
    assert "rigor_source" in scored_df.columns
    assert scored_df.loc[0, "rigor_source"] == "full_text"
    assert scored_df.loc[1, "rigor_source"] == "abstract_fallback"

    # Full text paper scores higher on compute and limitations
    assert scored_df.loc[0, "rubric_compute_resources"] > 0.8
    assert scored_df.loc[0, "rubric_limitations"] > 0.8
    assert scored_df.loc[0, "rubric_reproducibility"] > 0.8

    # Abstract only paper scores lower on compute
    assert scored_df.loc[1, "rubric_compute_resources"] < 0.5
