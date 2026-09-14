import json
from pathlib import Path
from unittest.mock import MagicMock
import pytest

from litreview.reporting.persistence import persist_research_artifacts, resolve_artifact_store_mode


def test_resolve_artifact_store_mode(tmp_path, monkeypatch):
    # Mode explicitly set
    assert resolve_artifact_store_mode("openspec", tmp_path) == "openspec"
    assert resolve_artifact_store_mode("engram", tmp_path) == "engram"
    assert resolve_artifact_store_mode("hybrid", tmp_path) == "hybrid"
    assert resolve_artifact_store_mode("none", tmp_path) == "none"

    # Auto mode: neither openspec dir nor engram executable
    monkeypatch.setattr("shutil.which", lambda cmd: None)
    assert resolve_artifact_store_mode("auto", tmp_path) == "none"

    # Auto mode: openspec dir exists
    (tmp_path / "openspec").mkdir()
    assert resolve_artifact_store_mode("auto", tmp_path) == "openspec"

    # Auto mode: engram exists too -> hybrid
    monkeypatch.setattr("shutil.which", lambda cmd: "C:\\bin\\engram.exe" if cmd == "engram" else None)
    assert resolve_artifact_store_mode("auto", tmp_path) == "hybrid"


def test_persist_openspec_mode(tmp_path):
    change = "feature-transformer-repair"
    md_content = "# Executive Research Briefing\n\nContent here."
    summary_data = {"status": "success", "corpus": {"total_papers": 5}}

    receipt = persist_research_artifacts(
        change_name=change,
        markdown_content=md_content,
        summary_data=summary_data,
        mode="openspec",
        workspace_root=tmp_path,
    )

    expected_dir = tmp_path / "openspec" / "changes" / change
    expected_md = expected_dir / "research.md"
    expected_json = expected_dir / "research-summary.json"

    assert expected_md.exists()
    assert expected_md.read_text(encoding="utf-8") == md_content
    assert expected_json.exists()
    with open(expected_json, "r", encoding="utf-8") as f:
        data = json.load(f)
        assert data["corpus"]["total_papers"] == 5

    assert receipt["change"] == change
    assert receipt["mode"] == "openspec"
    assert receipt["openspec_markdown"] == str(expected_md)
    assert receipt["openspec_json"] == str(expected_json)
    assert receipt["engram_saved"] is False


def test_persist_engram_mode(tmp_path, monkeypatch):
    change = "audio-diffusion-dsp"
    md_content = "# Research on Diffusion"
    summary_data = {"status": "success"}

    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    mock_run.return_value.stdout = "Memory saved: #800"
    mock_run.return_value.stderr = ""
    monkeypatch.setattr("subprocess.run", mock_run)

    receipt = persist_research_artifacts(
        change_name=change,
        markdown_content=md_content,
        summary_data=summary_data,
        mode="engram",
        workspace_root=tmp_path,
    )

    assert receipt["engram_saved"] is True
    assert receipt["engram_topic"] == f"sdd/{change}/research"
    assert receipt["openspec_markdown"] is None

    mock_run.assert_called_once()
    args = mock_run.call_args[0][0]
    assert args[0] == "engram"
    assert args[1] == "save"
    assert f"Research Briefing: {change}" in args[2]
    assert args[3] == md_content
    assert "--topic" in args
    assert f"sdd/{change}/research" in args


def test_persist_hybrid_mode(tmp_path, monkeypatch):
    change = "hybrid-agentic-workflow"
    md_content = "# Hybrid Briefing"
    summary_data = {"status": "success"}

    mock_run = MagicMock()
    mock_run.return_value.returncode = 0
    monkeypatch.setattr("subprocess.run", mock_run)

    receipt = persist_research_artifacts(
        change_name=change,
        markdown_content=md_content,
        summary_data=summary_data,
        mode="hybrid",
        workspace_root=tmp_path,
    )

    expected_md = tmp_path / "openspec" / "changes" / change / "research.md"
    assert expected_md.exists()
    assert receipt["openspec_markdown"] == str(expected_md)
    assert receipt["engram_saved"] is True
    assert receipt["engram_topic"] == f"sdd/{change}/research"


def test_persist_none_mode(tmp_path):
    change = "ephemeral-trial"
    md_content = "# Ephemeral"
    summary_data = {}

    receipt = persist_research_artifacts(
        change_name=change,
        markdown_content=md_content,
        summary_data=summary_data,
        mode="none",
        workspace_root=tmp_path,
    )

    assert receipt["mode"] == "none"
    assert receipt["openspec_markdown"] is None
    assert receipt["engram_saved"] is False
    assert not (tmp_path / "openspec").exists()


def test_report_export_markdown_with_change(tmp_path):
    import pandas as pd
    from unittest.mock import MagicMock
    from litreview.pipeline import Report

    df = pd.DataFrame([
        {"Title": "Paper Alpha", "Year": 2024, "score": 0.95, "Abstract Note": "Abstract 1", "topic": 0},
    ])
    mock_config = MagicMock()
    report = Report(
        df=df,
        corpus_stats={"total_papers": 1, "year_range": [2024, 2024]},
        topic_coverage={"num_topics": 1, "topic_sizes": {0: 1}},
        bertopic_results={"topic_words": {"0": [["alpha", 0.5]]}},
        zeroshot_results={"label_counts": {"SE": 1}},
        gap_analysis={},
        cross_analysis={},
        config=mock_config,
    )

    out_file = tmp_path / "report.md"
    saved = report.export_markdown(
        path=str(out_file),
        change_name="pipeline-change",
        artifact_store="openspec",
        workspace_root=tmp_path,
    )

    assert Path(saved).exists()
    openspec_md = tmp_path / "openspec" / "changes" / "pipeline-change" / "research.md"
    assert openspec_md.exists()
    assert "SDD-SOTA Executive Research Briefing" in openspec_md.read_text(encoding="utf-8")
