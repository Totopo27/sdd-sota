"""Reporting and briefing generation module for SDD-SOTA."""

from litreview.reporting.executive_report import (
    generate_executive_markdown,
    generate_topic_mermaid,
    export_markdown_report,
)
from litreview.reporting.persistence import (
    persist_research_artifacts,
    resolve_artifact_store_mode,
)

__all__ = [
    "generate_executive_markdown",
    "generate_topic_mermaid",
    "export_markdown_report",
    "persist_research_artifacts",
    "resolve_artifact_store_mode",
]
