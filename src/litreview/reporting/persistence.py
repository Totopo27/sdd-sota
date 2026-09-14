"""Persistence backend for SDD (Spec-Driven Development) research artifacts.

Coordinates automated persistence of executive markdown research briefings and machine
evidence receipts into OpenSpec directories and Engram persistent memory.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
from typing import Any

logger = logging.getLogger(__name__)


def resolve_artifact_store_mode(
    mode: str = "auto",
    workspace_root: str | Path = ".",
) -> str:
    """Resolve active artifact store mode (openspec, engram, hybrid, or none).

    Args:
        mode: Explicit mode ('openspec', 'engram', 'hybrid', 'none', 'auto').
        workspace_root: Project root directory to check for openspec/ folder.

    Returns:
        Resolved mode: 'openspec', 'engram', 'hybrid', or 'none'.
    """
    clean_mode = str(mode).strip().lower()
    if clean_mode in {"openspec", "engram", "hybrid", "none"}:
        return clean_mode

    root_path = Path(workspace_root)
    has_openspec_dir = (root_path / "openspec").is_dir()
    has_engram_cli = shutil.which("engram") is not None

    if has_openspec_dir and has_engram_cli:
        return "hybrid"
    elif has_openspec_dir:
        return "openspec"
    elif has_engram_cli:
        return "engram"
    else:
        return "none"


def persist_research_artifacts(
    change_name: str,
    markdown_content: str,
    summary_data: dict[str, Any],
    mode: str = "auto",
    workspace_root: str | Path = ".",
    project: str | None = None,
) -> dict[str, Any]:
    """Persist executive research briefing and summary receipt into active store.

    Args:
        change_name: SDD change slug (e.g. 'agentic-se-architectures').
        markdown_content: Generated executive markdown briefing text.
        summary_data: Complete litreview summary receipt dict.
        mode: Artifact store mode ('auto', 'openspec', 'engram', 'hybrid', 'none').
        workspace_root: Root path for workspace.
        project: Optional explicit project name for Engram memory.

    Returns:
        Receipt dict detailing persisted paths and topic keys.
    """
    resolved_mode = resolve_artifact_store_mode(mode, workspace_root=workspace_root)
    root = Path(workspace_root)

    receipt: dict[str, Any] = {
        "change": change_name,
        "mode": resolved_mode,
        "openspec_markdown": None,
        "openspec_json": None,
        "engram_saved": False,
        "engram_topic": None,
    }

    if resolved_mode == "none":
        return receipt

    # 1. OpenSpec Persistence
    if resolved_mode in {"openspec", "hybrid"}:
        change_dir = root / "openspec" / "changes" / change_name
        change_dir.mkdir(parents=True, exist_ok=True)

        md_path = change_dir / "research.md"
        md_path.write_text(markdown_content, encoding="utf-8")
        receipt["openspec_markdown"] = str(md_path)

        json_path = change_dir / "research-summary.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(summary_data, f, indent=2, default=str)
        receipt["openspec_json"] = str(json_path)

    # 2. Engram Persistence
    if resolved_mode in {"engram", "hybrid"}:
        topic_key = f"sdd/{change_name}/research"
        title = f"Research Briefing: {change_name}"
        cmd = [
            "engram",
            "save",
            title,
            markdown_content,
            "--type",
            "discovery",
            "--topic",
            topic_key,
        ]
        if project:
            cmd.extend(["--project", project])

        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=True,
            )
            receipt["engram_saved"] = True
            receipt["engram_topic"] = topic_key
            logger.info(f"Persisted research briefing to Engram ({topic_key}): {res.stdout.strip()}")
        except Exception as e:
            logger.warning(f"Failed to persist research briefing to Engram: {e}")
            receipt["engram_saved"] = False

    return receipt
