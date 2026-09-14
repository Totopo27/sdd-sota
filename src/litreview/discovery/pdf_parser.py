"""Academic paper PDF section parser and text cleaner.

Extracts text from academic PDFs using pypdf and segments papers into canonical
academic sections (Abstract, Introduction, Methodology, Experiments, Compute,
Limitations, Reproducibility, Conclusion, References) for targeted rigor evaluation.
"""

import io
import logging
from pathlib import Path
import re
from typing import Any, BinaryIO


import pypdf

logger = logging.getLogger(__name__)

# Canonical academic section headers with regex matching rules
SECTION_HEADER_PATTERNS = [
    (
        "limitations",
        re.compile(
            r"^(?:\d+[\.\)]?\s*)?(?:limitations?(?:\s+and\s+(?:ethical\s+considerations?|broader\s+impacts?|future\s+work))?|failure\s+modes?|threats\s+to\s+validity|ethical\s+considerations?|broader\s+impacts?)$",
            re.IGNORECASE,
        ),
    ),
    (
        "compute",
        re.compile(
            r"^(?:\d+[\.\)]?\s*)?(?:compute(?:\s+and\s+hardware)?(?:\s+resources)?|computational\s+resources|hardware(?:\s+resources|\s+setup|\s+details)?|training\s+details|computational\s+cost|gpu\s+hours?)$",
            re.IGNORECASE,
        ),
    ),
    (
        "reproducibility",
        re.compile(
            r"^(?:\d+[\.\)]?\s*)?(?:reproducibility(?:\s+statement)?|code\s+(?:and\s+data\s+)?availability|data\s+availability|artifacts?(?:\s+availability)?)$",
            re.IGNORECASE,
        ),
    ),
    (
        "experiments",
        re.compile(
            r"^(?:\d+[\.\)]?\s*)?(?:experiments?(?:\s+and\s+results)?|experimental\s+(?:results|setup|evaluation)|evaluation(?:\s+and\s+results)?|benchmarks?|results(?:\s+and\s+discussion)?)$",
            re.IGNORECASE,
        ),
    ),
    (
        "methodology",
        re.compile(
            r"^(?:\d+[\.\)]?\s*)?(?:methodology|methods?|proposed\s+(?:method|approach|framework|architecture|model)|system\s+architecture|architecture)$",
            re.IGNORECASE,
        ),
    ),
    (
        "introduction",
        re.compile(
            r"^(?:\d+[\.\)]?\s*)?(?:introduction|background)$",
            re.IGNORECASE,
        ),
    ),
    (
        "conclusion",
        re.compile(
            r"^(?:\d+[\.\)]?\s*)?(?:conclusions?(?:\s+and\s+future\s+work)?|concluding\s+remarks|summary|discussion)$",
            re.IGNORECASE,
        ),
    ),
    (
        "references",
        re.compile(
            r"^(?:\d+[\.\)]?\s*)?(?:references|bibliography)$",
            re.IGNORECASE,
        ),
    ),
    (
        "abstract",
        re.compile(
            r"^(?:\d+[\.\)]?\s*)?abstract$",
            re.IGNORECASE,
        ),
    ),
]


def clean_academic_text(text: str) -> str:
    """Clean academic text extracted from PDF.

    Removes hyphenation across line wraps, eliminates header/footer noise,
    and normalizes excessive whitespace while preserving paragraph structure.
    """
    if not text:
        return ""

    # 1. De-hyphenate words broken across line wraps: e.g. "trans-\nformer" -> "transformer"
    cleaned = re.sub(r"(\b[A-Za-z]+)-\s*\n\s*([A-Za-z]+\b)", r"\1\2", text)

    # 2. Remove standard page number artifacts: e.g. "Page 1 of 12", "Page 5"
    cleaned = re.sub(r"(?im)^\s*page\s+\d+(\s+of\s+\d+)?\s*$", "", cleaned)
    cleaned = re.sub(r"(?i)\bpage\s+\d+\s+of\s+\d+\b", "", cleaned)

    # 3. Clean trailing whitespace per line
    lines = [line.rstrip() for line in cleaned.splitlines()]
    cleaned = "\n".join(lines)

    # 4. Collapse 3+ consecutive newlines into 2
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)

    return cleaned.strip()


def segment_academic_sections(text: str) -> dict[str, Any]:
    """Segment cleaned academic paper text into canonical sections.

    Identifies standard academic headings and partitions text accordingly.
    Also provides boolean flags indicating section presence.
    """
    canonical_sections = [
        "abstract",
        "introduction",
        "methodology",
        "experiments",
        "compute",
        "limitations",
        "reproducibility",
        "conclusion",
        "references",
    ]
    sections: dict[str, Any] = {sec: "" for sec in canonical_sections}
    sections["other"] = ""

    current_section = "other"
    accumulators: dict[str, list[str]] = {sec: [] for sec in canonical_sections}
    accumulators["other"] = []

    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue

        clean_heading = line.rstrip(".: \t")
        matched_section = None
        for sec_name, pattern in SECTION_HEADER_PATTERNS:
            if pattern.match(clean_heading):
                matched_section = sec_name
                break

        if matched_section:
            current_section = matched_section
        else:
            accumulators[current_section].append(line)

    for sec in canonical_sections:
        sections[sec] = "\n".join(accumulators[sec]).strip()
    sections["other"] = "\n".join(accumulators["other"]).strip()

    sections["has_limitations_section"] = bool(sections["limitations"])
    sections["has_compute_section"] = bool(sections["compute"])
    sections["has_reproducibility_section"] = bool(sections["reproducibility"])

    return sections


class PDFSectionParser:
    """Extracts and segments academic papers from PDF inputs using pypdf."""

    def __init__(self):
        pass

    def parse_pdf(self, file_input: str | Path | bytes | BinaryIO) -> dict[str, Any]:
        """Extract text from PDF and segment into academic sections.

        Args:
            file_input: File path, raw bytes, or file-like binary stream.

        Returns:
            Dictionary with canonical section texts, boolean presence flags,
            page_count, and full_text.
        """
        if isinstance(file_input, (str, Path)):
            reader = pypdf.PdfReader(str(file_input))
        elif isinstance(file_input, bytes):
            reader = pypdf.PdfReader(io.BytesIO(file_input))
        else:
            reader = pypdf.PdfReader(file_input)

        pages_text = []
        for page in reader.pages:
            txt = page.extract_text()
            if txt:
                pages_text.append(txt)

        raw_combined = "\n\n".join(pages_text)
        cleaned_text = clean_academic_text(raw_combined)
        sections = segment_academic_sections(cleaned_text)
        sections["page_count"] = len(reader.pages)
        sections["full_text"] = cleaned_text
        return sections
