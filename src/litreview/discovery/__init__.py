"""Academic paper discovery and Zotero auto-population module."""

from litreview.discovery.openalex import OpenAlexClient, reconstruct_abstract
from litreview.discovery.zotero_populator import ZoteroPopulator
from litreview.discovery.oa_resolver import OAResolver, normalize_doi
from litreview.discovery.dedup import (
    PaperDeduplicator,
    normalize_title,
    title_token_similarity,
)
from litreview.discovery.pdf_parser import (
    PDFSectionParser,
    clean_academic_text,
    segment_academic_sections,
)

__all__ = [
    "OpenAlexClient",
    "reconstruct_abstract",
    "ZoteroPopulator",
    "OAResolver",
    "normalize_doi",
    "PaperDeduplicator",
    "normalize_title",
    "title_token_similarity",
    "PDFSectionParser",
    "clean_academic_text",
    "segment_academic_sections",
]

