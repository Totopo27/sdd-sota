"""Academic paper discovery and Zotero auto-population module."""

from litreview.discovery.openalex import OpenAlexClient, reconstruct_abstract
from litreview.discovery.zotero_populator import ZoteroPopulator

__all__ = ["OpenAlexClient", "reconstruct_abstract", "ZoteroPopulator"]
