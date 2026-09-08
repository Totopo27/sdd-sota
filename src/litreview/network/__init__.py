"""Citation network and knowledge graph module."""

from litreview.network.semantic_scholar import SemanticScholarClient
from litreview.network.citation_graph import CitationGraphBuilder
from litreview.network.visualization import plot_citation_network

__all__ = [
    "SemanticScholarClient",
    "CitationGraphBuilder",
    "plot_citation_network",
]
