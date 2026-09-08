"""Citation Graph Builder and Network Science metrics (PageRank, Bibliographic Coupling)."""

import logging
from collections import defaultdict
import networkx as nx
import pandas as pd

logger = logging.getLogger(__name__)


class CitationGraphBuilder:
    """Constructs directed citation graphs and extracts scientometric indicators."""

    def __init__(self):
        self.graph = nx.DiGraph()

    def add_paper_record(
        self,
        title: str,
        year: int | None = None,
        doi: str | None = None,
        in_corpus: bool = True,
        citation_count: int = 0,
        influential_count: int = 0,
    ) -> None:
        clean_title = title.strip()
        if not clean_title:
            return

        if not self.graph.has_node(clean_title):
            self.graph.add_node(
                clean_title,
                title=clean_title,
                year=year,
                doi=doi,
                in_corpus=in_corpus,
                citation_count=citation_count,
                influential_count=influential_count,
            )
        else:
            # Update attributes if already present as a cited reference
            node_data = self.graph.nodes[clean_title]
            if in_corpus:
                node_data["in_corpus"] = True
            if year and not node_data.get("year"):
                node_data["year"] = year
            if doi and not node_data.get("doi"):
                node_data["doi"] = doi

    def add_citation(self, source_title: str, target_title: str) -> None:
        """Record that source_title cites target_title (directed edge: source -> target)."""
        src = source_title.strip()
        dst = target_title.strip()
        if src and dst and src != dst:
            if not self.graph.has_node(src):
                self.add_paper_record(src, in_corpus=False)
            if not self.graph.has_node(dst):
                self.add_paper_record(dst, in_corpus=False)
            self.graph.add_edge(src, dst)

    def build_from_dataframe(
        self,
        df: pd.DataFrame,
        client = None,
        max_papers: int | None = 50,
    ) -> "CitationGraphBuilder":
        """Populate citation network from a DataFrame of papers using Semantic Scholar."""
        if client is None:
            from litreview.network.semantic_scholar import SemanticScholarClient
            client = SemanticScholarClient()

        papers_to_process = df.head(max_papers) if max_papers else df

        # Pass 1: Add all corpus papers as nodes
        for _, row in papers_to_process.iterrows():
            title = str(row.get("Title", "")).strip()
            if not title:
                continue
            year = row.get("Publication Year")
            doi = row.get("DOI")
            self.add_paper_record(title, year=year, doi=doi, in_corpus=True)

        # Pass 2: Query Semantic Scholar for references and internal/external citations
        for _, row in papers_to_process.iterrows():
            title = str(row.get("Title", "")).strip()
            doi = row.get("DOI")
            if not title:
                continue

            metadata = client.get_paper_for_record(title=title, doi=doi)
            if not metadata:
                continue

            # Update citation counts on the paper node
            if self.graph.has_node(title):
                self.graph.nodes[title]["citation_count"] = metadata.get("citationCount", 0)
                self.graph.nodes[title]["influential_count"] = metadata.get("influentialCitationCount", 0)

            # Add references (papers that this paper cites)
            refs = metadata.get("references", []) or []
            for ref in refs:
                ref_title = ref.get("title")
                if ref_title and str(ref_title).strip():
                    clean_ref_title = str(ref_title).strip()
                    self.add_paper_record(
                        clean_ref_title,
                        year=ref.get("year"),
                        in_corpus=False,
                    )
                    self.add_citation(title, clean_ref_title)

        return self

    def compute_pagerank(self, alpha: float = 0.85) -> dict[str, float]:
        """Compute PageRank centrality. Higher PageRank indicates foundational authority."""
        if len(self.graph) == 0:
            return {}
        try:
            return nx.pagerank(self.graph, alpha=alpha)
        except Exception as e:
            logger.warning(f"PageRank computation failed: {e}")
            return {node: 0.0 for node in self.graph.nodes}

    def get_foundational_papers(self, top_k: int = 5) -> list[dict]:
        """Identify top landmark/foundational papers via PageRank and In-Degree."""
        if len(self.graph) == 0:
            return []

        pagerank = self.compute_pagerank()
        in_degrees = dict(self.graph.in_degree())

        # Sort by PageRank primarily, in-degree secondarily
        sorted_nodes = sorted(
            self.graph.nodes,
            key=lambda n: (pagerank.get(n, 0.0), in_degrees.get(n, 0)),
            reverse=True,
        )

        results = []
        for node in sorted_nodes[:top_k]:
            data = self.graph.nodes[node]
            results.append({
                "title": data.get("title", node),
                "year": data.get("year"),
                "in_corpus": data.get("in_corpus", False),
                "pagerank": round(pagerank.get(node, 0.0), 5),
                "internal_citations_received": in_degrees.get(node, 0),
                "total_citations": data.get("citation_count", 0),
                "influential_citations": data.get("influential_count", 0),
            })
        return results

    def get_derivative_works(self, top_k: int = 5) -> list[dict]:
        """Identify derivative or survey papers that cite the most literature (highest Out-Degree)."""
        if len(self.graph) == 0:
            return []

        out_degrees = dict(self.graph.out_degree())
        # Filter for corpus papers that synthesize references
        corpus_nodes = [n for n in self.graph.nodes if self.graph.nodes[n].get("in_corpus", False)]
        target_nodes = corpus_nodes if corpus_nodes else list(self.graph.nodes)

        sorted_nodes = sorted(
            target_nodes,
            key=lambda n: out_degrees.get(n, 0),
            reverse=True,
        )

        results = []
        for node in sorted_nodes[:top_k]:
            if out_degrees.get(node, 0) == 0:
                continue
            data = self.graph.nodes[node]
            results.append({
                "title": data.get("title", node),
                "year": data.get("year"),
                "in_corpus": data.get("in_corpus", False),
                "references_cited_count": out_degrees.get(node, 0),
                "total_citations": data.get("citation_count", 0),
            })
        return results

    def compute_bibliographic_coupling(self, top_k: int = 5) -> list[dict]:
        """Find pairs of papers in the corpus that share the most common references."""
        corpus_nodes = [n for n in self.graph.nodes if self.graph.nodes[n].get("in_corpus", False)]
        if len(corpus_nodes) < 2:
            return []

        # Out-edges represent citations to references
        node_refs = {n: set(self.graph.successors(n)) for n in corpus_nodes}

        coupling_pairs = []
        for i in range(len(corpus_nodes)):
            for j in range(i + 1, len(corpus_nodes)):
                p1 = corpus_nodes[i]
                p2 = corpus_nodes[j]
                common = node_refs[p1].intersection(node_refs[p2])
                if common:
                    coupling_pairs.append({
                        "paper_1": p1,
                        "paper_2": p2,
                        "shared_references_count": len(common),
                        "shared_references": list(common)[:3],
                    })

        coupling_pairs.sort(key=lambda x: x["shared_references_count"], reverse=True)
        return coupling_pairs[:top_k]

    def summary(self) -> dict:
        """Generate structured summary for SDD Agent Evidence Receipt."""
        corpus_count = sum(1 for n in self.graph.nodes if self.graph.nodes[n].get("in_corpus", False))
        return {
            "total_nodes": self.graph.number_of_nodes(),
            "total_edges": self.graph.number_of_edges(),
            "corpus_papers_modeled": corpus_count,
            "foundational_papers": self.get_foundational_papers(top_k=5),
            "derivative_works": self.get_derivative_works(top_k=5),
            "bibliographic_coupling": self.compute_bibliographic_coupling(top_k=5),
        }
