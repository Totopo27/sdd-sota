"""Visualization of citation networks and knowledge graphs."""

import matplotlib.pyplot as plt
import networkx as nx
from litreview.network.citation_graph import CitationGraphBuilder


def plot_citation_network(graph_builder: CitationGraphBuilder, path: str) -> None:
    """Plot citation graph with visual hierarchy for foundational vs corpus papers."""
    g = graph_builder.graph
    if len(g) == 0:
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.text(0.5, 0.5, "Empty citation network", ha="center", va="center")
        ax.axis("off")
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    fig, ax = plt.subplots(figsize=(12, 9))

    # Layout using spring_layout with deterministic seed
    pos = nx.spring_layout(g, k=0.35, iterations=50, seed=42)

    # Distinguish corpus vs external reference nodes
    corpus_nodes = [n for n in g.nodes if g.nodes[n].get("in_corpus", False)]
    external_nodes = [n for n in g.nodes if not g.nodes[n].get("in_corpus", False)]

    # Compute node sizes based on in-degree (citations received)
    in_degrees = dict(g.in_degree())
    corpus_sizes = [max(120, (in_degrees.get(n, 0) + 1) * 160) for n in corpus_nodes]
    external_sizes = [max(80, (in_degrees.get(n, 0) + 1) * 120) for n in external_nodes]

    # Draw external references
    if external_nodes:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=external_nodes,
            node_color="#f59e0b",  # Amber/Gold for foundational external works
            node_size=external_sizes,
            alpha=0.75,
            ax=ax,
            label="Cited Foundation / External",
        )

    # Draw corpus papers
    if corpus_nodes:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=corpus_nodes,
            node_color="#2563eb",  # Blue for corpus papers
            node_size=corpus_sizes,
            alpha=0.9,
            ax=ax,
            label="In-Corpus Papers",
        )

    # Draw edges with arrows
    nx.draw_networkx_edges(
        g,
        pos,
        edge_color="#94a3b8",
        alpha=0.6,
        arrows=True,
        arrowsize=12,
        ax=ax,
    )

    # Draw labels for nodes with highest in-degree or small networks
    if len(g) <= 15:
        labels = {n: (n[:25] + "..." if len(n) > 25 else n) for n in g.nodes}
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, ax=ax)
    else:
        # Label top 7 most cited nodes only to avoid clutter
        top_nodes = sorted(g.nodes, key=lambda n: in_degrees.get(n, 0), reverse=True)[:7]
        labels = {n: (n[:25] + "..." if len(n) > 25 else n) for n in top_nodes}
        nx.draw_networkx_labels(g, pos, labels=labels, font_size=8, ax=ax)

    ax.set_title("Citation Network & Connected Literature (PageRank / S2AG)", fontsize=13, fontweight="bold")
    ax.legend(loc="upper left")
    ax.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
