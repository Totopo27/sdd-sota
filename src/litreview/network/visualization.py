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

    # Distinguish node roles
    roles = graph_builder.compute_graph_roles()
    
    foundation_nodes = [n for n in g.nodes if roles.get(n) == "foundation"]
    frontier_nodes = [n for n in g.nodes if roles.get(n) == "frontier"]
    bridge_nodes = [n for n in g.nodes if roles.get(n) == "bridge"]
    other_nodes = [n for n in g.nodes if n not in foundation_nodes and n not in frontier_nodes and n not in bridge_nodes]

    # Compute node sizes based on in-degree (citations received)
    in_degrees = dict(g.in_degree())
    def get_sizes(node_list):
        return [max(90, (in_degrees.get(n, 0) + 1) * 140) for n in node_list]

    # Draw Foundation nodes (Gold/Amber)
    if foundation_nodes:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=foundation_nodes,
            node_color="#f59e0b",
            node_size=get_sizes(foundation_nodes),
            alpha=0.85,
            ax=ax,
            label="Foundation (Landmark / S2AG)",
        )

    # Draw Frontier nodes (Emerald Green)
    if frontier_nodes:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=frontier_nodes,
            node_color="#10b981",
            node_size=get_sizes(frontier_nodes),
            alpha=0.85,
            ax=ax,
            label="Frontier (Recent Velocity)",
        )

    # Draw Bridge nodes (Purple / Interdisciplinary)
    if bridge_nodes:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=bridge_nodes,
            node_color="#8b5cf6",
            node_size=get_sizes(bridge_nodes),
            alpha=0.85,
            ax=ax,
            label="Bridge (Betweenness Centrality)",
        )

    # Draw standard corpus / other nodes (Blue)
    if other_nodes:
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=other_nodes,
            node_color="#2563eb",
            node_size=get_sizes(other_nodes),
            alpha=0.80,
            ax=ax,
            label="Corpus Literature",
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
