"""Visualization functions for literature review analysis."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import os

import pandas as pd
import numpy as np


# ---------------------------------------------------------------------------
# Shared color palette for topics
# ---------------------------------------------------------------------------

_TOPIC_PALETTE: list[str] = list(plt.get_cmap("tab20").colors)  # 20 colors


def _topic_color(topic_id: str | int, topic_names: dict | None = None) -> str:
    """Return a deterministic color for *topic_id* from a fixed palette."""
    try:
        idx = int(topic_id) % len(_TOPIC_PALETTE)
    except (ValueError, TypeError):
        # Non-numeric key — use a deterministic position
        idx = sum(ord(c) for c in str(topic_id)) % len(_TOPIC_PALETTE)
    return _TOPIC_PALETTE[idx]


def _topic_legend(ax, topics: list[str | int],
                  topic_names: dict | None = None) -> None:
    """Add a compact legend mapping topic IDs to their top labels.

    Parameters
    ----------
    ax : matplotlib Axes
    topics : list of topic IDs (str or int)
    topic_names : optional dict mapping topic ID → display label
    """
    if topic_names is None:
        topic_names = {}
    handles, labels = [], []
    for tid in topics:
        tid_str = str(tid)
        label = topic_names.get(tid_str, f"T{tid_str}")
        color = _topic_color(tid, topic_names)
        handles.append(mpatches.Patch(color=color, label=label))
    # Limit legend width so it doesn't dwarf the plot
    ax.legend(handles=handles, title="Topics", loc="center left",
              bbox_to_anchor=(1.02, 0.5), fontsize=7, title_fontsize=8,
              ncol=min(2, len(handles)), frameon=True)


def _normalize_topic_sizes(topic_sizes) -> dict:
    """Normalize topic_sizes to a dict, handling pandas DataFrame input.

    The pipeline may store topic_sizes as either a dict or a DataFrame
    depending on how BERTopic results are serialized.
    """
    if isinstance(topic_sizes, pd.DataFrame):
        if topic_sizes.empty:
            return {}
        topic_sizes = topic_sizes.to_dict(orient="list")
        topic_sizes = {k: v[0] if isinstance(v, list) else v for k, v in topic_sizes.items()}
    return topic_sizes if topic_sizes else {}


def plot_zeroshot_label_counts(zeroshot_results: dict, path: str,
                               top_n: int | None = None) -> None:
    """Plot horizontal bar chart of zero-shot label counts, ordered by count.

    Args:
        zeroshot_results: Dict with 'classifications' DataFrame and 'label_counts'.
        path: Output file path (.png).
        top_n: Show only top N labels. None shows all.
    """
    label_counts = zeroshot_results.get("label_counts", {})
    if not label_counts:
        return

    df = pd.Series(label_counts).sort_values(ascending=True)
    if top_n:
        df = df.tail(top_n)

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.6), max(4, len(df) * 0.4)))
    df.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Number of Papers")
    ax.set_title("Papers per Zero-Shot Label")
    for bar in ax.patches:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_zeroshot_confidence_by_label(zeroshot_results: dict, path: str) -> None:
    """Plot box+violin chart of confidence scores per label.

    Shows distribution of classification confidence for each label,
    revealing which labels have clear vs ambiguous boundaries.
    """
    classifications = zeroshot_results.get("classifications")
    if classifications is None or len(classifications) == 0:
        return

    fig, ax = plt.subplots(figsize=(max(8, len(classifications["label"].unique()) * 0.7), 6))
    label_order = classifications["label"].value_counts().index.tolist()
    data = [classifications[classifications["label"] == lbl]["score"].values
            for lbl in label_order]

    bp = ax.boxplot(data, tick_labels=label_order, patch_artist=True, showmeans=True)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
    ax.set_xlabel("Zero-Shot Label")
    ax.set_ylabel("Classification Confidence")
    ax.set_title("Confidence Distribution per Label")
    ax.set_xticklabels(label_order, rotation=45, ha="right")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bertopic_sizes(bertopic_results: dict, path: str) -> None:
    """Plot BERTopic topic sizes as horizontal bar chart, sorted by size.

    Outliers (topic -1) are excluded.
    """
    topic_sizes = _normalize_topic_sizes(bertopic_results.get("topic_sizes", {}))

    # Exclude outlier topic (-1)
    topic_sizes = {k: v for k, v in topic_sizes.items() if k != -1}
    if not topic_sizes:
        return

    df = pd.Series(topic_sizes).sort_values(ascending=True)

    fig, ax = plt.subplots(figsize=(max(8, len(df) * 0.6), max(4, len(df) * 0.4)))
    colors = [_topic_color(tid) for tid in df.index]
    ax.barh(df.index, df.values, color=colors)
    ax.set_xlabel("Number of Papers")
    ax.set_title("BERTopic Topic Sizes")
    for bar in ax.patches:
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height() / 2,
                str(int(bar.get_width())), va="center", fontsize=8)
    _topic_legend(ax, df.index.tolist())
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bertopic_topic_words(bertopic_results: dict, path: str,
                              top_words_per_topic: int = 10) -> None:
    """Plot top words per BERTopic topic as a horizontal bar grid.

    Each topic gets its own subplot with its top N words.
    """
    topic_words = bertopic_results.get("topic_words", {})
    if not topic_words:
        return

    n_topics = len(topic_words)
    cols = 3
    rows = (n_topics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3))
    axes = axes.flatten() if n_topics > 1 else [axes]

    for idx, (topic_id, words) in enumerate(sorted(topic_words.items())):
        ax = axes[idx]
        words = words[:top_words_per_topic]
        color = _topic_color(topic_id)
        ax.barh(range(len(words)), [1] * len(words), color=color)
        ax.set_yticks(range(len(words)))
        ax.set_yticklabels(words)
        ax.set_xlabel(f"Topic {topic_id}")
        ax.set_title(f"Topic {topic_id}: Top Words")
        ax.invert_yaxis()

    # Hide unused subplots
    for idx in range(n_topics, len(axes)):
        axes[idx].axis("off")

    fig.suptitle("BERTopic: Top Words per Topic", fontsize=14, y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_bertopic_representative_docs(bertopic_results: dict, path: str,
                                      max_docs_per_topic: int = 3) -> None:
    """Plot representative document titles per BERTopic topic.

    Shows the most representative paper title for each topic.
    """
    reps = bertopic_results.get("topic_representatives", {})
    if not reps:
        return

    rows = []
    for topic_id in sorted(reps.keys()):
        docs = reps[topic_id][:max_docs_per_topic]
        for doc in docs:
            rows.append({"Topic": f"T{topic_id}", "Title": doc})

    if not rows:
        return

    df = pd.DataFrame(rows)

    fig, ax = plt.subplots(figsize=(max(10, len(df) * 0.3), max(4, len(df) * 0.35)))
    y_pos = range(len(df))
    topic_ids = [int(r["Topic"][1:]) for r in df["Topic"]]
    patch_colors = [_topic_color(tid) for tid in topic_ids]

    ax.barh(y_pos, [1] * len(df), color=patch_colors)
    ax.set_yticks(y_pos)
    ax.set_yticklabels([f"{r['Topic']}: {r['Title'][:60]}..." for r in df["Title"]],
                       fontsize=8)
    ax.set_xlabel("Representative Documents")
    ax.set_title("BERTopic: Representative Documents per Topic")
    ax.invert_yaxis()
    # Unique topics for legend
    unique_topics = sorted(set(topic_ids))
    _topic_legend(ax, unique_topics)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_year_distribution(df: pd.DataFrame, path: str) -> None:
    """Plot publication year distribution histogram."""
    years = df["Publication Year"].dropna().astype(int)
    if len(years) == 0:
        return

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(years, bins=range(int(years.min()), int(years.max()) + 2), edgecolor="black")
    ax.set_xlabel("Year")
    ax.set_ylabel("Number of Papers")
    ax.set_title("Publication Year Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_topic_coverage(coverage: dict, path: str) -> None:
    """Plot papers per topic bar chart."""
    topic_sizes = _normalize_topic_sizes(coverage.get("topic_sizes", {}))
    if not topic_sizes:
        return

    topic_labels = [f"T{k}" for k in sorted(topic_sizes.keys())]
    values = [topic_sizes[k] for k in sorted(topic_sizes.keys())]
    topic_ids = sorted(topic_sizes.keys())

    fig, ax = plt.subplots(figsize=(max(10, len(topic_sizes) * 0.8), 6))
    colors = [_topic_color(tid) for tid in topic_ids]
    ax.bar(topic_labels, values, color=colors)
    ax.set_xlabel("Topic")
    ax.set_ylabel("Number of Papers")
    ax.set_title("Papers per Topic (BERTopic)")
    _topic_legend(ax, topic_ids)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_gap_analysis(gap_data: dict, path: str) -> None:
    """Plot identified gaps with severity coloring."""
    gaps = gap_data.get("gaps", [])
    if not gaps:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No significant gaps identified",
                ha="center", va="center", fontsize=14)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    severities = [g.get("severity", "low") for g in gaps]
    severity_order = {"low": 0, "medium": 1, "high": 2}
    severity_values = [severity_order.get(s, 0) for s in severities]
    colors = {"low": "green", "medium": "orange", "high": "red"}

    fig, ax = plt.subplots(figsize=(max(8, len(gaps) * 2), 4))
    ax.barh(range(len(gaps)), [1] * len(gaps),
            color=[colors.get(s, "gray") for s in severities])
    ax.set_yticks(range(len(gaps)))
    ax.set_yticklabels([g.get("seed", g.get("description", "unknown")) for g in gaps])
    ax.set_xlabel("Severity")
    ax.set_title("Identified Gaps")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_confidence_distribution(zeroshot_results: dict, path: str) -> None:
    """Plot zero-shot classification confidence distribution histogram."""
    classifications = zeroshot_results.get("classifications")
    if classifications is None or len(classifications) == 0:
        return

    scores = classifications["score"]
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.hist(scores, bins=50, edgecolor="black", color="steelblue")
    ax.set_xlabel("Classification Confidence")
    ax.set_ylabel("Number of Papers")
    ax.set_title("Zero-Shot Classification Confidence Distribution")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_topic_label_heatmap(bertopic_results: dict, zeroshot_results: dict,
                              path: str) -> None:
    """Heatmap of BERTopic topics × zero-shot labels.

    Shows which domain labels are most common in each unsupervised topic,
    revealing alignment (or misalignment) between the two methods.
    """
    topic_assignments = bertopic_results.get("topic_assignments")
    classifications = zeroshot_results.get("classifications")
    if topic_assignments is None or classifications is None:
        return

    # Build topic × label count matrix
    topic_ids = sorted(set(t for t in topic_assignments if t != -1))
    if not topic_ids:
        return

    # Combine topic assignments with classifications so we can group by topic
    combined = pd.DataFrame({
        "topic": topic_assignments,
        "label": classifications["label"],
    })

    label_counts = combined.groupby("topic")[
        "label"].value_counts().unstack(fill_value=0)
    # Filter to topics that exist
    label_counts = label_counts.loc[label_counts.index.isin(topic_ids)]
    if label_counts.empty:
        return

    # Sort labels by total count descending
    label_totals = label_counts.sum()
    label_counts = label_counts[label_totals.sort_values(ascending=False).index]

    fig, ax = plt.subplots(figsize=(max(8, label_counts.shape[1] * 0.6),
                                     max(5, label_counts.shape[0] * 0.5)))
    im = ax.imshow(label_counts.values, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(label_counts.columns)))
    ax.set_xticklabels(label_counts.columns, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(label_counts.index)))
    ax.set_yticklabels([f"T{k}" for k in label_counts.index])
    ax.set_xlabel("Zero-Shot Label")
    ax.set_ylabel("BERTopic Topic")
    ax.set_title("Topic × Label Alignment")
    fig.colorbar(im, ax=ax, label="Paper count")
    # Add topic legend on the side
    topic_ids = [int(k) for k in label_counts.index]
    _topic_legend(ax, topic_ids)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_colabel_matrix(zeroshot_results: dict, path: str) -> None:
    """Co-occurrence heatmap of zero-shot labels.

    Shows which domain labels tend to appear together in the same papers,
    revealing conceptual relationships in the field.
    """
    classifications = zeroshot_results.get("classifications")
    if classifications is None or len(classifications) == 0:
        return

    # Build paper × label matrix.
    # The classifications DataFrame has index=paper_id, columns=label/score/classified.
    # Each paper gets exactly one label, so we pivot the index against the label column.
    labels = classifications["label"].unique()
    if len(labels) > 25:
        top_labels = classifications["label"].value_counts().head(25).index
        classifications = classifications[classifications["label"].isin(top_labels)]
        labels = top_labels

    paper_label = classifications.pivot_table(
        index=classifications.index, columns="label", aggfunc="size", fill_value=0)
    # Only keep papers that have labels
    paper_label = paper_label.loc[paper_label.sum(axis=1) > 0]

    # Compute co-occurrence matrix
    cooccurrence = paper_label.values.T @ paper_label.values
    np.fill_diagonal(cooccurrence, 0)  # zero out diagonal for cleaner view

    fig, ax = plt.subplots(figsize=(max(8, len(labels) * 0.5),
                                     max(6, len(labels) * 0.5)))
    im = ax.imshow(cooccurrence, cmap="Reds", aspect="auto")
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("Label")
    ax.set_ylabel("Label")
    ax.set_title("Label Co-occurrence Matrix")
    fig.colorbar(im, ax=ax, label="Co-occurrence count")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_topic_confidence_scatter(bertopic_results: dict, zeroshot_results: dict,
                                   path: str) -> None:
    """Scatter plot: topic size vs mean classification confidence.

    Reveals whether larger topics are more or less confidently classified.
    """
    topic_sizes = _normalize_topic_sizes(bertopic_results.get("topic_sizes", {}))
    topic_assignments = bertopic_results.get("topic_assignments")
    classifications = zeroshot_results.get("classifications")
    if classifications is None or topic_assignments is None:
        return

    topic_sizes = {k: v for k, v in topic_sizes.items() if k != -1}
    if not topic_sizes:
        return

    # Merge topic assignments with classifications so we can group by topic
    combined = pd.DataFrame({
        "topic": topic_assignments,
        "score": classifications["score"],
    })

    # Mean confidence per topic
    topic_conf = combined.groupby("topic")["score"].mean()

    topics = sorted(topic_sizes.keys())
    sizes = [topic_sizes[t] for t in topics]
    confs = [topic_conf.get(t, 0) for t in topics]

    fig, ax = plt.subplots(figsize=(10, 6))
    colors = [_topic_color(t) for t in topics]
    ax.scatter(sizes, confs, s=100, alpha=0.8, color=colors, edgecolors="black")
    for i, t in enumerate(topics):
        ax.annotate(f"T{t}", (sizes[i], confs[i]), fontsize=7,
                    xytext=(5, 5), textcoords="offset points")
    ax.set_xlabel("Topic Size (number of papers)")
    ax.set_ylabel("Mean Classification Confidence")
    ax.set_title("Topic Size vs Classification Confidence")
    _topic_legend(ax, topics)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_method_overlap(cross_analysis: dict, path: str) -> None:
    """Bar chart showing classification overlap between BERTopic and zero-shot.

    Four bars: BERTopic only, zero-shot only, both, neither.
    """
    ca = cross_analysis
    if not ca:
        return

    labels = ["BERTopic only", "Zero-shot only", "Both", "Neither"]
    values = [
        ca.get("bertopic_classified", 0) - ca.get("both_classified", 0),
        ca.get("zeroshot_classified", 0) - ca.get("both_classified", 0),
        ca.get("both_classified", 0),
        ca.get("neither_classified", 0),
    ]
    colors = ["#e74c3c", "#3498db", "#2ecc71", "#95a5a6"]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.bar(labels, values, color=colors, edgecolor="white", linewidth=1.5)
    for i, (label, val) in enumerate(zip(labels, values)):
        if val > 0:
            ax.text(i, val + 0.5, str(int(val)), ha="center", fontsize=12, fontweight="bold")
    ax.set_ylabel("Number of Papers")
    ax.set_title("BERTopic vs Zero-Shot Classification Overlap")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_topic_network(bertopic_results: dict, zeroshot_results: dict,
                        path: str, min_cooccurrence: int = 2) -> None:
    """Network-style visualization of topic connections via shared labels.

    Topics are nodes; edges connect topics that share zero-shot labels.
    Edge thickness reflects co-occurrence strength.
    """
    topic_assignments = bertopic_results.get("topic_assignments")
    classifications = zeroshot_results.get("classifications")
    topic_sizes = _normalize_topic_sizes(bertopic_results.get("topic_sizes", {}))
    if topic_assignments is None or classifications is None:
        return

    topic_ids = sorted(set(t for t in topic_assignments if t != -1))
    if len(topic_ids) < 2:
        return

    # Build topic × label co-occurrence
    combined = pd.DataFrame({
        "topic": topic_assignments,
        "label": classifications["label"],
    })
    topic_labels = combined.groupby("topic")["label"].apply(set)
    topic_labels = topic_labels[topic_labels.index.isin(topic_ids)]

    # Compute pairwise label overlap between topics
    edges = []
    for i, t1 in enumerate(topic_ids):
        for t2 in topic_ids[i + 1:]:
            labels1 = topic_labels.get(t1, set())
            labels2 = topic_labels.get(t2, set())
            overlap = len(labels1 & labels2)
            if overlap >= min_cooccurrence:
                edges.append((t1, t2, overlap))

    if not edges:
        # Fallback: simple bar chart of topic sizes if no edges
        topic_sizes = {k: v for k, v in topic_sizes.items() if k != -1}
        if not topic_sizes:
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        keys = list(topic_sizes.keys())
        ax.bar([f"T{k}" for k in keys], [topic_sizes[k] for k in keys],
               color=[_topic_color(k) for k in keys])
        ax.set_xlabel("Topic")
        ax.set_ylabel("Number of Papers")
        ax.set_title("Topic Network (no connections found)")
        _topic_legend(ax, keys)
        fig.tight_layout()
        fig.savefig(path, dpi=150)
        plt.close(fig)
        return

    # Draw network using matplotlib (no graphviz dependency)
    import math
    n_topics = len(topic_ids)
    # Compute node positions using a simple force-directed layout approximation
    # Use circular layout as baseline
    angle_step = 2 * math.pi / n_topics
    positions = {
        t: (
            math.cos(angle_step * i),
            math.sin(angle_step * i),
        )
        for i, t in enumerate(topic_ids)
    }

    fig, ax = plt.subplots(figsize=(max(8, n_topics * 0.8), max(6, n_topics * 0.8)))
    # Draw edges
    for t1, t2, weight in edges:
        ax.plot([positions[t1][0], positions[t2][0]],
                [positions[t1][1], positions[t2][1]],
                "gray", alpha=min(weight * 0.3, 1.0), linewidth=weight * 0.5)
    # Draw nodes — color-coded by topic
    node_x = [positions[t][0] for t in topic_ids]
    node_y = [positions[t][1] for t in topic_ids]
    node_sizes = [topic_sizes.get(t, 1) * 20 for t in topic_ids]
    node_colors = [_topic_color(t) for t in topic_ids]
    ax.scatter(node_x, node_y, s=node_sizes, color=node_colors, alpha=0.7,
               edgecolors="black", linewidth=0.5)
    for t in topic_ids:
        ax.annotate(f"T{t}", (positions[t][0], positions[t][1]),
                    ha="center", va="center", fontsize=8, fontweight="bold",
                    color="white")
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_aspect("equal")
    ax.axis("off")

    # Build legend: topic ID + top label
    # Compute most common label per topic from the combined data
    label_counts_per_topic = combined.groupby("topic")["label"].value_counts()
    top_labels_map = {}
    for topic, group in label_counts_per_topic.groupby("topic"):
        top_labels_map[topic] = group.index[0]

    legend_handles = []
    legend_labels = []
    for t in topic_ids:
        size = topic_sizes.get(t, 1) * 20
        top_label = top_labels_map.get(t, "unknown")
        color = _topic_color(t)
        legend_handles.append(plt.scatter([], [], s=size, c=color,
                                          edgecolors="black", linewidth=0.5, alpha=0.7))
        legend_labels.append(f"T{t}: {top_label}")
    if legend_handles:
        ax.legend(legend_handles, legend_labels, title="Topics", loc="center left",
                  bbox_to_anchor=(1.02, 0.5), fontsize=8, title_fontsize=9,
                  frameon=True, fancybox=False)

    ax.set_title("Topic Network (edges = shared labels)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_topic_label_distribution(bertopic_results: dict, zeroshot_results: dict,
                                   path: str) -> None:
    """Stacked bar chart: dominant zero-shot labels per BERTopic topic.

    Shows the label composition of each topic, revealing what domain
    concepts each unsupervised cluster represents.
    """
    topic_assignments = bertopic_results.get("topic_assignments")
    classifications = zeroshot_results.get("classifications")
    topic_sizes = _normalize_topic_sizes(bertopic_results.get("topic_sizes", {}))
    if topic_assignments is None or classifications is None:
        return

    topic_ids = sorted(set(t for t in topic_assignments if t != -1))
    if not topic_ids:
        return

    # Count labels per topic
    combined = pd.DataFrame({
        "topic": topic_assignments,
        "label": classifications["label"],
    })
    label_per_topic = combined.groupby("topic")["label"].value_counts()
    # Keep only top labels per topic (by count)
    top_labels_per_topic = label_per_topic.groupby("topic").head(5)

    # Pivot: topic × label
    pivot = top_labels_per_topic.unstack(fill_value=0)
    if pivot.empty:
        return

    # Sort topics by size
    topic_sizes = {k: v for k, v in topic_sizes.items() if k != -1}
    pivot = pivot.reindex(sorted(pivot.index, key=lambda t: topic_sizes.get(t, 0)))

    ax = pivot.plot(kind="barh", stacked=True, figsize=(max(10, len(pivot) * 1.2),
                                                         max(5, len(pivot) * 0.6)),
                    colormap="tab20")
    ax.set_xlabel("Number of Papers")
    ax.set_ylabel("BERTopic Topic")
    ax.set_title("Label Distribution per Topic")
    handles, labels = ax.get_legend_handles_labels()
    if handles:
        ax.legend(handles, labels, title="Label", bbox_to_anchor=(1.02, 1),
                  fontsize=7, loc="upper left")
    fig = ax.get_figure()
    # Add topic legend on the right
    topic_ids = [int(k) for k in pivot.index]
    _topic_legend(ax, topic_ids)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_topic_distribution_by_year(
    bertopic_results: dict, df: pd.DataFrame, path: str
) -> None:
    """Stacked area chart: how topic prevalence shifts over time.

    Shows how the soft topic distribution evolves across years,
    revealing temporal trends in the literature.
    """
    dist_matrix = bertopic_results.get("distribution_matrix")
    if dist_matrix is None or df.empty:
        return

    years = df["Publication Year"].dropna().astype(int)
    if len(years) == 0:
        return

    # Align dist_matrix with year data
    valid_mask = df["Publication Year"].notna()
    valid_years = years[valid_mask].values
    valid_dist = dist_matrix[valid_mask.values]

    year_bins = sorted(set(valid_years))
    if len(year_bins) < 2:
        return

    # Compute mean distribution per year
    year_dist = {}
    for y in year_bins:
        mask = valid_years == y
        year_dist[int(y)] = valid_dist[mask].mean(axis=0)

    pivot = pd.DataFrame(year_dist).T  # rows=years, cols=topics
    pivot = pivot.div(pivot.sum(axis=1), axis=0)  # normalize per year

    if pivot.empty:
        return

    topic_ids = [int(c) for c in pivot.columns]
    ax = pivot.plot(kind="area", stacked=True, figsize=(max(12, len(pivot) * 0.3), 6),
                    colormap="tab20")
    ax.set_xlabel("Year")
    ax.set_ylabel("Mean Topic Probability")
    ax.set_title("Topic Distribution by Year")
    _topic_legend(ax, topic_ids)
    fig = ax.get_figure()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
