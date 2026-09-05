"""LitReview - Corpus characterization through topic discovery and validation."""

__version__ = "0.2.0"

from litreview.config import PipelineConfig, load_config
from litreview.pipeline import ReviewPipeline, Report
from litreview.fetchers.zotero import ZoteroFetcher
from litreview.analyzers.bertopic_wrapper import BERTopicAnalyzer
from litreview.analyzers.zeroshot import ZeroShotAnalyzer
from litreview.statistics import (
    compute_corpus_stats,
    compute_topic_coverage,
    compute_gap_analysis,
    compute_cross_analysis,
)
from litreview.visualization import (
    plot_year_distribution,
    plot_zeroshot_label_counts,
    plot_zeroshot_confidence_by_label,
    plot_bertopic_sizes,
    plot_bertopic_topic_words,
    plot_bertopic_representative_docs,
    plot_topic_coverage,
    plot_gap_analysis,
    plot_confidence_distribution,
    plot_topic_label_heatmap,
    plot_colabel_matrix,
    plot_topic_confidence_scatter,
    plot_method_overlap,
    plot_topic_network,
    plot_topic_label_distribution,
    plot_topic_distribution_by_year,
)

__all__ = [
    "__version__",
    "ReviewPipeline",
    "Report",
    "ZoteroFetcher",
    "load_config",
    "PipelineConfig",
    "BERTopicAnalyzer",
    "ZeroShotAnalyzer",
    "compute_corpus_stats",
    "compute_topic_coverage",
    "compute_gap_analysis",
    "compute_cross_analysis",
    "plot_year_distribution",
    "plot_zeroshot_label_counts",
    "plot_zeroshot_confidence_by_label",
    "plot_bertopic_sizes",
    "plot_bertopic_topic_words",
    "plot_bertopic_representative_docs",
    "plot_topic_coverage",
    "plot_gap_analysis",
    "plot_confidence_distribution",
    "plot_topic_label_heatmap",
    "plot_colabel_matrix",
    "plot_topic_confidence_scatter",
    "plot_method_overlap",
    "plot_topic_network",
    "plot_topic_label_distribution",
    "plot_topic_distribution_by_year",
]
