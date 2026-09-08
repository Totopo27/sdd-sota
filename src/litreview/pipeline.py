"""ReviewPipeline and Report classes.

Orchestrates the full literature review workflow:
fetch -> clean -> BERTopic discovery -> zero-shot validation -> statistics -> report
"""

import os
import re
import json

import pandas as pd

from litreview.config import PipelineConfig
from litreview.fetchers.base import Fetcher
from litreview.fetchers.zotero import ZoteroFetcher
from litreview.analyzers.bertopic import (
    BERTopicFitter,
    TopicDistributionAnalyzer,
    TopicWordExtractor,
    TopicRepresentativeDocs,
)
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


class ReviewPipeline:
    """Orchestrates the full literature review workflow.

    Usage:
        config = load_config("config.yaml")
        pipeline = ReviewPipeline(config)
        report = pipeline.run()
        report.export_csv("data/processed/classified.csv")
        report.generate_plots("results/plots/")
        print(report.summary())
    """

    def __init__(
        self,
        config: PipelineConfig,
        skip_bertopic: bool = False,
        fetcher: Fetcher | None = None,
    ):
        self.config = config
        self.skip_bertopic = skip_bertopic
        self.fetcher = fetcher or ZoteroFetcher(
            config.zotero.library_id,
            config.zotero.api_key,
            config.zotero.library_type,
        )
        self.zeroshot_analyzer = ZeroShotAnalyzer(config.zeroshot)

    def run(self) -> "Report":
        """Execute the full pipeline and return a Report.

        Returns:
            Report with DataFrame, statistics, and analysis results.
        """
        # 1. Fetch
        df = self.fetcher.fetch(self.config.zotero.collection_names)

        # 2. Clean
        df = self._clean(df)

        abstracts = df["Abstract Note"].dropna()

        # 3. BERTopic — fit model, then run configured analyses
        bertopic_results = {}
        topic_results = pd.DataFrame(index=df.index)
        bertopic_model = None

        if (not self.skip_bertopic
                and self.config.bertopic.compute
                and len(abstracts) > 0):
            # BERTopic needs a minimum number of documents to produce
            # meaningful clusters (UMAP needs n_neighbors >= 2, HDBSCAN
            # needs at least a few points per cluster).
            if len(abstracts) < 10:
                total_papers = len(df)
                non_empty = len(abstracts)
                raise ValueError(
                    f"Only {non_empty} of {total_papers} papers have abstracts. "
                    "BERTopic requires at least 10 documents with non-empty abstracts. "
                    "Options:\n"
                    "  1. Import abstracts into Zotero (item details → Notes → Abstract)\n"
                    "  2. Use --skip-bertopic to run zero-shot classification only\n"
                    "  3. Set bertopic.compute: false in config to skip entirely"
                )
            # Phase 1: Fit the model (always runs when compute=True)
            fitter = BERTopicFitter(self.config.bertopic)
            fitter.fit(abstracts)
            bertopic_model = fitter.topic_model
            bertopic_results = {k: v for k, v in fitter.results.items()
                                if k != "topic_model"}
            topic_results = pd.DataFrame(
                {"topic": fitter.topic_assignments}, index=df.index
            )

            # Phase 2: Run configured analyses, each receiving the fitted model
            analysis_map = {
                "distribution": TopicDistributionAnalyzer,
                "words": TopicWordExtractor,
                "representatives": TopicRepresentativeDocs,
            }
            for analysis_name in self.config.bertopic.analyses:
                analyzer_cls = analysis_map.get(analysis_name)
                if analyzer_cls is None:
                    continue
                if analysis_name == "distribution":
                    analyzer = analyzer_cls(
                        bertopic_model, self.config.bertopic,
                        top_n_topics=self.config.bertopic.top_n_topics,
                        window=self.config.bertopic.distribution_window,
                        stride=self.config.bertopic.distribution_stride,
                    )
                else:
                    analyzer = analyzer_cls(bertopic_model)
                analyzer.fit(abstracts)
                analysis_df = analyzer.transform(abstracts)
                topic_results = pd.concat([topic_results, analysis_df], axis=1)
                bertopic_results.update(analyzer.results)

        df = pd.concat([df, topic_results], axis=1)

        # 4. Validate topics (Zero-shot)
        self.zeroshot_analyzer.fit(abstracts)
        validation_results = self.zeroshot_analyzer.transform(abstracts)
        df = pd.concat([df, validation_results], axis=1)

        # 5. Compute statistics
        corpus_stats = compute_corpus_stats(df)
        bertopic_enabled = (not self.skip_bertopic
                            and self.config.bertopic.compute)
        if not bertopic_enabled:
            topic_coverage = {"coverage": {}, "num_topics": 0}
            gap_analysis = {"gaps": [], "num_gaps": 0}
            cross_analysis = {"agreement": 0.0}
        else:
            topic_coverage = compute_topic_coverage(
                topic_results, validation_results, self.config.bertopic.seed_topics
            )
            gap_analysis = compute_gap_analysis(
                topic_results, validation_results, self.config.bertopic.seed_topics
            )
            cross_analysis = compute_cross_analysis(topic_results, validation_results)

        # 6. Return report
        return Report(
            df=df,
            corpus_stats=corpus_stats,
            topic_coverage=topic_coverage,
            gap_analysis=gap_analysis,
            cross_analysis=cross_analysis,
            bertopic_results=bertopic_results,
            zeroshot_results=self.zeroshot_analyzer.results,
            config=self.config,
        )

    @staticmethod
    def _clean(df: pd.DataFrame) -> pd.DataFrame:
        """Normalize text and remove duplicates by title."""
        df["Title"] = df["Title"].apply(ReviewPipeline._normalize_text)
        df["Abstract Note"] = df["Abstract Note"].apply(ReviewPipeline._normalize_text)

        # Dedup by normalized title only
        df = df.drop_duplicates(subset=["Title"], keep="first")

        df.reset_index(drop=True, inplace=True)
        return df

    @staticmethod
    def _normalize_text(text):
        """Normalize text: remove non-word chars, lowercase, strip."""
        if pd.isna(text):
            return ""
        return re.sub(r"\W+", " ", str(text)).strip().lower()


class Report:
    """Container for pipeline results with export methods."""

    def __init__(
        self,
        df: pd.DataFrame,
        corpus_stats: dict,
        topic_coverage: dict,
        gap_analysis: dict,
        cross_analysis: dict,
        bertopic_results: dict,
        zeroshot_results: dict,
        config: PipelineConfig,
    ):
        self.df = df
        self.corpus_stats = corpus_stats
        self.topic_coverage = topic_coverage
        self.gap_analysis = gap_analysis
        self.cross_analysis = cross_analysis
        self.bertopic_results = bertopic_results
        self.zeroshot_results = zeroshot_results
        self.config = config

    def export_csv(self, path: str) -> None:
        """Export full DataFrame to CSV."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.df.to_csv(path, index=False)

    def export_json(self, path: str) -> None:
        """Export statistics summary to JSON."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        summary = {
            "corpus_stats": self.corpus_stats,
            "topic_coverage": self.topic_coverage,
            "gap_analysis": self.gap_analysis,
            "cross_analysis": self.cross_analysis,
        }
        with open(path, "w") as f:
            json.dump(summary, f, indent=2, default=str)

    def generate_plots(self, path: str) -> None:
        """Generate all plots to directory."""
        os.makedirs(path, exist_ok=True)

        # Corpus overview
        plot_year_distribution(self.df, os.path.join(path, "year_distribution.png"))

        # Zero-shot results
        plot_zeroshot_label_counts(
            self.zeroshot_results, os.path.join(path, "zeroshot_label_counts.png")
        )
        plot_zeroshot_confidence_by_label(
            self.zeroshot_results, os.path.join(path, "zeroshot_confidence_by_label.png")
        )
        plot_confidence_distribution(
            self.zeroshot_results, os.path.join(path, "confidence_distribution.png")
        )

        # BERTopic results
        plot_bertopic_sizes(
            self.bertopic_results, os.path.join(path, "bertopic_sizes.png")
        )
        plot_bertopic_topic_words(
            self.bertopic_results, os.path.join(path, "bertopic_topic_words.png")
        )
        plot_bertopic_representative_docs(
            self.bertopic_results, os.path.join(path, "bertopic_representative_docs.png")
        )

        # Cross-analysis
        plot_topic_coverage(self.topic_coverage, os.path.join(path, "topic_coverage.png"))
        plot_gap_analysis(self.gap_analysis, os.path.join(path, "gap_analysis.png"))

        # Topic × Label alignment
        plot_topic_label_heatmap(
            self.bertopic_results, self.zeroshot_results,
            os.path.join(path, "topic_label_heatmap.png")
        )

        # Label co-occurrence
        plot_colabel_matrix(
            self.zeroshot_results, os.path.join(path, "colabel_matrix.png")
        )

        # New: Topic size vs confidence
        plot_topic_confidence_scatter(
            self.bertopic_results, self.zeroshot_results,
            os.path.join(path, "topic_confidence_scatter.png")
        )

        # New: Method overlap
        plot_method_overlap(
            self.cross_analysis, os.path.join(path, "method_overlap.png")
        )

        # New: Topic network
        plot_topic_network(
            self.bertopic_results, self.zeroshot_results,
            os.path.join(path, "topic_network.png")
        )

        # New: Per-topic label distribution
        plot_topic_label_distribution(
            self.bertopic_results, self.zeroshot_results,
            os.path.join(path, "topic_label_distribution.png")
        )

        # Topic distribution by year (soft assignments)
        plot_topic_distribution_by_year(
            self.bertopic_results, self.df,
            os.path.join(path, "topic_distribution_by_year.png")
        )

    def summary(self) -> str:
        """Return human-readable summary string."""
        lines = [
            "=== LitReview Report ===",
            f"Total papers: {self.corpus_stats['total_papers']}",
            f"Year range: {self.corpus_stats['year_range']}",
            "",
            "--- BERTopic ---",
            f"Topics discovered: {self.bertopic_results.get('num_topics', 0)}",
            f"Outlier papers: {self.bertopic_results.get('outlier_count', 0)}",
        ]

        topic_sizes = self.bertopic_results.get("topic_sizes", {})

        # Handle DataFrame input — convert to dict
        if isinstance(topic_sizes, pd.DataFrame):
            if topic_sizes.empty:
                topic_sizes = {}
            else:
                topic_sizes = topic_sizes.to_dict(orient="list")
                topic_sizes = {k: v[0] if isinstance(v, list) else v for k, v in topic_sizes.items()}

        if topic_sizes:
            non_outlier = {k: v for k, v in topic_sizes.items() if k != -1}
            if non_outlier:
                largest = max(non_outlier, key=non_outlier.get)
                smallest = min(non_outlier, key=non_outlier.get)
                lines.append(f"Largest topic: T{largest} ({non_outlier[largest]} papers)")
                lines.append(f"Smallest topic: T{smallest} ({non_outlier[smallest]} papers)")

        lines.append("")
        lines.append("--- Zero-Shot Classification ---")
        label_counts = self.zeroshot_results.get("label_counts", {})
        if label_counts:
            sorted_labels = sorted(label_counts.items(), key=lambda x: x[1], reverse=True)
            for label, count in sorted_labels:
                pct = count / self.corpus_stats["total_papers"] * 100
                lines.append(f"  {label}: {count} papers ({pct:.1f}%)")

        if self.gap_analysis.get("gaps"):
            lines.append("")
            lines.append(f"--- Gaps ({self.gap_analysis['num_gaps']}) ---")
            for gap in self.gap_analysis["gaps"]:
                severity = gap.get("severity", "unknown")
                desc = gap.get("description", gap.get("seed", "unknown"))
                lines.append(f"  [{severity.upper()}] {desc}")

        lines.append("")
        lines.append("=== End Report ===")
        return "\n".join(lines)
