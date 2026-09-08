"""CLI entry point for SDD Agent / headless research summary.

Designed for AI Coding Agents (such as Pi / gentle-pi / sdd-sota).
Executes the literature review pipeline and outputs structured JSON
and markdown summaries with research gaps and topic maps.

Usage:
    litreview-agent-summary --config config.yaml --collection "Pneumonia-ML"
    litreview-agent-summary --config config.yaml --input-csv papers.csv --output-json results/sdd-research-summary.json
"""

import argparse
import json
import os
import sys
from pathlib import Path

from litreview import ReviewPipeline, load_config
from litreview.fetchers.csv_fetcher import CSVFetcher


def main():
    parser = argparse.ArgumentParser(
        description="Run literature review pipeline and generate SDD-ready JSON summary"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--collection", default=None, help="Zotero collection name to fetch from")
    parser.add_argument("--input-csv", default=None, help="Local CSV file path (bypasses Zotero)")
    parser.add_argument(
        "--labels-json",
        default=None,
        help="JSON string or file path with candidate labels mapping: '{\"KEY\": \"description\"}'",
    )
    parser.add_argument(
        "--output-json",
        default="results/sdd-research-summary.json",
        help="Path for agent JSON output summary",
    )
    parser.add_argument(
        "--output-csv",
        default="results/classified.csv",
        help="Path for classified papers CSV export",
    )
    parser.add_argument(
        "--plots-dir",
        default="results/plots",
        help="Directory to save generated charts",
    )
    parser.add_argument(
        "--skip-bertopic",
        action="store_true",
        help="Skip BERTopic unsupervised discovery, run zero-shot only",
    )
    parser.add_argument(
        "--no-plots",
        action="store_true",
        help="Skip plot generation to speed up headless execution",
    )
    args = parser.parse_args()

    # 1. Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.", file=sys.stderr)
        sys.exit(1)

    config = load_config(args.config)

    # 2. Apply CLI overrides
    if args.collection:
        config.zotero.collection_names = [args.collection]

    if args.labels_json:
        try:
            if os.path.exists(args.labels_json):
                with open(args.labels_json, "r", encoding="utf-8") as f:
                    labels_dict = json.load(f)
            else:
                labels_dict = json.loads(args.labels_json)

            if isinstance(labels_dict, dict):
                config.zeroshot.candidate_labels = labels_dict
                config.bertopic.candidate_labels = labels_dict
                config.bertopic.seed_words = list(labels_dict.values())
        except Exception as e:
            print(f"Warning: Failed to parse --labels-json: {e}", file=sys.stderr)

    # 3. Choose fetcher
    fetcher = None
    if args.input_csv:
        if not os.path.exists(args.input_csv):
            print(f"Error: Input CSV '{args.input_csv}' not found.", file=sys.stderr)
            sys.exit(1)
        fetcher = CSVFetcher(args.input_csv)

    # 4. Run pipeline
    pipeline = ReviewPipeline(config, skip_bertopic=args.skip_bertopic, fetcher=fetcher)
    try:
        report = pipeline.run()
    except Exception as e:
        err_response = {
            "status": "error",
            "error_type": type(e).__name__,
            "message": str(e),
        }
        print(json.dumps(err_response, indent=2))
        sys.exit(1)

    # 5. Export artifacts
    if args.output_csv:
        os.makedirs(os.path.dirname(args.output_csv) or ".", exist_ok=True)
        report.export_csv(args.output_csv)

    if not args.no_plots and args.plots_dir:
        try:
            report.generate_plots(args.plots_dir)
        except Exception as e:
            print(f"Warning: Plot generation failed: {e}", file=sys.stderr)

    # 6. Build structured summary for AI Agent
    summary_data = {
        "status": "success",
        "corpus": report.corpus_stats,
        "topics": {
            "num_topics": report.topic_coverage.get("num_topics", 0),
            "outlier_count": report.topic_coverage.get("outlier_count", 0),
            "topic_sizes": report.topic_coverage.get("topic_sizes", {}),
            "top_words": report.bertopic_results.get("topic_words", {}),
        },
        "taxonomy_validation": {
            "total_classified": report.topic_coverage.get("total_classified", 0),
            "mean_confidence": round(report.topic_coverage.get("mean_confidence", 0.0), 4),
            "median_confidence": round(report.topic_coverage.get("median_confidence", 0.0), 4),
            "label_counts": report.zeroshot_results.get("label_counts", {}),
        },
        "gap_analysis": report.gap_analysis,
        "cross_analysis": report.cross_analysis,
        "artifacts": {
            "csv_path": args.output_csv if args.output_csv else None,
            "plots_dir": args.plots_dir if not args.no_plots else None,
        },
    }

    # 7. Write output JSON
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=str)

    # 8. Print JSON summary to stdout
    print(json.dumps(summary_data, indent=2, default=str))


if __name__ == "__main__":
    main()
