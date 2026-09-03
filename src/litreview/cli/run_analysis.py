"""CLI entry point for the full analysis pipeline.

Usage:
    litreview-analysis --config config.yaml --fetch-zotero
    litreview-analysis --config config.yaml --fetch-zotero --output results/classified.csv --plots results/plots/
"""

import argparse
import os
import sys

from litreview import ReviewPipeline, load_config


def main():
    parser = argparse.ArgumentParser(
        description="Run literature review pipeline: fetch -> discover -> validate -> report"
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument("--fetch-zotero", action="store_true", help="Fetch from Zotero")
    parser.add_argument(
        "--output",
        default="data/processed/classified.csv",
        help="Output CSV path",
    )
    parser.add_argument(
        "--plots",
        default="results/plots",
        help="Plots output directory",
    )
    parser.add_argument(
        "--skip-bertopic",
        action="store_true",
        help="Skip BERTopic discovery, run zero-shot only",
    )
    args = parser.parse_args()

    # Load config
    if not os.path.exists(args.config):
        print(f"Error: Config file '{args.config}' not found.")
        sys.exit(1)

    config = load_config(args.config)

    # Run pipeline
    pipeline = ReviewPipeline(config, skip_bertopic=args.skip_bertopic)
    report = pipeline.run()

    # Export results
    report.export_csv(args.output)
    report.generate_plots(args.plots)
    print(report.summary())


if __name__ == "__main__":
    main()
