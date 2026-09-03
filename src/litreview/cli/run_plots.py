"""CLI entry point for generating plots from existing classified data.

Usage:
    litreview-plots --config config.yaml
    litreview-plots --config config.yaml --input data/processed/classified.csv --output results/plots/
"""

import argparse
import os
import sys

import pandas as pd

from litreview.statistics import compute_corpus_stats
from litreview.visualization import (
    plot_year_distribution,
    plot_source_distribution,
    plot_topic_coverage,
    plot_gap_analysis,
    plot_confidence_distribution,
)
from litreview import load_config


def main():
    parser = argparse.ArgumentParser(
        description="Generate plots from classified paper data"
    )
    parser.add_argument("--config", default="config.yaml", help="Config file path")
    parser.add_argument(
        "--input",
        default="data/processed/classified.csv",
        help="Input CSV path",
    )
    parser.add_argument(
        "--output",
        default="results/plots",
        help="Plots output directory",
    )
    args = parser.parse_args()

    # Load data
    if not os.path.exists(args.input):
        print(f"Error: Input file '{args.input}' not found.")
        sys.exit(1)

    df = pd.read_csv(args.input)
    stats = compute_corpus_stats(df)

    # Generate plots
    os.makedirs(args.output, exist_ok=True)
    plot_year_distribution(df, os.path.join(args.output, "year_distribution.png"))
    plot_source_distribution(df, os.path.join(args.output, "source_distribution.png"))

    print(f"Generated plots in {args.output}")
    print(f"Total papers: {stats['total_papers']}")
    print(f"Year range: {stats['year_range']}")


if __name__ == "__main__":
    main()
