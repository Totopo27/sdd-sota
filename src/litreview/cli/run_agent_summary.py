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
import logging
import os
import sys
from pathlib import Path
import requests
from dotenv import load_dotenv

from litreview import ReviewPipeline, load_config

from litreview.fetchers.csv_fetcher import CSVFetcher
from litreview.fetchers.parquet_fetcher import ParquetFetcher

logger = logging.getLogger(__name__)



def resolve_and_parse_pdfs(
    df,
    pdf_dir: str = "data/raw_pdfs",
) -> dict[str, dict]:
    """Resolve and extract academic sections from local or OA PDFs for papers in DataFrame."""
    if df is None or len(df) == 0:
        return {}

    os.makedirs(pdf_dir, exist_ok=True)
    from litreview.discovery.pdf_parser import PDFSectionParser

    parser = PDFSectionParser()

    parsed_sections = {}

    for _, row in df.iterrows():
        doi = str(row.get("DOI") or "").strip()
        title = str(row.get("Title") or "").strip()
        oa_url = str(row.get("oa_url") or "").strip()
        url = str(row.get("Url") or "").strip()

        # Sanitize filename for local cache
        safe_name = "".join(c if c.isalnum() or c in "._-" else "_" for c in (doi or title))[:80]
        local_pdf_path = os.path.join(pdf_dir, f"{safe_name}.pdf")

        pdf_bytes = None
        if os.path.exists(local_pdf_path):
            try:
                with open(local_pdf_path, "rb") as f:
                    pdf_bytes = f.read()
            except Exception as e:
                logger.warning(f"Failed to read local PDF {local_pdf_path}: {e}")

        # If not cached locally and OA URL is available, attempt download
        if pdf_bytes is None and (oa_url or (url and url.lower().endswith(".pdf"))):
            fetch_url = oa_url or url
            try:
                resp = requests.get(fetch_url, timeout=15, headers={"User-Agent": "SDD-SOTA-LitReview/1.0"})
                if resp.status_code == 200 and resp.content.startswith(b"%PDF"):
                    pdf_bytes = resp.content
                    try:
                        with open(local_pdf_path, "wb") as f:
                            f.write(pdf_bytes)
                    except Exception:
                        pass
            except Exception as e:
                logger.warning(f"Could not download PDF from {fetch_url}: {e}")

        if pdf_bytes:
            try:
                sections = parser.parse_pdf(pdf_bytes)
                if doi:
                    parsed_sections[doi] = sections
                if title:
                    parsed_sections[title] = sections
            except Exception as e:
                logger.warning(f"Failed to parse PDF for {doi or title}: {e}")

    return parsed_sections


def main():
    load_dotenv()
    parser = argparse.ArgumentParser(

        description="Run literature review pipeline and generate SDD-ready JSON summary"
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config.yaml")
    parser.add_argument("--collection", default=None, help="Zotero collection name to fetch from")
    parser.add_argument("--input-csv", default=None, help="Local CSV file path (bypasses Zotero)")
    parser.add_argument("--input-parquet", default=None, help="Local Parquet file path (bypasses Zotero and CSV)")
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
        "--output-parquet",
        default="results/classified.parquet",
        help="Path for compressed columnar Parquet export",
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
    parser.add_argument(
        "--skip-network",
        action="store_true",
        help="Skip Semantic Scholar citation graph analysis",
    )
    parser.add_argument(
        "--resolve-oa",
        action="store_true",
        help="Attempt to recover missing abstracts and identify Open Access links via Europe PMC and Unpaywall",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable automatic cross-source deduplication with year-slack",
    )
    parser.add_argument(
        "--parse-pdfs",
        action="store_true",
        help="Download/parse full-text academic PDFs for section-targeted NeurIPS rigor evaluation",
    )
    parser.add_argument(
        "--pdf-dir",
        default="data/raw_pdfs",
        help="Local directory to cache or read full-text PDFs",
    )
    parser.add_argument(
        "--output-report",
        default="results/sdd-research-report.md",
        help="Path for executive Markdown research briefing export",
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
    if args.input_parquet:
        if not os.path.exists(args.input_parquet):
            print(f"Error: Input Parquet '{args.input_parquet}' not found.", file=sys.stderr)
            sys.exit(1)
        fetcher = ParquetFetcher(args.input_parquet)
    elif args.input_csv:
        if not os.path.exists(args.input_csv):
            print(f"Error: Input CSV '{args.input_csv}' not found.", file=sys.stderr)
            sys.exit(1)
        fetcher = CSVFetcher(args.input_csv)

    # 4. Run pipeline
    pipeline = ReviewPipeline(
        config,
        skip_bertopic=args.skip_bertopic,
        fetcher=fetcher,
        resolve_oa=args.resolve_oa,
        dedup=not args.no_dedup,
    )
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

    if args.output_parquet:
        os.makedirs(os.path.dirname(args.output_parquet) or ".", exist_ok=True)
        report.export_parquet(args.output_parquet)

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
            "std_confidence": report.topic_coverage.get("std_confidence", 0.0),
            "variance_confidence": report.topic_coverage.get("variance_confidence", 0.0),
            "confidence_ci_95": report.topic_coverage.get("confidence_ci_95", []),
            "label_counts": report.zeroshot_results.get("label_counts", {}),
        },
        "gap_analysis": report.gap_analysis,
        "cross_analysis": report.cross_analysis,
        "artifacts": {
            "csv_path": args.output_csv if args.output_csv else None,
            "parquet_path": args.output_parquet if args.output_parquet else None,
            "plots_dir": args.plots_dir if not args.no_plots else None,
        },
    }

    if report.df is not None and "is_oa" in report.df.columns:
        oa_count = int(report.df["is_oa"].sum())
        total_p = len(report.df)
        summary_data["open_access"] = {
            "oa_count": oa_count,
            "total_papers": total_p,
            "oa_ratio": round(oa_count / total_p, 4) if total_p > 0 else 0.0,
        }

    # 7. Build Citation Network (Semantic Scholar / S2AG) and Scientometric Ranking
    if not args.skip_network and report.df is not None and len(report.df) > 0:
        try:
            from litreview.network import CitationGraphBuilder, plot_citation_network
            graph_builder = CitationGraphBuilder().build_from_dataframe(report.df)

            # Build topical relevance map from Zero-Shot confidence scores
            topical_map = {}
            if "score" in report.df.columns and "Title" in report.df.columns:
                for _, row in report.df.iterrows():
                    title_key = str(row.get("Title", "")).strip()
                    if title_key:
                        topical_map[title_key] = float(row.get("score", 0.5))

            # Run NeurIPS Rigor Rubric Analyzer on corpus abstracts or full-text sections
            methodology_map = {}
            rubric_summary = {}
            try:
                from litreview.analyzers.rubric import NeurIPSRubricAnalyzer
                rubric_analyzer = NeurIPSRubricAnalyzer()
                abstracts = report.df["Abstract Note"].dropna()
                if len(abstracts) > 0:
                    rubric_analyzer.fit(abstracts)
                    parsed_sections = {}
                    if getattr(args, "parse_pdfs", False):
                        parsed_sections = resolve_and_parse_pdfs(
                            report.df,
                            pdf_dir=args.pdf_dir,
                        )
                    rubric_df = rubric_analyzer.transform_sections(
                        report.df,
                        parsed_sections=parsed_sections,
                    )
                    rubric_summary = rubric_analyzer.results
                    for idx, row in rubric_df.iterrows():
                        paper_title = str(report.df.loc[idx, "Title"]).strip()
                        if paper_title:
                            methodology_map[paper_title] = float(row.get("rubric_score", 0.5))

                    # Enrich report.df with rubric scores and re-export if paths configured
                    for col in rubric_df.columns:
                        if col not in report.df.columns:
                            report.df[col] = rubric_df[col]
                    if args.output_csv:
                        report.export_csv(args.output_csv)
                    if args.output_parquet:
                        report.export_parquet(args.output_parquet)
            except Exception as re:
                logger.warning(f"NeurIPS rubric evaluation skipped/failed: {re}")


            summary_data["citation_network"] = graph_builder.summary(
                topical_relevance_map=topical_map,
                methodology_scores_map=methodology_map,
            )
            if rubric_summary:
                summary_data["neurips_rubric"] = rubric_summary

            if not args.no_plots and args.plots_dir:
                network_plot_path = os.path.join(args.plots_dir, "citation_network.png")
                plot_citation_network(graph_builder, network_plot_path)
                summary_data["artifacts"]["citation_network_plot"] = network_plot_path
        except Exception as e:
            print(f"Warning: Citation network analysis skipped/failed: {e}", file=sys.stderr)
            summary_data["citation_network"] = {"status": "unavailable", "reason": str(e)}

    # 8. Generate Executive Markdown Research Briefing
    if args.output_report:
        try:
            from litreview.reporting.executive_report import export_markdown_report
            scored_df = report.df if report is not None else None
            export_markdown_report(args.output_report, summary_data, scored_df=scored_df)
            summary_data["artifacts"]["markdown_report"] = args.output_report
        except Exception as me:
            logger.warning(f"Failed to generate executive markdown report: {me}")

    # 9. Write output JSON
    os.makedirs(os.path.dirname(args.output_json) or ".", exist_ok=True)
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(summary_data, f, indent=2, default=str)

    # 10. Print JSON summary to stdout
    print(json.dumps(summary_data, indent=2, default=str))



if __name__ == "__main__":
    main()
