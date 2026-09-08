"""CLI command for discovering academic literature and populating Zotero.

Usage:
    litreview-discover --query "quantum error correction" --limit 20 --collection "Quantum-EC"
"""

import argparse
import logging
import sys
from dotenv import load_dotenv

from litreview.discovery.openalex import OpenAlexClient
from litreview.discovery.zotero_populator import ZoteroPopulator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("litreview-discover")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Discover scholarly papers via OpenAlex and auto-populate Zotero."
    )
    parser.add_argument(
        "--query",
        "-q",
        type=str,
        required=True,
        help="Academic topic search query (e.g. 'retrieval augmented generation').",
    )
    parser.add_argument(
        "--limit",
        "-n",
        type=int,
        default=25,
        help="Number of papers with valid abstracts to retrieve (default: 25).",
    )
    parser.add_argument(
        "--collection",
        "-c",
        type=str,
        required=True,
        help="Target Zotero collection name to create/populate.",
    )
    parser.add_argument(
        "--min-year",
        type=int,
        default=None,
        help="Earliest publication year to accept (e.g. 2022).",
    )
    parser.add_argument(
        "--min-citations",
        type=int,
        default=0,
        help="Minimum citation threshold (default: 0).",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    logger.info(f"Initiating academic discovery for query: '{args.query}'")
    client = OpenAlexClient()
    papers = client.search_works(
        query=args.query,
        limit=args.limit,
        min_year=args.min_year,
        min_citations=args.min_citations,
        require_abstract=True,
    )

    if not papers:
        logger.warning(f"No papers found matching query: '{args.query}'")
        sys.exit(1)

    logger.info(f"Successfully harvested {len(papers)} papers with full abstracts from OpenAlex.")
    for idx, p in enumerate(papers[:3], 1):
        logger.info(f" [{idx}] {p['title']} ({p.get('year')}) | DOI: {p.get('doi') or 'N/A'}")

    try:
        populator = ZoteroPopulator()
        logger.info(f"Injecting papers into Zotero collection: '{args.collection}'...")
        inserted = populator.populate_papers(papers, collection_name=args.collection)
        logger.info(f"Complete! Successfully added {inserted} papers to Zotero collection '{args.collection}'.")
        print(f"\n[OK] {inserted} papers loaded into Zotero collection '{args.collection}'.")
        print(f"You can now run:\n  litreview-agent-summary --collection \"{args.collection}\"\n")
    except Exception as e:
        logger.error(f"Zotero population failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
