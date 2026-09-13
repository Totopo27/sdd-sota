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
from litreview.discovery.oa_resolver import OAResolver

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
    parser.add_argument(
        "--no-resolve-oa",
        action="store_true",
        help="Disable Open Access abstract recovery via Europe PMC and Unpaywall.",
    )
    parser.add_argument(
        "--no-dedup",
        action="store_true",
        help="Disable deduplication against existing Zotero collection items.",
    )
    return parser.parse_args()


def main():
    load_dotenv()
    args = parse_args()

    resolve_oa = not args.no_resolve_oa
    logger.info(f"Initiating academic discovery for query: '{args.query}' (resolve_oa={resolve_oa})")
    client = OpenAlexClient()

    # If OA recovery is active, we don't strictly require abstracts from OpenAlex initially,
    # as we can recover them from Europe PMC / Unpaywall
    fetch_limit = args.limit * 2 if resolve_oa else args.limit
    papers = client.search_works(
        query=args.query,
        limit=fetch_limit,
        min_year=args.min_year,
        min_citations=args.min_citations,
        require_abstract=not resolve_oa,
    )

    if not papers:
        logger.warning(f"No papers found matching query: '{args.query}'")
        sys.exit(1)

    # Resolve OA and missing abstracts if enabled
    if resolve_oa:
        oa_resolver = OAResolver()
        recovered_count = 0
        for p in papers:
            doi = p.get("doi")
            curr_abstract = p.get("abstract", "")
            if not curr_abstract or len(curr_abstract.strip()) < 50:
                if doi:
                    res = oa_resolver.resolve_by_doi(doi)
                    if res.get("abstract"):
                        p["abstract"] = res["abstract"]
                        recovered_count += 1
                    if res.get("is_oa"):
                        p["is_oa"] = True
                    if res.get("oa_url"):
                        p["oa_url"] = res["oa_url"]
                    if res.get("oa_status"):
                        p["oa_status"] = res["oa_status"]
            elif doi and not p.get("oa_url"):
                res = oa_resolver.resolve_by_doi(doi)
                if res.get("is_oa"):
                    p["is_oa"] = True
                if res.get("oa_url"):
                    p["oa_url"] = res["oa_url"]
                if res.get("oa_status"):
                    p["oa_status"] = res["oa_status"]

        if recovered_count > 0:
            logger.info(f"Successfully recovered {recovered_count} abstracts via Europe PMC / Unpaywall.")

        # Ensure we only keep papers with valid abstracts
        papers = [p for p in papers if p.get("abstract") and len(p["abstract"].strip()) >= 50][: args.limit]

    logger.info(f"Harvested {len(papers)} candidate papers with full abstracts.")
    for idx, p in enumerate(papers[:3], 1):
        oa_indicator = f" [OA: {p.get('oa_status')}]" if p.get("is_oa") else ""
        logger.info(f" [{idx}] {p['title']} ({p.get('year')}){oa_indicator} | DOI: {p.get('doi') or 'N/A'}")

    try:
        populator = ZoteroPopulator()

        # Deduplication against existing Zotero items
        if not args.no_dedup:
            papers, skipped_count = populator.filter_existing_duplicates(papers, collection_name=args.collection)
            if skipped_count > 0:
                logger.info(f"Deduplication: skipped {skipped_count} papers already present in '{args.collection}'.")

        if not papers:
            logger.info(f"All harvested papers are already present in Zotero collection '{args.collection}'.")
            print(f"\n[OK] 0 new papers needed (all already present in collection '{args.collection}').")
            return

        logger.info(f"Injecting {len(papers)} unique papers into Zotero collection: '{args.collection}'...")
        inserted = populator.populate_papers(papers, collection_name=args.collection)
        logger.info(f"Complete! Successfully added {inserted} papers to Zotero collection '{args.collection}'.")
        print(f"\n[OK] {inserted} papers loaded into Zotero collection '{args.collection}'.")
        print(f"You can now run:\n  litreview-agent-summary --collection \"{args.collection}\"\n")
    except Exception as e:
        logger.error(f"Zotero population failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
