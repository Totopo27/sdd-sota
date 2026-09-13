"""Open Access (OA) Resolver and Abstract Recovery Engine.

Queries Europe PMC and Unpaywall REST APIs to recover missing paper abstracts,
verify Open Access status, and retrieve direct full-text/PDF links without
requiring headless browsers or heavy scraping infrastructure.
"""

import logging
import os
import re
from typing import Any
import pandas as pd
import requests

logger = logging.getLogger(__name__)

DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Za-z0-9]+", re.IGNORECASE)
HTML_TAG_CLEANER = re.compile(r"<[^>]+>")


def normalize_doi(doi: str | None) -> str | None:
    """Extract and normalize a standard DOI string.

    Examples:
        "https://doi.org/10.1038/s41586-020-2649-2" -> "10.1038/s41586-020-2649-2"
        "doi: 10.1145/3372278.3390678" -> "10.1145/3372278.3390678"
    """
    if not doi or not isinstance(doi, str):
        return None

    cleaned = doi.strip()
    match = DOI_PATTERN.search(cleaned)
    if match:
        return match.group(0).strip().rstrip(".")
    return None


class OAResolver:
    """Resolves Open Access status and recovers abstracts from scholarly APIs."""

    EUROPE_PMC_SEARCH_URL = "https://www.ebi.ac.uk/europepmc/webservices/rest/search"
    UNPAYWALL_BASE_URL = "https://api.unpaywall.org/v2"

    def __init__(
        self,
        email: str | None = None,
        timeout: int = 10,
    ):
        self.email = email or os.getenv("UNPAYWALL_EMAIL", "researcher@sdd-sota.local")
        self.timeout = timeout

    def resolve_by_doi(self, raw_doi: str) -> dict[str, Any]:
        """Attempt to retrieve abstract, OA status, and PDF link for a DOI.

        Tries Europe PMC first (which directly provides abstracts without auth),
        then falls back to Unpaywall for OA metadata and open repository links.

        Returns:
            dict with keys: abstract, is_oa, oa_url, oa_status, source
        """
        result = {
            "abstract": None,
            "is_oa": False,
            "oa_url": None,
            "oa_status": None,
            "source": None,
        }

        doi = normalize_doi(raw_doi)
        if not doi:
            return result

        # 1. Query Europe PMC
        try:
            params = {
                "query": f'DOI:"{doi}"',
                "format": "json",
                "resultType": "core",
            }
            resp = requests.get(
                self.EUROPE_PMC_SEARCH_URL,
                params=params,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                results = data.get("resultList", {}).get("result", [])
                if results:
                    entry = results[0]
                    abstract_raw = entry.get("abstractText")
                    if abstract_raw:
                        # Clean HTML tags from Europe PMC abstract if present
                        cleaned_abstract = HTML_TAG_CLEANER.sub("", abstract_raw).strip()
                        if cleaned_abstract:
                            result["abstract"] = cleaned_abstract

                    is_oa_str = entry.get("isOpenAccess", "N")
                    if is_oa_str == "Y":
                        result["is_oa"] = True

                    # Extract full text PDF URL
                    urls = entry.get("fullTextUrlList", {}).get("fullTextUrl", [])
                    pdf_url = None
                    for u in urls:
                        if u.get("documentStyle") == "pdf":
                            pdf_url = u.get("url")
                            break
                    if not pdf_url and urls:
                        pdf_url = urls[0].get("url")

                    if pdf_url:
                        result["oa_url"] = pdf_url

                    if result["abstract"] or result["is_oa"]:
                        result["source"] = "europe_pmc"
                        # If we already got both abstract and OA status, return early
                        if result["abstract"] and result["is_oa"] and result["oa_url"]:
                            return result
        except requests.exceptions.RequestException as e:
            logger.debug(f"Europe PMC lookup failed for DOI {doi}: {e}")

        # 2. Query Unpaywall as fallback or supplement
        try:
            unpaywall_url = f"{self.UNPAYWALL_BASE_URL}/{doi}"
            params = {"email": self.email}
            resp = requests.get(
                unpaywall_url,
                params=params,
                timeout=self.timeout,
            )
            if resp.status_code == 200:
                data = resp.json()
                is_oa = data.get("is_oa", False)
                oa_status = data.get("oa_status")
                best_loc = data.get("best_oa_location") or {}
                oa_url = best_loc.get("url_for_pdf") or best_loc.get("url")

                if is_oa:
                    result["is_oa"] = True
                if oa_status:
                    result["oa_status"] = oa_status
                if oa_url and not result["oa_url"]:
                    result["oa_url"] = oa_url

                if not result["source"] and (result["is_oa"] or result["oa_url"]):
                    result["source"] = "unpaywall"
        except requests.exceptions.RequestException as e:
            logger.debug(f"Unpaywall lookup failed for DOI {doi}: {e}")

        return result

    def enrich_dataframe(
        self,
        df: pd.DataFrame,
        abstract_col: str = "Abstract Note",
        doi_col: str = "DOI",
        min_len: int = 50,
    ) -> pd.DataFrame:
        """Enrich DataFrame by recovering missing/short abstracts and tagging OA links.

        Args:
            df: Corpus DataFrame.
            abstract_col: Column containing document abstracts.
            doi_col: Column containing DOIs.
            min_len: Minimum acceptable length for existing abstract.

        Returns:
            Copy of DataFrame with populated abstracts and OA metadata.
        """
        enriched_df = df.copy()

        if "is_oa" not in enriched_df.columns:
            enriched_df["is_oa"] = False
        if "oa_url" not in enriched_df.columns:
            enriched_df["oa_url"] = None

        if doi_col not in enriched_df.columns:
            return enriched_df

        for idx, row in enriched_df.iterrows():
            curr_abstract = row.get(abstract_col)
            needs_abstract = (
                curr_abstract is None
                or not isinstance(curr_abstract, str)
                or len(curr_abstract.strip()) < min_len
            )

            raw_doi = row.get(doi_col)
            if not raw_doi or not isinstance(raw_doi, str):
                continue

            # If abstract is missing or we want OA link resolution
            if needs_abstract:
                res = self.resolve_by_doi(raw_doi)
                if res["abstract"]:
                    enriched_df.at[idx, abstract_col] = res["abstract"]
                if res["is_oa"]:
                    enriched_df.at[idx, "is_oa"] = True
                if res["oa_url"]:
                    enriched_df.at[idx, "oa_url"] = res["oa_url"]

        return enriched_df
