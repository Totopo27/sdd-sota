"""OpenAlex API client for academic paper discovery."""

import logging
import requests
from typing import Any

logger = logging.getLogger(__name__)

OPENALEX_WORKS_API = "https://api.openalex.org/works"


def reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    """Reconstruct full text abstract from OpenAlex inverted index.

    OpenAlex stores abstracts as an inverted index mapping words to integer
    word positions to optimize database storage.
    """
    if not inverted_index:
        return ""
    word_positions: list[tuple[int, str]] = []
    for word, positions in inverted_index.items():
        for pos in positions:
            word_positions.append((pos, word))
    word_positions.sort(key=lambda x: x[0])
    return " ".join(w for _, w in word_positions).strip()


class OpenAlexClient:
    """Client for querying the OpenAlex open scholarly dataset."""

    def __init__(self, mailto: str = "researcher@sdd-sota.local", timeout: int = 20):
        self.mailto = mailto
        self.timeout = timeout

    def search_works(
        self,
        query: str,
        limit: int = 25,
        min_year: int | None = None,
        min_citations: int = 0,
        require_abstract: bool = True,
    ) -> list[dict[str, Any]]:
        """Search OpenAlex works for a given natural language query.

        Args:
            query: Search query (e.g. "retrieval augmented generation")
            limit: Target number of valid papers to retrieve
            min_year: Optional earliest publication year (e.g. 2022)
            min_citations: Minimum citation count
            require_abstract: Only return papers with non-empty abstracts

        Returns:
            List of normalized paper dicts with keys:
            title, abstract, year, doi, authors, citations, venue, openalex_id
        """
        papers: list[dict[str, Any]] = []
        page = 1
        per_page = min(max(limit * 2, 25), 100)  # Fetch slightly more to account for filters

        filters = ["has_abstract:true"] if require_abstract else []
        if min_year:
            filters.append(f"from_publication_date:{min_year}-01-01")
        if min_citations > 0:
            filters.append(f"cited_by_count:>{min_citations - 1}")

        filter_str = ",".join(filters)

        headers = {
            "User-Agent": f"litreview-SSD-SOTA/0.2.0 (mailto:{self.mailto})"
        }

        while len(papers) < limit and page <= 5:
            params: dict[str, Any] = {
                "search": query,
                "per-page": per_page,
                "page": page,
                "mailto": self.mailto,
            }
            if filter_str:
                params["filter"] = filter_str

            try:
                resp = requests.get(
                    OPENALEX_WORKS_API,
                    params=params,
                    headers=headers,
                    timeout=self.timeout,
                )
                if resp.status_code != 200:
                    logger.warning(
                        f"OpenAlex returned HTTP {resp.status_code}: {resp.text[:150]}"
                    )
                    break

                data = resp.json()
                results = data.get("results", [])
                if not results:
                    break

                for work in results:
                    title = work.get("title") or ""
                    if not title.strip():
                        continue

                    abstract = reconstruct_abstract(work.get("abstract_inverted_index"))
                    if require_abstract and not abstract:
                        continue

                    year = work.get("publication_year")
                    doi_url = work.get("doi") or ""
                    # Clean up DOI from URL (e.g. https://doi.org/10.xxxx/yyyy -> 10.xxxx/yyyy)
                    doi = doi_url.replace("https://doi.org/", "").strip() if doi_url else ""

                    # Extract author names
                    authors = []
                    for authorship in work.get("authorships", []):
                        author = authorship.get("author", {})
                        display_name = author.get("display_name")
                        if display_name:
                            authors.append(display_name)

                    # Extract primary venue/journal
                    location = work.get("primary_location") or {}
                    source = location.get("source") or {}
                    venue = source.get("display_name") or ""

                    papers.append({
                        "title": title.strip(),
                        "abstract": abstract,
                        "year": year,
                        "doi": doi,
                        "authors": authors,
                        "citations": work.get("cited_by_count", 0),
                        "venue": venue,
                        "openalex_id": work.get("id", ""),
                    })
                    if len(papers) >= limit:
                        break

                if len(papers) >= limit or len(results) < per_page:
                    break

                page += 1

            except Exception as e:
                logger.error(f"OpenAlex search error: {e}")
                break

        return papers
