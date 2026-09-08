"""Semantic Scholar Academic Graph API client with persistent disk caching."""

import json
import logging
import os
import time
from pathlib import Path
import requests

logger = logging.getLogger(__name__)

S2_PAPER_API = "https://api.semanticscholar.org/graph/v1/paper"
S2_SEARCH_API = "https://api.semanticscholar.org/graph/v1/paper/search"
DEFAULT_FIELDS = "paperId,title,year,citationCount,influentialCitationCount,citations.paperId,citations.title,citations.year,references.paperId,references.title,references.year"


class SemanticScholarClient:
    """Client for the Allen Institute for AI Semantic Scholar Academic Graph (S2AG)."""

    def __init__(
        self,
        api_key: str | None = None,
        cache_path: str | Path | None = None,
        min_request_interval: float = 1.0,
    ):
        self.api_key = api_key or os.getenv("SEMANTIC_SCHOLAR_API_KEY")
        
        if cache_path is None:
            # Default to D:\huggingface_cache or local data/cache
            base_dir = Path("data/cache")
            base_dir.mkdir(parents=True, exist_ok=True)
            self.cache_path = base_dir / "semantic_scholar_cache.json"
        else:
            self.cache_path = Path(cache_path)
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)

        self.min_request_interval = min_request_interval
        self._last_request_time = 0.0
        self.cache = self._load_cache()

    def _load_cache(self) -> dict:
        if self.cache_path.exists():
            try:
                with open(self.cache_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Could not load Semantic Scholar cache from {self.cache_path}: {e}")
        return {}

    def _save_cache(self) -> None:
        try:
            with open(self.cache_path, "w", encoding="utf-8") as f:
                json.dump(self.cache, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.warning(f"Failed to persist Semantic Scholar cache to {self.cache_path}: {e}")

    def _update_cache(self, key: str, data: dict) -> None:
        self.cache[key] = data
        self._save_cache()

    def _get_headers(self) -> dict:
        headers = {
            "User-Agent": "litreview-SSD-SOTA/0.2.0 (research-lane-agent)"
        }
        if self.api_key:
            headers["x-api-key"] = self.api_key
        return headers

    def _throttle(self) -> None:
        elapsed = time.time() - self._last_request_time
        if elapsed < self.min_request_interval:
            time.sleep(self.min_request_interval - elapsed)
        self._last_request_time = time.time()

    def get_paper(self, paper_id: str) -> dict | None:
        """Retrieve paper by S2 paperId, DOI:<doi>, or arXiv:<id>."""
        paper_id = paper_id.strip()
        cache_key = paper_id
        if cache_key in self.cache:
            return self.cache[cache_key]

        self._throttle()
        url = f"{S2_PAPER_API}/{paper_id}?fields={DEFAULT_FIELDS}"
        try:
            resp = requests.get(url, headers=self._get_headers(), timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                self._update_cache(cache_key, data)
                return data
            elif resp.status_code == 404:
                return None
            elif resp.status_code == 429:
                logger.warning("Semantic Scholar rate limit hit (429). Retrying after short pause...")
                time.sleep(3.0)
                resp = requests.get(url, headers=self._get_headers(), timeout=15)
                if resp.status_code == 200:
                    data = resp.json()
                    self._update_cache(cache_key, data)
                    return data
        except Exception as e:
            logger.error(f"Semantic Scholar API error for {paper_id}: {e}")
        return None

    def search_paper_by_title(self, title: str) -> dict | None:
        """Search paper by title string if DOI is absent or not found."""
        clean_title = title.strip()
        if not clean_title:
            return None
        cache_key = f"TITLE:{clean_title.lower()}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        self._throttle()
        params = {
            "query": clean_title,
            "limit": 1,
            "fields": DEFAULT_FIELDS,
        }
        try:
            resp = requests.get(S2_SEARCH_API, params=params, headers=self._get_headers(), timeout=15)
            if resp.status_code == 200:
                data = resp.json()
                items = data.get("data", [])
                if items:
                    matched = items[0]
                    self._update_cache(cache_key, matched)
                    return matched
        except Exception as e:
            logger.error(f"Semantic Scholar search error for '{clean_title}': {e}")
        return None

    def get_paper_for_record(self, title: str, doi: str | None = None) -> dict | None:
        """Look up paper metadata using DOI first, then falling back to title search."""
        if doi and str(doi).strip() and not str(doi).lower() in ("nan", "none"):
            clean_doi = str(doi).strip()
            # Normalize DOI
            if not clean_doi.lower().startswith("doi:"):
                query_id = f"DOI:{clean_doi}"
            else:
                query_id = clean_doi
            paper = self.get_paper(query_id)
            if paper:
                return paper

        # Fallback to title search
        if title and str(title).strip():
            return self.search_paper_by_title(title)

        return None
