"""Cross-Source Paper Deduplication and Record Merging with Year-Slack.

Merges scholarly literature records gathered from diverse sources
(Zotero, Semantic Scholar, OpenAlex, CSV, Parquet) by combining:
1. Exact DOI normalization.
2. Stopword-stripped title fingerprinting with token similarity.
3. Publication year slack (tolerating discrepancies between preprint and print editions).
4. Best-attribute merging (longest abstract, max citation count, verified DOI).
"""

import logging
import re
from typing import Any
import pandas as pd

from litreview.discovery.oa_resolver import normalize_doi

logger = logging.getLogger(__name__)

STOPWORDS = {
    "a", "an", "the", "and", "or", "of", "in", "on", "at", "to", "for",
    "with", "by", "from", "as", "is", "that", "this", "it", "using", "based",
}


def normalize_title(title: str | None) -> str:
    """Normalize academic paper title for fingerprinting.

    Strips punctuation, lowercases, removes common English stopwords,
    and collapses multiple whitespace.
    """
    if not title or not isinstance(title, str):
        return ""

    cleaned = re.sub(r"[^\w\s]", " ", title.lower())
    tokens = [w for w in cleaned.split() if w not in STOPWORDS and len(w) > 1]
    return " ".join(tokens)


def title_token_similarity(t1: str, t2: str) -> float:
    """Compute Jaccard token overlap similarity between two normalized titles."""
    n1 = normalize_title(t1)
    n2 = normalize_title(t2)

    if not n1 or not n2:
        return 0.0
    if n1 == n2:
        return 1.0

    tokens1 = set(n1.split())
    tokens2 = set(n2.split())

    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)

    return intersection / union if union > 0 else 0.0


class PaperDeduplicator:
    """Deduplicates academic paper collections and merges duplicate metadata."""

    def __init__(
        self,
        year_slack: int = 1,
        title_similarity_threshold: float = 0.85,
    ):
        self.year_slack = year_slack
        self.title_similarity_threshold = title_similarity_threshold

    def are_duplicates(self, p1: dict | pd.Series, p2: dict | pd.Series) -> bool:
        """Determine if two paper representations represent the same work."""
        # 1. DOI check
        doi1 = normalize_doi(str(p1.get("DOI") or ""))
        doi2 = normalize_doi(str(p2.get("DOI") or ""))
        if doi1 and doi2 and doi1 == doi2:
            return True

        # 2. Title matching with Year Slack
        t1 = str(p1.get("Title") or p1.get("title") or "")
        t2 = str(p2.get("Title") or p2.get("title") or "")

        norm1 = normalize_title(t1)
        norm2 = normalize_title(t2)

        if not norm1 or not norm2:
            return False

        sim = title_token_similarity(norm1, norm2)
        if sim >= self.title_similarity_threshold:
            # Check year slack
            y1 = p1.get("Publication Year") if "Publication Year" in p1 else p1.get("year")
            y2 = p2.get("Publication Year") if "Publication Year" in p2 else p2.get("year")

            try:
                y1_int = int(y1) if y1 is not None and not pd.isna(y1) else None
            except (ValueError, TypeError):
                y1_int = None

            try:
                y2_int = int(y2) if y2 is not None and not pd.isna(y2) else None
            except (ValueError, TypeError):
                y2_int = None

            if y1_int is not None and y2_int is not None:
                if abs(y1_int - y2_int) <= self.year_slack:
                    return True
                return False

            # If one or both years are missing, match on title similarity alone
            return True

        return False

    def merge_records(self, records: list[dict | pd.Series]) -> dict[str, Any]:
        """Merge a group of duplicate records into a single consolidated record.

        Strategy:
        - Prefer canonical normalized DOI.
        - Prefer longest/cleanest title.
        - Prefer longest, non-empty abstract.
        - Take highest citation count.
        - Prefer earliest publication year.
        - Preserve any other non-null fields.
        """
        if not records:
            return {}

        merged: dict[str, Any] = {}
        # Convert all to dicts
        dict_records = [r.to_dict() if isinstance(r, pd.Series) else dict(r) for r in records]

        # 1. Collect all keys
        all_keys = set()
        for r in dict_records:
            all_keys.update(r.keys())

        # 2. Extract best values for key academic fields
        # Title
        best_title = max((r.get("Title") or r.get("title") or "" for r in dict_records), key=len)
        merged["Title"] = best_title

        # DOI
        best_doi = ""
        for r in dict_records:
            norm = normalize_doi(str(r.get("DOI") or ""))
            if norm:
                best_doi = norm
                break
        merged["DOI"] = best_doi

        # Abstract Note / abstract
        best_abstract = max(
            (str(r.get("Abstract Note") or r.get("abstract") or "") for r in dict_records),
            key=len,
        )
        if "Abstract Note" in all_keys or "abstract" not in all_keys:
            merged["Abstract Note"] = best_abstract
        else:
            merged["abstract"] = best_abstract

        # Year
        years = []
        for r in dict_records:
            y = r.get("Publication Year") if "Publication Year" in r else r.get("year")
            try:
                if y is not None and not pd.isna(y):
                    years.append(int(y))
            except (ValueError, TypeError):
                pass
        year_val = min(years) if years else None
        if "Publication Year" in all_keys:
            merged["Publication Year"] = year_val
        elif "year" in all_keys:
            merged["year"] = year_val

        # Citations
        citations_list = []
        for r in dict_records:
            c = r.get("citations") if "citations" in r else r.get("citation_count")
            try:
                if c is not None and not pd.isna(c):
                    citations_list.append(int(c))
            except (ValueError, TypeError):
                pass
        max_c = max(citations_list) if citations_list else 0
        if "citations" in all_keys:
            merged["citations"] = max_c
        elif "citation_count" in all_keys:
            merged["citation_count"] = max_c

        # Copy any other attributes not yet handled
        handled = {"Title", "title", "DOI", "Abstract Note", "abstract", "Publication Year", "year", "citations", "citation_count"}
        for k in all_keys - handled:
            for r in dict_records:
                val = r.get(k)
                if val is not None and not pd.isna(val) and val != "":
                    merged[k] = val
                    break
            if k not in merged:
                merged[k] = None

        return merged

    def deduplicate_dataframe(
        self,
        df: pd.DataFrame,
        doi_col: str = "DOI",
        title_col: str = "Title",
        year_col: str = "Publication Year",
        abstract_col: str = "Abstract Note",
    ) -> pd.DataFrame:
        """Deduplicate a DataFrame using Disjoint Set Union (DSU) clustering."""
        if df.empty:
            return df.copy()

        n = len(df)
        parent = list(range(n))

        def find(i: int) -> int:
            if parent[i] == i:
                return i
            parent[i] = find(parent[i])
            return parent[i]

        def union(i: int, j: int) -> None:
            root_i = find(i)
            root_j = find(j)
            if root_i != root_j:
                parent[root_i] = root_j

        # Fast DOI index to avoid O(N^2) comparisons where DOIs match
        doi_map: dict[str, int] = {}
        for i in range(n):
            raw_doi = df.iloc[i].get(doi_col)
            doi = normalize_doi(str(raw_doi or ""))
            if doi:
                if doi in doi_map:
                    union(doi_map[doi], i)
                else:
                    doi_map[doi] = i

        # Pairwise title comparison for remaining elements
        for i in range(n):
            for j in range(i + 1, n):
                if find(i) != find(j):
                    if self.are_duplicates(df.iloc[i], df.iloc[j]):
                        union(i, j)

        # Group indices by cluster root
        clusters: dict[int, list[int]] = {}
        for i in range(n):
            root = find(i)
            clusters.setdefault(root, []).append(i)

        # Build merged rows
        merged_rows = []
        for root, indices in clusters.items():
            if len(indices) == 1:
                row_dict = df.iloc[indices[0]].to_dict()
                # Normalize DOI in output
                if doi_col in row_dict:
                    norm = normalize_doi(str(row_dict[doi_col] or ""))
                    if norm:
                        row_dict[doi_col] = norm
                merged_rows.append(row_dict)
            else:
                cluster_rows = [df.iloc[idx] for idx in indices]
                merged_rows.append(self.merge_records(cluster_rows))

        result_df = pd.DataFrame(merged_rows)
        # Preserve original column order where possible
        cols = [c for c in df.columns if c in result_df.columns] + [
            c for c in result_df.columns if c not in df.columns
        ]
        return result_df[cols].reset_index(drop=True)
