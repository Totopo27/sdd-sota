"""Parquet / Columnar DataFrame fetcher for local datasets and high performance storage."""

import re
from pathlib import Path
import pandas as pd

from litreview.fetchers.base import Fetcher


class ParquetFetcher(Fetcher):
    """Fetch papers from a local Apache Parquet file.

    Supports custom column mappings and normalizes them to the
    standard litreview schema:
    Title, Abstract Note, Publication Year, Source, Item Type, DOI
    """

    def __init__(self, filepath: str | Path):
        self.filepath = Path(filepath)
        if not self.filepath.exists():
            raise FileNotFoundError(f"Parquet file not found: {self.filepath}")

    def fetch(self, collection_names: list[str] | None = None) -> pd.DataFrame:
        """Read Parquet file and normalize to litreview format."""
        df = pd.read_parquet(self.filepath, engine="pyarrow")

        # Standardize column names (case-insensitive mapping)
        col_map = {}
        for col in df.columns:
            lower = str(col).strip().lower()
            if lower in ("title", "paper_title", "article_title"):
                col_map[col] = "Title"
            elif lower in ("abstract note", "abstract", "summary", "description"):
                col_map[col] = "Abstract Note"
            elif lower in ("publication year", "year", "date", "pub_year"):
                col_map[col] = "Publication Year"
            elif lower in ("source", "journal", "publisher", "database"):
                col_map[col] = "Source"
            elif lower in ("item type", "type", "document_type"):
                col_map[col] = "Item Type"
            elif lower in ("doi", "article_doi", "identifier"):
                col_map[col] = "DOI"

        df = df.rename(columns=col_map)

        # Ensure required columns exist
        if "Title" not in df.columns:
            raise ValueError("Parquet file must contain a 'Title' or 'title' column.")
        if "Abstract Note" not in df.columns:
            raise ValueError("Parquet file must contain an 'Abstract Note' or 'abstract' column.")

        if "Publication Year" not in df.columns:
            df["Publication Year"] = None
        else:
            def _parse_year(val):
                if pd.isna(val):
                    return None
                m = re.search(r"\d{4}", str(val))
                return int(m.group()) if m else None

            df["Publication Year"] = df["Publication Year"].apply(_parse_year)

        if "Source" not in df.columns:
            df["Source"] = "Parquet"
        if "Item Type" not in df.columns:
            df["Item Type"] = "journalArticle"
        if "DOI" not in df.columns:
            df["DOI"] = None

        df = df[df["Abstract Note"].astype(str).str.strip() != ""].copy()
        return df[["Title", "Abstract Note", "Publication Year", "Source", "Item Type", "DOI"]]
