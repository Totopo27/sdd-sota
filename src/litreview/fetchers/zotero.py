"""Zotero API client for fetching papers."""

from pyzotero import zotero
import pandas as pd
import re
import sys

from litreview.fetchers.base import Fetcher


class ZoteroFetcher(Fetcher):
    """Fetch papers from a Zotero library or collection."""

    def __init__(self, library_id: str, api_key: str, library_type: str = "user"):
        self.zot = zotero.Zotero(library_id, library_type, api_key)
        self._library_type = library_type

    def _get_collection_id(self, name: str) -> str:
        """Find a collection key by its name."""
        collections = self.zot.collections()
        for col in collections:
            if col["data"]["name"].lower() == name.lower():
                return col["key"]

        available = [col["data"]["name"] for col in collections]
        raise ValueError(
            f"Collection '{name}' not found.\n"
            f"Available collections: {', '.join(available) if available else 'None'}"
        )

    def fetch(self, collection_names: list[str] | None = None) -> pd.DataFrame:
        """Fetch items from one or more Zotero collections.

        Args:
            collection_names: Optional list of collection names to fetch from.
                If None, fetches all top-level items.
                If a single name is provided, fetches from that collection.

        Returns:
            DataFrame with columns: Title, Abstract Note, Publication Year,
            Source, Item Type
        """
        try:
            if collection_names:
                all_items: list = []
                for name in collection_names:
                    coll_id = self._get_collection_id(name)
                    print(f"Fetching items from collection: {name} ({coll_id})")
                    items = self.zot.everything(self.zot.collection_items(coll_id))
                    all_items.extend(items)
                return self._to_dataframe(all_items)
            else:
                print("Fetching all top-level items from library...")
                items = self.zot.everything(self.zot.top())
                return self._to_dataframe(items)
        except Exception as e:
            err_msg = str(e)
            if "Invalid user ID" in err_msg or "400" in err_msg:
                print(
                    "\nError: Invalid Zotero User ID or Library ID.\n"
                    "Please ensure 'library_id' in config.yaml is your NUMERIC Zotero ID.\n"
                    "You can find your numeric ID at https://www.zotero.org/settings/keys"
                )
            raise

    # Only these item types are scholarly works with abstracts
    _PAPER_TYPES = {"journalArticle", "conferencePaper"}

    @staticmethod
    def _to_dataframe(items: list) -> pd.DataFrame:
        """Convert Zotero API items to a DataFrame.

        Filters out non-paper item types (attachments, notes, etc.),
        then drops rows with empty abstracts.
        """
        processed_data = []

        for item in items:
            data = item.get("data", {})
            if "title" not in data:
                continue

            # Only include scholarly work types
            item_type = data.get("itemType", "")
            if item_type not in ZoteroFetcher._PAPER_TYPES:
                continue

            full_date = data.get("date", "")
            year = None
            if full_date:
                match = re.search(r"\d{4}", full_date)
                if match:
                    year = int(match.group())

            # Extract DOI directly from item data or extra field
            doi = data.get("DOI") or None
            extra = data.get("extra", "")
            if not doi and extra:
                doi_match = re.search(r"doi\s*[:=]\s*(10\.\S+)", extra, re.IGNORECASE)
                if doi_match:
                    doi = doi_match.group(1)

            url = data.get("url", "")

            processed_data.append({
                "Title": data.get("title", ""),
                "Abstract Note": data.get("abstractNote", ""),
                "Publication Year": year,
                "DOI": doi,
                "Url": url,
                "Source": "Zotero",
                "Item Type": item_type,
            })


        df = pd.DataFrame(processed_data)
        if not df.empty:
            df = df[df["Abstract Note"].str.strip() != ""].copy()

        return df
