"""Zotero API client for writing collections and populating items."""

import logging
import os
import time
from typing import Any
from pyzotero import zotero

logger = logging.getLogger(__name__)


class ZoteroPopulator:
    """Creates collections and injects scholarly papers directly into Zotero via API."""

    def __init__(
        self,
        library_id: str | None = None,
        api_key: str | None = None,
        library_type: str | None = None,
    ):
        self.library_id = library_id or os.getenv("ZOTERO_LIBRARY_ID")
        self.api_key = api_key or os.getenv("ZOTERO_API_KEY")
        self.library_type = library_type or os.getenv("ZOTERO_LIBRARY_TYPE", "user")

        if not self.library_id or not self.api_key:
            raise ValueError(
                "Zotero credentials missing. Please set ZOTERO_LIBRARY_ID and ZOTERO_API_KEY in .env"
            )

        self.zot = zotero.Zotero(self.library_id, self.library_type, self.api_key)

    def get_or_create_collection(self, collection_name: str) -> str:
        """Find an existing collection by name, or create it if absent.

        Returns:
            Collection key string (e.g. 'ARX8AAVN')
        """
        collection_name = collection_name.strip()
        existing = self.zot.collections()
        for col in existing:
            if col["data"]["name"].lower() == collection_name.lower():
                logger.info(f"Using existing Zotero collection '{collection_name}' (key: {col['key']})")
                return col["key"]

        # Create new collection
        logger.info(f"Creating new Zotero collection: '{collection_name}'")
        res = self.zot.create_collections([{"name": collection_name}])
        if "successful" in res and "0" in res["successful"]:
            key = res["successful"]["0"]["key"]
            logger.info(f"Created collection '{collection_name}' with key: {key}")
            return key
        else:
            raise RuntimeError(f"Failed to create Zotero collection '{collection_name}': {res}")

    def populate_papers(
        self,
        papers: list[dict[str, Any]],
        collection_name: str,
        batch_size: int = 10,
    ) -> int:
        """Create Zotero journalArticle items under the specified collection.

        Args:
            papers: List of paper dicts with keys: title, abstract, year, doi, authors
            collection_name: Target collection name
            batch_size: Number of items to send per Zotero API call (max 50)

        Returns:
            Number of successfully inserted items
        """
        if not papers:
            return 0

        collection_key = self.get_or_create_collection(collection_name)
        template = self.zot.item_template("journalArticle")

        items_to_create = []
        for p in papers:
            item = template.copy()
            item["title"] = p.get("title", "Untitled")
            item["abstractNote"] = p.get("abstract", "")
            item["date"] = str(p.get("year", "")) if p.get("year") else ""
            item["DOI"] = p.get("doi", "") or ""
            item["publicationTitle"] = p.get("venue", "") or ""
            item["collections"] = [collection_key]

            # Authors
            creators = []
            for author_name in p.get("authors", []):
                parts = author_name.strip().split()
                if len(parts) > 1:
                    first = " ".join(parts[:-1])
                    last = parts[-1]
                else:
                    first = ""
                    last = author_name.strip()
                creators.append({
                    "creatorType": "author",
                    "firstName": first,
                    "lastName": last,
                })
            item["creators"] = creators
            items_to_create.append(item)

        total_inserted = 0
        for i in range(0, len(items_to_create), batch_size):
            chunk = items_to_create[i : i + batch_size]
            try:
                res = self.zot.create_items(chunk)
                successful = res.get("successful", {})
                inserted_count = len(successful)
                total_inserted += inserted_count
                logger.info(f"Inserted batch {i // batch_size + 1}: {inserted_count}/{len(chunk)} items into '{collection_name}'")
                time.sleep(0.5)  # Be polite to Zotero API
            except Exception as e:
                logger.error(f"Error inserting items batch to Zotero: {e}")

        return total_inserted
