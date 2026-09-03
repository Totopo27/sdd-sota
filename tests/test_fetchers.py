"""Tests for fetchers module."""

from litreview.fetchers.base import Fetcher
from litreview.fetchers.zotero import ZoteroFetcher


class TestFetcher:
    def test_is_abstract(self):
        try:
            Fetcher()
        except TypeError:
            pass  # Expected
        else:
            assert False, "Expected TypeError for abstract class"


class TestZoteroFetcher:
    def test_init(self):
        fetcher = ZoteroFetcher(library_id="abc123", api_key="xyz789")
        assert fetcher._library_type == "user"

    def test_to_dataframe_basic(self):
        """Test that _to_dataframe correctly extracts fields from Zotero items."""
        fetcher = ZoteroFetcher(library_id="abc123", api_key="xyz789")
        df = fetcher._to_dataframe([
            {
                "key": "abc123",
                "data": {
                    "title": "Test Paper",
                    "abstractNote": "This is a test abstract.",
                    "date": "2024-01-01",
                    "itemType": "journalArticle",
                    "extra": "doi = 10.1234/test",
                },
            }
        ])
        assert len(df) == 1
        assert df.iloc[0]["Title"] == "Test Paper"
        assert df.iloc[0]["Abstract Note"] == "This is a test abstract."
        assert df.iloc[0]["Publication Year"] == 2024
        assert df.iloc[0]["DOI"] == "10.1234/test"
        assert df.iloc[0]["Item Type"] == "journalArticle"

    def test_to_dataframe_filters_non_paper_types(self):
        """Test that non-paper item types (attachments, thesis, etc.) are excluded."""
        fetcher = ZoteroFetcher(library_id="abc123", api_key="xyz789")
        df = fetcher._to_dataframe([
            {
                "key": "att1",
                "data": {
                    "title": "Attached PDF",
                    "abstractNote": "",
                    "itemType": "attachment",
                },
            },
            {
                "key": "thesis1",
                "data": {
                    "title": "Master's Thesis",
                    "abstractNote": "Thesis abstract",
                    "itemType": "thesis",
                },
            },
            {
                "key": "paper1",
                "data": {
                    "title": "Journal Article",
                    "abstractNote": "Journal abstract",
                    "itemType": "journalArticle",
                },
            },
            {
                "key": "conf1",
                "data": {
                    "title": "Conference Paper",
                    "abstractNote": "Conference abstract",
                    "itemType": "conferencePaper",
                },
            },
        ])
        assert len(df) == 2
        types = set(df["Item Type"])
        assert types == {"journalArticle", "conferencePaper"}

    def test_to_dataframe_keeps_duplicate_titles_for_pipeline_dedup(self):
        """Test that items with same DOI but different titles are both kept.

        Deduplication is handled by the pipeline (_clean), not the fetcher,
        because papers may share DOIs across collections and the pipeline
        needs to see all of them to dedup by normalized title.
        """
        fetcher = ZoteroFetcher(library_id="abc123", api_key="xyz789")
        df = fetcher._to_dataframe([
            {
                "key": "item1",
                "data": {
                    "title": "Paper A (WoS)",
                    "abstractNote": "Abstract from Web of Science",
                    "itemType": "journalArticle",
                    "extra": "doi = 10.1234/test",
                },
            },
            {
                "key": "item2",
                "data": {
                    "title": "Paper A (Scopus)",
                    "abstractNote": "Abstract from Scopus",
                    "itemType": "journalArticle",
                    "extra": "doi = 10.1234/test",
                },
            },
        ])
        assert len(df) == 2

    def test_to_dataframe_drops_empty_abstracts(self):
        """Test that items with empty abstracts are dropped."""
        fetcher = ZoteroFetcher(library_id="abc123", api_key="xyz789")
        df = fetcher._to_dataframe([
            {
                "key": "paper1",
                "data": {
                    "title": "Paper With Abstract",
                    "abstractNote": "Real abstract",
                    "itemType": "journalArticle",
                },
            },
            {
                "key": "paper2",
                "data": {
                    "title": "Paper Without Abstract",
                    "abstractNote": "",
                    "itemType": "journalArticle",
                },
            },
        ])
        assert len(df) == 1
        assert df.iloc[0]["Title"] == "Paper With Abstract"

    def test_to_returns_empty_df(self, sample_df):
        """Test that fetch returns an empty DataFrame when no results."""
        fetcher = ZoteroFetcher(library_id="abc123", api_key="xyz789")
        df = fetcher._to_dataframe([])
        assert len(df) == 0
