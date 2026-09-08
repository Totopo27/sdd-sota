"""Unit tests for the discovery module (OpenAlex and ZoteroPopulator)."""

from unittest.mock import MagicMock, patch
from litreview.discovery.openalex import OpenAlexClient, reconstruct_abstract
from litreview.discovery.zotero_populator import ZoteroPopulator


def test_reconstruct_abstract():
    inverted_index = {
        "Retrieval-Augmented": [0],
        "Generation": [1],
        "is": [2],
        "powerful": [3],
    }
    abstract = reconstruct_abstract(inverted_index)
    assert abstract == "Retrieval-Augmented Generation is powerful"


def test_reconstruct_abstract_empty():
    assert reconstruct_abstract(None) == ""
    assert reconstruct_abstract({}) == ""


@patch("litreview.discovery.openalex.requests.get")
def test_openalex_client_search_works(mock_get):
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "results": [
            {
                "id": "https://openalex.org/W12345",
                "title": "Test Paper Title",
                "publication_year": 2024,
                "doi": "https://doi.org/10.1234/test.doi",
                "cited_by_count": 42,
                "abstract_inverted_index": {
                    "This": [0],
                    "is": [1],
                    "a": [2],
                    "test": [3],
                },
                "authorships": [
                    {"author": {"display_name": "Alan Turing"}},
                    {"author": {"display_name": "Ada Lovelace"}},
                ],
                "primary_location": {
                    "source": {"display_name": "Nature Machine Intelligence"}
                },
            }
        ]
    }
    mock_get.return_value = mock_response

    client = OpenAlexClient()
    papers = client.search_works(query="test query", limit=5)

    assert len(papers) == 1
    p = papers[0]
    assert p["title"] == "Test Paper Title"
    assert p["abstract"] == "This is a test"
    assert p["year"] == 2024
    assert p["doi"] == "10.1234/test.doi"
    assert p["authors"] == ["Alan Turing", "Ada Lovelace"]
    assert p["citations"] == 42
    assert p["venue"] == "Nature Machine Intelligence"


@patch("litreview.discovery.zotero_populator.zotero.Zotero")
def test_zotero_populator(mock_zotero_cls):
    mock_zot = MagicMock()
    mock_zotero_cls.return_value = mock_zot

    # Mock collection creation
    mock_zot.collections.return_value = []
    mock_zot.create_collections.return_value = {
        "successful": {"0": {"key": "COLL_KEY_123"}}
    }
    mock_zot.item_template.return_value = {
        "itemType": "journalArticle",
        "title": "",
        "abstractNote": "",
        "date": "",
        "creators": [],
    }
    mock_zot.create_items.return_value = {
        "successful": {"0": {"key": "ITEM_KEY_ABC"}}
    }

    populator = ZoteroPopulator(library_id="12345", api_key="dummy_key")
    papers = [
        {
            "title": "Quantum AI",
            "abstract": "Deep neural quantum models.",
            "year": 2025,
            "doi": "10.1000/qai",
            "authors": ["Richard Feynman"],
            "venue": "PRL",
        }
    ]

    inserted = populator.populate_papers(papers, collection_name="Quantum-AI")
    assert inserted == 1
    mock_zot.create_collections.assert_called_once_with([{"name": "Quantum-AI"}])
    assert mock_zot.create_items.call_count == 1
