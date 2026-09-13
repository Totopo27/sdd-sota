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
    # Check that item created contains fields
    created_item = mock_zot.create_items.call_args[0][0][0]
    assert created_item["title"] == "Quantum AI"
    assert created_item["DOI"] == "10.1000/qai"


@patch("litreview.discovery.zotero_populator.zotero.Zotero")
def test_zotero_populator_oa_url_and_metadata(mock_zotero_cls):
    mock_zot = MagicMock()
    mock_zotero_cls.return_value = mock_zot
    mock_zot.collections.return_value = [{"data": {"name": "AI-Repo"}, "key": "COL123"}]
    mock_zot.item_template.return_value = {
        "itemType": "journalArticle",
        "title": "",
        "abstractNote": "",
        "url": "",
        "extra": "",
        "date": "",
        "creators": [],
    }
    mock_zot.create_items.return_value = {"successful": {"0": {"key": "K1"}}}

    populator = ZoteroPopulator(library_id="123", api_key="key")
    papers = [
        {
            "title": "Open Access Transformer",
            "abstract": "Full abstract text.",
            "doi": "10.1111/oa.trans",
            "oa_url": "https://europepmc.org/articles/PMC999.pdf",
            "is_oa": True,
            "oa_status": "gold",
        }
    ]
    inserted = populator.populate_papers(papers, collection_name="AI-Repo")
    assert inserted == 1
    created_item = mock_zot.create_items.call_args[0][0][0]
    assert created_item["url"] == "https://europepmc.org/articles/PMC999.pdf"
    assert "Open Access: gold" in created_item["extra"]


@patch("litreview.discovery.zotero_populator.zotero.Zotero")
def test_zotero_populator_get_existing_and_filter_duplicates(mock_zotero_cls):
    mock_zot = MagicMock()
    mock_zotero_cls.return_value = mock_zot
    mock_zot.collections.return_value = [{"data": {"name": "TestCol"}, "key": "COL_TEST"}]
    mock_zot.collection_items.return_value = [
        {
            "data": {
                "title": "Existing Literature Paper",
                "DOI": "10.1000/existing",
                "date": "2023",
                "abstractNote": "Existing abstract.",
            }
        },
        {
            "data": {
                "title": "Another Paper In Library",
                "DOI": "",
                "date": "2022",
                "abstractNote": "Another abstract.",
            }
        },
    ]

    populator = ZoteroPopulator(library_id="123", api_key="key")
    existing = populator.get_existing_papers("TestCol")
    assert len(existing) == 2
    assert existing[0]["title"] == "Existing Literature Paper"
    assert existing[0]["doi"] == "10.1000/existing"

    # Now test filtering candidates
    candidates = [
        {
            "title": "Existing Literature Paper (Reprint)",
            "doi": "https://doi.org/10.1000/existing",  # Duplicate by DOI
            "year": 2024,
            "abstract": "Abstract",
        },
        {
            "title": "Another Paper In Library",  # Duplicate by Title + Year slack (2022 vs 2023)
            "doi": "",
            "year": 2023,
            "abstract": "Abstract",
        },
        {
            "title": "Brand New Breakthrough Method",  # Unique
            "doi": "10.2000/new",
            "year": 2025,
            "abstract": "Brand new abstract.",
        },
    ]

    filtered, skipped_count = populator.filter_existing_duplicates(candidates, "TestCol")
    assert skipped_count == 2
    assert len(filtered) == 1
    assert filtered[0]["title"] == "Brand New Breakthrough Method"


@patch("litreview.cli.run_discover.ZoteroPopulator")
@patch("litreview.cli.run_discover.OAResolver")
@patch("litreview.cli.run_discover.OpenAlexClient")
@patch("sys.argv", ["run_discover.py", "--query", "agentic workflows", "--collection", "Agent-Col"])
def test_run_discover_cli_integration(mock_openalex_cls, mock_oa_cls, mock_populator_cls):
    from litreview.cli.run_discover import main as discover_main

    mock_client = MagicMock()
    mock_openalex_cls.return_value = mock_client
    mock_client.search_works.return_value = [
        {
            "title": "Recoverable OA Paper",
            "abstract": "",  # Missing abstract from OpenAlex
            "doi": "10.1000/recoverable",
            "year": 2024,
            "authors": ["Geoffrey Hinton"],
        },
        {
            "title": "Already Existing Paper",
            "abstract": "Full abstract text here.",
            "doi": "10.1000/existing",
            "year": 2023,
            "authors": ["Yann LeCun"],
        },
    ]

    mock_resolver = MagicMock()
    mock_oa_cls.return_value = mock_resolver
    mock_resolver.resolve_by_doi.return_value = {
        "abstract": "Recovered abstract via Europe PMC.",
        "is_oa": True,
        "oa_url": "https://europepmc.org/articles/PMC1234.pdf",
        "oa_status": "gold",
        "source": "europe_pmc",
    }

    mock_populator = MagicMock()
    mock_populator_cls.return_value = mock_populator
    mock_populator.get_existing_papers.return_value = [
        {
            "title": "Already Existing Paper",
            "doi": "10.1000/existing",
            "year": 2023,
        }
    ]
    # filter_existing_duplicates behavior
    mock_populator.filter_existing_duplicates.return_value = (
        [
            {
                "title": "Recoverable OA Paper",
                "abstract": "Recovered abstract via Europe PMC.",
                "doi": "10.1000/recoverable",
                "year": 2024,
                "authors": ["Geoffrey Hinton"],
                "is_oa": True,
                "oa_url": "https://europepmc.org/articles/PMC1234.pdf",
                "oa_status": "gold",
            }
        ],
        1,  # skipped count
    )
    mock_populator.populate_papers.return_value = 1

    discover_main()

    mock_resolver.resolve_by_doi.assert_any_call("10.1000/recoverable")
    assert mock_resolver.resolve_by_doi.call_count == 2
    mock_populator.filter_existing_duplicates.assert_called_once()
    assert mock_populator.populate_papers.call_count == 1
    populated_papers = mock_populator.populate_papers.call_args[0][0]
    assert len(populated_papers) == 1
    assert populated_papers[0]["title"] == "Recoverable OA Paper"
    assert populated_papers[0]["abstract"] == "Recovered abstract via Europe PMC."
    assert populated_papers[0]["oa_url"] == "https://europepmc.org/articles/PMC1234.pdf"


