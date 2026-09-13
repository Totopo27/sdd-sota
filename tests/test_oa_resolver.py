"""Unit tests for Open Access (OA) Resolver (Europe PMC and Unpaywall)."""

from unittest.mock import MagicMock, patch
import pandas as pd
import pytest
import requests

from litreview.discovery.oa_resolver import OAResolver, normalize_doi


def test_normalize_doi():
    assert normalize_doi("https://doi.org/10.1038/s41586-020-2649-2") == "10.1038/s41586-020-2649-2"
    assert normalize_doi("http://dx.doi.org/10.1016/j.cell.2020.08.001") == "10.1016/j.cell.2020.08.001"
    assert normalize_doi("doi: 10.1145/3372278.3390678") == "10.1145/3372278.3390678"
    assert normalize_doi("  10.1109/ICSE.2021.0001  ") == "10.1109/ICSE.2021.0001"
    assert normalize_doi("not-a-doi") is None
    assert normalize_doi(None) is None
    assert normalize_doi("") is None


@patch("litreview.discovery.oa_resolver.requests.get")
def test_resolve_by_doi_europe_pmc_success(mock_get):
    # Europe PMC returns abstractText
    epmc_response = MagicMock()
    epmc_response.status_code = 200
    epmc_response.json.return_value = {
        "resultList": {
            "result": [
                {
                    "doi": "10.1038/s41586-020-2649-2",
                    "title": "Language Models are Few-Shot Learners",
                    "abstractText": "Recent work demonstrates substantial gains on many NLP tasks and benchmarks.",
                    "isOpenAccess": "Y",
                    "fullTextUrlList": {
                        "fullTextUrl": [
                            {"documentStyle": "pdf", "url": "https://europepmc.org/articles/PMC1234?pdf=render"}
                        ]
                    }
                }
            ]
        }
    }
    mock_get.return_value = epmc_response

    resolver = OAResolver()
    result = resolver.resolve_by_doi("10.1038/s41586-020-2649-2")

    assert result["abstract"] == "Recent work demonstrates substantial gains on many NLP tasks and benchmarks."
    assert result["is_oa"] is True
    assert result["oa_url"] == "https://europepmc.org/articles/PMC1234?pdf=render"
    assert result["source"] == "europe_pmc"


@patch("litreview.discovery.oa_resolver.requests.get")
def test_resolve_by_doi_unpaywall_fallback(mock_get):
    # First call to Europe PMC returns no abstract
    epmc_response = MagicMock()
    epmc_response.status_code = 200
    epmc_response.json.return_value = {"resultList": {"result": []}}

    # Second call to Unpaywall returns OA URL and metadata
    unpaywall_response = MagicMock()
    unpaywall_response.status_code = 200
    unpaywall_response.json.return_value = {
        "is_oa": True,
        "oa_status": "gold",
        "best_oa_location": {
            "url_for_pdf": "https://arxiv.org/pdf/2005.14165.pdf",
            "url": "https://arxiv.org/abs/2005.14165",
        },
    }

    mock_get.side_effect = [epmc_response, unpaywall_response]

    resolver = OAResolver(email="test@example.com")
    result = resolver.resolve_by_doi("10.1145/3372278.3390678")

    assert result["is_oa"] is True
    assert result["oa_status"] == "gold"
    assert result["oa_url"] == "https://arxiv.org/pdf/2005.14165.pdf"
    assert result["source"] == "unpaywall"


@patch("litreview.discovery.oa_resolver.requests.get")
def test_resolve_graceful_network_failure(mock_get):
    mock_get.side_effect = requests.exceptions.RequestException("Connection timeout")

    resolver = OAResolver()
    result = resolver.resolve_by_doi("10.1234/nonexistent")

    assert result["abstract"] is None
    assert result["is_oa"] is False
    assert result["oa_url"] is None
    assert result["source"] is None


@patch.object(OAResolver, "resolve_by_doi")
def test_enrich_dataframe(mock_resolve):
    df = pd.DataFrame([
        {
            "Title": "Paper with existing abstract",
            "Abstract Note": "This is a full existing abstract that is long enough.",
            "DOI": "10.1111/abc",
        },
        {
            "Title": "Paper without abstract",
            "Abstract Note": "",
            "DOI": "10.2222/def",
        },
        {
            "Title": "Paper with null abstract",
            "Abstract Note": None,
            "DOI": "10.3333/ghi",
        },
    ])

    mock_resolve.side_effect = [
        {
            "abstract": "Recovered abstract for paper 2.",
            "is_oa": True,
            "oa_url": "https://oa.org/pdf2",
            "oa_status": "gold",
            "source": "europe_pmc",
        },
        {
            "abstract": None,
            "is_oa": False,
            "oa_url": None,
            "oa_status": "closed",
            "source": None,
        },
    ]

    resolver = OAResolver()
    enriched = resolver.enrich_dataframe(df)

    assert enriched.loc[0, "Abstract Note"] == "This is a full existing abstract that is long enough."
    assert enriched.loc[1, "Abstract Note"] == "Recovered abstract for paper 2."
    assert bool(enriched.loc[1, "is_oa"]) is True
    assert enriched.loc[1, "oa_url"] == "https://oa.org/pdf2"
    assert pd.isna(enriched.loc[2, "Abstract Note"]) or enriched.loc[2, "Abstract Note"] == ""
    assert mock_resolve.call_count == 2
