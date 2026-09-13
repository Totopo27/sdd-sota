"""Unit tests for Cross-Source Deduplication with Year-Slack."""

import pandas as pd
import pytest

from litreview.discovery.dedup import (
    PaperDeduplicator,
    normalize_title,
    title_token_similarity,
)


def test_normalize_title():
    raw_1 = "Attention Is All You Need!"
    raw_2 = "attention is all you need"
    assert normalize_title(raw_1) == normalize_title(raw_2)

    raw_punct = "BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding"
    norm = normalize_title(raw_punct)
    assert "bert" in norm
    assert "transformers" in norm
    assert ":" not in norm
    assert "of" not in norm  # Stopword removed


def test_title_token_similarity():
    t1 = "Retrieval Augmented Generation for Knowledge-Intensive NLP Tasks"
    t2 = "Retrieval-Augmented Generation for Knowledge Intensive NLP Tasks"
    sim = title_token_similarity(t1, t2)
    assert sim == 1.0

    t3 = "Completely Different Topic in Computer Vision"
    assert title_token_similarity(t1, t3) < 0.2


def test_dedup_by_doi():
    dedup = PaperDeduplicator()
    p1 = {"Title": "Paper A", "DOI": "https://doi.org/10.1000/182", "Publication Year": 2023}
    p2 = {"Title": "Paper A (Reprint)", "DOI": "10.1000/182", "Publication Year": 2024}
    assert dedup.are_duplicates(p1, p2) is True


def test_dedup_by_title_with_year_slack():
    dedup = PaperDeduplicator(year_slack=1)
    # ArXiv preprint in 2023, Conference publication in 2024
    p1 = {
        "Title": "Deep Residual Learning for Image Recognition",
        "DOI": "",
        "Publication Year": 2023,
    }
    p2 = {
        "Title": "Deep Residual Learning for Image Recognition.",
        "DOI": None,
        "Publication Year": 2024,
    }
    # With year_slack=1, 2023 vs 2024 is accepted
    assert dedup.are_duplicates(p1, p2) is True

    # With year_slack=1, 2021 vs 2024 exceeds slack -> not duplicate
    p3 = {
        "Title": "Deep Residual Learning for Image Recognition",
        "Publication Year": 2021,
    }
    assert dedup.are_duplicates(p1, p3) is False


def test_merge_records():
    dedup = PaperDeduplicator()
    r1 = {
        "Title": "Generative Agents: Interactive Simulacra of Human Behavior",
        "DOI": "",
        "Abstract Note": "Short abstract",
        "Publication Year": 2023,
        "citations": 150,
    }
    r2 = {
        "Title": "Generative Agents: Interactive Simulacra of Human Behavior",
        "DOI": "10.1145/3586183.3606763",
        "Abstract Note": "Full detailed abstract describing believable proxies of human behavior...",
        "Publication Year": 2023,
        "citations": 500,
    }

    merged = dedup.merge_records([r1, r2])
    assert merged["DOI"] == "10.1145/3586183.3606763"
    assert "believable proxies" in merged["Abstract Note"]
    assert merged["citations"] == 500
    assert merged["Publication Year"] == 2023


def test_deduplicate_dataframe():
    df = pd.DataFrame([
        {
            "Title": "Paper Alpha",
            "DOI": "10.1001/alpha",
            "Abstract Note": "Short",
            "Publication Year": 2022,
            "citations": 10,
        },
        {
            "Title": "Paper Alpha",
            "DOI": "https://doi.org/10.1001/alpha",
            "Abstract Note": "Longer and more detailed abstract note for Paper Alpha.",
            "Publication Year": 2022,
            "citations": 25,
        },
        {
            "Title": "Paper Beta: A Novel Method",
            "DOI": None,
            "Abstract Note": "Abstract Beta",
            "Publication Year": 2023,
            "citations": 5,
        },
        {
            "Title": "Paper Beta A Novel Method",
            "DOI": "",
            "Abstract Note": "Abstract Beta",
            "Publication Year": 2024,  # Within year_slack=1
            "citations": 8,
        },
        {
            "Title": "Paper Gamma: Completely Unrelated",
            "DOI": "10.1001/gamma",
            "Abstract Note": "Abstract Gamma",
            "Publication Year": 2021,
            "citations": 100,
        },
    ])

    dedup = PaperDeduplicator(year_slack=1)
    deduped_df = dedup.deduplicate_dataframe(df)

    assert len(deduped_df) == 3
    # Check Paper Alpha merged
    alpha_row = deduped_df[deduped_df["DOI"] == "10.1001/alpha"].iloc[0]
    assert alpha_row["citations"] == 25
    assert "Longer and more detailed" in alpha_row["Abstract Note"]

    # Check Paper Beta merged with highest citations
    beta_row = deduped_df[deduped_df["Title"].str.contains("Paper Beta")].iloc[0]
    assert beta_row["citations"] == 8
