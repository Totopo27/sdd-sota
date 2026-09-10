"""Corpus characterization statistics.

Provides metrics for understanding the corpus, topic coverage,
research gaps, and cross-method agreement.
"""

import pandas as pd
import numpy as np


def compute_corpus_stats(df: pd.DataFrame) -> dict:
    """Compute corpus-level statistics.

    Args:
        df: DataFrame with columns: Publication Year, Source, Item Type,
            Abstract Note, Title.

    Returns:
        Dict with total_papers, papers_with_abstracts, year_range,
        sources, item_types.
    """
    years = df["Publication Year"].dropna().astype(int)
    return {
        "total_papers": len(df),
        "papers_with_abstracts": int(df["Abstract Note"].notna().sum()),
        "year_range": [int(years.min()), int(years.max())] if len(years) > 0 else [None, None],
        "sources": df["Source"].value_counts().to_dict(),
        "item_types": df["Item Type"].value_counts().to_dict(),
    }


def compute_topic_coverage(
    topic_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    seed_topics: list[list[str]] | None = None,
) -> dict:
    """Compute how many papers discuss each topic.

    Args:
        topic_df: DataFrame from BERTopicAnalyzer with 'topic' column.
        validation_df: DataFrame from ZeroShotAnalyzer with 'score', 'label',
            'classified' columns.
        seed_topics: Optional list of seed topic groups.

    Returns:
        Dict with topic_sizes, outlier_count, num_topics, total_classified,
        mean_confidence, median_confidence.
    """
    topic_sizes = topic_df["topic"].value_counts().to_dict()
    outlier_count = topic_sizes.get(-1, 0)
    topic_sizes_clean = {k: v for k, v in topic_sizes.items() if k != -1}

    scores = validation_df["score"].dropna()
    classified = validation_df["classified"].sum()
    n_scores = len(scores)

    mean_c = float(scores.mean()) if n_scores > 0 else 0.0
    median_c = float(scores.median()) if n_scores > 0 else 0.0
    std_c = float(scores.std(ddof=1)) if n_scores > 1 else 0.0
    var_c = float(scores.var(ddof=1)) if n_scores > 1 else 0.0

    # 95% Confidence Interval for the mean using normal approximation
    if n_scores > 1:
        import math
        std_err = std_c / math.sqrt(n_scores)
        ci_lower = max(0.0, mean_c - 1.96 * std_err)
        ci_upper = min(1.0, mean_c + 1.96 * std_err)
        ci_95 = [round(ci_lower, 4), round(ci_upper, 4)]
    else:
        ci_95 = [round(mean_c, 4), round(mean_c, 4)]

    return {
        "topic_sizes": topic_sizes_clean,
        "outlier_count": outlier_count,
        "num_topics": len(topic_sizes_clean),
        "total_classified": int(classified),
        "mean_confidence": round(mean_c, 4),
        "median_confidence": round(median_c, 4),
        "std_confidence": round(std_c, 4),
        "variance_confidence": round(var_c, 4),
        "confidence_ci_95": ci_95,
    }


def compute_gap_analysis(
    topic_df: pd.DataFrame,
    validation_df: pd.DataFrame,
    seed_topics: list[list[str]] | None = None,
) -> dict:
    """Identify gaps: topics with few/no papers, seeds with no matches.

    Args:
        topic_df: DataFrame from BERTopicAnalyzer with 'topic' column.
        validation_df: DataFrame from ZeroShotAnalyzer with 'label' column.
        seed_topics: Optional list of seed topic groups.

    Returns:
        Dict with gaps list and num_gaps count.
    """
    gaps = []

    # Check seed topics against zero-shot results
    if seed_topics:
        labels = validation_df["label"].value_counts()
        for seed_group in seed_topics:
            seed_label = seed_group[0] if isinstance(seed_group, list) else seed_group
            count = labels.get(seed_label, 0)
            if count == 0:
                gaps.append({
                    "type": "seed_no_matches",
                    "seed": seed_label,
                    "papers": 0,
                    "severity": "high",
                })
            elif count < 5:
                gaps.append({
                    "type": "seed_few_matches",
                    "seed": seed_label,
                    "papers": int(count),
                    "severity": "medium",
                })

    # Check for large outlier groups
    outlier_count = int((topic_df["topic"] == -1).sum())
    total = len(topic_df)
    if total > 0 and outlier_count > total * 0.3:
        gaps.append({
            "type": "large_outliers",
            "description": (
                f"{outlier_count} papers ({outlier_count / total * 100:.0f}%) "
                "don't match any topic"
            ),
            "severity": "medium",
        })

    return {
        "gaps": gaps,
        "num_gaps": len(gaps),
    }


def compute_cross_analysis(
    topic_df: pd.DataFrame,
    validation_df: pd.DataFrame,
) -> dict:
    """Compare BERTopic and zero-shot results for agreement.

    Args:
        topic_df: DataFrame from BERTopicAnalyzer with 'topic' column.
        validation_df: DataFrame from ZeroShotAnalyzer with 'classified' column.

    Returns:
        Dict with bertopic_classified, zeroshot_classified,
        both_classified, neither_classified.
    """
    topic_papers = (topic_df["topic"] != -1).sum()
    classified_papers = validation_df["classified"].sum()

    return {
        "bertopic_classified": int(topic_papers),
        "zeroshot_classified": int(classified_papers),
        "both_classified": int(
            ((topic_df["topic"] != -1) & validation_df["classified"]).sum()
        ),
        "neither_classified": int(
            ((topic_df["topic"] == -1) & ~validation_df["classified"]).sum()
        ),
    }
