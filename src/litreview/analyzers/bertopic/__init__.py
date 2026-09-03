"""Composable BERTopic analysis modules.

Each module receives a fitted BERTopic model on __init__ and performs
a specific analysis. The pipeline orchestrates which ones to run
based on the config file.

Usage:
    fitter = BERTopicFitter(config)
    model = fitter.fit(texts)          # returns fitted model

    dist = TopicDistributionAnalyzer(model, config)
    dist_df = dist.fit_transform(texts)

    words = TopicWordExtractor(model)
    words_results = words.results

    reps = TopicRepresentativeDocs(model)
    reps_results = reps.results
"""

from litreview.analyzers.bertopic.fitter import BERTopicFitter
from litreview.analyzers.bertopic.distribution import TopicDistributionAnalyzer
from litreview.analyzers.bertopic.words import TopicWordExtractor
from litreview.analyzers.bertopic.representatives import TopicRepresentativeDocs

__all__ = [
    "BERTopicFitter",
    "TopicDistributionAnalyzer",
    "TopicWordExtractor",
    "TopicRepresentativeDocs",
]
