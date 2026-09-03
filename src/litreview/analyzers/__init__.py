"""BERTopic analyzers — composable modules and legacy wrapper."""

from litreview.analyzers.base import Analyzer
from litreview.analyzers.bertopic import (
    BERTopicFitter,
    TopicDistributionAnalyzer,
    TopicWordExtractor,
    TopicRepresentativeDocs,
)
from litreview.analyzers.bertopic_wrapper import BERTopicAnalyzer  # deprecated wrapper
from litreview.analyzers.zeroshot import ZeroShotAnalyzer

__all__ = [
    "Analyzer",
    "BERTopicFitter",
    "TopicDistributionAnalyzer",
    "TopicWordExtractor",
    "TopicRepresentativeDocs",
    "BERTopicAnalyzer",  # deprecated — use BERTopicFitter + analysis modules
    "ZeroShotAnalyzer",
]
