"""Base class for analyzers."""

from abc import ABC, abstractmethod
import pandas as pd


class Analyzer(ABC):
    """Abstract base class for analyzers.

    Analyzers follow a scikit-learn-like API:
    - fit(texts): learn from texts, return self
    - transform(texts): return results DataFrame
    - fit_transform(texts): fit + transform in one call
    - results: property returning analysis results as dict
    """

    @abstractmethod
    def fit(self, texts: pd.Series) -> "Analyzer":
        """Learn from texts. Return self for chaining."""
        ...

    @abstractmethod
    def transform(self, texts: pd.Series) -> pd.DataFrame:
        """Transform texts, return results DataFrame."""
        ...

    def fit_transform(self, texts: pd.Series) -> pd.DataFrame:
        """Fit and transform in one call."""
        return self.fit(texts).transform(texts)

    @property
    @abstractmethod
    def results(self) -> dict:
        """Return analysis results as dict."""
        ...
