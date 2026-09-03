"""Base class for data fetchers."""

from abc import ABC, abstractmethod
import pandas as pd


class Fetcher(ABC):
    """Abstract base class for data fetchers."""

    @abstractmethod
    def fetch(self, collection_names: list[str] | None = None) -> pd.DataFrame:
        """Fetch data and return as DataFrame.

        Expected columns: Title, Abstract Note, Publication Year, Source, Item Type
        """
        ...
