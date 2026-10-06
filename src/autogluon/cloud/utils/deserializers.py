import io
from abc import ABC, abstractmethod

import pandas as pd


class PandasDeserializeStrategy(ABC):
    @property
    @abstractmethod
    def supported_content_type(self):
        """The supported content type this strategy supports"""
        raise NotImplementedError

    @abstractmethod
    def deserialize(self, stream) -> pd.DataFrame:
        """Deserialize data from stream to a ``pd.DataFrame``

        Args:
            stream (botocore.response.StreamingBody): Data to be deserialized.

        Returns:
            pd.DataFrame: The data deserialized into a ``pd.DataFrame``.
        """
        raise NotImplementedError


class ParquetPandasDeserializeStrategy(PandasDeserializeStrategy):
    @property
    def supported_content_type(self):
        return "application/x-parquet"

    def deserialize(self, stream) -> pd.DataFrame:
        return pd.read_parquet(io.BytesIO(stream.read()))


class CSVPandasDeserializeStrategy(PandasDeserializeStrategy):
    @property
    def supported_content_type(self):
        return "text/csv"

    def deserialize(self, stream) -> pd.DataFrame:
        return pd.read_csv(stream)


class JsonPandasDeserializeStrategy(PandasDeserializeStrategy):
    @property
    def supported_content_type(self):
        return "application/json"

    def deserialize(self, stream) -> pd.DataFrame:
        return pd.read_json(stream)


class PandasDeserializeStrategyFactory:
    __supported_strategy = [
        ParquetPandasDeserializeStrategy,
        CSVPandasDeserializeStrategy,
        JsonPandasDeserializeStrategy,
    ]
    __content_type_to_strategy = {cls().supported_content_type: cls for cls in __supported_strategy}

    @staticmethod
    def get_strategy(content_type: str) -> PandasDeserializeStrategy:
        assert content_type in PandasDeserializeStrategyFactory.__content_type_to_strategy, (
            f"{content_type} not supported"
        )
        return PandasDeserializeStrategyFactory.__content_type_to_strategy[content_type]()


class PandasDeserializer:
    """Deserialize Parquet, CSV or JSON data from an inference endpoint into a ``pd.DataFrame``."""

    def __init__(self, accept=("application/x-parquet", "text/csv", "application/json")):
        """Initialize a ``PandasDeserializer`` instance.

        Args:
            accept (union[str, tuple[str]]): The MIME type (or tuple of allowable MIME types) that
                is expected from the inference endpoint (default: ("application/x-parquet", "text/csv","application/json")).
        """
        self.accept = (accept,) if isinstance(accept, str) else tuple(accept)

    def deserialize(self, stream, content_type):
        """Deserialize CSV or JSON data from an inference endpoint into a ``pd.DataFrame``.

        If the data is JSON, the data should be formatted in the 'columns' orient.
        See https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_json.html

        Args:
            stream (botocore.response.StreamingBody): Data to be deserialized.
            content_type (str): The MIME type of the data.

        Returns:
            pd.DataFrame: The data deserialized into a ``pd.DataFrame``.
        """
        return PandasDeserializeStrategyFactory.get_strategy(content_type).deserialize(stream)
