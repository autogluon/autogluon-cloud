import pickle
from dataclasses import dataclass
from typing import Any, Dict

import numpy as np
import pandas as pd
from sagemaker.serializers import NumpySerializer, SimpleBaseSerializer


@dataclass
class AutoGluonSerializationWrapper:
    data: pd.DataFrame
    inference_kwargs: Dict[str, Any]


class ParquetSerializer(SimpleBaseSerializer):
    """Serialize data to a buffer using the .parquet format."""

    def __init__(self, content_type="application/x-parquet"):
        """Initialize a ``ParquetSerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "application/x-parquet").
        """
        super(ParquetSerializer, self).__init__(content_type=content_type)

    def serialize(self, data):
        """Serialize data to a buffer using the .parquet format.

        Args:
            data (object): Data to be serialized. Can be a Pandas Dataframe,
                file, or buffer.

        Returns:
            io.BytesIO: A buffer containing data serialized in the .parquet format.
        """
        if isinstance(data, pd.DataFrame):
            return data.to_parquet()

        # files and buffers. Assumed to hold parquet-formatted data.
        if hasattr(data, "read"):
            return data.read()

        raise ValueError(f"{data} format is not supported. Please provide a DataFrame, parquet file, or buffer.")


class AutoGluonSerializer(SimpleBaseSerializer):
    """Serialize data to a buffer with data itself and optional AutoGluon inference arguments."""

    def __init__(self, content_type="application/x-autogluon"):
        """Initialize a ``AutoGluonSerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "application/x-autogluon").
        """
        super(AutoGluonSerializer, self).__init__(content_type=content_type)
        self.parquet_serializer = ParquetSerializer()

    def serialize(self, data: AutoGluonSerializationWrapper):
        """Serialize data to a buffer using the .parquet format.

        Args:
            data (object): Data to be serialized. An AutoGluonSerializationWrapper object

        Returns:
            bytese: Bytes containing both data and inference args
        """
        if isinstance(data, AutoGluonSerializationWrapper):
            package = {"data": self.parquet_serializer.serialize(data.data), "inference_kwargs": data.inference_kwargs}
            return pickle.dumps(package)

        raise ValueError(f"{data} format is not supported. Please provide a `AutoGluonSerializationWrapper`.")


class MultiModalSerializer(SimpleBaseSerializer):
    """
    Serializer for multi-modal use case.
    When passed in a dataframe, the serializer will serialize the data to be parquet format.
    When passed in a numpy array, the serializer will serialize the data to be numpy format.
    Both allow passing optional inference arguments along with the data
    """

    def __init__(self, content_type="application/x-autogluon-parquet"):
        """Initialize a ``MultiModalSerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "application/x-autogluon-parquet").
                To BE NOTICED, this content_type will not used by MultiModalSerializer
                as it doesn't support dynamic updating. Instead, we pass expected content_type to
                `initial_args` of `predict()` call to endpoints.
        """
        super(MultiModalSerializer, self).__init__(content_type=content_type)
        self.parquet_serializer = ParquetSerializer()
        self.numpy_serializer = NumpySerializer()

    def serialize(self, data):
        """Serialize data to a buffer using the .parquet format or numpy format.

        Args:
            data (object): Data to be serialized.
                An AutoGluonSerializationWrapper, which its data can be a Pandas Dataframe,
                or a numpy array

        Returns:
            bytese: Bytes containing both data and inference args
        """
        if isinstance(data, AutoGluonSerializationWrapper):
            if isinstance(data.data, pd.DataFrame):
                package = {
                    "data": self.parquet_serializer.serialize(data.data),
                    "inference_kwargs": data.inference_kwargs,
                }
                return pickle.dumps(package)

            if isinstance(data.data, np.ndarray):
                package = {
                    "data": self.numpy_serializer.serialize(data.data),
                    "inference_kwargs": data.inference_kwargs,
                }
                return pickle.dumps(package)

            raise ValueError(
                f"{data} format is not supported. Please provide a `DataFrame, or numpy array.` being wrapped by `AutoGluonSerializationWrapper`"
            )

        raise ValueError(f"{data} format is not supported. Please provide a `AutoGluonSerializationWrapper`")


class JsonLineSerializer(SimpleBaseSerializer):
    """Serialize data to a buffer using the .jsonl format."""

    def __init__(self, content_type="application/jsonl"):
        """Initialize a ``JsonLineSerializer`` instance.

        Args:
            content_type (str): The MIME type to signal to the inference endpoint when sending
                request data (default: "application/jsonl").
        """
        super(JsonLineSerializer, self).__init__(content_type=content_type)

    def serialize(self, data):
        """Serialize data to a buffer using the .jsonl format.

        Args:
            data (pd.DataFrame): Data to be serialized.

        Returns:
            io.StringIO: A buffer containing data serialized in the .jsonl format.
        """
        if isinstance(data, pd.DataFrame):
            return data.to_json(orient="records", lines=True)

        raise ValueError(f"{data} format is not supported. Please provide a DataFrame.")
