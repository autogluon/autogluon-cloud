import base64
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

AUTOGLUON_SERDE_VERSION = 1


def _dataframe_to_b64(df: pd.DataFrame) -> str:
    return base64.b64encode(df.to_parquet()).decode("ascii")


def _ensure_json_serializable(inference_kwargs: dict[str, Any]) -> None:
    try:
        json.dumps(inference_kwargs)
    except (TypeError, ValueError) as e:
        raise ValueError(
            "`inference_kwargs` must be JSON-serializable; got value that cannot be encoded as JSON."
        ) from e


@dataclass
class AutoGluonSerializationWrapper:
    """Container for data, inference kwargs, and optional side inputs to be serialized into a single request payload."""

    data: pd.DataFrame
    inference_kwargs: dict[str, Any]
    train_data: pd.DataFrame | None = field(default=None)
    static_features: pd.DataFrame | None = field(default=None)
    known_covariates: pd.DataFrame | None = field(default=None)


class AutoGluonSerializer:
    """Serialize data to a buffer with data itself and optional AutoGluon inference arguments."""

    def __init__(self, content_type="application/x-autogluon"):
        """Initialize a ``AutoGluonSerializer`` instance.

        Parameters
        ----------
        content_type: str, default = "application/x-autogluon"
            The MIME type to signal to the inference endpoint when sending request data.
        """
        self.content_type = content_type

    def serialize(self, data: AutoGluonSerializationWrapper):
        """Serialize data to a JSON envelope with base64-encoded parquet payloads.

        Parameters
        ----------
        data: AutoGluonSerializationWrapper
            Data to be serialized.

        Returns
        -------
        bytes
            UTF-8 JSON containing base64-encoded parquet bytes and inference args.
        """
        if not isinstance(data, AutoGluonSerializationWrapper):
            raise ValueError(f"{data} format is not supported. Please provide a `AutoGluonSerializationWrapper`.")

        inference_kwargs = data.inference_kwargs or {}
        _ensure_json_serializable(inference_kwargs)
        package = {
            "version": AUTOGLUON_SERDE_VERSION,
            "data": _dataframe_to_b64(data.data),
            "inference_kwargs": inference_kwargs,
        }
        if data.train_data is not None:
            package["train_data"] = _dataframe_to_b64(data.train_data)
        if data.static_features is not None:
            package["static_features"] = _dataframe_to_b64(data.static_features)
        if data.known_covariates is not None:
            package["known_covariates"] = _dataframe_to_b64(data.known_covariates)
        return json.dumps(package).encode("utf-8")


class MultiModalSerializer:
    """Serializer for multi-modal use case.

    Produces a JSON envelope containing either base64-encoded parquet (for ``pd.DataFrame`` objects) or a
    JSON list of base85-encoded image strings (for numpy arrays), plus inference kwargs.
    """

    def __init__(self, content_type="application/x-autogluon-parquet"):
        """Initialize a ``MultiModalSerializer`` instance.

        Parameters
        ----------
        content_type: str, default = "application/x-autogluon-parquet"
            The MIME type to signal to the inference endpoint when sending request data.
            Requests with image data pass their own content type to the endpoint call instead.
        """
        self.content_type = content_type

    def serialize(self, data):
        """Serialize data to a JSON envelope.

        For ``pd.DataFrame`` inputs, ``data`` is base64-encoded parquet bytes.
        For numpy/list image inputs, ``data`` is a JSON list of base85-encoded image strings.

        Parameters
        ----------
        data: AutoGluonSerializationWrapper
            Data to be serialized. Its data can be a ``pd.DataFrame``,
            or a numpy array of base85-encoded image strings.

        Returns
        -------
        bytes
            UTF-8 JSON containing both data and inference args.
        """
        if not isinstance(data, AutoGluonSerializationWrapper):
            raise ValueError(f"{data} format is not supported. Please provide a `AutoGluonSerializationWrapper`")

        inference_kwargs = data.inference_kwargs or {}
        _ensure_json_serializable(inference_kwargs)

        if isinstance(data.data, pd.DataFrame):
            package = {
                "version": AUTOGLUON_SERDE_VERSION,
                "data": _dataframe_to_b64(data.data),
                "inference_kwargs": inference_kwargs,
            }
            return json.dumps(package).encode("utf-8")

        if isinstance(data.data, np.ndarray):
            package = {
                "version": AUTOGLUON_SERDE_VERSION,
                "data": data.data.tolist(),
                "inference_kwargs": inference_kwargs,
            }
            return json.dumps(package).encode("utf-8")

        raise ValueError(
            f"{data.data} format is not supported. Please provide a DataFrame or numpy array"
            " wrapped by `AutoGluonSerializationWrapper`."
        )
