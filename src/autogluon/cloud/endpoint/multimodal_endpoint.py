from typing import Any

import numpy as np
import pandas as pd

from autogluon.common.loaders import load_pd

from ..utils.deserializers import PandasDeserializer
from ..utils.sagemaker_api import invoke_endpoint
from ..utils.serializers import AutoGluonSerializationWrapper, MultiModalSerializer
from ..utils.utils import (
    convert_image_path_to_encoded_bytes_in_dataframe,
    is_image_file,
    read_image_bytes_and_encode,
    split_pred_and_pred_proba,
)
from .endpoint import Endpoint

DataInput = str | list[str] | pd.DataFrame
Prediction = pd.DataFrame | pd.Series


class MultiModalEndpoint(Endpoint):
    """High-level handle for an AutoGluon-Cloud multimodal endpoint.

    Returned by :meth:`autogluon.cloud.MultiModalCloudPredictor.deploy`. Construct it directly to attach to an
    existing endpoint by name.
    """

    @staticmethod
    def _load_data(data: DataInput, image_column: str | None) -> tuple[pd.DataFrame | np.ndarray, str]:
        if isinstance(data, str):
            data = [data] if is_image_file(data) else load_pd.load(data)
        if isinstance(data, list):
            encoded_images = np.array([read_image_bytes_and_encode(image) for image in data], dtype="object")
            return encoded_images, "application/x-autogluon-npy"
        if image_column is not None:
            data = convert_image_path_to_encoded_bytes_in_dataframe(dataframe=data, image_column=image_column)
        return data, "application/x-autogluon-parquet"

    def _predict(self, data: DataInput, inference_kwargs: dict[str, Any]) -> tuple[pd.Series, Prediction]:
        inference_kwargs = dict(inference_kwargs)
        data, content_type = self._load_data(data, image_column=inference_kwargs.pop("image_column", None))
        raw = invoke_endpoint(
            self._endpoint_name,
            self._session,
            AutoGluonSerializationWrapper(data=data, inference_kwargs=inference_kwargs),
            serializer=MultiModalSerializer(),
            deserializer=PandasDeserializer(),
            # The serializer's content type is fixed, so the per-request content type is passed explicitly.
            content_type=content_type,
            accept="application/x-parquet",
        )
        pred, pred_proba = split_pred_and_pred_proba(raw)
        if pred_proba is None:
            pred_proba = pred
        return pred, pred_proba

    def predict(self, data: DataInput, **inference_kwargs: Any) -> pd.Series:
        """Predict ``data`` with the deployed endpoint.

        This is intended for low-latency inference. For larger inputs, use
        :meth:`autogluon.cloud.MultiModalCloudPredictor.predict` instead.

        Parameters
        ----------
        data: str | list[str] | pd.DataFrame
            Data to predict. One of:

            * a ``pd.DataFrame`` or a local path to a data file.
            * a local path to a single image file, or a list of local paths to image files.
        **inference_kwargs: Any
            Additional args passed to the ``predict`` call of the AutoGluon predictor on the endpoint. If ``data`` has
            an image column, pass ``image_column`` to name the column with absolute paths to local images; the images
            are encoded and sent with the request.

        Returns
        -------
        pd.Series
            Predictions for ``data``.

        SageMaker API
        -------------
        * :sm-runtime-api:`InvokeEndpoint`: sends the data to the endpoint and returns the predictions. The payload is
          limited to 6 MB (4 MB for serverless endpoints).
        """
        pred, _ = self._predict(data, inference_kwargs=inference_kwargs)
        return pred

    def predict_proba(
        self,
        data: DataInput,
        *,
        include_predict: bool = True,
        **inference_kwargs: Any,
    ) -> tuple[pd.Series, Prediction] | Prediction:
        """Predict class probabilities for ``data`` with the deployed endpoint.

        For regression, the probability result is identical to the prediction.

        Parameters
        ----------
        data: str | list[str] | pd.DataFrame
            Data to predict. One of:

            * a ``pd.DataFrame`` or a local path to a data file.
            * a local path to a single image file, or a list of local paths to image files.
        include_predict: bool, default = True
            Whether to return the predictions along with the probabilities. Both are computed in the same request.
        **inference_kwargs: Any
            Additional args passed to the ``predict_proba`` call of the AutoGluon predictor on the endpoint. If ``data`` has
            an image column, pass ``image_column`` to name the column with absolute paths to local images; the images
            are encoded and sent with the request.

        Returns
        -------
        tuple[pd.Series, pd.DataFrame | pd.Series] | pd.DataFrame | pd.Series
            ``(prediction, predict_probability)`` if ``include_predict`` is True, otherwise ``predict_probability``.

        SageMaker API
        -------------
        * :sm-runtime-api:`InvokeEndpoint`: sends the data to the endpoint and returns the predictions. The payload is
          limited to 6 MB (4 MB for serverless endpoints).
        """
        pred, pred_proba = self._predict(data, inference_kwargs=inference_kwargs)
        if include_predict:
            return pred, pred_proba
        return pred_proba
