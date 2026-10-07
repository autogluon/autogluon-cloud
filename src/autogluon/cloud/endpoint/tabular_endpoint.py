from pathlib import Path
from typing import Any

import pandas as pd

from autogluon.common.loaders import load_pd

from ..utils.deserializers import PandasDeserializer
from ..utils.sagemaker_api import invoke_endpoint
from ..utils.serializers import AutoGluonSerializationWrapper, AutoGluonSerializer
from ..utils.utils import convert_image_path_to_encoded_bytes_in_dataframe, split_pred_and_pred_proba
from .endpoint import Endpoint

DataInput = str | Path | pd.DataFrame
Prediction = pd.DataFrame | pd.Series


class TabularEndpoint(Endpoint):
    """High-level handle for an AutoGluon-Cloud tabular endpoint.

    Returned by :meth:`autogluon.cloud.TabularCloudPredictor.deploy` and
    :meth:`autogluon.cloud.TabularFoundationModel.deploy`. Construct it directly to attach to an existing endpoint by
    name.

    * **Trained predictor endpoints** (:meth:`TabularCloudPredictor.deploy`) serve the fitted predictor: pass only
      ``data``.
    * **Foundation model endpoints** (:meth:`TabularFoundationModel.deploy`) fit a request-scoped predictor on the
      labeled examples sent with each request: pass ``train_data`` and ``label`` along with ``data``.
    """

    @staticmethod
    def _load_data(data: DataInput) -> pd.DataFrame:
        if isinstance(data, (str, Path)):
            return load_pd.load(str(data))
        return data

    def _predict(
        self,
        data: DataInput,
        train_data: DataInput | None = None,
        label: str | None = None,
        inference_kwargs: dict[str, Any] | None = None,
    ) -> tuple[pd.Series, Prediction]:
        data = self._load_data(data)
        inference_kwargs = dict(inference_kwargs or {})
        self._pop_as_pandas(inference_kwargs)
        image_column = inference_kwargs.pop("image_column", None)

        if (train_data is None) != (label is None):
            raise ValueError(
                "`train_data` and `label` must be passed together: pass both for foundation model endpoints, "
                "and neither for endpoints deployed with `TabularCloudPredictor.deploy`."
            )
        if train_data is not None:
            if image_column is not None:
                raise ValueError(
                    "`image_column` is only supported by endpoints deployed with `TabularCloudPredictor.deploy`; "
                    "foundation model endpoints do not support image features."
                )
            train_data = self._load_data(train_data)
            if label not in train_data.columns:
                raise ValueError(f"Label column {label!r} is not present in `train_data`.")
            feature_columns = [column for column in train_data.columns if column != label]
            missing_columns = [column for column in feature_columns if column not in data.columns]
            if missing_columns:
                raise ValueError(f"`data` is missing feature columns present in `train_data`: {missing_columns}.")
            inference_kwargs = {"label": label, **inference_kwargs}

        if image_column is not None:
            data = convert_image_path_to_encoded_bytes_in_dataframe(data, image_column)

        payload = AutoGluonSerializationWrapper(
            data=data,
            train_data=train_data,
            inference_kwargs=inference_kwargs,
        )
        raw = invoke_endpoint(
            self._endpoint_name,
            self._session,
            payload,
            serializer=AutoGluonSerializer(),
            deserializer=PandasDeserializer(),
            accept="application/x-parquet",
        )
        pred, pred_proba = split_pred_and_pred_proba(raw)
        if pred_proba is None:
            pred_proba = pred
        return pred, pred_proba

    def predict(
        self,
        data: DataInput,
        train_data: DataInput | None = None,
        label: str | None = None,
        **inference_kwargs: Any,
    ) -> pd.Series:
        """Predict ``data`` with the deployed endpoint.

        This is intended for low-latency inference. For inputs above the payload limit, use
        :meth:`autogluon.cloud.TabularCloudPredictor.predict` or
        :meth:`autogluon.cloud.TabularFoundationModel.predict` instead.

        Parameters
        ----------
        data: str | pathlib.Path | pd.DataFrame
            Rows to predict, as a ``pd.DataFrame`` or local/S3 path to a data file.
        train_data: str | pathlib.Path | pd.DataFrame | None, default = None
            Labeled examples the foundation model is fit on. Required for foundation model endpoints; must be
            ``None`` for trained predictor endpoints.
        label: str | None, default = None
            Name of the label column in ``train_data``. Required if and only if ``train_data`` is passed.
        **inference_kwargs: Any
            Additional args passed to the ``predict`` call of the AutoGluon predictor on the endpoint.

        Returns
        -------
        pd.Series
            Predictions for ``data``.

        SageMaker API
        -------------
        * :sm-runtime-api:`InvokeEndpoint`: sends the data to the endpoint and returns the predictions. The payload is
          limited to 6 MB (4 MB for serverless endpoints).
        """
        pred, _ = self._predict(
            data=data,
            train_data=train_data,
            label=label,
            inference_kwargs=inference_kwargs,
        )
        return pred

    def predict_proba(
        self,
        data: DataInput,
        train_data: DataInput | None = None,
        label: str | None = None,
        *,
        include_predict: bool = True,
        **inference_kwargs: Any,
    ) -> tuple[pd.Series, Prediction] | Prediction:
        """Predict class probabilities for ``data`` with the deployed endpoint.

        For regression, the probability result is identical to the prediction. For inputs above the payload limit,
        use :meth:`autogluon.cloud.TabularCloudPredictor.predict_proba` or
        :meth:`autogluon.cloud.TabularFoundationModel.predict_proba` instead.

        Parameters
        ----------
        data: str | pathlib.Path | pd.DataFrame
            Rows to predict, as a ``pd.DataFrame`` or local/S3 path to a data file.
        train_data: str | pathlib.Path | pd.DataFrame | None, default = None
            Labeled examples the foundation model is fit on. Required for foundation model endpoints; must be
            ``None`` for trained predictor endpoints.
        label: str | None, default = None
            Name of the label column in ``train_data``. Required if and only if ``train_data`` is passed.
        include_predict: bool, default = True
            Whether to return the predictions along with the probabilities. Both are computed in the same request.
        **inference_kwargs: Any
            Additional args passed to the ``predict_proba`` call of the AutoGluon predictor on the endpoint.

        Returns
        -------
        tuple[pd.Series, pd.DataFrame | pd.Series] | pd.DataFrame | pd.Series
            ``(prediction, predict_probability)`` if ``include_predict`` is True, otherwise ``predict_probability``.

        SageMaker API
        -------------
        * :sm-runtime-api:`InvokeEndpoint`: sends the data to the endpoint and returns the predictions. The payload is
          limited to 6 MB (4 MB for serverless endpoints).
        """
        pred, pred_proba = self._predict(
            data=data,
            train_data=train_data,
            label=label,
            inference_kwargs=inference_kwargs,
        )
        if include_predict:
            return pred, pred_proba
        return pred_proba
