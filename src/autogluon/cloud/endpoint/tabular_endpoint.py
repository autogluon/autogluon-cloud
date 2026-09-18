from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

import boto3
import pandas as pd
from sagemaker.predictor import Predictor

from autogluon.common.loaders import load_pd

from ..utils.aws_utils import setup_sagemaker_session
from ..utils.deserializers import PandasDeserializer
from ..utils.serializers import AutoGluonSerializationWrapper, AutoGluonSerializer
from ..utils.utils import split_pred_and_pred_proba

DataInput = Union[str, Path, pd.DataFrame]
Prediction = Union[pd.DataFrame, pd.Series]


class TabularEndpoint:
    """High-level handle for an AutoGluon-Cloud tabular foundation-model endpoint."""

    def __init__(self, endpoint_name: str, session: Optional[boto3.Session] = None):
        """
        Parameters
        ----------
        endpoint_name
            Name of an existing SageMaker endpoint deployed through
            :meth:`autogluon.cloud.TabularFoundationModel.deploy`.
        session
            ``boto3.Session`` used to invoke and delete the endpoint. If ``None``, the default ambient session is used.
        """
        self._predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=setup_sagemaker_session(boto_session=session),
            serializer=AutoGluonSerializer(),
            deserializer=PandasDeserializer(),
        )

    @property
    def endpoint_name(self) -> str:
        return self._predictor.endpoint_name

    @staticmethod
    def _load_data(data: DataInput) -> pd.DataFrame:
        if isinstance(data, (str, Path)):
            return load_pd.load(str(data))
        return data

    def _predict(
        self,
        data: DataInput,
        train_data: DataInput,
        label: str,
        inference_kwargs: Optional[Dict[str, Any]] = None,
    ) -> Tuple[pd.Series, Prediction]:
        data = self._load_data(data)
        train_data = self._load_data(train_data)

        if label not in train_data.columns:
            raise ValueError(f"Label column {label!r} is not present in `train_data`.")
        feature_columns = [column for column in train_data.columns if column != label]
        missing_columns = [column for column in feature_columns if column not in data.columns]
        if missing_columns:
            raise ValueError(f"`data` is missing feature columns present in `train_data`: {missing_columns}.")

        payload = AutoGluonSerializationWrapper(
            data=data,
            train_data=train_data,
            inference_kwargs={"label": label, **(inference_kwargs or {})},
        )
        raw = self._predictor.predict(payload, initial_args={"Accept": "application/x-parquet"})
        pred, pred_proba = split_pred_and_pred_proba(raw)
        if pred_proba is None:
            pred_proba = pred
        return pred, pred_proba

    def predict(
        self,
        data: DataInput,
        train_data: DataInput,
        label: str,
        **inference_kwargs: Any,
    ) -> pd.Series:
        """Fit the foundation model on ``train_data`` and predict ``data``.

        The serialized request includes both ``train_data`` and ``data`` and must not exceed SageMaker's
        6 MiB real-time invocation payload limit. Use
        :meth:`autogluon.cloud.TabularFoundationModel.predict` for larger inputs.
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
        train_data: DataInput,
        label: str,
        *,
        include_predict: bool = True,
        **inference_kwargs: Any,
    ) -> Union[Tuple[pd.Series, Prediction], Prediction]:
        """Fit the foundation model and return class probabilities.

        For regression, the probability result is identical to the prediction.

        The serialized request includes both ``train_data`` and ``data`` and must not exceed SageMaker's
        6 MiB real-time invocation payload limit. Use
        :meth:`autogluon.cloud.TabularFoundationModel.predict_proba` for larger inputs.
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

    def delete_endpoint(self) -> None:
        """Delete the endpoint and its backing model + endpoint config."""
        self._predictor.delete_model()
        self._predictor.delete_endpoint(delete_endpoint_config=True)
