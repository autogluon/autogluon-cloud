from pathlib import Path
from typing import Any, Dict, Union

import pandas as pd

from autogluon.common.loaders import load_pd

from ..utils.serializers import AutoGluonSerializationWrapper
from ..utils.utils import split_pred_and_pred_proba
from .endpoint import Endpoint


class TabularEndpoint:
    """High-level endpoint for tabular foundation-model prediction.

    Each request carries both the labeled few-shot context (``train_data``) and the unlabeled
    rows to predict on (``test_data``); the endpoint fits a fresh predictor per request.
    """

    def __init__(self, endpoint: Endpoint):
        self._endpoint = endpoint

    @property
    def endpoint_name(self) -> str:
        return self._endpoint.endpoint_name

    def predict(
        self,
        train_data: Union[str, Path, pd.DataFrame],
        test_data: Union[str, Path, pd.DataFrame],
        label: str = "target",
        accept: str = "application/x-parquet",
    ) -> pd.Series:
        """Run real-time prediction. Returns the predicted label column."""
        result = self._invoke(train_data, test_data, label=label, accept=accept)
        if result.shape[1] == 1:
            return result.iloc[:, 0]
        pred, _ = split_pred_and_pred_proba(result)
        return pred

    def predict_proba(
        self,
        train_data: Union[str, Path, pd.DataFrame],
        test_data: Union[str, Path, pd.DataFrame],
        label: str = "target",
        include_predict: bool = False,
        accept: str = "application/x-parquet",
    ) -> Union[pd.DataFrame, "tuple"]:
        """Run real-time prediction. Returns class probabilities (classification only).

        If ``include_predict`` is True, returns ``(prediction, predict_probability)``.
        """
        result = self._invoke(train_data, test_data, label=label, accept=accept)
        if result.shape[1] == 1:
            raise ValueError(
                "predict_proba is not supported for regression endpoints — only a single column was returned."
            )
        pred, pred_proba = split_pred_and_pred_proba(result)
        if include_predict:
            return pred, pred_proba
        return pred_proba

    def _invoke(
        self,
        train_data: Union[str, Path, pd.DataFrame],
        test_data: Union[str, Path, pd.DataFrame],
        *,
        label: str,
        accept: str,
    ) -> pd.DataFrame:
        train_df = load_pd.load(str(train_data)) if not isinstance(train_data, pd.DataFrame) else train_data
        test_df = load_pd.load(str(test_data)) if not isinstance(test_data, pd.DataFrame) else test_data

        inference_kwargs: Dict[str, Any] = {"label": label}
        payload = AutoGluonSerializationWrapper(
            data=test_df,
            inference_kwargs=inference_kwargs,
            train_data=train_df,
        )
        return self._endpoint.predict(payload, initial_args={"Accept": accept})

    def delete_endpoint(self) -> None:
        """Delete the endpoint and cleanup artifacts."""
        self._endpoint.delete_endpoint()
