import logging
from typing import Any, Optional, Union

import pandas as pd
from sagemaker.predictor import Predictor
from sagemaker.predictor_async import AsyncPredictor

from .endpoint import Endpoint
from .prediction_future import PredictionFuture

logger = logging.getLogger(__name__)


class SagemakerEndpoint(Endpoint):
    def __init__(self, endpoint: Union[Predictor, AsyncPredictor]) -> None:
        self._endpoint: Union[Predictor, AsyncPredictor] = endpoint

    @property
    def endpoint_name(self) -> str:
        """Name of the endpoint"""
        if self._endpoint is not None:
            return self._endpoint.endpoint_name
        return None

    @property
    def is_async(self) -> bool:
        """Whether this endpoint was deployed in async mode."""
        return isinstance(self._endpoint, AsyncPredictor)

    def predict(self, test_data: Union[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """
        Predict with the endpoint
        """
        if self.is_async:
            raise RuntimeError("Endpoint was deployed with `inference_mode='async'`; use `predict_async()`.")
        return self._endpoint.predict(test_data, **kwargs)

    def predict_async(
        self,
        test_data: Union[str, pd.DataFrame],
        accept: str,
        initial_args: Optional[dict] = None,
        **kwargs: Any,
    ) -> PredictionFuture:
        """Submit an async inference request and return a future for the eventual result."""
        if not self.is_async:
            raise RuntimeError("Endpoint was not deployed with `inference_mode='async'`; use `predict()`.")
        response = self._endpoint.predict_async(
            data=test_data, initial_args={"Accept": accept, **(initial_args or {})}, **kwargs
        )
        return PredictionFuture._from_async_response(response, accept=accept)

    def delete_endpoint(self) -> None:
        """
        Delete the endpoint and cleanup artifacts
        """
        self._delete_endpoint_model()
        self._delete_endpoint()

    def _delete_endpoint_model(self):
        assert self._endpoint is not None, "There is no endpoint deployed yet"
        logger.log(20, "Deleting endpoint model")
        self._endpoint.delete_model()
        logger.log(20, "Endpoint model deleted")

    def _delete_endpoint(self, delete_endpoint_config=True):
        assert self._endpoint is not None, "There is no endpoint deployed yet"
        logger.log(20, "Deleteing endpoint")
        self._endpoint.delete_endpoint(delete_endpoint_config=delete_endpoint_config)
        logger.log(20, "Endpoint deleted")
        self._endpoint = None
