import logging
from typing import Union

import pandas as pd
from sagemaker.predictor import Predictor

from .endpoint import Endpoint

logger = logging.getLogger(__name__)


class SagemakerEndpoint(Endpoint):
    def __init__(self, endpoint: Predictor) -> None:
        self._endpoint: Predictor = endpoint

    @property
    def endpoint_name(self) -> str:
        """Name of the endpoint"""
        if self._endpoint is not None:
            return self._endpoint.endpoint_name
        return None

    def predict(self, test_data: Union[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """
        Predict with the endpoint
        """
        return self._endpoint.predict(test_data, **kwargs)

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
        self._endpoint is not None, "There is no endpoint deployed yet"
        logger.log(20, "Deleteing endpoint")
        self._endpoint.delete_endpoint(delete_endpoint_config=delete_endpoint_config)
        logger.log(20, "Endpoint deleted")
        self._endpoint = None
