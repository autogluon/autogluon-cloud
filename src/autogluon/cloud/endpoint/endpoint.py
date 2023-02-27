from abc import ABC, abstractmethod
from typing import Union

import pandas as pd


class Endpoint(ABC):
    @property
    @abstractmethod
    def endpoint_name(self) -> str:
        """Name of the endpoint"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, test_data: Union[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """
        Predict with the endpoint
        """
        raise NotImplementedError

    @abstractmethod
    def delete_endpoint(self) -> None:
        """
        Delete the endpoint and cleanup artifacts
        """
        raise NotImplementedError
