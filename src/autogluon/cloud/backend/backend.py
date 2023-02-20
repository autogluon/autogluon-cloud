from abc import ABC, abstractmethod
from typing import Any, Dict, Union

import pandas as pd

from ..endpoint.endpoint import Endpoint


class Backend(ABC):
    def __init__(self, **kwargs) -> None:
        self.initialize(**kwargs)

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the backend."""
        raise NotImplementedError

    @abstractmethod
    def generate_default_permission(self, **kwargs) -> Dict[str, str]:
        """Generate default permission file user could use to setup the corresponding entity, i.e. IAM Role in AWS"""
        raise NotImplementedError

    @abstractmethod
    def fit(self, **kwargs) -> None:
        """Fit AG on the backend"""
        raise NotImplementedError

    @abstractmethod
    def deploy(self, **kwargs) -> Endpoint:
        """Deploy and endpoint"""
        raise NotImplementedError

    @abstractmethod
    def predict_realtime(self, test_data: Union[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """Realtime prediction with the endpoint"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, test_data: Union[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """Batch inference"""
        raise NotImplementedError
