from abc import ABC, abstractmethod
from typing import Any, Dict, List, Union

import pandas as pd

from ..endpoint.endpoint import Endpoint


class Backend(ABC):
    def __init__(self, **kwargs) -> None:
        self.initialize(**kwargs)

    @abstractmethod
    @property
    def name(self) -> str:
        """Name of this backend"""
        raise NotImplementedError

    @abstractmethod
    def initialize(self, **kwargs) -> None:
        """Initialize the backend."""
        raise NotImplementedError

    @abstractmethod
    def generate_default_permission(self, **kwargs) -> Dict[str, str]:
        """Generate default permission file user could use to setup the corresponding entity, i.e. IAM Role in AWS"""
        raise NotImplementedError

    @abstractmethod
    def parse_backend_fit_kwargs(self, kwargs: Dict) -> List[Dict]:
        """Parse backend specific kwargs and get them ready to be sent to fit call"""
        raise NotImplementedError

    @abstractmethod
    def attach_job(self, job_name: str) -> None:
        """
        Attach to a existing training job.
        This is useful when the local process crashed and you want to reattach to the previous job

        Parameters
        ----------
        job_name: str
            The name of the job being attached
        """
        raise NotImplementedError

    @abstractmethod
    def get_fit_job_status(self) -> str:
        """
        Get the status of the training job.
        This is useful when the user made an asynchronous call to the `fit()` function

        Returns
        -------
        str,
            Status of the job
        """
        raise NotImplementedError

    @abstractmethod
    def fit(self, **kwargs) -> None:
        """Fit AG on the backend"""
        raise NotImplementedError

    @abstractmethod
    def parse_backend_deploy_kwargs(self, kwargs: Dict) -> List[Dict]:
        """Parse backend specific kwargs and get them ready to be sent to deploy call"""
        raise NotImplementedError

    @abstractmethod
    def prepare_deploy(self, **kwargs) -> None:
        """Things to be configured before deploy goes here"""
        raise NotImplementedError

    @abstractmethod
    def deploy(self, **kwargs) -> None:
        """Deploy and endpoint"""
        raise NotImplementedError

    @abstractmethod
    def attach_endpoint(self, endpoint: Endpoint) -> None:
        """Attach the backend to an existing endpoint"""
        raise NotImplementedError

    @abstractmethod
    def detach_endpoint(self) -> Endpoint:
        """Detach the current endpoint and return it"""
        raise NotImplementedError

    @abstractmethod
    def predict_realtime(self, test_data: Union[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """Realtime prediction with the endpoint"""
        raise NotImplementedError

    @abstractmethod
    def predict(self, test_data: Union[str, pd.DataFrame], **kwargs) -> Union[pd.DataFrame, pd.Series]:
        """Batch inference"""
        raise NotImplementedError
