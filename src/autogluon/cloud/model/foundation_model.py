"""FoundationModel — deploy and predict with pretrained foundation models on AWS."""

from __future__ import annotations

from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd

from autogluon.cloud.endpoint.endpoint import Endpoint
from autogluon.cloud.job.remote_job import RemoteJob

from .registry import get_model_config


class FoundationModel:
    """
    Pretrained foundation model inference on AWS.

    Factory: FoundationModel("chronos-bolt-base", ...) returns the appropriate
    task-specific subclass (TimeSeriesFoundationModel, TabularFoundationModel).

    Examples
    --------
    >>> model = FoundationModel("chronos-bolt-base", role_arn="arn:...")
    >>> endpoint = model.deploy()
    >>> predictions = endpoint.predict(data)
    >>> endpoint.delete_endpoint()
    """

    def __new__(cls, model_id: str, **kwargs) -> "FoundationModel":
        if cls is not FoundationModel:
            return super().__new__(cls)
        config = get_model_config(model_id)
        task = config["task"]
        if task == "timeseries":
            return super().__new__(TimeSeriesFoundationModel)
        elif task == "tabular":
            return super().__new__(TabularFoundationModel)
        raise ValueError(f"Unsupported task: {task}")

    def __init__(
        self,
        model_id: str,
        backend: Literal["sagemaker"] = "sagemaker",
        role_arn: Optional[str] = None,
        region: Optional[str] = None,
        s3_output_path: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
    ):
        self.model_id = model_id
        self.role_arn = role_arn
        self.region = region
        self.s3_output_path = s3_output_path
        self._config = get_model_config(model_id)
        # Merge user overrides on top of registry defaults
        self.model_config = {**self._config.get("model_config", {}), **(model_config or {})}
        # TODO: instantiate backend via BackendFactory
        self._backend_type = backend

    def deploy(
        self,
        instance_type: Optional[str] = None,
        mode: Literal["realtime", "serverless", "async"] = "realtime",
        endpoint_name: Optional[str] = None,
        model_artifact_path: Optional[str] = None,
        model_config: Optional[Dict[str, Any]] = None,
        wait: bool = True,
    ) -> Endpoint:
        """
        Deploy model to an endpoint.

        Parameters
        ----------
        instance_type
            Instance type for the endpoint.
            If None, will use the default from the model registry.
        mode
            Endpoint type.
        endpoint_name
            Custom endpoint name.
            If None, will auto-generate a unique name.
        model_artifact_path
            S3 path to pre-cached model weights (for VPC / fast cold start).
            If None, weights are downloaded from HuggingFace on cold start.
        model_config
            Override default inference config (prediction_length, quantile_levels, etc.)
        wait
            Whether to block until the endpoint is ready.

        Returns
        -------
        Endpoint
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, data: Union[str, pd.DataFrame], wait: bool = True, **kwargs) -> Union[pd.DataFrame, RemoteJob]:
        """Subclasses override with task-specific signature."""
        ...

    def fit(
        self,
        train_data: Union[str, pd.DataFrame],
        output_path: Optional[str] = None,
        instance_type: Optional[str] = None,
        wait: bool = True,
        **kwargs,
    ) -> "FoundationModel":
        """
        Fine-tune the model. Returns a new FoundationModel pointing to the fine-tuned artifact.

        Parameters
        ----------
        train_data
            Training data as DataFrame or S3 path.
        output_path
            S3 path to store fine-tuned model.
            If None, will auto-generate under s3_output_path.
        instance_type
            Instance type for the training job.
            If None, will use the default from the model registry.
        wait
            If True, block until training completes.

        Returns
        -------
        FoundationModel
            New instance with model_config pointing to the fine-tuned artifact.
        """
        raise NotImplementedError

    def cache_model_artifact(self, s3_path: str) -> str:
        """
        Pre-cache model weights to S3 for VPC or production use.

        Launches a small job that downloads weights from HuggingFace
        and writes them to S3, avoiding large local downloads.

        Parameters
        ----------
        s3_path
            S3 path where the model weights should be cached.

        Returns
        -------
        str
            S3 path to the cached artifact.
        """
        raise NotImplementedError


class TimeSeriesFoundationModel(FoundationModel):
    """Foundation model for time series forecasting (Chronos, etc.)."""

    def predict(
        self,
        data: Union[str, pd.DataFrame],
        known_covariates: Optional[Union[str, pd.DataFrame]] = None,
        prediction_length: Optional[int] = None,
        quantile_levels: Optional[List[float]] = None,
        output_path: Optional[str] = None,
        instance_type: Optional[str] = None,
        wait: bool = True,
    ) -> Union[pd.DataFrame, RemoteJob]:
        """
        Run batch prediction for time series.

        Parameters
        ----------
        data
            Historical time series in long format (DataFrame or S3 path).
        known_covariates
            Future values of known covariates (DataFrame or S3 path).
        prediction_length
            Number of time steps to forecast.
            If None, will use the default from the model registry.
        quantile_levels
            Quantiles to predict.
            If None, will use the default from the model registry.
        output_path
            S3 path to store predictions.
            If None, will auto-generate under s3_output_path.
        instance_type
            Instance type for the prediction job.
            If None, will use the default from the model registry.
        wait
            If True, block and return DataFrame. If False, return the job handle.

        Returns
        -------
        Union[pd.DataFrame, RemoteJob]
        """
        raise NotImplementedError


class TabularFoundationModel(FoundationModel):
    """Foundation model for tabular prediction (Mitra, TabICL, etc.)."""

    def predict(
        self,
        train_data: Union[str, pd.DataFrame],
        test_data: Union[str, pd.DataFrame],
        label: str = "target",
        output_path: Optional[str] = None,
        instance_type: Optional[str] = None,
        wait: bool = True,
    ) -> Union[pd.DataFrame, RemoteJob]:
        """
        Run batch prediction for tabular tasks.

        Parameters
        ----------
        train_data
            Labeled few-shot context for the foundation model.
        test_data
            Unlabeled data to predict on.
        label
            Target column name in train_data.
        output_path
            S3 path to store predictions.
            If None, will auto-generate under s3_output_path.
        instance_type
            Instance type for the prediction job.
            If None, will use the default from the model registry.
        wait
            If True, block and return DataFrame. If False, return the job handle.

        Returns
        -------
        Union[pd.DataFrame, RemoteJob]
        """
        raise NotImplementedError
