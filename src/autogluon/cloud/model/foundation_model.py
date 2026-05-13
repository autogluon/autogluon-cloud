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
    >>> model = FoundationModel("chronos-bolt-base", role_arn="arn:...", hyperparameters={"model_path": "s3://cached/"})
    >>> endpoint = model.deploy()
    >>> predictions = endpoint.predict(data)
    >>> endpoint.delete_endpoint()
    """

    def __new__(cls, model_id: str, **kwargs) -> "FoundationModel":
        if cls is not FoundationModel:
            return super().__new__(cls)
        config = get_model_config(model_id)
        task = config["task"]
        if task == "forecasting":
            return super().__new__(TimeSeriesFoundationModel)
        elif task in ("classification", "regression"):
            return super().__new__(TabularFoundationModel)
        raise ValueError(f"Unsupported task: {task}")

    def __init__(
        self,
        model_id: str,
        backend: Literal["sagemaker"] = "sagemaker",
        role_arn: Optional[str] = None,
        region: Optional[str] = None,
        s3_output_path: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        self.model_id = model_id
        self.role_arn = role_arn
        self.region = region
        self.s3_output_path = s3_output_path
        self._config = get_model_config(model_id)
        self._hyperparameter_overrides = hyperparameters or {}
        # TODO: instantiate backend via BackendFactory
        self._backend_type = backend

    def _get_hyperparameters(
        self, context: Literal["inference", "training"], overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        config_key = "inference_hyperparameters" if context == "inference" else "training_hyperparameters"
        return self._config.get(config_key, {}) | self._hyperparameter_overrides | (overrides or {})

    def deploy(
        self,
        instance_type: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        wait: bool = True,
        **backend_kwargs,
    ) -> Endpoint:
        """
        Deploy model to an endpoint.

        Parameters
        ----------
        instance_type
            Instance type for the endpoint.
            If None, will use the default from the model registry.
        endpoint_name
            Custom endpoint name.
            If None, will auto-generate a unique name.
        hyperparameters
            Model hyperparameters for inference. Overrides values passed to the constructor.
            Available hyperparameters for each model are listed in the AutoGluon documentation.
        wait
            Whether to block until the endpoint is ready.
        **backend_kwargs
            Backend-specific arguments. Use these to configure serverless, async, or
            autoscaling (e.g. memory_size_in_mb, max_concurrency, initial_instance_count).

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
        hyperparameters: Optional[Dict[str, Any]] = None,
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
        hyperparameters
            Model hyperparameters for training. Overrides values passed to the constructor.
            Available hyperparameters for each model are listed in the AutoGluon documentation.
        wait
            If True, block until training completes.

        Returns
        -------
        FoundationModel
            New instance with hyperparameters pointing to the fine-tuned artifact.
        """
        raise NotImplementedError

    def cache_model_artifact(self, s3_path: str) -> str:
        """
        Pre-cache model weights to S3 (for VPC-deployed endpoints).

        Launches a small job that downloads weights from HuggingFace
        and uploads them to S3.

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
        target: str = "target",
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        known_covariates: Optional[Union[str, pd.DataFrame]] = None,
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        prediction_length: int = 1,
        quantile_levels: Optional[List[float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        instance_type: Optional[str] = None,
        wait: bool = True,
        **backend_kwargs,
    ) -> Union[pd.DataFrame, RemoteJob]:
        """
        Run batch prediction for time series.

        Parameters
        ----------
        data
            Historical time series in long format (DataFrame or S3 path).
        target
            Name of the target column to forecast.
        id_column
            Name of the item ID column.
        timestamp_column
            Name of the timestamp column.
        known_covariates
            Future values of known covariates (DataFrame or S3 path).
        static_features
            Metadata attributes of individual items (DataFrame or S3 path).
        prediction_length
            Number of time steps to forecast.
        quantile_levels
            Quantiles to predict.
        hyperparameters
            Model hyperparameters for inference. Overrides values passed to the constructor.
            Available hyperparameters for each model are listed in the AutoGluon documentation.
        output_path
            S3 path to store predictions.
            If None, will auto-generate under s3_output_path.
        instance_type
            Instance type for the prediction job.
            If None, will use the default from the model registry.
        wait
            If True, block and return DataFrame. If False, return the job handle.
        **backend_kwargs
            Additional backend-specific arguments (e.g. job_name, custom_image_uri,
            framework_version, volume_size).

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
        hyperparameters: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        instance_type: Optional[str] = None,
        wait: bool = True,
        **backend_kwargs,
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
        hyperparameters
            Model hyperparameters for inference. Overrides values passed to the constructor.
            Available hyperparameters for each model are listed in the AutoGluon documentation.
        output_path
            S3 path to store predictions.
            If None, will auto-generate under s3_output_path.
        instance_type
            Instance type for the prediction job.
            If None, will use the default from the model registry.
        wait
            If True, block and return DataFrame. If False, return the job handle.
        **backend_kwargs
            Additional backend-specific arguments (e.g. job_name, custom_image_uri,
            framework_version, volume_size).

        Returns
        -------
        Union[pd.DataFrame, RemoteJob]
        """
        raise NotImplementedError

    def predict_proba(
        self,
        train_data: Union[str, pd.DataFrame],
        test_data: Union[str, pd.DataFrame],
        label: str = "target",
        hyperparameters: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        instance_type: Optional[str] = None,
        wait: bool = True,
        **backend_kwargs,
    ) -> Union[pd.DataFrame, RemoteJob]:
        """
        Run batch prediction returning class probabilities.

        Parameters
        ----------
        train_data
            Labeled few-shot context for the foundation model.
        test_data
            Unlabeled data to predict on.
        label
            Target column name in train_data.
        hyperparameters
            Model hyperparameters for inference. Overrides values passed to the constructor.
            Available hyperparameters for each model are listed in the AutoGluon documentation.
        output_path
            S3 path to store predictions.
            If None, will auto-generate under s3_output_path.
        instance_type
            Instance type for the prediction job.
            If None, will use the default from the model registry.
        wait
            If True, block and return DataFrame. If False, return the job handle.
        **backend_kwargs
            Additional backend-specific arguments (e.g. job_name, custom_image_uri,
            framework_version, volume_size).

        Returns
        -------
        Union[pd.DataFrame, RemoteJob]
        """
        raise NotImplementedError
