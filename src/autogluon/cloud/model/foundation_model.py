"""FoundationModel — predict with pretrained foundation models on AWS."""

from __future__ import annotations

import tempfile
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd

from ..backend.backend_factory import BackendFactory
from ..backend.constant import SAGEMAKER, TABULAR_SAGEMAKER, TIMESERIES_SAGEMAKER
from ..endpoint.endpoint import Endpoint
from .registry import get_model_config


class FoundationModel:
    """
    Pretrained foundation model inference on AWS.

    Factory: FoundationModel("chronos-bolt-base", ...) returns the appropriate
    task-specific subclass (TimeSeriesFoundationModel, TabularFoundationModel).

    Examples
    --------
    >>> model = FoundationModel("chronos-bolt-base")
    >>> predictions = model.predict(data, prediction_length=12)
    """

    _backend_map: Dict[str, str] = {}
    _predictor_type: str

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
        cloud_output_path: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
    ):
        self.model_id = model_id
        self.cloud_output_path = cloud_output_path
        self._config = get_model_config(model_id)
        self._hyperparameter_overrides = hyperparameters or {}
        self._tmpdir = tempfile.TemporaryDirectory(prefix="ag_fm_")

        backend_name = self._backend_map.get(backend)
        if backend_name is None:
            raise ValueError(
                f"Backend '{backend}' is not supported for {self.__class__.__name__}. "
                f"Available: {list(self._backend_map.keys())}"
            )
        self._backend = BackendFactory.get_backend(
            backend=backend_name,
            local_output_path=self._tmpdir.name,
            cloud_output_path=cloud_output_path,
            predictor_type=self._predictor_type,
        )

    def _get_hyperparameters(
        self, context: Literal["inference", "training"], overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Merge registry defaults → constructor overrides → call-site overrides."""
        config_key = "inference_hyperparameters" if context == "inference" else "training_hyperparameters"
        return self._config.get(config_key, {}) | self._hyperparameter_overrides | (overrides or {})

    @abstractmethod
    def _build_predictor_init_args(self, **user_kwargs) -> Dict[str, Any]:
        """Build predictor_init_args dict from user-provided kwargs.

        Subclasses override to map their public API kwargs (e.g., prediction_length,
        target, known_covariates_names) to the dict that TimeSeriesPredictor/TabularPredictor expects.
        """
        ...

    @abstractmethod
    def _build_predictor_fit_args(
        self, train_data, hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build predictor_fit_args dict. Subclasses override with task-specific logic."""
        ...

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
    def predict(self, data: Union[str, pd.DataFrame], wait: bool = True, **kwargs) -> Optional[pd.DataFrame]:
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
            If None, will auto-generate under cloud_output_path.
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
        if not self._config.get("fine_tunable", False):
            raise ValueError(f"Model '{self.model_id}' does not support fine-tuning.")
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

    _backend_map = {SAGEMAKER: TIMESERIES_SAGEMAKER}
    _predictor_type = "timeseries"

    def _build_predictor_fit_args(
        self, train_data, hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        model_name = self._config["model_name"]
        merged_hp = self._get_hyperparameters("inference", hyperparameters)
        return {
            "train_data": train_data,
            "hyperparameters": {model_name: merged_hp},
            "skip_model_selection": True,
        }

    def _build_predictor_init_args(
        self,
        target: str = "target",
        prediction_length: int = 1,
        quantile_levels: Optional[List[float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Map user kwargs to TimeSeriesPredictor init args."""
        args: Dict[str, Any] = {
            "target": target,
            "prediction_length": prediction_length,
        }
        if quantile_levels is not None:
            args["quantile_levels"] = quantile_levels
        return args

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
        instance_type: Optional[str] = None,
        framework_version: str = "latest",
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        **backend_kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        Run batch prediction for time series via a SageMaker training job (fit_predict pattern).

        Launches a job that loads the foundation model, runs .fit() + .predict() in one shot,
        and saves predictions to the job output.

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
            Covariate column names are inferred from the DataFrame columns
            (excluding id_column and timestamp_column).
        static_features
            Metadata attributes of individual items (DataFrame or S3 path).
        prediction_length
            Number of time steps to forecast.
        quantile_levels
            Quantiles to predict.
        hyperparameters
            Model hyperparameters for inference. Overrides values passed to the constructor.
        instance_type
            Instance type for the prediction job. If None, uses registry default.
        framework_version
            Container framework version.
        custom_image_uri
            Custom Docker image URI for the container.
        wait
            If True, block and return DataFrame. If False, return the job handle.
        **backend_kwargs
            Additional backend-specific arguments (e.g., job_name, volume_size,
            autogluon_sagemaker_estimator_kwargs).

        Returns
        -------
        Optional[pd.DataFrame]
        """
        if instance_type is None:
            instance_type = self._config["predict_instance_type"]

        predictor_init_args = self._build_predictor_init_args(
            target=target,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )

        predictor_fit_args = self._build_predictor_fit_args(data, hyperparameters)

        if known_covariates is not None:
            predictor_fit_args["known_covariates"] = known_covariates

        self._backend.fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            id_column=id_column,
            timestamp_column=timestamp_column,
            static_features=static_features,
            framework_version=framework_version,
            instance_type=instance_type,
            custom_image_uri=custom_image_uri,
            wait=wait,
            predict_after_fit=True,
            **backend_kwargs,
        )

        if not wait:
            # TODO: return a handle that supports polling status and fetching results
            return None

        return self._backend.get_fit_predict_results()


class TabularFoundationModel(FoundationModel):
    """Foundation model for tabular prediction (Mitra, TabICL, etc.)."""

    _backend_map = {SAGEMAKER: TABULAR_SAGEMAKER}
    _predictor_type = "tabular"

    def _build_predictor_init_args(self, label: str = "target", **kwargs) -> Dict[str, Any]:
        """Map user kwargs to TabularPredictor init args."""
        return {"label": label}

    def predict(
        self,
        train_data: Union[str, pd.DataFrame],
        test_data: Union[str, pd.DataFrame],
        label: str = "target",
        hyperparameters: Optional[Dict[str, Any]] = None,
        instance_type: Optional[str] = None,
        framework_version: str = "latest",
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        **backend_kwargs,
    ) -> Optional[pd.DataFrame]:
        """
        Run batch prediction for tabular tasks.

        For tabular foundation models (e.g., Mitra), train_data provides the few-shot
        context and test_data contains the rows to predict on.

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
        instance_type
            Instance type for the prediction job. If None, uses registry default.
        framework_version
            Container framework version.
        custom_image_uri
            Custom Docker image URI for the container.
        wait
            If True, block and return DataFrame. If False, return the job handle.
        **backend_kwargs
            Additional backend-specific arguments.

        Returns
        -------
        Optional[pd.DataFrame]
        """
        # TODO: requires fit_predict support for TabularCloudPredictor
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
    ) -> Optional[pd.DataFrame]:
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
            If None, will auto-generate under cloud_output_path.
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
        Optional[pd.DataFrame]
        """
        raise NotImplementedError
