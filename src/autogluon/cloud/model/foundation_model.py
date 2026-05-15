"""FoundationModel — deploy and predict with pretrained foundation models on AWS."""

from __future__ import annotations

import logging
import tempfile
from abc import abstractmethod
from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd

from ..backend.backend_factory import BackendFactory
from ..backend.constant import SAGEMAKER, TABULAR_SAGEMAKER, TIMESERIES_SAGEMAKER
from ..endpoint.endpoint import Endpoint
from ..job.remote_job import RemoteJob
from ..scripts.script_manager import ScriptManager
from .registry import get_model_config

logger = logging.getLogger(__name__)


class FoundationModel:
    """
    Pretrained foundation model inference on AWS.

    Factory: FoundationModel("chronos-bolt-base", ...) returns the appropriate
    task-specific subclass (TimeSeriesFoundationModel, TabularFoundationModel).

    Examples
    --------
    >>> model = FoundationModel("chronos-bolt-base", role_arn="arn:...")
    >>> predictions = model.predict(data, prediction_length=12)
    >>> endpoint = model.deploy()
    >>> predictions = endpoint.predict(data)
    >>> endpoint.delete_endpoint()
    """

    _backend_map: Dict[str, str] = {}
    _predictor_type: str
    _serve_script_path: str

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
        self._backend_type = backend

        self._tmpdir = tempfile.TemporaryDirectory(prefix="ag_fm_")

        # Instantiate the backend
        backend_name = self._backend_map.get(backend)
        if backend_name is None:
            raise ValueError(
                f"Backend '{backend}' is not supported for {self.__class__.__name__}. "
                f"Available: {list(self._backend_map.keys())}"
            )
        self._backend = BackendFactory.get_backend(
            backend=backend_name,
            local_output_path=self._tmpdir.name,
            cloud_output_path=s3_output_path,
            predictor_type=self._predictor_type,
        )

    def _get_hyperparameters(
        self, context: Literal["inference", "training"], overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Merge registry defaults → constructor overrides → call-site overrides."""
        config_key = "inference_hyperparameters" if context == "inference" else "training_hyperparameters"
        return self._config.get(config_key, {}) | self._hyperparameter_overrides | (overrides or {})

    def _build_predictor_init_args(self, **user_kwargs) -> Dict[str, Any]:
        """Build predictor_init_args dict from user-provided kwargs.

        Subclasses override to map their public API kwargs (e.g., prediction_length,
        target, known_covariates_names) to the dict that TimeSeriesPredictor/TabularPredictor expects.
        """
        raise NotImplementedError

    def _build_predictor_fit_args(
        self, train_data, hyperparameters: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Build predictor_fit_args dict.

        Wraps the model hyperparameters into the AG hyperparameters format:
            {"ModelName": {"model_path": "...", ...}}
        """
        model_name = self._config["model_name"]
        merged_hp = self._get_hyperparameters("inference", hyperparameters)
        return {
            "train_data": train_data,
            "hyperparameters": {model_name: merged_hp},
        }

    def deploy(
        self,
        instance_type: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        framework_version: str = "latest",
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        **backend_kwargs,
    ) -> Endpoint:
        """
        Deploy model to a real-time endpoint.

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
        framework_version
            Container framework version. If 'latest', uses the most recent available.
        custom_image_uri
            Custom Docker image URI for the inference container.
        wait
            Whether to block until the endpoint is ready.
        **backend_kwargs
            Backend-specific arguments (e.g., initial_instance_count, volume_size,
            model_kwargs, deploy_kwargs for SageMaker).

        Returns
        -------
        Endpoint
        """
        if instance_type is None:
            instance_type = self._config["inference_instance_type"]

        serve_config = {
            "model_name": self._config["model_name"],
            "hyperparameters": self._get_hyperparameters("inference", hyperparameters),
        }

        self._backend.deploy(
            predictor_path=None,
            endpoint_name=endpoint_name,
            framework_version=framework_version,
            instance_type=instance_type,
            custom_image_uri=custom_image_uri,
            wait=wait,
            serve_script=self._serve_script_path,
            serve_config=serve_config,
            **backend_kwargs,
        )
        assert self._backend.endpoint is not None
        return self._backend.endpoint

    @abstractmethod
    def predict(self, data: Union[str, pd.DataFrame], wait: bool = True, **kwargs) -> Union[pd.DataFrame, RemoteJob]:
        """Subclasses override with task-specific signature."""
        ...

    def fit(self, train_data: Union[str, pd.DataFrame], **kwargs) -> "FoundationModel":
        """Fine-tune the model. Not yet implemented."""
        raise NotImplementedError

    def cache_model_artifact(self, s3_path: str) -> str:
        """Pre-cache model weights to S3. Not yet implemented."""
        raise NotImplementedError


class TimeSeriesFoundationModel(FoundationModel):
    """Foundation model for time series forecasting (Chronos, etc.)."""

    _backend_map = {SAGEMAKER: TIMESERIES_SAGEMAKER}
    _predictor_type = "timeseries"
    _serve_script_path = ScriptManager.SAGEMAKER_TIMESERIES_FM_SERVE_SCRIPT_PATH

    def _build_predictor_init_args(
        self,
        target: str = "target",
        prediction_length: int = 1,
        known_covariates_names: Optional[List[str]] = None,
        quantile_levels: Optional[List[float]] = None,
        **kwargs,
    ) -> Dict[str, Any]:
        """Map user kwargs to TimeSeriesPredictor init args."""
        args: Dict[str, Any] = {
            "target": target,
            "prediction_length": prediction_length,
        }
        if known_covariates_names:
            args["known_covariates_names"] = known_covariates_names
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
        output_path: Optional[str] = None,
        instance_type: Optional[str] = None,
        framework_version: str = "latest",
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        **backend_kwargs,
    ) -> Union[pd.DataFrame, RemoteJob]:
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
        output_path
            S3 path to store predictions. If None, auto-generates under s3_output_path.
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
        Union[pd.DataFrame, RemoteJob]
        """
        if instance_type is None:
            instance_type = self._config["inference_instance_type"]

        # Derive known_covariates_names from the DataFrame columns
        known_covariates_names: Optional[List[str]] = None
        if known_covariates is not None and isinstance(known_covariates, pd.DataFrame):
            known_covariates_names = [c for c in known_covariates.columns if c not in (id_column, timestamp_column)]

        # Build the AG predictor args from user-facing kwargs
        predictor_init_args = self._build_predictor_init_args(  # noqa: F841
            target=target,
            prediction_length=prediction_length,
            known_covariates_names=known_covariates_names,
            quantile_levels=quantile_levels,
        )

        predictor_fit_args = self._build_predictor_fit_args(data, hyperparameters)

        # Add known_covariates to fit_args (handled by train.py's predict_after_fit branch)
        if known_covariates is not None:
            predictor_fit_args["known_covariates"] = known_covariates

        # TODO: call backend.fit() with predict_after_fit=True
        # This reuses the existing fit_predict pattern from the fit_predict branch.
        # self._backend.fit(
        #     predictor_init_args=predictor_init_args,
        #     predictor_fit_args=predictor_fit_args,
        #     id_column=id_column,
        #     timestamp_column=timestamp_column,
        #     static_features=static_features,
        #     framework_version=framework_version,
        #     instance_type=instance_type,
        #     custom_image_uri=custom_image_uri,
        #     wait=wait,
        #     predict_after_fit=True,
        #     **backend_kwargs,
        # )
        #
        # if not wait:
        #     # TODO: return a RemoteJob handle that can retrieve predictions later
        #     return self._backend._fit_job  # or wrap in a RemoteJob adapter
        #
        # return self._backend.get_fit_predict_results()

        raise NotImplementedError


class TabularFoundationModel(FoundationModel):
    """Foundation model for tabular prediction (Mitra, TabICL, etc.)."""

    _backend_map = {SAGEMAKER: TABULAR_SAGEMAKER}
    _predictor_type = "tabular"
    _serve_script_path = ScriptManager.SAGEMAKER_TABULAR_FM_SERVE_SCRIPT_PATH

    def _build_predictor_init_args(self, label: str = "target", **kwargs) -> Dict[str, Any]:
        """Map user kwargs to TabularPredictor init args."""
        return {"label": label}

    def predict(
        self,
        train_data: Union[str, pd.DataFrame],
        test_data: Union[str, pd.DataFrame],
        label: str = "target",
        hyperparameters: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        instance_type: Optional[str] = None,
        framework_version: str = "latest",
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        **backend_kwargs,
    ) -> Union[pd.DataFrame, RemoteJob]:
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
        output_path
            S3 path to store predictions. If None, auto-generates under s3_output_path.
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
        Union[pd.DataFrame, RemoteJob]
        """
        # TODO: tabular FM predict is different — needs both train_data (context) and test_data
        # The existing train.py doesn't handle this split. Options:
        #   a) Extend train.py to support a "context_data" + "test_data" split
        #   b) Write a separate tabular FM predict script
        #   c) Concatenate train+test with a marker column, split in the script
        raise NotImplementedError

    def predict_proba(
        self,
        train_data: Union[str, pd.DataFrame],
        test_data: Union[str, pd.DataFrame],
        label: str = "target",
        hyperparameters: Optional[Dict[str, Any]] = None,
        output_path: Optional[str] = None,
        instance_type: Optional[str] = None,
        framework_version: str = "latest",
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        **backend_kwargs,
    ) -> Union[pd.DataFrame, RemoteJob]:
        """Run batch prediction returning class probabilities."""
        # TODO: same implementation as predict() but with predict_proba flag
        raise NotImplementedError
