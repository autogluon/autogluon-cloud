"""FoundationModel — predict with pretrained foundation models on AWS."""

from __future__ import annotations

import json
import logging
import tarfile
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Union, overload

import pandas as pd

from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix

from ..backend.backend_factory import BackendFactory
from ..backend.constant import SAGEMAKER, TABULAR_SAGEMAKER, TIMESERIES_SAGEMAKER
from ..endpoint.prediction_future import JobPredictionFuture
from ..endpoint.timeseries_endpoint import TimeSeriesEndpoint
from ..scripts.script_manager import ScriptManager
from ..utils.aws_utils import resolve_cloud_output_path
from ..version import __version__
from .registry import get_model_config

logger = logging.getLogger(__name__)

# SageMaker extracts model.tar.gz to /opt/ml/model in the container.
_BUNDLED_WEIGHTS_SUBDIR = "weights"
_CONTAINER_WEIGHTS_DIR = f"/opt/ml/model/{_BUNDLED_WEIGHTS_SUBDIR}"

_AG_CLOUD_VERSION_METADATA_KEY = "autogluon-cloud-version"


def _s3_head_or_none(s3_client: Any, bucket: str, key: str) -> Optional[Dict[str, Any]]:
    """Return ``head_object`` response if the key exists, ``None`` for 404. Other errors propagate."""
    from botocore.exceptions import ClientError

    try:
        return s3_client.head_object(Bucket=bucket, Key=key)
    except ClientError as e:
        if e.response.get("Error", {}).get("Code") in ("404", "NoSuchKey", "NotFound"):
            return None
        raise


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
        *,
        hyperparameters: Optional[Dict[str, Any]] = None,
        model_artifact_uri: Optional[str] = None,
        backend: Literal["sagemaker"] = "sagemaker",
        cloud_output_path: Optional[str] = None,
        role: Optional[str] = None,
    ):
        """
        Parameters
        ----------
        model_id
            ID of the foundation model from the model registry.
        hyperparameters
            Default hyperparameters applied to inference and (when supported) training.
        model_artifact_uri
            S3 URI of a pre-bundled ``model.tar.gz`` produced by
            :meth:`cache_model_artifact`. When set, deploys skip the runtime
            HuggingFace download and load weights from the bundled artifact.
        backend
            Cloud backend to use.
        cloud_output_path
            S3 location where intermediate artifacts are stored. Accepts:

            * ``s3://bucket`` — a unique timestamped subfolder ``ag-<timestamp>`` is appended.
            * ``s3://bucket/prefix`` — used verbatim. Re-running with the same prefix
              will overwrite previously written artifacts.
            * ``None`` (default) — use the bucket saved in ``~/.autogluon/cloud.yaml`` (set
              by :func:`autogluon.cloud.bootstrap` / :func:`autogluon.cloud.register`) and
              append a timestamped subfolder. Raises if no bucket is configured.
        role
            ARN of the SageMaker execution role used to run training and inference jobs. If ``None``, falls back to
            ``role_arn`` in ``~/.autogluon/cloud.yaml`` (set by :func:`autogluon.cloud.bootstrap` /
            :func:`autogluon.cloud.register`), and finally to ``sagemaker.get_execution_role()``.
        """
        self.model_id = model_id
        self.model_artifact_uri = model_artifact_uri
        self.cloud_output_path = resolve_cloud_output_path(cloud_output_path, backend_name=backend)
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
            cloud_output_path=self.cloud_output_path,
            predictor_type=self._predictor_type,
            role=role,
        )

    def _get_hyperparameters(
        self, context: Literal["inference", "training"], overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Merge registry defaults → constructor overrides → call-site overrides,
        defaulting ``model_path`` to ``model_source_uri`` if not set."""
        config_key = "inference_hyperparameters" if context == "inference" else "training_hyperparameters"
        merged = self._config.get(config_key, {}) | self._hyperparameter_overrides | (overrides or {})
        merged.setdefault("model_path", self._config["model_source_uri"])
        return merged

    @abstractmethod
    def _build_predictor_init_args(self, **user_kwargs) -> Dict[str, Any]:
        """Build predictor_init_args dict from user-provided kwargs.

        Subclasses override to map their public API kwargs (e.g., prediction_length,
        target, known_covariates_names) to the dict that TimeSeriesPredictor/TabularPredictor expects.
        """
        ...

    @abstractmethod
    def _build_predictor_fit_args(self, hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Build predictor_fit_args dict. Subclasses override with task-specific logic."""
        ...

    @property
    @abstractmethod
    def _serve_script_path(self) -> str:
        """Path to the serve script for this model type."""
        ...

    @abstractmethod
    def deploy(self, **kwargs):
        """Deploy model to a real-time endpoint.

        Subclasses implement this and return a task-specific endpoint
        (e.g., TimeSeriesEndpoint, TabularEndpoint).
        """
        ...

    @abstractmethod
    def predict(self, data: Union[str, Path, pd.DataFrame], wait: bool = True, **kwargs) -> Optional[pd.DataFrame]:
        """Subclasses override with task-specific signature."""
        ...

    def _deploy_backend(
        self,
        instance_type: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        framework_version: str = "latest",
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        inference_mode: Literal["realtime", "serverless"] = "realtime",
        inference_config: Optional[Dict[str, Any]] = None,
        **backend_kwargs,
    ) -> None:
        """Shared deploy logic. Subclasses call this then wrap the endpoint."""
        if inference_mode == "serverless" and instance_type is not None:
            raise ValueError("`instance_type` must not be set when `inference_mode='serverless'`.")
        if instance_type is None and inference_mode != "serverless":
            instance_type = self._config["deploy_instance_type"]

        merged_hp = self._get_hyperparameters("inference", hyperparameters)
        if self.model_artifact_uri is not None:
            merged_hp["model_path"] = _CONTAINER_WEIGHTS_DIR
        fm_serve_config = {
            "model_name": self._config["model_name"],
            "hyperparameters": merged_hp,
        }

        model_kwargs = backend_kwargs.pop("model_kwargs", {})
        model_kwargs["entry_point"] = self._serve_script_path

        self._backend.deploy(
            predictor_path=self.model_artifact_uri,
            endpoint_name=endpoint_name,
            framework_version=framework_version,
            instance_type=instance_type,
            custom_image_uri=custom_image_uri,
            wait=wait,
            model_kwargs=model_kwargs,
            fm_serve_config=fm_serve_config,
            inference_mode=inference_mode,
            inference_config=inference_config,
            repack=self.model_artifact_uri is None,
            **backend_kwargs,
        )
        assert self._backend.endpoint is not None

    def fit(
        self,
        train_data: Union[str, Path, pd.DataFrame],
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
            Training data, as a DataFrame or local/S3 path to a data file.
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

        :meta private:
        """
        if not self._config.get("fine_tunable", False):
            raise ValueError(f"Model '{self.model_id}' does not support fine-tuning.")
        raise NotImplementedError

    def cache_model_artifact(self, cache_path: str, *, overwrite: bool = False) -> "FoundationModel":
        """
        Download model weights from HuggingFace, bundle them with the FM serve script
        into a SageMaker-compatible ``model.tar.gz``, and upload to S3.

        Lets :meth:`deploy` skip the runtime HuggingFace download — required for
        network-isolated endpoints (e.g. SageMaker Serverless Inference). Returns a
        new :class:`FoundationModel` with ``model_artifact_uri`` set to the uploaded
        tarball.

        Destination key: ``{cache_path}/{model_id}/model.tar.gz``. If it already
        exists, upload is skipped unless ``overwrite=True``; a warning is logged
        when the cached artifact's autogluon-cloud version differs from the
        current version.

        Parameters
        ----------
        cache_path
            S3 prefix under which the artifact will be uploaded. Multiple foundation
            models can share one prefix.
        overwrite
            If True, re-upload even when the destination key exists.

        Returns
        -------
        FoundationModel
            A new instance with ``model_artifact_uri`` populated. The original is unchanged.
        """
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise ImportError(
                "cache_model_artifact requires `huggingface_hub`. Install with: pip install huggingface_hub"
            ) from e

        if not cache_path.startswith("s3://"):
            raise ValueError(f"cache_path must be an s3:// URI, got: {cache_path!r}")

        source_uri = self._config["model_source_uri"]
        cache_key = f"{cache_path.rstrip('/')}/{self.model_id}/model.tar.gz"
        bucket, key = s3_path_to_bucket_prefix(cache_key)
        s3 = self._backend.sagemaker_session.boto_session.client("s3")

        head = None if overwrite else _s3_head_or_none(s3, bucket, key)
        if head is not None:
            cached_version = head["Metadata"].get(_AG_CLOUD_VERSION_METADATA_KEY)
            if cached_version != __version__:
                logger.warning(
                    f"Cached artifact at {cache_key} was bundled with autogluon-cloud "
                    f"{cached_version!r}, current is {__version__!r}. "
                    f"Pass overwrite=True to refresh."
                )
            else:
                logger.log(20, f"Cached artifact already exists at {cache_key}; skipping upload")
        else:
            with tempfile.TemporaryDirectory(prefix="ag_fm_cache_") as tmp:
                tmp_path = Path(tmp)
                weights_dir = tmp_path / _BUNDLED_WEIGHTS_SUBDIR
                logger.log(20, f"Downloading {source_uri} from HuggingFace to {weights_dir}")
                snapshot_download(repo_id=source_uri, local_dir=str(weights_dir))

                code_dir = tmp_path / "code"
                code_dir.mkdir()
                serve_script = Path(self._serve_script_path)
                (code_dir / serve_script.name).write_bytes(serve_script.read_bytes())

                tarball = tmp_path / "model.tar.gz"
                logger.log(20, f"Bundling weights + serve script into {tarball}")
                with tarfile.open(tarball, "w:gz") as tar:
                    tar.add(weights_dir, arcname=_BUNDLED_WEIGHTS_SUBDIR)
                    tar.add(code_dir, arcname="code")
                logger.log(20, f"Uploading to {cache_key}")
                s3.upload_file(
                    str(tarball),
                    bucket,
                    key,
                    ExtraArgs={"Metadata": {_AG_CLOUD_VERSION_METADATA_KEY: __version__}},
                )

        return self.__class__(
            model_id=self.model_id,
            hyperparameters=self._hyperparameter_overrides or None,
            model_artifact_uri=cache_key,
            cloud_output_path=self.cloud_output_path,
            role=self._backend.role_arn,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the model identity. Runtime context (``role``, ``cloud_output_path``)
        is excluded so configs can be shared across users."""
        out: Dict[str, Any] = {"model_id": self.model_id}
        if self._hyperparameter_overrides:
            out["hyperparameters"] = self._hyperparameter_overrides
        if self.model_artifact_uri:
            out["model_artifact_uri"] = self.model_artifact_uri
        return out

    def to_json(self) -> str:
        """Serialize :meth:`to_dict` output as a JSON string."""
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, config: Dict[str, Any], **runtime_context: Any) -> "FoundationModel":
        """Restore from :meth:`to_dict` output. Pass ``role`` / ``cloud_output_path``
        as ``runtime_context``."""
        return cls(**config, **runtime_context)

    @classmethod
    def from_json(cls, s: str, **runtime_context: Any) -> "FoundationModel":
        """Restore from a :meth:`to_json` string."""
        return cls.from_dict(json.loads(s), **runtime_context)


class TimeSeriesFoundationModel(FoundationModel):
    """Foundation model for time series forecasting (Chronos, etc.)."""

    _backend_map = {SAGEMAKER: TIMESERIES_SAGEMAKER}
    _predictor_type = "timeseries"

    @property
    def _serve_script_path(self) -> str:
        return ScriptManager.SAGEMAKER_TIMESERIES_FM_SERVE_SCRIPT_PATH

    def deploy(
        self,
        instance_type: Optional[str] = None,
        endpoint_name: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        framework_version: str = "latest",
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        inference_mode: Literal["realtime", "serverless"] = "realtime",
        inference_config: Optional[Dict[str, Any]] = None,
        **backend_kwargs,
    ) -> TimeSeriesEndpoint:
        """
        Deploy model to an inference endpoint.

        Parameters
        ----------
        instance_type
            Instance type for the endpoint. Defaults to the model registry value. Must be ``None``
            when ``inference_mode="serverless"``.
        endpoint_name
            Custom endpoint name. If None, will auto-generate a unique name.
        hyperparameters
            Model hyperparameters for inference. Overrides values passed to the constructor.
        framework_version
            Container framework version. If 'latest', uses the most recent available.
        custom_image_uri
            Custom Docker image URI for the inference container.
        wait
            Whether to block until the endpoint is ready.
        inference_mode
            Endpoint type. ``"serverless"`` provisions a SageMaker Serverless Inference endpoint
            (no instance management, scales to zero).
        inference_config
            Mode-specific overrides forwarded to ``sagemaker.serverless.ServerlessInferenceConfig``
            (e.g. ``memory_size_in_mb``, ``max_concurrency``).
        **backend_kwargs
            Backend-specific arguments (e.g., initial_instance_count, volume_size,
            model_kwargs, deploy_kwargs).
        """
        self._deploy_backend(
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            hyperparameters=hyperparameters,
            framework_version=framework_version,
            custom_image_uri=custom_image_uri,
            wait=wait,
            inference_mode=inference_mode,
            inference_config=inference_config,
            **backend_kwargs,
        )
        return TimeSeriesEndpoint(self._backend.endpoint)

    def _build_predictor_fit_args(self, hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        model_name = self._config["model_name"]
        merged_hp = self._get_hyperparameters("inference", hyperparameters)
        return {
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

    @overload
    def predict(
        self,
        data: Union[str, Path, pd.DataFrame],
        target: str = ...,
        id_column: str = ...,
        timestamp_column: str = ...,
        known_covariates: Optional[Union[str, Path, pd.DataFrame]] = ...,
        static_features: Optional[Union[str, Path, pd.DataFrame]] = ...,
        prediction_length: int = ...,
        quantile_levels: Optional[List[float]] = ...,
        hyperparameters: Optional[Dict[str, Any]] = ...,
        instance_type: Optional[str] = ...,
        framework_version: str = ...,
        custom_image_uri: Optional[str] = ...,
        wait: Literal[True] = True,
        predictions_path: Optional[str] = ...,
        **backend_kwargs: Any,
    ) -> pd.DataFrame: ...

    @overload
    def predict(
        self,
        data: Union[str, Path, pd.DataFrame],
        target: str = ...,
        id_column: str = ...,
        timestamp_column: str = ...,
        known_covariates: Optional[Union[str, Path, pd.DataFrame]] = ...,
        static_features: Optional[Union[str, Path, pd.DataFrame]] = ...,
        prediction_length: int = ...,
        quantile_levels: Optional[List[float]] = ...,
        hyperparameters: Optional[Dict[str, Any]] = ...,
        instance_type: Optional[str] = ...,
        framework_version: str = ...,
        custom_image_uri: Optional[str] = ...,
        *,
        wait: Literal[False],
        predictions_path: Optional[str] = ...,
        **backend_kwargs: Any,
    ) -> JobPredictionFuture: ...

    def predict(
        self,
        data: Union[str, Path, pd.DataFrame],
        target: str = "target",
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        known_covariates: Optional[Union[str, Path, pd.DataFrame]] = None,
        static_features: Optional[Union[str, Path, pd.DataFrame]] = None,
        prediction_length: int = 1,
        quantile_levels: Optional[List[float]] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        instance_type: Optional[str] = None,
        framework_version: str = "latest",
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        predictions_path: Optional[str] = None,
        **backend_kwargs,
    ) -> Union[pd.DataFrame, JobPredictionFuture]:
        """
        Run batch prediction for time series.

        Parameters
        ----------
        data
            Historical time series to forecast from, in long format, as a DataFrame or local/S3 path to
            a data file. See the `TimeSeriesPredictor docs <https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html>`_
            for the expected format.
        target
            Name of the column that contains the target values to forecast.
        id_column
            Name of the column with the unique identifier of each time series (item).
        timestamp_column
            Name of the column with the observation timestamps.
        known_covariates
            Future values of the known covariates over the forecast horizon. Covariate column names are
            inferred from the columns (excluding ``id_column`` and ``timestamp_column``).
        static_features
            Static (time-independent) features describing each individual time series.
        prediction_length
            Forecast horizon: how many time steps into the future the model should predict.
        quantile_levels
            List of increasing decimals between 0 and 1 specifying which quantiles to estimate. Defaults
            to ``[0.1, 0.2, ..., 0.9]``.
        hyperparameters
            Model hyperparameters for inference. Overrides values passed to the constructor.
        instance_type
            Instance type for the prediction job. If None, uses registry default.
        framework_version
            Container framework version.
        custom_image_uri
            Custom Docker image URI for the container.
        wait
            If True, block and return a DataFrame. If False, return a
            :class:`JobPredictionFuture` immediately — call ``.result()`` on it later to
            retrieve the DataFrame, or ``.status()`` to check progress.
        predictions_path
            S3 URL where predictions will be written by the prediction job (e.g.
            ``s3://my-bucket/runs/2024-05-01/predictions.csv``). The container's SageMaker execution
            role must have ``s3:PutObject`` permission for this location. Defaults to
            ``{cloud_output_path}/{job_name}/predictions.csv``. Predictions use AutoGluon's canonical
            column names ``item_id`` and ``timestamp``, regardless of the ``id_column`` /
            ``timestamp_column`` passed in.
        **backend_kwargs
            Additional backend-specific arguments (e.g., job_name, volume_size,
            autogluon_sagemaker_estimator_kwargs).

        Returns
        -------
        pd.DataFrame or JobPredictionFuture
            DataFrame if ``wait=True``; a :class:`JobPredictionFuture` otherwise.
        """
        if instance_type is None:
            instance_type = self._config["predict_instance_type"]

        predictor_init_args = self._build_predictor_init_args(
            target=target,
            prediction_length=prediction_length,
            quantile_levels=quantile_levels,
        )

        predictor_fit_args = self._build_predictor_fit_args(hyperparameters)
        data_channels = {
            "train_data": data,
            "known_covariates": known_covariates,
            "static_features": static_features,
        }

        extra_ag_args: Dict[str, Any] = {"predict_after_fit": True}
        if predictions_path is not None:
            extra_ag_args["predictions_path"] = predictions_path

        self._backend.fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            data_channels=data_channels,
            id_column=id_column,
            timestamp_column=timestamp_column,
            framework_version=framework_version,
            instance_type=instance_type,
            custom_image_uri=custom_image_uri,
            wait=wait,
            extra_ag_args=extra_ag_args,
            **backend_kwargs,
        )

        if not wait:
            return JobPredictionFuture(
                job=self._backend._fit_job,
                result_loader=self._backend.get_fit_predict_results,
            )

        return self._backend.get_fit_predict_results()


class TabularFoundationModel(FoundationModel):
    """Foundation model for tabular prediction (Mitra, TabICL, etc.)."""

    _backend_map = {SAGEMAKER: TABULAR_SAGEMAKER}
    _predictor_type = "tabular"

    @property
    def _serve_script_path(self) -> str:
        raise NotImplementedError("Tabular FM deploy is not yet supported")

    def deploy(self, **kwargs):
        raise NotImplementedError("Tabular FM deploy is not yet supported")

    def _build_predictor_init_args(self, label: str = "target", **kwargs) -> Dict[str, Any]:
        """Map user kwargs to TabularPredictor init args."""
        return {"label": label}

    def predict(
        self,
        train_data: Union[str, Path, pd.DataFrame],
        test_data: Union[str, Path, pd.DataFrame],
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
        train_data: Union[str, Path, pd.DataFrame],
        test_data: Union[str, Path, pd.DataFrame],
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
