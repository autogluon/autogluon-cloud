"""FoundationModel — predict with pretrained foundation models on AWS."""

from __future__ import annotations

import json
import logging
import tarfile
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from packaging.version import InvalidVersion, Version
from typing_extensions import Self, Unpack

from autogluon.common.loaders import load_pd
from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix

from ..backend.backend_factory import BackendFactory
from ..backend.constant import SAGEMAKER, TABULAR_SAGEMAKER, TIMESERIES_SAGEMAKER
from ..endpoint.prediction_future import JobPredictionFuture
from ..endpoint.tabular_endpoint import TabularEndpoint
from ..endpoint.timeseries_endpoint import TimeSeriesEndpoint
from ..scripts.script_manager import ScriptManager
from ..utils.aws_utils import resolve_cloud_output_path
from ..utils.constants import DEFAULT_FRAMEWORK_VERSION
from ..utils.sagemaker_api import (
    IGNORED_FM_JOB_KWARGS,
    DeployKwargs,
    TrainingJobKwargs,
    check_backend_kwargs,
    reject_legacy_kwargs,
)
from ..utils.utils import split_pred_and_pred_proba
from ..version import __version__
from .registry import FOUNDATION_MODEL_REGISTRY, get_model_config

logger = logging.getLogger(__name__)

# SageMaker extracts model.tar.gz to /opt/ml/model in the container.
_CONTAINER_WEIGHTS_DIR = "/opt/ml/model/weights"

_AG_CLOUD_VERSION_METADATA_KEY = "autogluon-cloud-version"


def _s3_head_or_none(s3_client: Any, bucket: str, key: str) -> dict[str, Any] | None:
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

    Factory: ``FoundationModel(model_id, ...)`` dispatches on the model's task and returns the
    appropriate task-specific subclass (:class:`TimeSeriesFoundationModel`, ``TabularFoundationModel``).
    Most users instantiate the subclass directly instead.

    Examples
    --------
    >>> model = FoundationModel("chronos-2")  # returns a TimeSeriesFoundationModel
    >>> predictions = model.predict(data, prediction_length=24)
    """

    _backend_map: dict[str, str] = {}
    _predictor_type: str
    _problem_types: tuple[str, ...] = ("forecasting", "multiclass", "regression")

    @classmethod
    def list_models(cls) -> list[str]:
        """
        List the supported ``model_id`` values.

        Returns
        -------
        list[str]
            IDs of the foundation models that this class can load.
        """
        return [
            model_id
            for model_id, config in FOUNDATION_MODEL_REGISTRY.items()
            if config.problem_type in cls._problem_types
        ]

    def __new__(cls, model_id: str, **kwargs) -> Self:
        if cls is not FoundationModel:
            return super().__new__(cls)
        config = get_model_config(model_id)
        problem_type = config.problem_type
        if problem_type == "forecasting":
            return super().__new__(TimeSeriesFoundationModel)
        elif problem_type in ("multiclass", "regression"):
            return super().__new__(TabularFoundationModel)
        raise ValueError(f"Unsupported problem_type: {problem_type}")

    def __init__(
        self,
        model_id: str,
        *,
        cloud_output_path: str | None = None,
        role: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        model_artifact_uri: str | None = None,
        backend: Literal["sagemaker"] = "sagemaker",
    ):
        """
        Parameters
        ----------
        model_id: str
            ID of the foundation model, e.g. ``"chronos-2"`` or ``"mitra-classifier"``. Use :meth:`list_models` to
            get the supported IDs.
        cloud_output_path: str | None, default = None
            S3 location where intermediate artifacts are stored. Accepts:

            * ``s3://bucket`` — a unique timestamped subfolder ``ag-<timestamp>`` is appended.
            * ``s3://bucket/prefix`` — used verbatim. Re-running with the same prefix will overwrite previously written
              artifacts.
            * ``None`` (default) — use the bucket saved in ``~/.autogluon/cloud.yaml`` (set by
              :func:`autogluon.cloud.bootstrap` / :func:`autogluon.cloud.register`) and append a timestamped subfolder.
              Raises if no bucket is configured.
        role: str | None, default = None
            ARN of the SageMaker execution role used to run training and inference jobs. If ``None``, falls back to
            ``role_arn`` in ``~/.autogluon/cloud.yaml`` (set by :func:`autogluon.cloud.bootstrap` /
            :func:`autogluon.cloud.register`), and finally to the role of the current AWS identity.
        hyperparameters: dict[str, Any] | None, default = None
            Model hyperparameters. Hyperparameters passed to ``predict()`` or ``deploy()`` take precedence.
        model_artifact_uri: str | None, default = None
            S3 URI of a ``model.tar.gz`` created by :meth:`cache_model_artifact`. If set, ``deploy()`` loads the
            weights from this artifact instead of downloading them from Hugging Face.
        backend: Literal["sagemaker"], default = "sagemaker"
            Cloud backend to use.
        """
        available_models = self.list_models()
        if model_id not in available_models:
            raise ValueError(
                f"Unknown model_id {model_id!r} for {type(self).__name__}. Available models: {available_models}"
            )
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
            resource_prefix=f"ag-cloud-{self.model_id}",
            role=role,
        )

    def _get_hyperparameters(
        self, context: Literal["inference", "training"], overrides: dict[str, Any] | None = None
    ) -> dict[str, Any]:
        """Merge registry defaults → constructor overrides → call-site overrides, defaulting the model's
        weights-source hyperparameter (``model_source_hyperparameter``) to ``model_source_uri`` if not set."""
        if context == "inference":
            registry_defaults = self._config.inference_hyperparameters
        else:
            registry_defaults = self._config.training_hyperparameters
        merged = registry_defaults | self._hyperparameter_overrides | (overrides or {})
        if self._config.model_source_hyperparameter is not None:
            merged.setdefault(self._config.model_source_hyperparameter, self._config.model_source_uri)
        return merged

    def _check_framework_version(self, framework_version: str, custom_image_uri: str | None) -> None:
        """Raise if ``framework_version`` is older than the model's ``min_framework_version``.

        Skipped for custom images, whose AutoGluon version is unknown.
        """
        if custom_image_uri is not None or framework_version == "latest":
            return
        min_version = self._config.min_framework_version
        try:
            requested = Version(framework_version).release[:2]
        except InvalidVersion:
            return  # the backend raises a descriptive error for invalid versions
        if requested < Version(min_version).release[:2]:
            raise ValueError(
                f"Model '{self.model_id}' requires AutoGluon {min_version} or newer, got "
                f"framework_version={framework_version!r}."
            )

    @abstractmethod
    def _build_predictor_init_args(self, **user_kwargs) -> dict[str, Any]:
        """Build predictor_init_args dict from user-provided kwargs.

        Subclasses override to map their public API kwargs (e.g., prediction_length,
        target, known_covariates_names) to the dict that TimeSeriesPredictor/TabularPredictor expects.
        """
        ...

    @abstractmethod
    def _build_predictor_fit_args(self, hyperparameters: dict[str, Any] | None = None) -> dict[str, Any]:
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
    def predict(self, data: str | Path | pd.DataFrame, wait: bool = True, **kwargs) -> pd.DataFrame | None:
        """Subclasses override with task-specific signature."""
        ...

    def _deploy_backend(
        self,
        instance_type: str | None = None,
        endpoint_name: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        framework_version: str = DEFAULT_FRAMEWORK_VERSION,
        wait: bool = True,
        inference_mode: Literal["realtime", "serverless"] = "realtime",
        inference_config: dict[str, Any] | None = None,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs,
    ) -> None:
        """Shared deploy logic. Subclasses call this then wrap the endpoint."""
        kwargs = check_backend_kwargs(kwargs, DeployKwargs, "deploy")
        self._check_framework_version(framework_version, kwargs.get("custom_image_uri"))
        if inference_mode == "serverless" and instance_type is not None:
            raise ValueError("`instance_type` must not be set when `inference_mode='serverless'`.")
        if instance_type is None and inference_mode == "realtime":
            instance_type = self._config.deploy_instance_type

        merged_hp = self._get_hyperparameters("inference", hyperparameters)
        if self.model_artifact_uri is not None:
            source_hp = self._config.model_source_hyperparameter
            user_model_path = (hyperparameters or {}).get(source_hp) or self._hyperparameter_overrides.get(source_hp)
            if user_model_path is not None:
                raise ValueError(
                    f"Cannot set hyperparameters['{source_hp}'] when model_artifact_uri is in use — the bundled "
                    f"artifact determines the in-container weights path ({_CONTAINER_WEIGHTS_DIR}). Drop "
                    f"'{source_hp}', or call deploy() on a FoundationModel without model_artifact_uri."
                )
            merged_hp[source_hp] = _CONTAINER_WEIGHTS_DIR
        fm_serve_config = {
            "ag_model_key": self._config.ag_model_key,
            "hyperparameters": merged_hp,
            "problem_type": self._config.problem_type,
        }
        if self._config.optional_dependencies:
            fm_serve_config["optional_dependencies"] = self._config.optional_dependencies

        # FM deploys never repack: predictor_path is either None (script-only tarball is built locally) or a
        # pre-bundled cache artifact that already contains the serve script.
        self._backend.deploy(
            predictor_path=self.model_artifact_uri,
            endpoint_name=endpoint_name,
            framework_version=framework_version,
            instance_type=instance_type,
            wait=wait,
            backend_overrides=backend_overrides,
            entry_point=self._serve_script_path,
            fm_serve_config=fm_serve_config,
            inference_mode=inference_mode,
            inference_config=inference_config,
            repack=False,
            extra_tags=[{"Key": "autogluon-cloud-model-id", "Value": self.model_id}],
            **kwargs,
        )
        assert self._backend.endpoint_name is not None

    def fit(
        self,
        train_data: str | Path | pd.DataFrame,
        output_path: str | None = None,
        instance_type: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        wait: bool = True,
        **kwargs,
    ) -> Self:
        """
        Fine-tune the model. Returns a new FoundationModel pointing to the fine-tuned artifact.

        Parameters
        ----------
        train_data: str | Path | pd.DataFrame
            Training data, as a ``pd.DataFrame`` or local/S3 path to a data file.
        output_path: str | None, default = None
            S3 path to store fine-tuned model.
            If None, will auto-generate under cloud_output_path.
        instance_type: str | None, default = None
            Instance type for the training job.
            If None, will use the default from the model registry.
        hyperparameters: dict[str, Any] | None, default = None
            Model hyperparameters for training. Overrides values passed to the constructor.
            Available hyperparameters for each model are listed in the AutoGluon documentation.
        wait: bool, default = True
            If True, block until training completes.

        Returns
        -------
        FoundationModel
            New instance with hyperparameters pointing to the fine-tuned artifact.

        :meta private:
        """
        if not self._config.fine_tunable:
            raise ValueError(f"Model '{self.model_id}' does not support fine-tuning.")
        raise NotImplementedError

    def cache_model_artifact(self, cache_path: str, *, overwrite: bool = False) -> Self:
        """
        Bundle the model weights from Hugging Face and the serve script into a ``model.tar.gz`` on S3.

        Deploying from the bundled artifact skips the weight download, which is required for network-isolated
        endpoints such as serverless ones (``deploy(inference_mode="serverless")``).

        The artifact is written to ``{cache_path}/{model_id}/model.tar.gz``. If it already exists, the upload is
        skipped unless ``overwrite=True``, and ``RuntimeError`` is raised if it was bundled with a different
        autogluon-cloud version.

        Parameters
        ----------
        cache_path: str
            S3 prefix to upload the artifact to. Multiple foundation models can share one prefix.
        overwrite: bool, default = False
            If True, re-bundle and upload the artifact even if it already exists.

        Returns
        -------
        TimeSeriesFoundationModel | TabularFoundationModel
            A copy of this model with ``model_artifact_uri`` set to the uploaded artifact. The original is unchanged.
        """
        from huggingface_hub import snapshot_download

        if not cache_path.startswith("s3://"):
            raise ValueError(f"cache_path must be an s3:// URI, got: {cache_path!r}")
        if self._config.model_source_hyperparameter is None:
            raise ValueError(
                f"Model '{self.model_id}' does not support cache_model_artifact: its weights are downloaded by "
                f"AutoGluon at runtime and cannot be loaded from a bundled artifact."
            )

        source_uri = self._config.model_source_uri
        cache_key = f"{cache_path.rstrip('/')}/{self.model_id}/model.tar.gz"
        bucket, key = s3_path_to_bucket_prefix(cache_key)
        s3 = self._backend.sagemaker_session.boto_session.client("s3")

        head = None if overwrite else _s3_head_or_none(s3, bucket, key)
        if head is not None:
            cached_version = head["Metadata"].get(_AG_CLOUD_VERSION_METADATA_KEY)
            if cached_version != __version__:
                raise RuntimeError(
                    f"Cached artifact at {cache_key} was bundled with autogluon-cloud "
                    f"{cached_version!r}, current is {__version__!r}. "
                    f"Pass overwrite=True to re-bundle and re-upload."
                )
            logger.info(f"Cached artifact already exists at {cache_key}; skipping upload")
        else:
            with tempfile.TemporaryDirectory(prefix="ag_fm_cache_") as tmp:
                tmp_path = Path(tmp)
                weights_dir = tmp_path / "weights"
                logger.info(f"Downloading {source_uri} from HuggingFace to {weights_dir}")
                # trusted AG-owned repo, numeric-only outputs, no code-execution path
                snapshot_download(repo_id=source_uri, local_dir=str(weights_dir))  # nosec B615

                # Mirror the layout produced by SagemakerBackend._create_serve_script_tarball:
                # entry-point script + serving_utils/ under code/, so the cached endpoint can
                # `from serving_utils.timeseries import ...` exactly like a fresh deploy.
                serve_script = Path(self._serve_script_path)
                tarball = tmp_path / "model.tar.gz"
                logger.info(f"Bundling weights + serve script into {tarball}")
                with tarfile.open(tarball, "w:gz") as tar:
                    tar.add(weights_dir, arcname="weights")
                    tar.add(serve_script, arcname=f"code/{serve_script.name}")
                    tar.add(ScriptManager.SAGEMAKER_SERVING_UTILS_DIR, arcname="code/serving_utils")
                logger.info(f"Uploading to {cache_key}")
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

    def to_dict(self) -> dict[str, Any]:
        """Serialize ``model_id``, ``hyperparameters``, and ``model_artifact_uri`` to a dict.

        ``role`` and ``cloud_output_path`` are excluded so the config can be shared across users and accounts.
        """
        out: dict[str, Any] = {"model_id": self.model_id}
        if self._hyperparameter_overrides:
            out["hyperparameters"] = self._hyperparameter_overrides
        if self.model_artifact_uri:
            out["model_artifact_uri"] = self.model_artifact_uri
        return out

    def to_json(self) -> str:
        """Serialize ``model_id``, ``hyperparameters``, and ``model_artifact_uri`` to a JSON string.

        ``role`` and ``cloud_output_path`` are excluded so the config can be shared across users and accounts.
        """
        return json.dumps(self.to_dict())

    @classmethod
    def from_dict(cls, config: dict[str, Any], **runtime_context: Any) -> Self:
        """Create a model from the output of :meth:`to_dict`.

        Pass ``role`` and ``cloud_output_path`` as keyword arguments if needed.
        """
        return cls(**config, **runtime_context)

    @classmethod
    def from_json(cls, s: str, **runtime_context: Any) -> Self:
        """Create a model from the output of :meth:`to_json`.

        Pass ``role`` and ``cloud_output_path`` as keyword arguments if needed.
        """
        return cls.from_dict(json.loads(s), **runtime_context)


class TimeSeriesFoundationModel(FoundationModel):
    """Pretrained time series foundation model for zero-shot forecasting on Amazon SageMaker.

    Wraps pretrained models like `Chronos-2 <https://huggingface.co/autogluon/chronos-2>`_, Chronos-Bolt, and
    Toto-2.0, with no training required. Use :meth:`list_models` to get the supported ``model_id`` values, and see
    `the foundation model tutorial <https://auto.gluon.ai/cloud/stable/tutorials/foundation-model-timeseries.html>`_
    for a full walkthrough.

    Predictions can be produced in three modes:

    * **Batch** — :meth:`predict` runs a one-off SageMaker training job and writes forecasts to S3.
    * **Real-time** — :meth:`deploy` provisions a real-time endpoint; call
      :meth:`TimeSeriesEndpoint.predict` for low-latency inference, then
      :meth:`TimeSeriesEndpoint.delete_endpoint` to tear it down.
    * **Serverless** — :meth:`deploy` with ``inference_mode="serverless"`` provisions a SageMaker
      Serverless Inference endpoint that scales to zero. Requires a cached model artifact (see
      :meth:`cache_model_artifact`).
    """

    _backend_map = {SAGEMAKER: TIMESERIES_SAGEMAKER}
    _predictor_type = "timeseries"
    _problem_types = ("forecasting",)

    @property
    def _serve_script_path(self) -> str:
        return ScriptManager.SAGEMAKER_TIMESERIES_FM_SERVE_SCRIPT_PATH

    @reject_legacy_kwargs
    def deploy(
        self,
        *,
        instance_type: str | None = None,
        endpoint_name: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        framework_version: str = DEFAULT_FRAMEWORK_VERSION,
        wait: bool = True,
        inference_mode: Literal["realtime", "serverless"] = "realtime",
        inference_config: dict[str, Any] | None = None,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[DeployKwargs],
    ) -> TimeSeriesEndpoint:
        """
        Deploy the model to a real-time or serverless endpoint.

        Parameters
        ----------
        instance_type: str | None, default = None
            Instance type for the endpoint. Defaults to the model registry value. Must be ``None``
            when ``inference_mode="serverless"``.
        endpoint_name: str | None, default = None
            Name of the endpoint. If None, a unique name is generated.
        hyperparameters: dict[str, Any] | None, default = None
            Model hyperparameters for inference. Overrides values passed to the constructor.
        framework_version: str, optional
            AutoGluon version, e.g. "1.6". Uses the official AutoGluon DLC image for this version.
        wait: bool, default = True
            Whether to block until the endpoint is ready.
        inference_mode: Literal["realtime", "serverless"], default = "realtime"
            Endpoint type. ``"serverless"`` provisions a SageMaker Serverless Inference endpoint
            (no instance management, scales to zero).
        inference_config: dict[str, Any] | None, default = None
            Serverless settings: ``memory_size_in_mb`` (default 4096), ``max_concurrency`` (default 5), and
            ``provisioned_concurrency``.

        Returns
        -------
        TimeSeriesEndpoint
            Handle to the deployed endpoint.

        Other Parameters
        ----------------
        backend_overrides: dict[str, dict[str, Any]] | None, default = None
            Raw SageMaker request fields for settings without a dedicated argument.

            * Keys: request names from the *SageMaker API* section below.
            * Values: request fields in PascalCase, as in the SageMaker API and boto3. Deep-merged over the request
              built by AutoGluon-Cloud; lists and other non-dict values replace the generated ones.
            * Example: ``{"ProductionVariant": {"ModelDataDownloadTimeoutInSeconds": 1200}}``
        initial_instance_count: int, default = 1
            Number of instances for the endpoint. Ignored when ``inference_mode="serverless"``.
        volume_size: int | None, default = None
            Size in GB of the EBS volume to use for the endpoint. Ignored for GPU instances.
        custom_image_uri: str | None, default = None
            Custom inference container image URI. If set, ``framework_version`` is ignored.

        SageMaker API
        -------------
        * :sm-api:`CreateModel`: registers the model artifact and inference image as a SageMaker model.
        * :sm-api:`CreateEndpointConfig`: defines the endpoint's single :sm-api:`ProductionVariant`: instance type and
          count, or the serverless settings.
        * :sm-api:`CreateEndpoint`: launches the endpoint.

        The endpoint is billed until :meth:`TimeSeriesEndpoint.delete_endpoint` deletes it.
        """
        self._deploy_backend(
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            hyperparameters=hyperparameters,
            framework_version=framework_version,
            wait=wait,
            inference_mode=inference_mode,
            inference_config=inference_config,
            backend_overrides=backend_overrides,
            **kwargs,
        )
        return TimeSeriesEndpoint(
            endpoint_name=self._backend.endpoint_name,
            session=self._backend.sagemaker_session.boto_session,
        )

    def _build_predictor_fit_args(self, hyperparameters: dict[str, Any] | None = None) -> dict[str, Any]:
        merged_hp = self._get_hyperparameters("inference", hyperparameters)
        return {
            "hyperparameters": {self._config.ag_model_key: merged_hp},
            "skip_model_selection": True,
        }

    def _build_predictor_init_args(
        self,
        target: str = "target",
        prediction_length: int = 1,
        quantile_levels: list[float] | None = None,
        **kwargs,
    ) -> dict[str, Any]:
        """Map user kwargs to TimeSeriesPredictor init args."""
        args: dict[str, Any] = {
            "target": target,
            "prediction_length": prediction_length,
        }
        if quantile_levels is not None:
            args["quantile_levels"] = quantile_levels
        return args

    @reject_legacy_kwargs
    def predict(
        self,
        data: str | Path | pd.DataFrame,
        *,
        target: str = "target",
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        known_covariates: str | Path | pd.DataFrame | None = None,
        static_features: str | Path | pd.DataFrame | None = None,
        prediction_length: int = 1,
        quantile_levels: list[float] | None = None,
        predictions_path: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        instance_type: str | None = None,
        framework_version: str = DEFAULT_FRAMEWORK_VERSION,
        wait: bool = True,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[TrainingJobKwargs],
    ) -> pd.DataFrame | JobPredictionFuture:
        """
        Forecast the future values of ``data`` in a one-off SageMaker job.

        Parameters
        ----------
        data: str | Path | pd.DataFrame
            Historical time series to forecast from, in long format, as a ``pd.DataFrame`` or local/S3 path to
            a data file. See the `TimeSeriesPredictor docs <https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html>`_
            for the expected format.
        target: str, default = "target"
            Name of the column with the values to forecast.
        id_column: str, default = "item_id"
            Name of the column with the ID of each time series.
        timestamp_column: str, default = "timestamp"
            Name of the column with the observation timestamps.
        known_covariates: str | Path | pd.DataFrame | None, default = None
            Future values of the known covariates over the forecast horizon. All columns except ``id_column`` and
            ``timestamp_column`` are used as known covariates.
        static_features: str | Path | pd.DataFrame | None, default = None
            Static (time-independent) features describing each individual time series.
        prediction_length: int, default = 1
            Number of time steps to forecast.
        quantile_levels: list[float] | None, default = None
            Quantiles to forecast, as floats between 0 and 1. Defaults to ``[0.1, 0.2, ..., 0.9]``.
        predictions_path: str | None, default = None
            S3 URL ending in ``.csv`` or ``.parquet`` where the job writes the predictions, e.g.
            ``s3://my-bucket/predictions.csv``. The SageMaker execution role needs ``s3:PutObject`` permission for
            it. Defaults to ``{cloud_output_path}/{job_name}/predictions.csv``. The predictions always use the
            column names ``item_id`` and ``timestamp``, regardless of ``id_column`` and ``timestamp_column``.
        hyperparameters: dict[str, Any] | None, default = None
            Model hyperparameters for inference. Overrides values passed to the constructor.
        instance_type: str | None, default = None
            Instance type for the prediction job. Defaults to the model registry value.
        framework_version: str, optional
            AutoGluon version, e.g. "1.6". Uses the official AutoGluon DLC image for this version.
        wait: bool, default = True
            If True, block until the job completes and return the forecasts. If False, return a
            :class:`~autogluon.cloud.endpoint.prediction_future.JobPredictionFuture` immediately; call its
            ``.status()`` to check progress and ``.result()`` to get the forecasts.

        Returns
        -------
        pd.DataFrame | JobPredictionFuture
            Forecasts with ``item_id`` and ``timestamp`` columns, a ``mean`` column, and one column per quantile
            level if ``wait=True``; a :class:`~autogluon.cloud.endpoint.prediction_future.JobPredictionFuture`
            otherwise.

        Other Parameters
        ----------------
        backend_overrides: dict[str, dict[str, Any]] | None, default = None
            Raw SageMaker request fields for settings without a dedicated argument.

            * Keys: request names from the *SageMaker API* section below.
            * Values: request fields in PascalCase, as in the SageMaker API and boto3. Deep-merged over the request
              built by AutoGluon-Cloud; lists and other non-dict values replace the generated ones.
            * Example: ``{"CreateTrainingJob": {"RetryStrategy": {"MaximumRetryAttempts": 2}}}``
        job_name: str | None, default = None
            Name of the training job that runs the prediction. Auto-generated if not set.
        volume_size: int, default = 100
            Size in GB of the storage volume to use for the job.
        custom_image_uri: str | None, default = None
            Custom container image URI. If set, ``framework_version`` is ignored.
        timeout: int, default = 86400
            Maximum job runtime in seconds. Defaults to 24 hours.

        SageMaker API
        -------------
        * :sm-api:`CreateTrainingJob`: runs the prediction as a training job (not a batch transform job) on
          ``instance_type``. Predictions are written to ``predictions_path``.
        """
        kwargs = check_backend_kwargs(kwargs, TrainingJobKwargs, "predict", IGNORED_FM_JOB_KWARGS)
        self._check_framework_version(framework_version, kwargs.get("custom_image_uri"))
        if instance_type is None:
            instance_type = self._config.predict_instance_type

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

        extra_ag_args: dict[str, Any] = {"predict_after_fit": True, "save_predictor": False}
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
            wait=wait,
            backend_overrides=backend_overrides,
            extra_ag_args=extra_ag_args,
            extra_tags=[{"Key": "autogluon-cloud-model-id", "Value": self.model_id}],
            **kwargs,
        )

        if not wait:
            return JobPredictionFuture(
                job=self._backend._fit_job,
                result_loader=self._backend.get_fit_predict_results,
            )

        return self._backend.get_fit_predict_results()


class TabularFoundationModel(FoundationModel):
    """Foundation model for tabular prediction on Amazon SageMaker.

    Wraps pretrained tabular models like `Mitra <https://huggingface.co/autogluon/mitra-classifier>`_, with no
    training required. Each ``model_id`` targets a single task, either classification (``*-classifier``) or
    regression (``*-regressor``). Use :meth:`list_models` to get the supported ``model_id`` values.

    Predictions can be produced in a one-off SageMaker job with :meth:`predict` / :meth:`predict_proba`, or
    through a real-time endpoint created with :meth:`deploy`. In both modes, labeled ``train_data`` provides the
    in-context examples for each prediction.
    """

    _backend_map = {SAGEMAKER: TABULAR_SAGEMAKER}
    _predictor_type = "tabular"
    _problem_types = ("multiclass", "regression")

    @property
    def _serve_script_path(self) -> str:
        return ScriptManager.SAGEMAKER_TABULAR_FM_SERVE_SCRIPT_PATH

    @reject_legacy_kwargs
    def deploy(
        self,
        *,
        instance_type: str | None = None,
        endpoint_name: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        framework_version: str = DEFAULT_FRAMEWORK_VERSION,
        wait: bool = True,
        inference_mode: Literal["realtime"] = "realtime",
        inference_config: dict[str, Any] | None = None,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[DeployKwargs],
    ) -> TabularEndpoint:
        """Deploy the model to a real-time endpoint.

        Each request to the endpoint sends the labeled ``train_data`` along with the rows to predict. Serverless
        endpoints are not supported.

        Parameters
        ----------
        instance_type: str | None, default = None
            Instance type for the endpoint. Defaults to the model registry value.
        endpoint_name: str | None, default = None
            Name of the endpoint. If None, a unique name is generated.
        hyperparameters: dict[str, Any] | None, default = None
            Model hyperparameters for inference. Overrides values passed to the constructor.
        framework_version: str, optional
            AutoGluon version, e.g. "1.6". Uses the official AutoGluon DLC image for this version.
        wait: bool, default = True
            Whether to block until the endpoint is ready.
        inference_mode: Literal["realtime"], default = "realtime"
            Endpoint type. Only ``"realtime"`` is supported.
        inference_config: dict[str, Any] | None, default = None
            Not supported; must be None.

        Returns
        -------
        TabularEndpoint
            Handle to the deployed endpoint.

        Other Parameters
        ----------------
        backend_overrides: dict[str, dict[str, Any]] | None, default = None
            Raw SageMaker request fields for settings without a dedicated argument.

            * Keys: request names from the *SageMaker API* section below.
            * Values: request fields in PascalCase, as in the SageMaker API and boto3. Deep-merged over the request
              built by AutoGluon-Cloud; lists and other non-dict values replace the generated ones.
            * Example: ``{"ProductionVariant": {"ModelDataDownloadTimeoutInSeconds": 1200}}``
        initial_instance_count: int, default = 1
            Number of instances for the endpoint.
        volume_size: int | None, default = None
            Size in GB of the EBS volume to use for the endpoint. Ignored for GPU instances.
        custom_image_uri: str | None, default = None
            Custom inference container image URI. If set, ``framework_version`` is ignored.

        SageMaker API
        -------------
        * :sm-api:`CreateModel`: registers the model artifact and inference image as a SageMaker model.
        * :sm-api:`CreateEndpointConfig`: defines the endpoint's single :sm-api:`ProductionVariant`: instance type and
          count.
        * :sm-api:`CreateEndpoint`: launches the endpoint.

        The endpoint is billed until :meth:`TabularEndpoint.delete_endpoint` deletes it.
        """
        if inference_mode != "realtime":
            raise ValueError(
                "TabularFoundationModel.deploy only supports `inference_mode='realtime'`; "
                "SageMaker Serverless Inference does not provide sufficient resources for tabular foundation models."
            )
        if inference_config is not None:
            raise ValueError(
                "`inference_config` is not supported by TabularFoundationModel.deploy because tabular foundation "
                "models do not support SageMaker Serverless Inference."
            )
        self._deploy_backend(
            instance_type=instance_type,
            endpoint_name=endpoint_name,
            hyperparameters=hyperparameters,
            framework_version=framework_version,
            wait=wait,
            inference_mode="realtime",
            backend_overrides=backend_overrides,
            **kwargs,
        )
        return TabularEndpoint(
            endpoint_name=self._backend.endpoint_name,
            session=self._backend.sagemaker_session.boto_session,
        )

    def _build_predictor_init_args(self, label: str = "target", **kwargs) -> dict[str, Any]:
        """Map user kwargs to TabularPredictor init args."""
        return {"label": label, "problem_type": self._config.problem_type}

    def _build_predictor_fit_args(self, hyperparameters: dict[str, Any] | None = None) -> dict[str, Any]:
        merged_hp = self._get_hyperparameters("inference", hyperparameters)
        return {
            "hyperparameters": {self._config.ag_model_key: merged_hp},
            "fit_weighted_ensemble": False,
        }

    def _load_results(
        self, *, include_predict: bool, predict_only: bool = False
    ) -> tuple[pd.Series, pd.DataFrame | pd.Series] | pd.DataFrame | pd.Series:
        # The training container writes [pred, <class>_proba...]; regression has only the pred column.
        raw = self._backend.get_fit_predict_results()
        pred, pred_proba = split_pred_and_pred_proba(raw)
        if pred_proba is None:  # regression: proba mirrors pred, matching TabularPredictor.predict_proba
            pred_proba = pred
        if predict_only:
            return pred
        elif include_predict:
            return pred, pred_proba
        else:
            return pred_proba

    @reject_legacy_kwargs
    def predict(
        self,
        test_data: str | Path | pd.DataFrame,
        train_data: str | Path | pd.DataFrame,
        label: str,
        *,
        predictions_path: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        instance_type: str | None = None,
        framework_version: str = DEFAULT_FRAMEWORK_VERSION,
        wait: bool = True,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[TrainingJobKwargs],
    ) -> pd.Series | JobPredictionFuture:
        """
        Predict ``test_data`` in a one-off SageMaker job, using the labeled ``train_data`` as in-context examples.

        Parameters
        ----------
        test_data: str | Path | pd.DataFrame
            Rows to predict, as a ``pd.DataFrame`` or local/S3 path to a data file. Must contain all feature columns
            of ``train_data``.
        train_data: str | Path | pd.DataFrame
            Labeled examples, as a ``pd.DataFrame`` or local/S3 path to a data file.
        label: str
            Name of the label column in ``train_data``.
        predictions_path: str | None, default = None
            S3 URL ending in ``.csv`` or ``.parquet`` where the job writes the predictions, e.g.
            ``s3://my-bucket/predictions.csv``. Defaults to ``{cloud_output_path}/{job_name}/predictions.csv``.
        hyperparameters: dict[str, Any] | None, default = None
            Model hyperparameters for inference. Overrides values passed to the constructor.
        instance_type: str | None, default = None
            Instance type for the prediction job. Defaults to the model registry value.
        framework_version: str, optional
            AutoGluon version, e.g. "1.6". Uses the official AutoGluon DLC image for this version.
        wait: bool, default = True
            If True, block until the job completes and return the predictions. If False, return a
            :class:`~autogluon.cloud.endpoint.prediction_future.JobPredictionFuture` immediately; call its
            ``.status()`` to check progress and ``.result()`` to get the predictions.

        Returns
        -------
        pd.Series | JobPredictionFuture
            Predictions if ``wait=True``; a
            :class:`~autogluon.cloud.endpoint.prediction_future.JobPredictionFuture` otherwise.

        Other Parameters
        ----------------
        backend_overrides: dict[str, dict[str, Any]] | None, default = None
            Raw SageMaker request fields for settings without a dedicated argument.

            * Keys: request names from the *SageMaker API* section below.
            * Values: request fields in PascalCase, as in the SageMaker API and boto3. Deep-merged over the request
              built by AutoGluon-Cloud; lists and other non-dict values replace the generated ones.
            * Example: ``{"CreateTrainingJob": {"RetryStrategy": {"MaximumRetryAttempts": 2}}}``
        job_name: str | None, default = None
            Name of the training job that runs the prediction. Auto-generated if not set.
        volume_size: int, default = 100
            Size in GB of the storage volume to use for the job.
        custom_image_uri: str | None, default = None
            Custom container image URI. If set, ``framework_version`` is ignored.
        timeout: int, default = 86400
            Maximum job runtime in seconds. Defaults to 24 hours.

        SageMaker API
        -------------
        * :sm-api:`CreateTrainingJob`: runs the prediction as a training job (not a batch transform job) on
          ``instance_type``. Predictions are written to ``predictions_path``.
        """
        # Checked here too so warnings and errors name `predict()`; the check in `predict_proba()` is then a no-op.
        kwargs = check_backend_kwargs(kwargs, TrainingJobKwargs, "predict", IGNORED_FM_JOB_KWARGS)
        result = self.predict_proba(
            test_data,
            train_data,
            label=label,
            include_predict=True,
            predictions_path=predictions_path,
            hyperparameters=hyperparameters,
            instance_type=instance_type,
            framework_version=framework_version,
            wait=wait,
            backend_overrides=backend_overrides,
            **kwargs,
        )
        if not wait:
            return JobPredictionFuture(
                job=self._backend._fit_job,
                result_loader=lambda: self._load_results(include_predict=True, predict_only=True),
            )
        pred, _ = result
        return pred

    @reject_legacy_kwargs
    def predict_proba(
        self,
        test_data: str | Path | pd.DataFrame,
        train_data: str | Path | pd.DataFrame,
        label: str,
        *,
        include_predict: bool = True,
        predictions_path: str | None = None,
        hyperparameters: dict[str, Any] | None = None,
        instance_type: str | None = None,
        framework_version: str = DEFAULT_FRAMEWORK_VERSION,
        wait: bool = True,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[TrainingJobKwargs],
    ) -> tuple[pd.Series, pd.DataFrame | pd.Series] | pd.DataFrame | pd.Series | JobPredictionFuture:
        """
        Predict class probabilities for ``test_data`` in a one-off SageMaker job, using the labeled ``train_data`` as
        in-context examples.

        For regression, the probability result is identical to the prediction.

        Parameters
        ----------
        test_data: str | Path | pd.DataFrame
            Rows to predict, as a ``pd.DataFrame`` or local/S3 path to a data file. Must contain all feature columns
            of ``train_data``.
        train_data: str | Path | pd.DataFrame
            Labeled examples, as a ``pd.DataFrame`` or local/S3 path to a data file.
        label: str
            Name of the label column in ``train_data``.
        include_predict: bool, default = True
            Whether to return the predictions along with the probabilities. Both are computed in the same job.
        predictions_path: str | None, default = None
            S3 URL ending in ``.csv`` or ``.parquet`` where the job writes the predictions, e.g.
            ``s3://my-bucket/predictions.csv``. Defaults to ``{cloud_output_path}/{job_name}/predictions.csv``.
        hyperparameters: dict[str, Any] | None, default = None
            Model hyperparameters for inference. Overrides values passed to the constructor.
        instance_type: str | None, default = None
            Instance type for the prediction job. Defaults to the model registry value.
        framework_version: str, optional
            AutoGluon version, e.g. "1.6". Uses the official AutoGluon DLC image for this version.
        wait: bool, default = True
            If True, block until the job completes and return the result. If False, return a
            :class:`~autogluon.cloud.endpoint.prediction_future.JobPredictionFuture` immediately; call its
            ``.status()`` to check progress and ``.result()`` to get the result.

        Returns
        -------
        tuple[pd.Series, pd.DataFrame | pd.Series] | pd.DataFrame | pd.Series | JobPredictionFuture
            ``(prediction, predict_probability)`` if ``include_predict`` is True, otherwise ``predict_probability``.
            A :class:`~autogluon.cloud.endpoint.prediction_future.JobPredictionFuture` if ``wait=False``.

        Other Parameters
        ----------------
        backend_overrides: dict[str, dict[str, Any]] | None, default = None
            Raw SageMaker request fields for settings without a dedicated argument.

            * Keys: request names from the *SageMaker API* section below.
            * Values: request fields in PascalCase, as in the SageMaker API and boto3. Deep-merged over the request
              built by AutoGluon-Cloud; lists and other non-dict values replace the generated ones.
            * Example: ``{"CreateTrainingJob": {"RetryStrategy": {"MaximumRetryAttempts": 2}}}``
        job_name: str | None, default = None
            Name of the training job that runs the prediction. Auto-generated if not set.
        volume_size: int, default = 100
            Size in GB of the storage volume to use for the job.
        custom_image_uri: str | None, default = None
            Custom container image URI. If set, ``framework_version`` is ignored.
        timeout: int, default = 86400
            Maximum job runtime in seconds. Defaults to 24 hours.

        SageMaker API
        -------------
        * :sm-api:`CreateTrainingJob`: runs the prediction as a training job (not a batch transform job) on
          ``instance_type``. Predictions are written to ``predictions_path``.
        """
        kwargs = check_backend_kwargs(kwargs, TrainingJobKwargs, "predict_proba", IGNORED_FM_JOB_KWARGS)
        self._check_framework_version(framework_version, kwargs.get("custom_image_uri"))
        if instance_type is None:
            instance_type = self._config.predict_instance_type

        if isinstance(train_data, (str, Path)):
            train_data = load_pd.load(str(train_data))
        # Duplicate two tuning rows so AutoGluon does not hold out any rows from the prediction context. Two rather
        # than one, since some models (e.g. Nori) return a 0-d array when predicting a single row.
        tuning_data = train_data.iloc[:2].copy()

        extra_ag_args: dict[str, Any] = {"predict_after_fit": True, "save_predictor": False}
        if self._config.optional_dependencies:
            extra_ag_args["optional_dependencies"] = self._config.optional_dependencies
        if predictions_path is not None:
            extra_ag_args["predictions_path"] = predictions_path
        kwargs["leaderboard"] = False

        self._backend.fit(
            predictor_init_args=self._build_predictor_init_args(label=label),
            predictor_fit_args=self._build_predictor_fit_args(hyperparameters),
            data_channels={"train_data": train_data, "tuning_data": tuning_data, "test_data": test_data},
            framework_version=framework_version,
            instance_type=instance_type,
            wait=wait,
            backend_overrides=backend_overrides,
            extra_ag_args=extra_ag_args,
            extra_tags=[{"Key": "autogluon-cloud-model-id", "Value": self.model_id}],
            **kwargs,
        )

        if not wait:
            return JobPredictionFuture(
                job=self._backend._fit_job,
                result_loader=lambda: self._load_results(include_predict=include_predict),
            )
        return self._load_results(include_predict=include_predict)
