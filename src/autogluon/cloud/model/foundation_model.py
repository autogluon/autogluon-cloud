"""FoundationModel — predict with pretrained foundation models on AWS."""

from __future__ import annotations

import json
import logging
import tarfile
import tempfile
from abc import abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Tuple, Union

import pandas as pd
from typing_extensions import Self

from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix

from ..backend.backend_factory import BackendFactory
from ..backend.constant import SAGEMAKER, TABULAR_SAGEMAKER, TIMESERIES_SAGEMAKER
from ..endpoint.prediction_future import JobPredictionFuture
from ..endpoint.timeseries_endpoint import TimeSeriesEndpoint
from ..scripts.script_manager import ScriptManager
from ..utils.aws_utils import resolve_cloud_output_path
from ..utils.utils import split_pred_and_pred_proba
from ..version import __version__
from .registry import get_model_config

logger = logging.getLogger(__name__)

# SageMaker extracts model.tar.gz to /opt/ml/model in the container.
_CONTAINER_WEIGHTS_DIR = "/opt/ml/model/weights"

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

    Factory: ``FoundationModel(model_id, ...)`` dispatches on the model's task and returns the
    appropriate task-specific subclass (:class:`TimeSeriesFoundationModel`, ``TabularFoundationModel``).
    Most users instantiate the subclass directly instead.

    Examples
    --------
    >>> model = FoundationModel("chronos-2")  # returns a TimeSeriesFoundationModel
    >>> predictions = model.predict(data, prediction_length=24)
    """

    _backend_map: Dict[str, str] = {}
    _predictor_type: str

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
        cloud_output_path: Optional[str] = None,
        role: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        model_artifact_uri: Optional[str] = None,
        backend: Literal["sagemaker"] = "sagemaker",
    ):
        """
        Parameters
        ----------
        model_id
            ID of the foundation model from the model registry. See
            `Available models <https://auto.gluon.ai/cloud/stable/tutorials/foundation-model-timeseries.html#available-models>`_
            in the foundation model tutorial for the list of supported values.
        cloud_output_path
            S3 location where intermediate artifacts are stored. Accepts:

            * ``s3://bucket`` — a unique timestamped subfolder ``ag-<timestamp>`` is appended.
            * ``s3://bucket/prefix`` — used verbatim. Re-running with the same prefix will overwrite previously written
              artifacts.
            * ``None`` (default) — use the bucket saved in ``~/.autogluon/cloud.yaml`` (set by
              :func:`autogluon.cloud.bootstrap` / :func:`autogluon.cloud.register`) and append a timestamped subfolder.
              Raises if no bucket is configured.
        role
            ARN of the SageMaker execution role used to run training and inference jobs. If ``None``, falls back to
            ``role_arn`` in ``~/.autogluon/cloud.yaml`` (set by :func:`autogluon.cloud.bootstrap` /
            :func:`autogluon.cloud.register`), and finally to ``sagemaker.get_execution_role()``.
        hyperparameters
            Default hyperparameters applied to inference and (when supported) training.
        model_artifact_uri
            S3 URI of a pre-bundled ``model.tar.gz`` produced by :meth:`cache_model_artifact`. When set, deploys skip
            the runtime HuggingFace download and load weights from the bundled artifact.
        backend
            Cloud backend to use.
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
        }

        model_kwargs = backend_kwargs.pop("model_kwargs", {})
        model_kwargs["entry_point"] = self._serve_script_path

        # FM deploys never want SDK repack: predictor_path is either None (script-only tarball is built locally) or a
        # pre-bundled cache artifact that already contains the serve script.
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
            repack=False,
            extra_tags=[{"Key": "autogluon-cloud-model-id", "Value": self.model_id}],
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
    ) -> Self:
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
        if not self._config.fine_tunable:
            raise ValueError(f"Model '{self.model_id}' does not support fine-tuning.")
        raise NotImplementedError

    def cache_model_artifact(self, cache_path: str, *, overwrite: bool = False) -> Self:
        """
        Download model weights from HuggingFace, bundle them with the FM serve script into a SageMaker-compatible
        ``model.tar.gz``, and upload to S3.

        Lets :meth:`deploy` skip the runtime HuggingFace download — required for network-isolated endpoints (e.g.
        SageMaker Serverless Inference). Returns a new :class:`FoundationModel` with ``model_artifact_uri`` set to the
        uploaded tarball.

        Destination key: ``{cache_path}/{model_id}/model.tar.gz``. If it already exists, upload is skipped unless
        ``overwrite=True``; a stale-cache mismatch between the bundled artifact's autogluon-cloud version and the
        current version raises ``RuntimeError`` and prompts the caller to re-bundle.

        Parameters
        ----------
        cache_path
            S3 prefix under which the artifact will be uploaded. Multiple foundation models can share one prefix.
        overwrite
            If True, re-upload even when the destination key exists.

        Returns
        -------
        FoundationModel
            A new instance with ``model_artifact_uri`` populated. The original is unchanged.
        """
        from huggingface_hub import snapshot_download

        if not cache_path.startswith("s3://"):
            raise ValueError(f"cache_path must be an s3:// URI, got: {cache_path!r}")

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

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the model identity. Runtime context (``role``, ``cloud_output_path``) is excluded so configs can
        be shared across users."""
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
    def from_dict(cls, config: Dict[str, Any], **runtime_context: Any) -> Self:
        """Restore from :meth:`to_dict` output. Pass ``role`` / ``cloud_output_path`` as ``runtime_context``."""
        return cls(**config, **runtime_context)

    @classmethod
    def from_json(cls, s: str, **runtime_context: Any) -> Self:
        """Restore from a :meth:`to_json` string."""
        return cls.from_dict(json.loads(s), **runtime_context)


class TimeSeriesFoundationModel(FoundationModel):
    """Pretrained time series foundation model for zero-shot forecasting on Amazon SageMaker.

    Wraps pretrained models like `Chronos-2 <https://huggingface.co/autogluon/chronos-2>`_ and
    Chronos-Bolt and runs prediction as a managed SageMaker job, with no training required. See
    `the foundation model tutorial <https://auto.gluon.ai/cloud/stable/tutorials/foundation-model-timeseries.html>`_
    for the supported ``model_id`` values and a full walkthrough.

    Predictions can be produced in three modes:

    * **Batch** — :meth:`predict` runs a one-off SageMaker training job and writes forecasts to S3.
      Best for one-shot inference.
    * **Real-time** — :meth:`deploy` provisions a real-time endpoint; call
      :meth:`TimeSeriesEndpoint.predict` for low-latency inference, then
      :meth:`TimeSeriesEndpoint.delete_endpoint` to tear it down.
    * **Serverless** — :meth:`deploy` with ``inference_mode="serverless"`` provisions a SageMaker
      Serverless Inference endpoint that scales to zero. Requires a cached model artifact (see
      :meth:`cache_model_artifact`).
    """

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
        return TimeSeriesEndpoint(
            endpoint_name=self._backend.endpoint.endpoint_name,
            session=self._backend.sagemaker_session.boto_session,
        )

    def _build_predictor_fit_args(self, hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged_hp = self._get_hyperparameters("inference", hyperparameters)
        return {
            "hyperparameters": {self._config.ag_model_key: merged_hp},
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
        data: Union[str, Path, pd.DataFrame],
        target: str = "target",
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        known_covariates: Optional[Union[str, Path, pd.DataFrame]] = None,
        static_features: Optional[Union[str, Path, pd.DataFrame]] = None,
        prediction_length: int = 1,
        quantile_levels: Optional[List[float]] = None,
        predictions_path: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        instance_type: Optional[str] = None,
        framework_version: str = "latest",
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
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
        predictions_path
            S3 URL where predictions will be written by the prediction job (e.g.
            ``s3://my-bucket/runs/2024-05-01/predictions.csv``). The container's SageMaker execution
            role must have ``s3:PutObject`` permission for this location. Defaults to
            ``{cloud_output_path}/{job_name}/predictions.csv``. Predictions use AutoGluon's canonical
            column names ``item_id`` and ``timestamp``, regardless of the ``id_column`` /
            ``timestamp_column`` passed in.
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
        **backend_kwargs
            Additional backend-specific arguments (e.g., job_name, volume_size,
            autogluon_sagemaker_estimator_kwargs).

        Returns
        -------
        pd.DataFrame or JobPredictionFuture
            DataFrame if ``wait=True``; a :class:`JobPredictionFuture` otherwise.
        """
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
            extra_tags=[{"Key": "autogluon-cloud-model-id", "Value": self.model_id}],
            **backend_kwargs,
        )

        if not wait:
            return JobPredictionFuture(
                job=self._backend._fit_job,
                result_loader=self._backend.get_fit_predict_results,
            )

        return self._backend.get_fit_predict_results()


class TabularFoundationModel(FoundationModel):
    """Foundation model for tabular prediction on Amazon SageMaker.

    Wraps pretrained tabular models like `Mitra <https://huggingface.co/autogluon/mitra-classifier>`_ and
    runs prediction as a managed SageMaker job, with no training required. Each ``model_id`` targets a
    single task — ``mitra-classifier`` for classification and ``mitra-regressor`` for regression.

    Predictions are produced in batch mode: :meth:`predict` (and :meth:`predict_proba`) runs a one-off
    SageMaker training job where the labeled ``train_data`` provides the in-context examples and the
    predictions for ``test_data`` are written to S3. Both fitting and prediction happen inside that
    single job. Best for one-shot inference.
    """

    _backend_map = {SAGEMAKER: TABULAR_SAGEMAKER}
    _predictor_type = "tabular"

    @property
    def _serve_script_path(self) -> str:
        raise NotImplementedError("Tabular FM deploy is not yet supported")

    def deploy(self, **kwargs):
        raise NotImplementedError("Tabular FM deploy is not yet supported")

    def _build_predictor_init_args(self, label: str = "target", **kwargs) -> Dict[str, Any]:
        """Map user kwargs to TabularPredictor init args. Pins ``problem_type`` from the registry so the
        selected checkpoint's task is enforced rather than inferred from the label column."""
        return {"label": label, "problem_type": self._config.problem_type}

    def _build_predictor_fit_args(self, hyperparameters: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        merged_hp = self._get_hyperparameters("inference", hyperparameters)
        return {
            "hyperparameters": {self._config.ag_model_key: merged_hp},
            "fit_weighted_ensemble": False,
        }

    def predict(
        self,
        test_data: Union[str, Path, pd.DataFrame],
        train_data: Union[str, Path, pd.DataFrame],
        label: str,
        *,
        predictions_path: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        instance_type: Optional[str] = None,
        framework_version: str = "latest",
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        **backend_kwargs,
    ) -> Union[pd.Series, JobPredictionFuture]:
        """
        Run batch prediction for tabular tasks.

        For tabular foundation models (e.g., Mitra), ``train_data`` provides the few-shot context and
        ``test_data`` contains the rows to predict on. Both fitting and prediction run inside a single
        SageMaker training job.

        Parameters
        ----------
        test_data
            Data to predict on. Must contain every feature column present in ``train_data`` (the label column
            is not required).
        train_data
            Labeled few-shot context for the foundation model, as a DataFrame or local/S3 path to a data file.
        label
            Target column name in ``train_data``.
        predictions_path
            S3 URL where predictions will be written by the training container (e.g.
            ``s3://my-bucket/runs/2024-05-01/predictions.csv``). Defaults to
            ``{cloud_output_path}/{job_name}/predictions.csv``.
        hyperparameters
            Model hyperparameters for inference. Overrides values passed to the constructor.
        instance_type
            Instance type for the prediction job. If None, uses registry default.
        framework_version
            Container framework version.
        custom_image_uri
            Custom Docker image URI for the container.
        wait
            If True, block and return the predictions. If False, return a :class:`JobPredictionFuture`
            immediately — call ``.result()`` on it later to retrieve the predictions.
        **backend_kwargs
            Additional backend-specific arguments (e.g., job_name, volume_size).

        Returns
        -------
        pd.Series or JobPredictionFuture
            Predictions as a Series if ``wait=True``; a :class:`JobPredictionFuture` otherwise.
        """
        result = self.predict_proba(
            test_data,
            train_data,
            label=label,
            include_predict=True,
            predictions_path=predictions_path,
            hyperparameters=hyperparameters,
            instance_type=instance_type,
            framework_version=framework_version,
            custom_image_uri=custom_image_uri,
            wait=wait,
            **backend_kwargs,
        )
        if not wait:
            return JobPredictionFuture(job=self._backend._fit_job, result_loader=lambda: result.result()[0])
        pred, _ = result
        return pred

    def predict_proba(
        self,
        test_data: Union[str, Path, pd.DataFrame],
        train_data: Union[str, Path, pd.DataFrame],
        label: str,
        *,
        include_predict: bool = True,
        predictions_path: Optional[str] = None,
        hyperparameters: Optional[Dict[str, Any]] = None,
        instance_type: Optional[str] = None,
        framework_version: str = "latest",
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        **backend_kwargs,
    ) -> Union[Tuple[pd.Series, Union[pd.DataFrame, pd.Series]], Union[pd.DataFrame, pd.Series], JobPredictionFuture]:
        """
        Run batch prediction returning class probabilities.

        Identical to :meth:`predict` but returns class probabilities. For regression the probabilities are
        identical to the predictions.

        Parameters
        ----------
        test_data
            Data to predict on. Must contain every feature column present in ``train_data``.
        train_data
            Labeled few-shot context for the foundation model, as a DataFrame or local/S3 path to a data file.
        label
            Target column name in ``train_data``.
        include_predict
            Whether to return the predictions along with the probabilities. Comes for free — the job always
            computes both.
        predictions_path
            S3 URL where predictions will be written by the training container. Defaults to
            ``{cloud_output_path}/{job_name}/predictions.csv``.
        hyperparameters
            Model hyperparameters for inference. Overrides values passed to the constructor.
        instance_type
            Instance type for the prediction job. If None, uses registry default.
        framework_version
            Container framework version.
        custom_image_uri
            Custom Docker image URI for the container.
        wait
            If True, block and return the result. If False, return a :class:`JobPredictionFuture` immediately.
        **backend_kwargs
            Additional backend-specific arguments (e.g., job_name, volume_size).

        Returns
        -------
        (pd.Series, pd.DataFrame | pd.Series) or (pd.DataFrame | pd.Series) or JobPredictionFuture
            If ``include_predict`` is True, returns ``(prediction, predict_probability)``; otherwise just
            ``predict_probability``. Returns a :class:`JobPredictionFuture` when ``wait=False``.
        """
        if instance_type is None:
            instance_type = self._config.predict_instance_type

        extra_ag_args: Dict[str, Any] = {"predict_after_fit": True}
        if predictions_path is not None:
            extra_ag_args["predictions_path"] = predictions_path

        self._backend.fit(
            predictor_init_args=self._build_predictor_init_args(label=label),
            predictor_fit_args=self._build_predictor_fit_args(hyperparameters),
            data_channels={"train_data": train_data, "test_data": test_data},
            framework_version=framework_version,
            instance_type=instance_type,
            custom_image_uri=custom_image_uri,
            wait=wait,
            extra_ag_args=extra_ag_args,
            extra_tags=[{"Key": "autogluon-cloud-model-id", "Value": self.model_id}],
            **backend_kwargs,
        )

        def _load() -> Union[Tuple[pd.Series, Union[pd.DataFrame, pd.Series]], Union[pd.DataFrame, pd.Series]]:
            # The training container writes [pred, <class>_proba...]; regression has only the pred column.
            raw = self._backend.get_fit_predict_results()
            pred, pred_proba = split_pred_and_pred_proba(raw)
            if pred_proba is None:  # regression: proba mirrors pred, matching TabularPredictor.predict_proba
                pred_proba = pred
            if include_predict:
                return pred, pred_proba
            return pred_proba

        if not wait:
            return JobPredictionFuture(job=self._backend._fit_job, result_loader=_load)
        return _load()
