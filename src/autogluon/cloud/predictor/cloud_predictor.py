from __future__ import annotations

import io
import logging
import os
import posixpath
import tarfile
import warnings
from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Any, Generic, Literal, TypeVar

import boto3
import pandas as pd
from typing_extensions import Self, Unpack, deprecated

from autogluon.common.loaders import load_pkl
from autogluon.common.savers import save_pkl
from autogluon.common.utils.log_utils import set_logger_verbosity
from autogluon.common.utils.s3_utils import is_s3_url, s3_path_to_bucket_prefix
from autogluon.common.utils.utils import setup_outputdir

from ..backend.backend import Backend
from ..backend.backend_factory import BackendFactory
from ..backend.constant import SAGEMAKER
from ..endpoint.endpoint import Endpoint
from ..utils.aws_utils import resolve_cloud_output_path
from ..utils.constants import DEFAULT_FRAMEWORK_VERSION
from ..utils.sagemaker_api import (
    IGNORED_TRAINING_KWARGS,
    BatchTransformKwargs,
    DeployKwargs,
    TrainingJobKwargs,
    check_backend_kwargs,
    reject_legacy_kwargs,
)
from ..utils.utils import safe_unpack_archive

logger = logging.getLogger(__name__)

EndpointT = TypeVar("EndpointT", bound=Endpoint)

# Marks methods as deprecated for IDEs and type checkers only (category=None); the methods emit their own
# FutureWarning at runtime, naming the concrete predictor and endpoint classes.
_DEPRECATED_REAL_TIME = "Deploy an endpoint with `deploy()` and call `predict()` / `predict_proba()` on it instead."


def _reject_image_column(kwargs: dict[str, Any], name: str) -> None:
    """Raise a pointer to MultiModalCloudPredictor if ``name`` is passed to a predictor without image support."""
    if name in kwargs:
        raise ValueError(
            f"`{name}` is no longer supported for tabular predictors: image models in AutoGluon-Tabular require "
            "autogluon.multimodal, which is being deprecated. Use `MultiModalCloudPredictor` for image data."
        )


class CloudPredictor(ABC, Generic[EndpointT]):
    predictor_file_name = "CloudPredictor.pkl"
    backend_map = {}
    _endpoint_cls: type[EndpointT]

    def __init__(
        self,
        local_output_path: str | None = None,
        cloud_output_path: str | None = None,
        backend: str = SAGEMAKER,
        role: str | None = None,
        verbosity: int = 2,
    ) -> None:
        """
        Parameters
        ----------
        local_output_path: str | None, default = None
            Local directory for the saved predictor, downloaded artifacts, and intermediate files. If ``None``, a
            timestamped folder ``AutogluonCloudPredictor/ag-<timestamp>`` is created in the working directory.
            Reusing the same path for two predictors overwrites the files of the first one.
        cloud_output_path: str | None, default = None
            S3 location where intermediate artifacts and trained models are stored. Accepts:

            * ``s3://bucket`` — a unique timestamped subfolder ``ag-<timestamp>`` is appended,
              so each call gets its own folder and repeated runs don't overwrite each other.
            * ``s3://bucket/prefix`` — used verbatim. Re-running with the same prefix will
              overwrite previously written artifacts.
            * ``None`` (default) — use the bucket saved in ``~/.autogluon/cloud.yaml`` (set
              by :func:`autogluon.cloud.bootstrap` / :func:`autogluon.cloud.register`) and
              append a timestamped subfolder. Raises if no bucket is configured.
        backend: str, default = "sagemaker"
            Cloud backend to use. Currently only ``"sagemaker"`` is supported.
        role: str | None, default = None
            ARN of the SageMaker execution role used to run training and inference jobs. If ``None``, falls back to
            ``role_arn`` in ``~/.autogluon/cloud.yaml`` (set by :func:`autogluon.cloud.bootstrap` /
            :func:`autogluon.cloud.register`), and finally to the role of the current AWS identity.
        verbosity: int, default = 2
            Logging verbosity from 0 (errors only) to 4 (debug).
        """
        self.verbosity = verbosity
        cloud_logger = logging.getLogger("autogluon.cloud")
        set_logger_verbosity(self.verbosity, logger=cloud_logger)
        self.local_output_path = self._setup_local_output_path(local_output_path)
        if backend in ("ray", "ray_aws"):
            raise ValueError("The Ray backend was removed in AutoGluon-Cloud v0.7.0. Use backend='sagemaker' instead.")
        if backend not in self.backend_map:
            raise ValueError(f"Unsupported backend {backend!r}. Supported backends: {sorted(self.backend_map)}.")
        self.cloud_output_path = resolve_cloud_output_path(cloud_output_path, backend_name=backend)
        self.backend: Backend = BackendFactory.get_backend(
            backend=self.backend_map[backend],
            local_output_path=self.local_output_path,
            cloud_output_path=self.cloud_output_path,
            predictor_type=self.predictor_type,
            role=role,
        )

    @property
    @abstractmethod
    def predictor_type(self) -> str:
        """
        Type of the underlying AutoGluon predictor.
        """
        raise NotImplementedError

    @property
    def is_fit(self) -> bool:
        """
        Whether the training job has completed successfully.
        """
        return self.backend.is_fit

    @property
    def endpoint_name(self) -> str | None:
        """
        Name of the most recent endpoint deployed by this predictor, or ``None``.
        """
        return self.backend.endpoint_name

    def info(self) -> dict[str, Any]:
        """
        Return a summary of the predictor: output paths, the training job, batch inference jobs, and the endpoint.
        """
        info = dict(
            local_output_path=self.local_output_path,
            cloud_output_path=self.cloud_output_path,
            fit_job=self.backend.get_fit_job_info(),
            recent_batch_inference_job=self.backend.get_batch_inference_job_info(),
            batch_inference_jobs=self.backend.get_batch_inference_jobs(),
            endpoint=self.endpoint_name,
        )
        return info

    def leaderboard(self) -> pd.DataFrame:
        """
        Return the leaderboard of models trained by the completed ``fit()`` job.

        Returns
        -------
        pd.DataFrame
            Output of the underlying predictor's ``leaderboard()``. Empty if the leaderboard can't be read, e.g.
            when the job hasn't finished or was fit with ``leaderboard=False``.
        """
        model_path = self.backend.get_fit_job_output_path()
        if model_path is None:
            return pd.DataFrame()
        # SageMaker writes the output data (incl. the leaderboard) to output.tar.gz next to model.tar.gz
        bucket, key = s3_path_to_bucket_prefix(model_path)
        key = posixpath.join(posixpath.dirname(key), "output.tar.gz")
        s3 = boto3.client("s3")
        try:
            wholefile = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            fileobj = io.BytesIO(wholefile)
            tarf = tarfile.open(fileobj=fileobj)
            leaderboard = tarf.extractfile("leaderboard.csv")
            df = pd.read_csv(leaderboard)
            return df
        except Exception:
            empty = pd.DataFrame()
            return empty

    def _setup_local_output_path(self, path):
        if path is None:
            utcnow = datetime.utcnow()
            timestamp = utcnow.strftime("%Y%m%d_%H%M%S")
            path = f"AutogluonCloudPredictor{os.path.sep}ag-{timestamp}{os.path.sep}"
        path = setup_outputdir(path)
        util_path = os.path.join(path, "utils")
        try:
            os.makedirs(util_path)
        except FileExistsError:
            logger.warning(
                f"Warning: path already exists! This predictor may overwrite an existing predictor! path={path!r}"
            )
        return os.path.abspath(path)

    @reject_legacy_kwargs
    def fit(
        self,
        train_data: str | Path | pd.DataFrame | None = None,
        *,
        tuning_data: str | Path | pd.DataFrame | None = None,
        predictor_init_args: dict[str, Any],
        predictor_fit_args: dict[str, Any] | None = None,
        framework_version: str = DEFAULT_FRAMEWORK_VERSION,
        instance_type: str = "ml.m5.2xlarge",
        wait: bool = True,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[TrainingJobKwargs],
    ) -> Self:
        """
        Fit the predictor in a SageMaker training job.

        Parameters
        ----------
        train_data: str | pathlib.Path | pd.DataFrame
            Training data, as a ``pd.DataFrame`` or local/S3 path to a data file.
        tuning_data: str | pathlib.Path | pd.DataFrame | None, default = None
            Optional tuning data.
        predictor_init_args: dict
            Arguments forwarded to the underlying predictor's constructor, e.g. ``{"label": "target"}``.
        predictor_fit_args: dict | None, default = None
            Additional fit args forwarded to the underlying predictor's ``fit()``. Must NOT contain
            ``train_data`` or ``tuning_data`` — pass those as explicit arguments above.
        framework_version: str, optional
            AutoGluon version, e.g. ``"1.6"``. Training uses the official AutoGluon DLC image for this version.
            Ignored if ``custom_image_uri`` is set.
        instance_type: str, default = "ml.m5.2xlarge"
            Instance type of the training job.
        wait: bool, default = True
            Whether to block until the job completes. If ``False``, returns once the job is launched; use
            :meth:`get_fit_job_status` to poll it.

        Returns
        -------
        CloudPredictor
            The fitted predictor (``self``).

        Other Parameters
        ----------------
        backend_overrides: dict[str, dict[str, Any]] | None, default = None
            Raw SageMaker request fields for settings without a dedicated argument.

            * Keys: request names from the *SageMaker API* section below.
            * Values: request fields in PascalCase, as in the SageMaker API and boto3. Deep-merged over the request
              built by AutoGluon-Cloud; lists and other non-dict values replace the generated ones.
            * Example: ``{"CreateTrainingJob": {"RetryStrategy": {"MaximumRetryAttempts": 2}}}``
        job_name: str | None, default = None
            Name of the training job. Defaults to a unique name with prefix ``ag-cloud-<predictor_type>``.
        volume_size: int, default = 100
            Size in GB of the EBS volume that stores the training data and model artifacts.
        custom_image_uri: str | None, default = None
            Custom training container image URI. If set, ``framework_version`` is ignored.
        timeout: int, default = 86400
            Maximum training job runtime in seconds. Defaults to 24 hours.

        SageMaker API
        -------------
        * :sm-api:`CreateTrainingJob`: trains the predictor on ``instance_type`` and writes the artifact to
          ``cloud_output_path``.
        """  # noqa: E501
        _reject_image_column(kwargs, "image_column")
        kwargs = check_backend_kwargs(kwargs, TrainingJobKwargs, "fit", IGNORED_TRAINING_KWARGS)
        self._fit(
            data_channels={"train_data": train_data, "tuning_data": tuning_data},
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            framework_version=framework_version,
            instance_type=instance_type,
            wait=wait,
            backend_overrides=backend_overrides,
            **kwargs,
        )
        return self

    def _fit(
        self,
        *,
        data_channels: dict[str, str | Path | pd.DataFrame | None],
        predictor_init_args: dict[str, Any],
        predictor_fit_args: dict[str, Any] | None,
        **backend_args,
    ) -> None:
        """Validate the inputs shared by ``fit()`` and ``fit_predict()``, then launch the training job.

        ``backend_args`` are forwarded to ``backend.fit()``, including internal ones such as ``extra_ag_args``.
        """
        assert not self.backend.is_fit, (
            "Predictor is already fit! To fit additional models, create a new `CloudPredictor`"
        )
        predictor_fit_args = {} if predictor_fit_args is None else dict(predictor_fit_args)
        for key in data_channels:
            if key in predictor_fit_args:
                raise TypeError(
                    f"`{key}` can no longer be passed via `predictor_fit_args`. "
                    f"Pass `{key}` as an explicit argument instead."
                )
        if data_channels["train_data"] is None:
            raise TypeError("missing required argument: 'train_data'")
        self.backend.fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            data_channels=data_channels,
            **backend_args,
        )

    def attach_job(self, job_name: str) -> None:
        """
        Attach to an existing SageMaker training job, e.g. after the local process that launched it crashed.

        Parameters
        ----------
        job_name: str
            Name of the training job.

        SageMaker API
        -------------
        * :sm-api:`DescribeTrainingJob`: polled until the job finishes.
        """
        self.backend.attach_job(job_name)

    def get_fit_job_status(self) -> str:
        """
        Get the status of the training job, e.g. after calling ``fit(wait=False)``.

        Returns
        -------
        str
            One of ``InProgress``, ``Completed``, ``Failed``, ``Stopping``, ``Stopped``, or ``NotCreated``.

        SageMaker API
        -------------
        * :sm-api:`DescribeTrainingJob`: reads the job status.
        """
        return self.backend.get_fit_job_status()

    def get_fit_job_output_path(self) -> str:
        """
        Get the S3 path of the trained predictor artifact (``model.tar.gz``).

        Returns
        -------
        str | None
            S3 path of the artifact, or ``None`` if the training job hasn't completed.
        """
        return self.backend.get_fit_job_output_path()

    def download_trained_predictor(self, predictor_path: str | None = None, save_path: str | None = None) -> str:
        """
        Download and extract the trained predictor.

        Parameters
        ----------
        predictor_path: str | None, default = None
            S3 path of the predictor tarball. If ``None``, uses the artifact of this predictor's training job.
        save_path: str | None, default = None
            Local directory to download to. The predictor is extracted to ``<save_path>/AutoGluonModels``.
            Defaults to ``local_output_path``.

        Returns
        -------
        str
            Path to the extracted predictor directory.
        """
        path = predictor_path
        if not path:
            path = self.backend.get_fit_job_output_path()
        assert path is not None, (
            "No fit job associated with this CloudPredictor. Either attach to a fit job with `attach_job()` or start one with `fit()`"
        )
        assert is_s3_url(path), "Please provide a valid s3 path to the predictor tarball."
        if not save_path:
            save_path = self.local_output_path
        save_path = self._download_predictor(path, save_path)
        if not save_path.endswith("/"):
            save_path += "/"
        return save_path

    def _get_local_predictor_cls(self):
        raise NotImplementedError

    def to_local_predictor(self, predictor_path: str | None = None, save_path: str | None = None, **kwargs):
        """
        Download the trained predictor and load it as a local AutoGluon predictor.

        Parameters
        ----------
        predictor_path: str | None, default = None
            S3 path of the predictor tarball. If ``None``, uses the artifact of this predictor's training job.
        save_path: str | None, default = None
            Local directory to download to. Defaults to ``local_output_path``.
        **kwargs: Any
            Additional args forwarded to the underlying predictor's ``load()``.

        Returns
        -------
        TabularPredictor | TimeSeriesPredictor
            The loaded local predictor, matching the predictor type.
        """
        predictor_cls = self._get_local_predictor_cls()
        local_model_path = self.download_trained_predictor(predictor_path=predictor_path, save_path=save_path)
        return predictor_cls.load(local_model_path, **kwargs)

    @reject_legacy_kwargs
    def deploy(
        self,
        *,
        predictor_path: str | None = None,
        endpoint_name: str | None = None,
        framework_version: str | None = None,
        instance_type: str | None = None,
        wait: bool = True,
        inference_mode: Literal["realtime", "serverless"] = "realtime",
        inference_config: dict[str, Any] | None = None,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[DeployKwargs],
    ) -> EndpointT:
        """
        Deploy a predictor to a real-time inference endpoint.

        Returns a handle to the endpoint; call its ``predict()`` for low-latency inference, then
        ``delete_endpoint()`` to tear it down.

        Parameters
        ----------
        predictor_path: str | None, default = None
            Local or S3 path of the predictor tarball to deploy. If ``None``, deploys the predictor trained by
            :meth:`fit`.
        endpoint_name: str | None, default = None
            Name of the endpoint. If ``None``, a unique name with prefix ``ag-cloud-<predictor_type>`` is generated.
        framework_version: str | None, default = None
            AutoGluon version, e.g. ``"1.6"``. Inference uses the official AutoGluon DLC image for this version.
            Defaults to the version used by :meth:`fit`. Ignored if ``custom_image_uri`` is set.
        instance_type: str | None, default = None
            Instance type of the endpoint. Defaults to ``ml.m5.2xlarge``. Must be ``None`` when
            ``inference_mode="serverless"``.
        wait: bool, default = True
            Whether to block until the endpoint is in service.
        inference_mode: {"realtime", "serverless"}, default = "realtime"
            Endpoint type. ``"serverless"`` provisions a SageMaker Serverless Inference endpoint
            (no instance management, scales to zero).
        inference_config: dict[str, Any] | None, default = None
            Serverless settings (``memory_size_in_mb``, ``max_concurrency``, ``provisioned_concurrency``).

        Returns
        -------
        TabularEndpoint | TimeSeriesEndpoint | MultiModalEndpoint
            Handle to the deployed endpoint, matching the predictor type.

        Other Parameters
        ----------------
        backend_overrides: dict[str, dict[str, Any]] | None, default = None
            Raw SageMaker request fields for settings without a dedicated argument.

            * Keys: request names from the *SageMaker API* section below.
            * Values: request fields in PascalCase, as in the SageMaker API and boto3. Deep-merged over the request
              built by AutoGluon-Cloud; lists and other non-dict values replace the generated ones.
            * Example: ``{"ProductionVariant": {"ModelDataDownloadTimeoutInSeconds": 1200}}``
        initial_instance_count: int, default = 1
            Initial number of endpoint instances. Ignored when ``inference_mode="serverless"``.
        volume_size: int | None, default = None
            Size in GB of the endpoint's EBS volume. Ignored for GPU instances (``ml.g*`` / ``ml.p*``).
        custom_image_uri: str | None, default = None
            Custom inference container image URI. If set, ``framework_version`` is ignored.

        SageMaker API
        -------------
        * :sm-api:`CreateModel`: registers the model artifact and inference image as a SageMaker model.
        * :sm-api:`CreateEndpointConfig`: defines the endpoint's single :sm-api:`ProductionVariant`: instance type and
          count, or the serverless settings.
        * :sm-api:`CreateEndpoint`: launches the endpoint.

        The endpoint is billed until ``delete_endpoint()`` of the returned endpoint deletes it.
        """
        if inference_mode == "serverless" and instance_type is not None:
            raise ValueError("`instance_type` must not be set when `inference_mode='serverless'`.")
        if instance_type is None and inference_mode == "realtime":
            instance_type = "ml.m5.2xlarge"
        kwargs = check_backend_kwargs(kwargs, DeployKwargs, "deploy")
        self._warn_if_endpoint_active()
        self.backend.deploy(
            predictor_path=predictor_path,
            endpoint_name=endpoint_name,
            framework_version=framework_version,
            instance_type=instance_type,
            wait=wait,
            inference_mode=inference_mode,
            inference_config=inference_config,
            backend_overrides=backend_overrides,
            **kwargs,
        )
        return self._endpoint_cls(
            endpoint_name=self.backend.endpoint_name,
            session=self.backend.sagemaker_session.boto_session,
        )

    def _warn_if_endpoint_active(self) -> None:
        # The predictor only remembers the name of its last endpoint, which may since have been deleted through the
        # returned endpoint handle, so ask SageMaker before warning.
        previous_endpoint = self.backend.endpoint_name
        if previous_endpoint is None:
            return
        try:
            status = self.backend.sagemaker_session.sagemaker_client.describe_endpoint(EndpointName=previous_endpoint)[
                "EndpointStatus"
            ]
        except Exception:
            return
        if status not in ("Deleting", "Failed"):
            logger.warning(
                f"This predictor already deployed endpoint {previous_endpoint} (status: {status}). Deploying a new "
                f"endpoint; {previous_endpoint} keeps running and incurring charges until you delete it with "
                f"`{self._endpoint_cls.__name__}('{previous_endpoint}').delete_endpoint()`."
            )

    def _warn_deprecated(self, method: str, replacement: str, stacklevel: int = 3) -> None:
        warnings.warn(
            f"`{type(self).__name__}.{method}` is deprecated and will be removed in a future release. {replacement}",
            FutureWarning,
            stacklevel=stacklevel,
        )

    def _warn_deprecated_real_time(self, method: str) -> None:
        endpoint_method = method.removesuffix("_real_time")
        self._warn_deprecated(
            method,
            f"Use the endpoint returned by `deploy()` instead: `endpoint = predictor.deploy()`, then "
            f"`endpoint.{endpoint_method}(...)`.",
            stacklevel=4,
        )

    @deprecated("Construct the endpoint class directly from the endpoint name instead.", category=None)
    def attach_endpoint(self, endpoint: str) -> None:
        """
        Attach the current CloudPredictor to an existing endpoint.

        :meta private:

        .. deprecated::
            Construct the endpoint class directly to get a handle to an existing endpoint instead, e.g.
            ``TabularEndpoint(endpoint_name)``.

        Parameters
        ----------
        endpoint: str
            Name of the endpoint being attached to.
        """
        self._warn_deprecated(
            "attach_endpoint",
            f"Use `{self._endpoint_cls.__name__}(endpoint_name)` to get a handle to an existing endpoint instead.",
        )
        self.backend.attach_endpoint(endpoint)

    @deprecated("The endpoint returned by `deploy()` is independent of the predictor.", category=None)
    def detach_endpoint(self) -> str:
        """
        Detach the current endpoint and return its name.

        :meta private:

        .. deprecated::
            The endpoint returned by :meth:`deploy` is independent of the predictor, so there is nothing to detach.

        Returns
        -------
        str
            Name of the detached endpoint. Pass it to :meth:`attach_endpoint` to attach it again.
        """
        self._warn_deprecated(
            "detach_endpoint",
            "The endpoint returned by `deploy()` is independent of the predictor, so there is nothing to detach.",
        )
        return self.backend.detach_endpoint()

    @deprecated(_DEPRECATED_REAL_TIME, category=None)
    def predict_real_time(
        self,
        test_data: str | pd.DataFrame,
        test_data_image_column: str | None = None,
        accept: str = "application/x-parquet",
        **kwargs,
    ) -> pd.Series:
        """
        Predict with the deployed endpoint. A deployed endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        :meta private:

        .. deprecated::
            Use ``predict()`` of the endpoint returned by :meth:`deploy` instead.

        Parameters
        ----------
        test_data: str | pd.DataFrame
            The test data to be inferenced. Can be a ``pd.DataFrame``, or a local path to csv file.
        test_data_image_column: default = None
            If provided a csv file or ``pd.DataFrame`` as the test_data and test_data involves image modality,
            you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json
        **kwargs: Any
            Additional args that you would pass to `predict` calls of an AutoGluon logic

        Returns
        -------
        pd.Series
            Predict results in ``pd.Series``

        SageMaker API
        -------------
        * :sm-runtime-api:`InvokeEndpoint`: sends the data to the endpoint and returns the predictions. The payload is
          limited to 6 MB (4 MB for serverless endpoints).
        """
        self._warn_deprecated_real_time("predict_real_time")
        self._validate_inference_kwargs(inference_kwargs=kwargs)
        return self.backend.predict_real_time(
            test_data=test_data, test_data_image_column=test_data_image_column, accept=accept, inference_kwargs=kwargs
        )

    @deprecated(_DEPRECATED_REAL_TIME, category=None)
    def predict_proba_real_time(
        self,
        test_data: str | pd.DataFrame,
        test_data_image_column: str | None = None,
        accept: str = "application/x-parquet",
        **kwargs,
    ) -> pd.DataFrame | pd.Series:
        """
        Predict probability with the deployed endpoint. A deployed endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict_proba()` instead.
        If your problem_type is regression, this functions identically to `predict_real_time`, returning the same output.

        :meta private:

        .. deprecated::
            Use ``predict_proba(..., include_predict=False)`` of the endpoint returned by :meth:`deploy` instead.

        Parameters
        ----------
        test_data: str | pd.DataFrame
            The test data to be inferenced. Can be a ``pd.DataFrame``, or a local path to csv file.
        test_data_image_column: default = None
            If provided a csv file or ``pd.DataFrame`` as the test_data and test_data involves image modality,
            you must specify the column name corresponding to image paths.
            The path MUST be an abspath
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json
        **kwargs: Any
            Additional args that you would pass to `predict` calls of an AutoGluon logic

        Returns
        -------
        pd.DataFrame | pd.Series
            Will return a ``pd.Series`` when it's a regression problem. Will return a ``pd.DataFrame`` otherwise

        SageMaker API
        -------------
        * :sm-runtime-api:`InvokeEndpoint`: sends the data to the endpoint and returns the predictions. The payload is
          limited to 6 MB (4 MB for serverless endpoints).
        """
        self._warn_deprecated_real_time("predict_proba_real_time")
        self._validate_inference_kwargs(inference_kwargs=kwargs)
        return self.backend.predict_proba_real_time(
            test_data=test_data, test_data_image_column=test_data_image_column, accept=accept, inference_kwargs=kwargs
        )

    @reject_legacy_kwargs
    def predict(
        self,
        test_data: str | pd.DataFrame,
        *,
        predictor_path: str | None = None,
        framework_version: str | None = None,
        instance_type: str = "ml.m5.2xlarge",
        wait: bool = True,
        predictions_path: str | None = None,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[BatchTransformKwargs],
    ) -> pd.Series | None:
        """
        Predict with a SageMaker batch transform job.

        Suited for large datasets. For low-latency predictions, deploy an endpoint with :meth:`deploy` instead.

        Parameters
        ----------
        test_data: str | pd.DataFrame
            Data to predict on, as a ``pd.DataFrame`` or local path to a data file.
        predictor_path: str | None, default = None
            Local or S3 path of the predictor tarball. If ``None``, uses the predictor trained by :meth:`fit`.
        framework_version: str | None, default = None
            AutoGluon version, e.g. ``"1.6"``. Inference uses the official AutoGluon DLC image for this version.
            Defaults to the version used by :meth:`fit`. Ignored if ``custom_image_uri`` is set.
        instance_type: str, default = "ml.m5.2xlarge"
            Instance type of the batch transform job.
        wait: bool, default = True
            Whether to block until the job completes and return the predictions. If ``False``, returns ``None`` once
            the job is launched; use :meth:`get_batch_inference_job_status` to poll it.
        predictions_path: str | None, default = None
            S3 prefix under which the batch transform job writes its results (``<predictions_path>/<input file>.out``).
            Defaults to ``{cloud_output_path}/batch_transform/<timestamp>/results``.

        Returns
        -------
        pd.Series | None
            Predictions, or ``None`` if ``wait=False``.

        Other Parameters
        ----------------
        backend_overrides: dict[str, dict[str, Any]] | None, default = None
            Raw SageMaker request fields for settings without a dedicated argument.

            * Keys: request names from the *SageMaker API* section below.
            * Values: request fields in PascalCase, as in the SageMaker API and boto3. Deep-merged over the request
              built by AutoGluon-Cloud; lists and other non-dict values replace the generated ones.
            * Example: ``{"CreateTransformJob": {"BatchStrategy": "SingleRecord", "MaxPayloadInMB": 20}}``
        job_name: str | None, default = None
            Name of the batch transform job. Defaults to a unique name with prefix ``ag-cloud-<predictor_type>``.
        instance_count: int, default = 1
            Number of batch transform instances.
        custom_image_uri: str | None, default = None
            Custom inference container image URI. If set, ``framework_version`` is ignored.

        SageMaker API
        -------------
        * :sm-api:`CreateModel`: registers the predictor artifact and inference image as a SageMaker model.
        * :sm-api:`CreateTransformJob`: runs batch inference on ``instance_count`` x ``instance_type``. Results are
          written to ``predictions_path``.

        The model is deleted when the job finishes. With ``wait=False`` it is kept; delete it with
        :sm-api:`DeleteModel`.
        """
        _reject_image_column(kwargs, "test_data_image_column")
        kwargs = check_backend_kwargs(kwargs, BatchTransformKwargs, "predict")
        return self.backend.predict(
            test_data=test_data,
            predictor_path=predictor_path,
            framework_version=framework_version,
            instance_type=instance_type,
            wait=wait,
            predictions_path=predictions_path,
            backend_overrides=backend_overrides,
            **kwargs,
        )

    @reject_legacy_kwargs
    def predict_proba(
        self,
        test_data: str | pd.DataFrame,
        *,
        include_predict: bool = True,
        predictor_path: str | None = None,
        framework_version: str | None = None,
        instance_type: str = "ml.m5.2xlarge",
        wait: bool = True,
        predictions_path: str | None = None,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[BatchTransformKwargs],
    ) -> tuple[pd.Series, pd.DataFrame | pd.Series] | pd.DataFrame | pd.Series | None:
        """
        Predict class probabilities with a SageMaker batch transform job.

        Suited for large datasets. For low-latency predictions, deploy an endpoint with :meth:`deploy` instead.
        For regression, the "probabilities" are identical to the predictions.

        Parameters
        ----------
        test_data: str | pd.DataFrame
            Data to predict on, as a ``pd.DataFrame`` or local path to a data file.
        include_predict: bool, default = True
            Whether to also return the predictions, saving a second batch transform job.
        predictor_path: str | None, default = None
            Local or S3 path of the predictor tarball. If ``None``, uses the predictor trained by :meth:`fit`.
        framework_version: str | None, default = None
            AutoGluon version, e.g. ``"1.6"``. Inference uses the official AutoGluon DLC image for this version.
            Defaults to the version used by :meth:`fit`. Ignored if ``custom_image_uri`` is set.
        instance_type: str, default = "ml.m5.2xlarge"
            Instance type of the batch transform job.
        wait: bool, default = True
            Whether to block until the job completes and return the predictions. If ``False``, returns ``None`` once
            the job is launched; use :meth:`get_batch_inference_job_status` to poll it.
        predictions_path: str | None, default = None
            S3 prefix under which the batch transform job writes its results (``<predictions_path>/<input file>.out``).
            Defaults to ``{cloud_output_path}/batch_transform/<timestamp>/results``.

        Returns
        -------
        tuple[pd.Series, pd.DataFrame | pd.Series] | pd.DataFrame | pd.Series | None
            ``(prediction, predict_probability)`` if ``include_predict=True``, otherwise ``predict_probability``.
            ``predict_probability`` is a ``pd.DataFrame`` with one column per class, or a ``pd.Series`` for
            regression. If ``wait=False``, returns ``(None, None)`` or ``None``, respectively.

        Other Parameters
        ----------------
        backend_overrides: dict[str, dict[str, Any]] | None, default = None
            Raw SageMaker request fields for settings without a dedicated argument.

            * Keys: request names from the *SageMaker API* section below.
            * Values: request fields in PascalCase, as in the SageMaker API and boto3. Deep-merged over the request
              built by AutoGluon-Cloud; lists and other non-dict values replace the generated ones.
            * Example: ``{"CreateTransformJob": {"BatchStrategy": "SingleRecord", "MaxPayloadInMB": 20}}``
        job_name: str | None, default = None
            Name of the batch transform job. Defaults to a unique name with prefix ``ag-cloud-<predictor_type>``.
        instance_count: int, default = 1
            Number of batch transform instances.
        custom_image_uri: str | None, default = None
            Custom inference container image URI. If set, ``framework_version`` is ignored.

        SageMaker API
        -------------
        * :sm-api:`CreateModel`: registers the predictor artifact and inference image as a SageMaker model.
        * :sm-api:`CreateTransformJob`: runs batch inference on ``instance_count`` x ``instance_type``. Results are
          written to ``predictions_path``.

        The model is deleted when the job finishes. With ``wait=False`` it is kept; delete it with
        :sm-api:`DeleteModel`.
        """
        _reject_image_column(kwargs, "test_data_image_column")
        kwargs = check_backend_kwargs(kwargs, BatchTransformKwargs, "predict_proba")
        return self.backend.predict_proba(
            test_data=test_data,
            include_predict=include_predict,
            predictor_path=predictor_path,
            framework_version=framework_version,
            instance_type=instance_type,
            wait=wait,
            predictions_path=predictions_path,
            backend_overrides=backend_overrides,
            **kwargs,
        )

    def get_batch_inference_job_info(self, job_name: str | None = None) -> dict[str, Any]:
        """
        Get info about a batch inference job launched by this predictor.

        Parameters
        ----------
        job_name: str | None, default = None
            Name of the job. If ``None``, uses the most recent job.

        Returns
        -------
        dict[str, Any] | None
            Job name, status, and result path, or ``None`` if no such job exists.
        """
        return self.backend.get_batch_inference_job_info(job_name)

    def get_batch_inference_job_status(self, job_name: str | None = None) -> str:
        """
        Get the status of a batch inference job, e.g. after calling ``predict(wait=False)``.

        Parameters
        ----------
        job_name: str | None, default = None
            Name of the job. If ``None``, uses the most recent job.

        Returns
        -------
        str
            One of ``InProgress``, ``Completed``, ``Failed``, ``Stopping``, ``Stopped``, or ``NotCreated``.

        SageMaker API
        -------------
        * :sm-api:`DescribeTransformJob`: reads the job status.
        """
        return self.backend.get_batch_inference_job_status(job_name)

    @deprecated("Call `delete_endpoint()` on the endpoint returned by `deploy()` instead.", category=None)
    def cleanup_deployment(self) -> None:
        """
        Delete the deployed endpoint and other artifacts

        :meta private:

        .. deprecated::
            Use ``delete_endpoint()`` of the endpoint returned by :meth:`deploy` instead.

        SageMaker API
        -------------
        * :sm-api:`DeleteEndpoint`, :sm-api:`DeleteEndpointConfig` and :sm-api:`DeleteModel`: delete the endpoint and
          the endpoint config and model created with it.
        """
        self._warn_deprecated(
            "cleanup_deployment", "Use `delete_endpoint()` of the endpoint returned by `deploy()` instead."
        )
        self.backend.cleanup_deployment()

    def _download_predictor(self, path, save_path):
        logger.log(20, "Downloading trained models to local directory")
        predictor_bucket, predictor_key_prefix = s3_path_to_bucket_prefix(path)
        tarball_path = os.path.join(save_path, "model.tar.gz")
        s3 = boto3.client("s3")
        s3.download_file(predictor_bucket, predictor_key_prefix, tarball_path)
        logger.log(20, "Extracting the trained model tarball")
        save_path = os.path.join(save_path, "AutoGluonModels")
        safe_unpack_archive(tarball_path, save_path)
        return save_path

    def save(self, silent: bool = False) -> None:
        """
        Save the predictor to ``local_output_path``, so it can be restored later with :meth:`load`.

        Parameters
        ----------
        silent: bool, default = False
            Whether to suppress the log message.
        """
        path = self.local_output_path
        predictor_file_name = self.predictor_file_name
        save_pkl.save(path=os.path.join(path, predictor_file_name), object=self)

        if not silent:
            logger.log(
                20,
                f"{type(self).__name__} saved. To load, use: predictor = {type(self).__name__}.load({self.local_output_path!r})",
            )

    @classmethod
    def load(cls, path: str, verbosity: int | None = None) -> CloudPredictor:
        """
        Load a predictor saved with :meth:`save`.

        Parameters
        ----------
        path: str
            The ``local_output_path`` of the saved predictor.
        verbosity: int | None, default = None
            If set, overrides the logging verbosity.

        Returns
        -------
        CloudPredictor
            The loaded predictor.
        """
        if verbosity is not None:
            set_logger_verbosity(verbosity, logger=logger)  # Reset logging after load (may be in new Python session)
        if path is None:
            raise ValueError("path cannot be None in load()")

        path = setup_outputdir(path, warn_if_exist=False)  # replace ~ with absolute path if it exists
        predictor: CloudPredictor = load_pkl.load(path=os.path.join(path, cls.predictor_file_name))
        # TODO: Version compatibility check
        return predictor

    def _validate_inference_kwargs(self, inference_kwargs):
        if inference_kwargs.pop("as_pandas", True) is not True:
            logger.warning("as_pandas must be true for real time prediction.")
