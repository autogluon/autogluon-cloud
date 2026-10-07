from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import pandas as pd
from typing_extensions import Self, Unpack, deprecated

from ..backend.constant import SAGEMAKER, TIMESERIES_SAGEMAKER
from ..endpoint.timeseries_endpoint import TimeSeriesEndpoint
from ..utils.constants import DEFAULT_FRAMEWORK_VERSION
from ..utils.sagemaker_api import (
    IGNORED_TRAINING_KWARGS,
    BatchTransformKwargs,
    TrainingJobKwargs,
    check_backend_kwargs,
    reject_legacy_kwargs,
)
from .cloud_predictor import _DEPRECATED_REAL_TIME, CloudPredictor

logger = logging.getLogger(__name__)


class TimeSeriesCloudPredictor(CloudPredictor[TimeSeriesEndpoint]):
    """Train and deploy AutoGluon time series forecasting models on Amazon SageMaker.

    Wraps :class:`autogluon.timeseries.TimeSeriesPredictor` (`docs <https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html>`_)
    and runs ``fit``, ``predict``, and endpoint deployment as managed SageMaker jobs.
    """

    predictor_file_name = "TimeSeriesCloudPredictor.pkl"
    backend_map = {SAGEMAKER: TIMESERIES_SAGEMAKER}
    _endpoint_cls = TimeSeriesEndpoint

    @property
    def predictor_type(self):
        """
        Type of the underlying AutoGluon predictor.
        """
        return "timeseries"

    def _get_local_predictor_cls(self):
        from autogluon.timeseries import TimeSeriesPredictor

        return TimeSeriesPredictor

    @reject_legacy_kwargs
    def fit(
        self,
        train_data: str | Path | pd.DataFrame | None = None,
        *,
        predictor_init_args: dict[str, Any],
        predictor_fit_args: dict[str, Any] | None = None,
        tuning_data: str | Path | pd.DataFrame | None = None,
        known_covariates: str | Path | pd.DataFrame | None = None,
        static_features: str | Path | pd.DataFrame | None = None,
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
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
            Training time series in long format, as a ``pd.DataFrame`` or local/S3 path to a data file.
            See the `TimeSeriesPredictor.fit docs <https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.fit.html>`_
            for the expected format.
        predictor_init_args: dict
            Arguments forwarded to ``TimeSeriesPredictor()``. See the
            `TimeSeriesPredictor docs <https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html>`_
            for available options (e.g. ``target``, ``prediction_length``, ``freq``, ``eval_metric``,
            ``quantile_levels``, ``known_covariates_names``).
        predictor_fit_args: dict | None, default = None
            Additional fit args forwarded to ``TimeSeriesPredictor.fit()``. See the
            `TimeSeriesPredictor.fit docs <https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.fit.html>`_
            for available options. Must NOT contain ``train_data`` or ``tuning_data`` — pass those as
            explicit arguments above.
        tuning_data: str | pathlib.Path | pd.DataFrame | None, default = None
            Optional tuning data in long format, as a ``pd.DataFrame`` or local/S3 path to a data file.
        known_covariates: str | pathlib.Path | pd.DataFrame | None, default = None
            Values of the known covariates over the training period. Must be provided if
            ``known_covariates_names`` is set in ``predictor_init_args``.
        static_features: str | pathlib.Path | pd.DataFrame | None, default = None
            Static (time-independent) features describing each individual time series.
        id_column: str, default = "item_id"
            Name of the column with the unique identifier of each time series (item).
        timestamp_column: str, default = "timestamp"
            Name of the column with the observation timestamps.
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
        TimeSeriesCloudPredictor
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
            Name of the training job. Defaults to a unique name with prefix ``ag-cloud-timeseries``.
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
        """
        kwargs = check_backend_kwargs(kwargs, TrainingJobKwargs, "fit", IGNORED_TRAINING_KWARGS)
        self._fit(
            data_channels={
                "train_data": train_data,
                "tuning_data": tuning_data,
                "known_covariates": known_covariates,
                "static_features": static_features,
            },
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            id_column=id_column,
            timestamp_column=timestamp_column,
            framework_version=framework_version,
            instance_type=instance_type,
            wait=wait,
            backend_overrides=backend_overrides,
            **kwargs,
        )
        return self

    @deprecated(_DEPRECATED_REAL_TIME, category=None)
    def predict_real_time(
        self,
        data: str | pd.DataFrame,
        static_features: str | pd.DataFrame | None = None,
        known_covariates: pd.DataFrame | None = None,
        accept: str = "application/x-parquet",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Predict with the deployed SageMaker endpoint. A deployed SageMaker endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        :meta private:

        .. deprecated::
            Use ``predict()`` of the endpoint returned by :meth:`deploy` instead.

        ``data`` must use the same ``id_column`` / ``timestamp_column`` names that were passed to ``fit()``.

        Parameters
        ----------
        data: str | pd.DataFrame
            Historical time series to forecast from, in long format, as a ``pd.DataFrame`` or local/S3 path to
            a data file.
        static_features: pd.DataFrame | None
            Static (time-independent) features describing each individual time series.
        known_covariates: pd.DataFrame | None
            Future values of the known covariates over the forecast horizon. Must be provided if
            ``known_covariates_names`` was specified at fit time.
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json
        **kwargs: Any
            Additional args that you would pass to `predict` calls of an AutoGluon logic

        Returns
        -------
        pd.DataFrame
            Predict results in ``pd.DataFrame``

        SageMaker API
        -------------
        * :sm-runtime-api:`InvokeEndpoint`: sends the data to the endpoint and returns the predictions. The payload is
          limited to 6 MB (4 MB for serverless endpoints).
        """
        self._warn_deprecated_real_time("predict_real_time")
        return self.backend.predict_real_time(
            test_data=data,
            static_features=static_features,
            known_covariates=known_covariates,
            accept=accept,
            inference_kwargs=kwargs,
        )

    def predict_proba_real_time(self, **kwargs) -> pd.DataFrame:
        """
        :meta private:
        """
        raise ValueError(f"{self.__class__.__name__} does not support predict_proba operation.")

    @reject_legacy_kwargs
    def predict(
        self,
        data: str | pd.DataFrame,
        *,
        static_features: str | pd.DataFrame | None = None,
        known_covariates: str | pd.DataFrame | None = None,
        predictor_path: str | None = None,
        framework_version: str | None = None,
        instance_type: str = "ml.m5.2xlarge",
        wait: bool = True,
        predictions_path: str | None = None,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[BatchTransformKwargs],
    ) -> pd.DataFrame | None:
        """
        Forecast with a SageMaker batch transform job.

        Suited for large datasets. For low-latency predictions, deploy an endpoint with :meth:`deploy` instead.
        ``data`` must use the same ``id_column`` / ``timestamp_column`` names that were passed to :meth:`fit`.

        Parameters
        ----------
        data: str | pd.DataFrame
            Historical time series to forecast from, in long format, as a ``pd.DataFrame`` or local/S3 path to
            a data file.
        static_features: str | pd.DataFrame | None, default = None
            Static (time-independent) features describing each individual time series.
        known_covariates: str | pd.DataFrame | None, default = None
            Future values of the known covariates over the forecast horizon. Must be provided if
            ``known_covariates_names`` was specified at fit time.
        predictor_path: str | None, default = None
            Local or S3 path of the predictor tarball. If ``None``, uses the predictor trained by :meth:`fit`.
        framework_version: str | None, default = None
            AutoGluon version, e.g. ``"1.6"``. Inference uses the official AutoGluon DLC image for this version.
            Defaults to the version used by :meth:`fit`. Ignored if ``custom_image_uri`` is set.
        instance_type: str, default = "ml.m5.2xlarge"
            Instance type of the batch transform job.
        wait: bool, default = True
            Whether to block until the job completes and return the forecast. If ``False``, returns ``None`` once
            the job is launched; use :meth:`get_batch_inference_job_status` to poll it.
        predictions_path: str | None, default = None
            S3 prefix under which the batch transform job writes its results (``<predictions_path>/<input file>.out``).
            Defaults to ``{cloud_output_path}/batch_transform/<timestamp>/results``.

        Returns
        -------
        pd.DataFrame | None
            Forecast in long format with ``item_id``, ``timestamp``, ``mean``, and one column per quantile, or
            ``None`` if ``wait=False``.

        Other Parameters
        ----------------
        backend_overrides: dict[str, dict[str, Any]] | None, default = None
            Raw SageMaker request fields for settings without a dedicated argument.

            * Keys: request names from the *SageMaker API* section below.
            * Values: request fields in PascalCase, as in the SageMaker API and boto3. Deep-merged over the request
              built by AutoGluon-Cloud; lists and other non-dict values replace the generated ones.
            * Example: ``{"CreateTransformJob": {"BatchStrategy": "SingleRecord", "MaxPayloadInMB": 20}}``
        job_name: str | None, default = None
            Name of the batch transform job. Defaults to a unique name with prefix ``ag-cloud-timeseries``.
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
        kwargs = check_backend_kwargs(kwargs, BatchTransformKwargs, "predict")
        return self.backend.predict(
            test_data=data,
            static_features=static_features,
            known_covariates=known_covariates,
            predictor_path=predictor_path,
            framework_version=framework_version,
            instance_type=instance_type,
            wait=wait,
            predictions_path=predictions_path,
            backend_overrides=backend_overrides,
            **kwargs,
        )

    def predict_proba(
        self,
        **kwargs,
    ) -> pd.DataFrame | None:
        """
        :meta private:
        """
        raise ValueError(f"{self.__class__.__name__} does not support predict_proba operation.")

    @reject_legacy_kwargs
    def fit_predict(
        self,
        train_data: str | Path | pd.DataFrame,
        *,
        predictor_init_args: dict[str, Any],
        predictor_fit_args: dict[str, Any] | None = None,
        known_covariates: str | Path | pd.DataFrame | None = None,
        static_features: str | Path | pd.DataFrame | None = None,
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        predictions_path: str | None = None,
        framework_version: str = DEFAULT_FRAMEWORK_VERSION,
        instance_type: str = "ml.m5.2xlarge",
        wait: bool = True,
        backend_overrides: dict[str, dict[str, Any]] | None = None,
        **kwargs: Unpack[TrainingJobKwargs],
    ) -> pd.DataFrame | None:
        """
        Fit and predict in a single SageMaker training job.

        Fits a ``TimeSeriesPredictor`` on ``train_data`` and, in the same job, forecasts the next
        ``prediction_length`` steps after the end of each series in ``train_data``.

        Parameters
        ----------
        train_data: str | pathlib.Path | pd.DataFrame
            Historical time series to train on and forecast from, in long format, as a ``pd.DataFrame`` or
            local/S3 path to a data file.
        predictor_init_args: dict
            Arguments forwarded to ``TimeSeriesPredictor()``. Must include ``prediction_length``. See the
            `TimeSeriesPredictor docs <https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html>`_
            for available options.
        predictor_fit_args: dict | None, default = None
            Additional fit args forwarded to ``TimeSeriesPredictor.fit()``. See the
            `TimeSeriesPredictor.fit docs <https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.fit.html>`_
            for available options. Must NOT contain ``train_data``, ``tuning_data``, or
            ``known_covariates`` — pass those as explicit arguments above.
        known_covariates: str | pathlib.Path | pd.DataFrame | None, default = None
            Future values of the known covariates over the forecast horizon. Must be provided if
            ``known_covariates_names`` was specified in ``predictor_init_args``.
        static_features: str | pathlib.Path | pd.DataFrame | None, default = None
            Static (time-independent) features describing each individual time series.
        id_column: str, default = "item_id"
            Name of the column with the unique identifier of each time series (item).
        timestamp_column: str, default = "timestamp"
            Name of the column with the observation timestamps.
        predictions_path: str | None, default = None
            S3 URL of the predictions file, ending in ``.csv`` or ``.parquet``. The SageMaker execution role must be
            able to write to it. Defaults to ``{cloud_output_path}/{job_name}/predictions.csv``.
        framework_version: str, optional
            AutoGluon version, e.g. ``"1.6"``. Training uses the official AutoGluon DLC image for this version.
            Ignored if ``custom_image_uri`` is set.
        instance_type: str, default = "ml.m5.2xlarge"
            Instance type of the training job.
        wait: bool, default = True
            Whether to block until the job completes. If ``False``, returns ``None`` once the job is launched.

        Returns
        -------
        pd.DataFrame | None
            Forecast in long format with ``item_id``, ``timestamp``, ``mean``, and one column per quantile, or
            ``None`` if ``wait=False``; fetch it later with :meth:`get_fit_predict_results`. Columns are named
            ``item_id`` and ``timestamp`` regardless of ``id_column`` / ``timestamp_column``.

        Other Parameters
        ----------------
        backend_overrides: dict[str, dict[str, Any]] | None, default = None
            Raw SageMaker request fields for settings without a dedicated argument.

            * Keys: request names from the *SageMaker API* section below.
            * Values: request fields in PascalCase, as in the SageMaker API and boto3. Deep-merged over the request
              built by AutoGluon-Cloud; lists and other non-dict values replace the generated ones.
            * Example: ``{"CreateTrainingJob": {"RetryStrategy": {"MaximumRetryAttempts": 2}}}``
        job_name: str | None, default = None
            Name of the training job. Defaults to a unique name with prefix ``ag-cloud-timeseries``.
        volume_size: int, default = 100
            Size in GB of the EBS volume that stores the training data and model artifacts.
        custom_image_uri: str | None, default = None
            Custom training container image URI. If set, ``framework_version`` is ignored.
        timeout: int, default = 86400
            Maximum training job runtime in seconds. Defaults to 24 hours.

        SageMaker API
        -------------
        * :sm-api:`CreateTrainingJob`: trains the predictor and predicts in the same job on ``instance_type``.
          Predictions are written to ``predictions_path``.
        """
        kwargs = check_backend_kwargs(kwargs, TrainingJobKwargs, "fit_predict", IGNORED_TRAINING_KWARGS)
        extra_ag_args = {"predict_after_fit": True}
        if predictions_path is not None:
            extra_ag_args["predictions_path"] = predictions_path

        self._fit(
            data_channels={
                "train_data": train_data,
                "tuning_data": None,
                "known_covariates": known_covariates,
                "static_features": static_features,
            },
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            id_column=id_column,
            timestamp_column=timestamp_column,
            framework_version=framework_version,
            instance_type=instance_type,
            wait=wait,
            backend_overrides=backend_overrides,
            extra_ag_args=extra_ag_args,
            **kwargs,
        )

        if not wait:
            logger.info(
                "fit_predict job launched asynchronously. Use `get_fit_job_status()` "
                "to poll, then `get_fit_predict_results()` to fetch predictions."
            )
            return None

        return self.get_fit_predict_results()

    def get_fit_predict_results(self) -> pd.DataFrame:
        """
        Retrieve the forecast of a completed :meth:`fit_predict` job.

        Returns
        -------
        pd.DataFrame
            Predictions for the forecast horizon.
        """
        return self.backend.get_fit_predict_results()
