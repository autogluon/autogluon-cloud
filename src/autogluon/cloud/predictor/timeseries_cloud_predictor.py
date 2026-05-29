from __future__ import annotations

import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import pandas as pd

from autogluon.common.utils.s3_utils import is_s3_url

from ..backend.constant import SAGEMAKER, TIMESERIES_SAGEMAKER
from .cloud_predictor import CloudPredictor

logger = logging.getLogger(__name__)


class TimeSeriesCloudPredictor(CloudPredictor):
    predictor_file_name = "TimeSeriesCloudPredictor.pkl"
    backend_map = {SAGEMAKER: TIMESERIES_SAGEMAKER}

    @property
    def predictor_type(self):
        """
        Type of the underneath AutoGluon Predictor
        """
        return "timeseries"

    def _get_local_predictor_cls(self):
        from autogluon.timeseries import TimeSeriesPredictor

        return TimeSeriesPredictor

    def fit(
        self,
        train_data: Optional[Union[str, Path, pd.DataFrame]] = None,
        *,
        tuning_data: Optional[Union[str, Path, pd.DataFrame]] = None,
        known_covariates: Optional[Union[str, Path, pd.DataFrame]] = None,
        static_features: Optional[Union[str, Path, pd.DataFrame]] = None,
        predictor_init_args: Dict[str, Any],
        predictor_fit_args: Optional[Dict[str, Any]] = None,
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        framework_version: str = "latest",
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size: int = 100,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        backend_kwargs: Optional[Dict] = None,
    ) -> TimeSeriesCloudPredictor:
        """
        Fit the predictor with SageMaker.
        This function will first upload necessary config and train data to s3 bucket.
        Then launch a SageMaker training job with the AutoGluon training container.

        Parameters
        ----------
        train_data: Union[str, pathlib.Path, pd.DataFrame]
            Training time-series data, as a DataFrame or local/S3 path to a data file.
        tuning_data: Optional[Union[str, pathlib.Path, pd.DataFrame]], default = None
            Optional tuning data.
        known_covariates: Optional[Union[str, pathlib.Path, pd.DataFrame]], default = None
            Future values of the known covariates over the forecast horizon.
        static_features: Optional[Union[str, pathlib.Path, pd.DataFrame]], default = None
            Optional metadata attributes for each item. For details, see the ``TimeSeriesDataFrame``
            documentation: https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesDataFrame.html
        predictor_init_args: dict
            Init args for the predictor.
        predictor_fit_args: Optional[dict], default = None
            Additional fit args forwarded to ``TimeSeriesPredictor.fit()``. Must NOT contain ``train_data``,
            ``tuning_data``, or ``known_covariates`` — pass those as explicit arguments above.
        framework_version: str, default = `latest`
            Training container version of autogluon.
            If `latest`, will use the latest available container version.
            If provided a specific version, will use this version.
            If `custom_image_uri` is set, this argument will be ignored.
        job_name: str, default = None
            Name of the launched training job.
            If None, CloudPredictor will create one with prefix ag-cloudpredictor
        instance_type: str, default = 'ml.m5.2xlarge'
            Instance type the predictor will be trained on with SageMaker.
        instance_count: int, default = 1
            Number of instance used to fit the predictor.
        volumes_size: int, default = 30
            Size in GB of the EBS volume to use for storing input data during training (default: 30).
            Must be large enough to store training data if File Mode is used (which is the default).
        wait: bool, default = True
            Whether the call should wait until the job completes
            To be noticed, the function won't return immediately because there are some preparations needed prior fit.
            Use `get_fit_job_status` to get job status.
        backend_kwargs: dict, default = None
            Any extra arguments needed to pass to the underneath backend.
            For SageMaker backend, valid keys are:
                1. autogluon_sagemaker_estimator_kwargs
                    Any extra arguments needed to initialize AutoGluonSagemakerEstimator
                    Please refer to https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Estimator for all options
                2. fit_kwargs
                    Any extra arguments needed to pass to fit.
                    Please refer to https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Estimator.fit for all options

        Returns
        -------
        `TimeSeriesCloudPredictor` object. Returns self.
        """
        assert not self.backend.is_fit, (
            "Predictor is already fit! To fit additional models, create a new `CloudPredictor`"
        )
        if backend_kwargs is None:
            backend_kwargs = {}

        predictor_fit_args = {} if predictor_fit_args is None else dict(predictor_fit_args)
        data_channels = {
            "train_data": train_data,
            "tuning_data": tuning_data,
            "known_covariates": known_covariates,
            "static_features": static_features,
        }
        for key in ("train_data", "tuning_data", "known_covariates"):
            if key in predictor_fit_args:
                warnings.warn(
                    f"Passing `{key}` via `predictor_fit_args` is deprecated and will be removed in autogluon.cloud 0.6.0. "
                    f"Pass `{key}` as an explicit argument to `fit()` instead.",
                    FutureWarning,
                    stacklevel=2,
                )
                if data_channels[key] is None:
                    data_channels[key] = predictor_fit_args.pop(key)
                else:
                    raise TypeError(
                        f"`{key}` was passed both as an explicit argument and via `predictor_fit_args`. "
                        f"Pass it only as an explicit argument."
                    )
        if data_channels["train_data"] is None:
            raise TypeError("fit() missing required argument: 'train_data'")

        backend_kwargs = self.backend.parse_backend_fit_kwargs(backend_kwargs)
        self.backend.fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            data_channels=data_channels,
            id_column=id_column,
            timestamp_column=timestamp_column,
            framework_version=framework_version,
            job_name=job_name,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size=volume_size,
            custom_image_uri=custom_image_uri,
            wait=wait,
            **backend_kwargs,
        )

        return self

    def predict_real_time(
        self,
        data: Union[str, pd.DataFrame],
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        known_covariates: Optional[pd.DataFrame] = None,
        accept: str = "application/x-parquet",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Predict with the deployed SageMaker endpoint. A deployed SageMaker endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        ``data`` must use the same ``id_column`` / ``timestamp_column`` names that were passed to ``fit()``.

        Parameters
        ----------
        data: Union(str, pandas.DataFrame)
            The data to forecast from.
            Can be a pandas.DataFrame or a local path to a csv file.
        static_features: Optional[pd.DataFrame]
             An optional data frame describing the metadata attributes of individual items in the item index.
             For more detail, please refer to `TimeSeriesDataFrame` documentation:
             https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesDataFrame.html
        known_covariates : Optional[pd.DataFrame]
            If ``known_covariates_names`` were specified when creating the predictor, it is necessary to provide the
            values of the known covariates for each time series during the forecast horizon.
            For more details, please refer to the `TimeSeriesPredictor.predictor` documentation:
            https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.predict.html
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json
        kwargs:
            Additional args that you would pass to `predict` calls of an AutoGluon logic

        Returns
        -------
        Pandas.DataFrame
        Predict results in DataFrame
        """
        return self.backend.predict_real_time(
            test_data=data,
            static_features=static_features,
            known_covariates=known_covariates,
            accept=accept,
            inference_kwargs=kwargs,
        )

    def predict_proba_real_time(self, **kwargs) -> pd.DataFrame:
        raise ValueError(f"{self.__class__.__name__} does not support predict_proba operation.")

    def predict(
        self,
        data: Union[str, pd.DataFrame],
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        known_covariates: Optional[Union[str, pd.DataFrame]] = None,
        predictor_path: Optional[str] = None,
        framework_version: str = "latest",
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        backend_kwargs: Optional[Dict] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Predict using SageMaker batch transform.
        When minimizing latency isn't a concern, then the batch transform functionality may be easier, more scalable, and more appropriate.
        If you want to minimize latency, use `predict_real_time()` instead.
        To learn more: https://docs.aws.amazon.com/sagemaker/latest/dg/batch-transform.html
        This method would first create a AutoGluonSagemakerInferenceModel with the trained predictor,
        then create a transformer with it, and call transform in the end.

        ``data`` must use the same ``id_column`` / ``timestamp_column`` names that were passed to ``fit()``.

        Parameters
        ----------
        data: Union(str, pandas.DataFrame)
            The data to forecast from.
            Can be a pandas.DataFrame or a local path to a csv file.
        static_features: Optional[Union[str, pd.DataFrame]]
             An optional data frame describing the metadata attributes of individual items in the item index.
             For more detail, please refer to `TimeSeriesDataFrame` documentation:
             https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesDataFrame.html
        known_covariates: Optional[Union[str, pd.DataFrame]]
            Future values of the known covariates over the forecast horizon.
        predictor_path: str
            Path to the predictor tarball you want to use to predict.
            Path can be both a local path or a S3 location.
            If None, will use the most recent trained predictor trained with `fit()`.
        framework_version: str, default = `latest`
            Inference container version of autogluon.
            If `latest`, will use the latest available container version.
            If provided a specific version, will use this version.
            If `custom_image_uri` is set, this argument will be ignored.
        job_name: str, default = None
            Name of the launched training job.
            If None, CloudPredictor will create one with prefix ag-cloudpredictor.
        instance_count: int, default = 1,
            Number of instances used to do batch transform.
        instance_type: str, default = 'ml.m5.2xlarge'
            Instance to be used for batch transform.
        wait: bool, default = True
            Whether to wait for batch transform to complete.
            To be noticed, the function won't return immediately because there are some preparations needed prior transform.
        backend_kwargs: dict, default = None
            Any extra arguments needed to pass to the underneath backend.
            For SageMaker backend, valid keys are:
                1. download: bool, default = True
                    Whether to download the batch transform results to the disk and load it after the batch transform finishes.
                    Will be ignored if `wait` is `False`.
                2. persist: bool, default = True
                    Whether to persist the downloaded batch transform results on the disk.
                    Will be ignored if `download` is `False`
                3. save_path: str, default = None,
                    Path to save the downloaded result.
                    Will be ignored if `download` is `False`.
                    If None, CloudPredictor will create one.
                    If `persist` is `False`, file would first be downloaded to this path and then removed.
                4. model_kwargs: dict, default = dict()
                    Any extra arguments needed to initialize Sagemaker Model
                    Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/model.html#model for all options
                5. transformer_kwargs: dict
                    Any extra arguments needed to pass to transformer.
                    Please refer to https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html#sagemaker.transformer.Transformer for all options.
                6. transform_kwargs:
                    Any extra arguments needed to pass to transform.
                    Please refer to
                    https://sagemaker.readthedocs.io/en/stable/api/inference/transformer.html#sagemaker.transformer.Transformer.transform for all options.
        """
        if backend_kwargs is None:
            backend_kwargs = {}
        backend_kwargs = self.backend.parse_backend_predict_kwargs(backend_kwargs)
        return self.backend.predict(
            test_data=data,
            static_features=static_features,
            known_covariates=known_covariates,
            predictor_path=predictor_path,
            framework_version=framework_version,
            job_name=job_name,
            instance_type=instance_type,
            instance_count=instance_count,
            custom_image_uri=custom_image_uri,
            wait=wait,
            **backend_kwargs,
        )

    def predict_proba(
        self,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        raise ValueError(f"{self.__class__.__name__} does not support predict_proba operation.")

    def fit_predict(
        self,
        train_data: Union[str, Path, pd.DataFrame],
        *,
        known_covariates: Optional[Union[str, Path, pd.DataFrame]] = None,
        static_features: Optional[Union[str, Path, pd.DataFrame]] = None,
        predictor_init_args: Dict[str, Any],
        predictor_fit_args: Optional[Dict[str, Any]] = None,
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        framework_version: str = "latest",
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size: int = 100,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        predictions_path: Optional[str] = None,
        backend_kwargs: Optional[Dict] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fit and predict in a single SageMaker training job.

        This is useful for foundation-model forecasting workflows (e.g. Chronos-2) where "fit" is essentially loading
        a pretrained model. Running fit and predict in the same job avoids the SageMaker startup overhead twice.

        Predictions are generated inside the training container against ``train_data`` (the standard time-series
        forecasting flow where the last ``prediction_length`` steps of each series are forecast) and written
        directly to S3.

        Parameters
        ----------
        train_data: Union[str, pathlib.Path, pd.DataFrame]
            Historical time-series data to train on and forecast from, as a DataFrame or local/S3 path to
            a data file.
        known_covariates: Optional[Union[str, pathlib.Path, pd.DataFrame]], default = None
            Values of the known covariates for each time series during the forecast horizon. Forwarded to
            ``TimeSeriesPredictor.predict`` in the container. For details, see:
            https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.predict.html
        static_features: Optional[Union[str, pathlib.Path, pd.DataFrame]], default = None
            Optional metadata attributes per item.
        predictor_init_args: dict
            Init args for the predictor (must include ``prediction_length``).
        predictor_fit_args: Optional[dict], default = None
            Additional fit args for the predictor. Must NOT contain ``train_data``, ``tuning_data``, or
            ``known_covariates`` — pass those as explicit arguments above.
        id_column: str, default = "item_id"
            Name of the item ID column.
        timestamp_column: str, default = "timestamp"
            Name of the timestamp column.
        framework_version, job_name, instance_type, instance_count, volume_size, custom_image_uri, wait,
        backend_kwargs:
            Same semantics as ``fit()``.
        predictions_path: Optional[str]
            S3 URL where predictions will be written by the training container (e.g.
            ``s3://my-bucket/runs/2024-05-01/predictions.csv``). The container's SageMaker execution role must
            have ``s3:PutObject`` permission for this location. Defaults to
            ``{cloud_output_path}/{job_name}/predictions.csv``. Predictions use AutoGluon's canonical column
            names ``item_id`` and ``timestamp``, regardless of the ``id_column`` / ``timestamp_column`` passed in.

        Returns
        -------
        Optional[pd.DataFrame]
            Predictions as a DataFrame. Returns ``None`` when ``wait`` is False.
        """
        if predictions_path is not None:
            if not is_s3_url(predictions_path) or not predictions_path.endswith((".csv", ".parquet")):
                raise ValueError(
                    f"`predictions_path` must be a full S3 URL ending in '.csv' or '.parquet' "
                    f"(e.g. 's3://bucket/key/predictions.parquet'), got {predictions_path!r}."
                )
        if backend_kwargs is None:
            backend_kwargs = {}
        else:
            backend_kwargs = dict(backend_kwargs)
        extra_ag_args = {"predict_after_fit": True}
        if predictions_path is not None:
            extra_ag_args["predictions_path"] = predictions_path
        backend_kwargs["extra_ag_args"] = extra_ag_args

        self.fit(
            train_data=train_data,
            known_covariates=known_covariates,
            static_features=static_features,
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            id_column=id_column,
            timestamp_column=timestamp_column,
            framework_version=framework_version,
            job_name=job_name,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size=volume_size,
            custom_image_uri=custom_image_uri,
            wait=wait,
            backend_kwargs=backend_kwargs,
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
        Retrieve predictions produced by a completed ``fit_predict`` job.

        Returns
        -------
        pd.DataFrame
            Predictions for the forecast horizon.
        """
        return self.backend.get_fit_predict_results()
