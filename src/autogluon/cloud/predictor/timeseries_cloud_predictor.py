from __future__ import annotations

import logging
from typing import Any, Dict, Optional, Union

import pandas as pd

from ..backend.constant import SAGEMAKER, TIMESERIES_SAGEMAKER
from .cloud_predictor import CloudPredictor

logger = logging.getLogger(__name__)


class TimeSeriesCloudPredictor(CloudPredictor):
    predictor_file_name = "TimeSeriesCloudPredictor.pkl"
    backend_map = {SAGEMAKER: TIMESERIES_SAGEMAKER}

    def __init__(
        self,
        local_output_path: Optional[str] = None,
        cloud_output_path: Optional[str] = None,
        backend: str = SAGEMAKER,
        verbosity: int = 2,
    ) -> None:
        super().__init__(
            local_output_path=local_output_path,
            cloud_output_path=cloud_output_path,
            backend=backend,
            verbosity=verbosity,
        )
        self.target_column: Optional[str] = None
        self.id_column: Optional[str] = None
        self.timestamp_column: Optional[str] = None

    @property
    def predictor_type(self):
        """
        Type of the underneath AutoGluon Predictor
        """
        return "timeseries"

    def _get_local_predictor_cls(self):
        from autogluon.timeseries import TimeSeriesPredictor

        predictor_cls = TimeSeriesPredictor
        return predictor_cls

    def fit(
        self,
        *,
        predictor_init_args: Dict[str, Any],
        predictor_fit_args: Dict[str, Any],
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        static_features: Optional[Union[str, pd.DataFrame]] = None,
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
        predictor_init_args: dict
            Init args for the predictor
        predictor_fit_args: dict
            Fit args for the predictor
        id_column: str, default = "item_id"
            Name of the item ID column
        timestamp_column: str, default = "timestamp"
            Name of the timestamp column
        static_features: Optional[pd.DataFrame]
             An optional data frame describing the metadata attributes of individual items in the item index.
             For more detail, please refer to `TimeSeriesDataFrame` documentation:
             https://auto.gluon.ai/stable/api/autogluon.predictor.html#timeseriesdataframe
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
        assert (
            not self.backend.is_fit
        ), "Predictor is already fit! To fit additional models, create a new `CloudPredictor`"
        if backend_kwargs is None:
            backend_kwargs = {}

        self.target_column = predictor_init_args.get("target", "target")
        self.id_column = id_column
        self.timestamp_column = timestamp_column

        backend_kwargs = self.backend.parse_backend_fit_kwargs(backend_kwargs)
        self.backend.fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            id_column=id_column,
            timestamp_column=timestamp_column,
            static_features=static_features,
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
        test_data: Union[str, pd.DataFrame],
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        accept: str = "application/x-parquet",
        **kwargs,
    ) -> pd.DataFrame:
        """
        Predict with the deployed SageMaker endpoint. A deployed SageMaker endpoint is required.
        This is intended to provide a low latency inference.
        If you want to inference on a large dataset, use `predict()` instead.

        Parameters
        ----------
        test_data: Union(str, pandas.DataFrame)
            The test data to be inferenced.
            Can be a pandas.DataFrame or a local path to a csv file.
        static_features: Optional[pd.DataFrame]
             An optional data frame describing the metadata attributes of individual items in the item index.
             For more detail, please refer to `TimeSeriesDataFrame` documentation:
             https://auto.gluon.ai/stable/api/autogluon.predictor.html#timeseriesdataframe
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
            test_data=test_data,
            id_column=self.id_column,
            timestamp_column=self.timestamp_column,
            target=self.target_column,
            static_features=static_features,
            accept=accept,
        )

    def predict_proba_real_time(self, **kwargs) -> pd.DataFrame:
        raise ValueError(f"{self.__class__.__name__} does not support predict_proba operation.")

    def predict(
        self,
        test_data: Union[str, pd.DataFrame],
        static_features: Optional[Union[str, pd.DataFrame]] = None,
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

        Parameters
        ----------
        test_data: str
            The test data to be inferenced.
            Can be a pandas.DataFrame or a local path to a csv file.
        static_features: Optional[Union[str, pd.DataFrame]]
             An optional data frame describing the metadata attributes of individual items in the item index.
             For more detail, please refer to `TimeSeriesDataFrame` documentation:
             https://auto.gluon.ai/stable/api/autogluon.predictor.html#timeseriesdataframe
        target: str
            Name of column that contains the target values to forecast
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
            test_data=test_data,
            id_column=self.id_column,
            timestamp_column=self.timestamp_column,
            target=self.target_column,
            static_features=static_features,
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
