from __future__ import annotations
import copy
import logging
from typing import Optional, Dict, Any

import pandas as pd

from autogluon.common.loaders import load_pd

from .cloud_predictor import CloudPredictor

logger = logging.getLogger(__name__)


class TimeSeriesCloudPredictor(CloudPredictor):
    predictor_file_name = "TimeSeriesCloudPredictor.pkl"

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
    
    def _preprocess_data(self, data, id_column, timestamp_column):
        if isinstance(data, str):
            data = load_pd.load(data)
        else:
            data = copy.copy(data)
        cols = data.columns.to_list()
        # Make sure id and timestamp columns are the first two columns
        timestamp_index = cols.index(timestamp_column)
        cols.insert(0, cols.pop(timestamp_index))
        id_index = cols.index(id_column)
        cols.insert(0, cols.pop(id_index))
        data = data[cols]
        
        return data
    
    def fit(
        self,
        *,
        predictor_init_args: Dict[str, Any],
        predictor_fit_args: Dict[str, Any],
        id_column: str,
        timestamp_column: str,
        framework_version: str = "latest",
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size: int = 100,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        autogluon_sagemaker_estimator_kwargs: Dict = None,
        **kwargs,
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
        image_column: str, default = None
            The column name in the training/tuning data that contains the image paths.
            The image paths MUST be absolute paths to you local system.
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
        autogluon_sagemaker_estimator_kwargs: dict, default = dict()
            Any extra arguments needed to initialize AutoGluonSagemakerEstimator
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Framework for all options
        **kwargs:
            Any extra arguments needed to pass to fit.
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Framework.fit for all options

        Returns
        -------
        `TimeSeriesCloudPredictor` object. Returns self.
        """
        predictor_fit_args = copy.deepcopy(predictor_fit_args)
        train_data = predictor_fit_args.pop("train_data")
        tuning_data = predictor_fit_args.pop("tuning_data", None)
        train_data = self._preprocess_data(data=train_data, id_column=id_column, timestamp_column=timestamp_column)
        if tuning_data is not None:
            tuning_data = self._preprocess_data(data=tuning_data, id_column=id_column, timestamp_column=timestamp_column)
        predictor_fit_args["train_data"] = train_data
        predictor_fit_args["tuning_data"] = tuning_data
        print(train_data.dtypes)
        print(train_data)
        return super().fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            framework_version=framework_version,
            job_name=job_name,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size=volume_size,
            custom_image_uri=custom_image_uri,
            wait=wait,
            autogluon_sagemaker_estimator_kwargs=autogluon_sagemaker_estimator_kwargs,
            **kwargs,
        )
        