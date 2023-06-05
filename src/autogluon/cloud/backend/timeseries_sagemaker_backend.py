import copy
from typing import Any, Dict, Optional, Union

import pandas as pd

from autogluon.common.loaders import load_pd

from .constant import TIMESERIES_SAGEMAKER
from .sagemaker_backend import SagemakerBackend


class TimeSeriesSagemakerBackend(SagemakerBackend):
    name = TIMESERIES_SAGEMAKER

    def _preprocess_data(
        self,
        data: Union[pd.DataFrame, str],
        id_column: str,
        timestamp_column: str,
        target: str,
        static_features: Optional[Union[pd.DataFrame, str]] = None,
    ) -> pd.DataFrame:
        if isinstance(data, str):
            data = load_pd.load(data)
        else:
            data = copy.copy(data)
        cols = data.columns.to_list()
        # Make sure id and timestamp columns are the first two columns, and target column is in the end
        # This is to ensure in the container we know how to find id and timestamp columns, and whether there are static features being merged
        timestamp_index = cols.index(timestamp_column)
        cols.insert(0, cols.pop(timestamp_index))
        id_index = cols.index(id_column)
        cols.insert(0, cols.pop(id_index))
        target_index = cols.index(target)
        cols.append(cols.pop(target_index))
        data = data[cols]

        if static_features is not None:
            # Merge static features so only one dataframe needs to be sent to remote container
            if isinstance(static_features, str):
                static_features = load_pd.load(static_features)
            data = pd.merge(data, static_features, how="left", on=id_column)

        return data

    def fit(
        self,
        *,
        predictor_init_args: Dict[str, Any],
        predictor_fit_args: Dict[str, Any],
        id_column: str,
        timestamp_column: str,
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        framework_version: str = "latest",
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size: int = 100,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        autogluon_sagemaker_estimator_kwargs: Optional[Dict] = None,
        fit_kwargs: Optional[Dict] = None,
    ) -> None:
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
        leaderboard: bool, default = True
            Whether to include the leaderboard in the output artifact
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
        fit_kwargs:
            Any extra arguments needed to pass to fit.
            Please refer to https://sagemaker.readthedocs.io/en/stable/api/training/estimators.html#sagemaker.estimator.Framework.fit for all options
        """
        predictor_fit_args = copy.deepcopy(predictor_fit_args)
        train_data = predictor_fit_args.pop("train_data")
        tuning_data = predictor_fit_args.pop("tuning_data", None)
        target = predictor_init_args.get("target")
        train_data = self._preprocess_data(
            data=train_data,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=target,
            static_features=static_features,
        )
        if tuning_data is not None:
            tuning_data = self._preprocess_data(
                data=tuning_data,
                id_column=id_column,
                timestamp_column=timestamp_column,
                target=target,
                static_features=static_features,
            )
        predictor_fit_args["train_data"] = train_data
        predictor_fit_args["tuning_data"] = tuning_data
        super().fit(
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
            fit_kwargs=fit_kwargs,
        )

    def predict_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        id_column: str,
        timestamp_column: str,
        target: str,
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        accept: str = "application/x-parquet",
        inference_kwargs: Optional[Dict[str, Any]] = None,
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
        id_column: str
            Name of the 'item_id' column
        timestamp_column: str
            Name of the 'timestamp' column
        static_features: Optional[pd.DataFrame]
             An optional data frame describing the metadata attributes of individual items in the item index.
             For more detail, please refer to `TimeSeriesDataFrame` documentation:
             https://auto.gluon.ai/stable/api/autogluon.predictor.html#timeseriesdataframe
        target: str
            Name of column that contains the target values to forecast
        accept: str, default = application/x-parquet
            Type of accept output content.
            Valid options are application/x-parquet, text/csv, application/json
        inference_kwargs: Optional[Dict[str, Any]], default = None
            Additional args that you would pass to `predict` calls of an AutoGluon logic

        Returns
        -------
        Pandas.DataFrame
        Predict results in DataFrame
        """
        self._validate_predict_real_time_args(accept)
        test_data = self._preprocess_data(
            data=test_data,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=target,
            static_features=static_features,
        )
        pred, _ = self._predict_real_time(
            test_data=test_data, accept=accept, split_pred_proba=False, inference_kwargs=inference_kwargs
        )
        return pred

    def predict_proba_real_time(self, **kwargs) -> pd.DataFrame:
        raise ValueError(f"{self.__class__.__name__} does not support predict_proba operation.")

    def predict(
        self,
        test_data: Union[str, pd.DataFrame],
        id_column: str,
        timestamp_column: str,
        target: str,
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        **kwargs,
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
        id_column: str
            Name of the 'item_id' column
        timestamp_column: str
            Name of the 'timestamp' column
        static_features: Optional[Union[str, pd.DataFrame]]
             An optional data frame describing the metadata attributes of individual items in the item index.
             For more detail, please refer to `TimeSeriesDataFrame` documentation:
             https://auto.gluon.ai/stable/api/autogluon.predictor.html#timeseriesdataframe
        target: str
            Name of column that contains the target values to forecast
        kwargs:
            Refer to `SagemakerBackend.predict()`
        """
        test_data = self._preprocess_data(
            data=test_data,
            id_column=id_column,
            timestamp_column=timestamp_column,
            target=target,
            static_features=static_features,
        )
        pred, _ = super()._predict(
            test_data=test_data,
            split_pred_proba=False,
            **kwargs,
        )
        return pred

    def predict_proba(
        self,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        raise ValueError(f"{self.__class__.__name__} does not support predict_proba operation.")
