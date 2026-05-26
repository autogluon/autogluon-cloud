from typing import Any, Dict, Optional, Union

import pandas as pd

from autogluon.common.loaders import load_pd

from ..utils.serializers import AutoGluonSerializationWrapper
from .constant import TIMESERIES_SAGEMAKER
from .sagemaker_backend import SagemakerBackend


class TimeSeriesSagemakerBackend(SagemakerBackend):
    name = TIMESERIES_SAGEMAKER

    def fit(
        self,
        *,
        predictor_init_args: Dict[str, Any],
        predictor_fit_args: Dict[str, Any],
        data_channels: Dict[str, Optional[Union[str, pd.DataFrame]]],
        id_column: str,
        timestamp_column: str,
        framework_version: str = "latest",
        job_name: Optional[str] = None,
        instance_type: str = "ml.m5.2xlarge",
        instance_count: int = 1,
        volume_size: int = 100,
        custom_image_uri: Optional[str] = None,
        wait: bool = True,
        autogluon_sagemaker_estimator_kwargs: Optional[Dict] = None,
        fit_kwargs: Optional[Dict] = None,
        extra_ag_args: Optional[Dict[str, Any]] = None,
    ) -> None:
        """Fit a TimeSeriesPredictor in SageMaker.

        ``id_column`` / ``timestamp_column`` are forwarded to the training script via ``ag_args.pkl``.
        ``known_covariates`` (if present in ``data_channels``) is only honored when
        ``extra_ag_args["predict_after_fit"]`` is True.
        """
        merged_extra_ag_args: Dict[str, Any] = {
            "id_column": id_column,
            "timestamp_column": timestamp_column,
            **(extra_ag_args or {}),
        }
        if data_channels.get("known_covariates") is not None and not merged_extra_ag_args.get(
            "predict_after_fit", False
        ):
            raise ValueError("`known_covariates` should only be provided if `predict_after_fit=True`.")

        super().fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            data_channels=data_channels,
            framework_version=framework_version,
            job_name=job_name,
            instance_type=instance_type,
            instance_count=instance_count,
            volume_size=volume_size,
            custom_image_uri=custom_image_uri,
            wait=wait,
            autogluon_sagemaker_estimator_kwargs=autogluon_sagemaker_estimator_kwargs,
            fit_kwargs=fit_kwargs,
            extra_ag_args=merged_extra_ag_args,
        )

    def predict_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        id_column: str,
        timestamp_column: str,
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        known_covariates: Optional[Union[str, pd.DataFrame]] = None,
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
        static_features: Optional[Union[str, pd.DataFrame]]
             An optional data frame describing the metadata attributes of individual items in the item index.
             For more detail, please refer to `TimeSeriesDataFrame` documentation:
             https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesDataFrame.html
        known_covariates: Optional[Union[str, pd.DataFrame]]
            Future values of the known covariates over the forecast horizon.
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

        if isinstance(test_data, str):
            test_data = load_pd.load(test_data)
        if isinstance(static_features, str):
            static_features = load_pd.load(static_features)
        if isinstance(known_covariates, str):
            known_covariates = load_pd.load(known_covariates)

        if inference_kwargs is None:
            inference_kwargs = {}
        inference_kwargs["id_column"] = id_column
        inference_kwargs["timestamp_column"] = timestamp_column

        wrapper = AutoGluonSerializationWrapper(
            data=test_data,
            inference_kwargs=inference_kwargs,
            static_features=static_features,
            known_covariates=known_covariates,
        )
        pred, _ = self._predict_real_time(test_data=wrapper, accept=accept, split_pred_proba=False)
        return pred

    def predict_proba_real_time(self, **kwargs) -> pd.DataFrame:
        raise ValueError(f"{self.__class__.__name__} does not support predict_proba operation.")

    def predict(
        self,
        test_data: Union[str, pd.DataFrame],
        id_column: str,
        timestamp_column: str,
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        known_covariates: Optional[Union[str, pd.DataFrame]] = None,
        **kwargs,
    ) -> Optional[pd.DataFrame]:
        """Predict using SageMaker batch transform.

        Batch transform sends a single CSV body without per-record metadata, so the serve script
        relies on positional column inference: ``id_column`` at position 0, ``timestamp_column`` at
        position 1. We reorder ``test_data`` here to match that contract.

        ``static_features`` and ``known_covariates`` are not supported in this code path — SageMaker
        batch transform splits the request body and we have no second channel to attach them to. Use
        ``predict_real_time()`` instead.
        """
        if static_features is not None:
            raise NotImplementedError(
                "`static_features` is not supported for batch prediction. Use `predict_real_time()` instead."
            )
        if known_covariates is not None:
            raise NotImplementedError(
                "`known_covariates` is not supported for batch prediction. Use `predict_real_time()` instead."
            )
        if isinstance(test_data, str):
            test_data = load_pd.load(test_data)
        for required in (id_column, timestamp_column):
            if required not in test_data.columns:
                raise ValueError(f"`test_data` must contain column '{required}'.")
        # Reorder so the serve script's positional inference picks up the right columns.
        cols = test_data.columns.to_list()
        reordered = [id_column, timestamp_column] + [c for c in cols if c not in (id_column, timestamp_column)]
        test_data = test_data[reordered]
        pred, _ = super()._predict(
            test_data=test_data,
            split_pred_proba=False,
            **kwargs,
        )
        return pred

    def predict_proba(self, **kwargs) -> Optional[pd.DataFrame]:
        raise ValueError(f"{self.__class__.__name__} does not support predict_proba operation.")
