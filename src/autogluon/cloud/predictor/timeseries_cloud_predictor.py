from __future__ import annotations

import copy
import logging
from typing import Any, Dict, Optional, Union

import pandas as pd

from autogluon.common.loaders import load_pd

from ..backend.constant import RAY, SAGEMAKER, TIMESERIES_SAGEMAKER
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

    @property
    def backend_map(self) -> Dict:
        """
        Map between general backend to module specific backend
        """
        return {SAGEMAKER: TIMESERIES_SAGEMAKER}

    def _get_local_predictor_cls(self):
        from autogluon.timeseries import TimeSeriesPredictor

        predictor_cls = TimeSeriesPredictor
        return predictor_cls

    def predict_real_time(
        self,
        test_data: Union[str, pd.DataFrame],
        id_column: str,
        timestamp_column: str,
        target: str,
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        accept: str = "application/x-parquet",
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
        pred, _ = self._predict_real_time(test_data=test_data, accept=accept, split_pred_proba=False)
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
            Refer to `CloudPredictor.predict()`
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
