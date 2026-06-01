from typing import Any, Dict, List, Literal, Optional, Union

import pandas as pd

from autogluon.common.loaders import load_pd

from ..utils.serializers import AutoGluonSerializationWrapper
from .prediction_future import PredictionFuture
from .sagemaker_endpoint import SagemakerEndpoint

AsyncAccept = Literal["application/x-parquet", "text/csv"]


class TimeSeriesEndpoint:
    """High-level endpoint for time series prediction.

    Wraps an Endpoint and handles serialization/deserialization,
    providing a clean predict() interface.
    """

    def __init__(self, endpoint: SagemakerEndpoint):
        # TODO: replace with sagemaker.Predictor directly (remove Endpoint/SagemakerEndpoint layer)
        self._endpoint = endpoint

    @property
    def endpoint_name(self) -> str:
        return self._endpoint.endpoint_name

    def _build_payload(
        self,
        data: Union[str, pd.DataFrame],
        known_covariates: Optional[Union[str, pd.DataFrame]],
        static_features: Optional[Union[str, pd.DataFrame]],
        prediction_length: int,
        target: str,
        id_column: str,
        timestamp_column: str,
        quantile_levels: Optional[List[float]],
    ) -> AutoGluonSerializationWrapper:
        if isinstance(data, str):
            data = load_pd.load(data)
        if isinstance(known_covariates, str):
            known_covariates = load_pd.load(known_covariates)
        if isinstance(static_features, str):
            static_features = load_pd.load(static_features)

        inference_kwargs: Dict[str, Any] = {
            "prediction_length": prediction_length,
            "target": target,
            "id_column": id_column,
            "timestamp_column": timestamp_column,
        }
        if quantile_levels is not None:
            inference_kwargs["quantile_levels"] = quantile_levels

        return AutoGluonSerializationWrapper(
            data=data,
            inference_kwargs=inference_kwargs,
            static_features=static_features,
            known_covariates=known_covariates,
        )

    def predict(
        self,
        data: Union[str, pd.DataFrame],
        known_covariates: Optional[Union[str, pd.DataFrame]] = None,
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        prediction_length: int = 1,
        target: str = "target",
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        quantile_levels: Optional[List[float]] = None,
        accept: str = "application/x-parquet",
    ) -> pd.DataFrame:
        """
        Run real-time prediction on the deployed endpoint.

        Parameters
        ----------
        data
            Historical time series to forecast from, in long format, as a DataFrame or local/S3 path to
            a data file. See the `TimeSeriesPredictor docs <https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html>`_
            for the expected format.
        known_covariates
            Future values of the known covariates over the forecast horizon.
        static_features
            Static (time-independent) features describing each individual time series.
        prediction_length
            Forecast horizon: how many time steps into the future the model should predict.
        target
            Name of the column that contains the target values to forecast.
        id_column
            Name of the column with the unique identifier of each time series (item).
        timestamp_column
            Name of the column with the observation timestamps.
        quantile_levels
            List of increasing decimals between 0 and 1 specifying which quantiles to estimate. Defaults
            to ``[0.1, 0.2, ..., 0.9]``.
        accept
            Response format. Options: 'application/x-parquet', 'text/csv', 'application/json'.

        Returns
        -------
        pd.DataFrame
        """
        payload = self._build_payload(
            data,
            known_covariates,
            static_features,
            prediction_length,
            target,
            id_column,
            timestamp_column,
            quantile_levels,
        )
        return self._endpoint.predict(payload, initial_args={"Accept": accept})

    def predict_async(
        self,
        data: Union[str, pd.DataFrame],
        known_covariates: Optional[Union[str, pd.DataFrame]] = None,
        static_features: Optional[Union[str, pd.DataFrame]] = None,
        prediction_length: int = 1,
        target: str = "target",
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        quantile_levels: Optional[List[float]] = None,
        accept: AsyncAccept = "application/x-parquet",
    ) -> PredictionFuture:
        """Submit an asynchronous prediction request.

        Returns a :class:`PredictionFuture` immediately. Forecasting parameters match
        :meth:`predict`. ``accept`` controls the response format written to S3.
        """
        payload = self._build_payload(
            data,
            known_covariates,
            static_features,
            prediction_length,
            target,
            id_column,
            timestamp_column,
            quantile_levels,
        )
        return self._endpoint.predict_async(payload, accept=accept)

    def delete_endpoint(self) -> None:
        """Delete the endpoint and cleanup artifacts."""
        self._endpoint.delete_endpoint()
