from typing import Any, Dict, List, Optional, Union

import pandas as pd

from ..utils.serializers import AutoGluonSerializationWrapper
from .endpoint import Endpoint


class TimeSeriesEndpoint:
    """High-level endpoint for time series prediction.

    Wraps an Endpoint and handles serialization/deserialization,
    providing a clean predict() interface.
    """

    def __init__(
        self,
        endpoint: Endpoint,
        default_target: Optional[str] = None,
        default_id_column: Optional[str] = None,
        default_timestamp_column: Optional[str] = None,
    ):
        self._endpoint = endpoint
        self._default_target = default_target
        self._default_id_column = default_id_column
        self._default_timestamp_column = default_timestamp_column

    @property
    def endpoint_name(self) -> str:
        return self._endpoint.endpoint_name

    def predict(
        self,
        data: Union[str, pd.DataFrame],
        known_covariates: Optional[pd.DataFrame] = None,
        static_features: Optional[pd.DataFrame] = None,
        prediction_length: int = 1,
        target: Optional[str] = None,
        id_column: Optional[str] = None,
        timestamp_column: Optional[str] = None,
        quantile_levels: Optional[List[float]] = None,
        accept: str = "application/x-parquet",
    ) -> pd.DataFrame:
        """
        Run real-time prediction on the deployed endpoint.

        Parameters
        ----------
        data
            Historical time series in long format.
        known_covariates
            Future values of known covariates.
        static_features
            Static metadata for individual items.
        prediction_length
            Number of time steps to forecast.
        target
            Name of the target column.
        id_column
            Name of the item ID column.
        timestamp_column
            Name of the timestamp column.
        quantile_levels
            Quantiles to predict.
        accept
            Response format. Options: 'application/x-parquet', 'text/csv', 'application/json'.

        Returns
        -------
        pd.DataFrame
        """
        target = target or self._default_target
        id_column = id_column or self._default_id_column
        timestamp_column = timestamp_column or self._default_timestamp_column

        inference_kwargs: Dict[str, Any] = {"prediction_length": prediction_length}
        if target is not None:
            inference_kwargs["target"] = target
        if id_column is not None:
            inference_kwargs["id_column"] = id_column
        if timestamp_column is not None:
            inference_kwargs["timestamp_column"] = timestamp_column
        if quantile_levels is not None:
            inference_kwargs["quantile_levels"] = quantile_levels

        # TODO: known_covariates and static_features support in follow-up serde PR
        if known_covariates is not None:
            raise NotImplementedError("known_covariates for real-time endpoints is not yet supported")
        if static_features is not None:
            raise NotImplementedError("static_features for real-time endpoints is not yet supported")

        payload = AutoGluonSerializationWrapper(data=data, inference_kwargs=inference_kwargs)
        return self._endpoint.predict(payload, initial_args={"Accept": accept})

    def delete_endpoint(self) -> None:
        """Delete the endpoint and cleanup artifacts."""
        self._endpoint.delete_endpoint()
