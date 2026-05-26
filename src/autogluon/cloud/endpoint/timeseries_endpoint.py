from typing import Any, Dict, List, Optional, Union

import pandas as pd

from autogluon.common.loaders import load_pd

from ..utils.serializers import AutoGluonSerializationWrapper
from .endpoint import Endpoint


class TimeSeriesEndpoint:
    """High-level endpoint for time series prediction.

    Wraps an Endpoint and handles serialization/deserialization,
    providing a clean predict() interface.
    """

    def __init__(self, endpoint: Endpoint):
        # TODO: replace with sagemaker.Predictor directly (remove Endpoint/SagemakerEndpoint layer)
        self._endpoint = endpoint

    @property
    def endpoint_name(self) -> str:
        return self._endpoint.endpoint_name

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
