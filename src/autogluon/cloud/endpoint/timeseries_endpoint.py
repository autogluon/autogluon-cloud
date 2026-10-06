from typing import Any

import pandas as pd

from autogluon.common.loaders import load_pd

from ..utils.deserializers import PandasDeserializer
from ..utils.sagemaker_api import invoke_endpoint
from ..utils.serializers import AutoGluonSerializationWrapper, AutoGluonSerializer
from .endpoint import Endpoint


class TimeSeriesEndpoint(Endpoint):
    """High-level handle for an AutoGluon-Cloud time series endpoint.

    Returned by :meth:`autogluon.cloud.TimeSeriesCloudPredictor.deploy` and
    :meth:`autogluon.cloud.TimeSeriesFoundationModel.deploy`. Construct it directly to attach to an existing endpoint
    by name.

    * **Trained predictor endpoints** (:meth:`TimeSeriesCloudPredictor.deploy`) use the ``prediction_length``,
      ``quantile_levels``, and ``target`` set at fit time.
    * **Foundation model endpoints** (:meth:`TimeSeriesFoundationModel.deploy`) read them from each request.
    """

    def predict(
        self,
        data: str | pd.DataFrame,
        known_covariates: str | pd.DataFrame | None = None,
        static_features: str | pd.DataFrame | None = None,
        prediction_length: int | None = None,
        target: str | None = None,
        id_column: str | None = None,
        timestamp_column: str | None = None,
        quantile_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Run real-time prediction on the deployed endpoint.

        Parameters
        ----------
        data: str | pd.DataFrame
            Historical time series to forecast from, in long format, as a ``pd.DataFrame`` or local/S3 path to a data file.
            See the `TimeSeriesPredictor docs <https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html>`_
            for the expected format.
        known_covariates: str | pd.DataFrame | None, default = None
            Future values of the known covariates over the forecast horizon.
        static_features: str | pd.DataFrame | None, default = None
            Static (time-independent) features describing each individual time series.
        prediction_length: int | None, default = None
            Foundation model endpoints only. Forecast horizon: how many time steps into the future the model should
            predict. Defaults to 1.
        target: str | None, default = None
            Foundation model endpoints only. Name of the column that contains the target values to forecast.
            Defaults to ``"target"``.
        id_column: str | None, default = None
            Name of the column with the unique identifier of each time series (item). Defaults to the column used at
            fit time for trained predictor endpoints, and to ``"item_id"`` for foundation model endpoints.
        timestamp_column: str | None, default = None
            Name of the column with the observation timestamps. Defaults to the column used at fit time for trained
            predictor endpoints, and to ``"timestamp"`` for foundation model endpoints.
        quantile_levels: list[float] | None, default = None
            Foundation model endpoints only. List of increasing decimals between 0 and 1 specifying which quantiles
            to estimate. Defaults to ``[0.1, 0.2, ..., 0.9]``.

        Returns
        -------
        pd.DataFrame
            Predicted forecasts.

        SageMaker API
        -------------
        * :sm-runtime-api:`InvokeEndpoint`: sends the data to the endpoint and returns the predictions. The payload is
          limited to 6 MB (4 MB for serverless endpoints).
        """
        if isinstance(data, str):
            data = load_pd.load(data)
        if isinstance(known_covariates, str):
            known_covariates = load_pd.load(known_covariates)
        if isinstance(static_features, str):
            static_features = load_pd.load(static_features)

        # Only send the args that were set: the endpoint falls back to its own defaults for the rest.
        inference_kwargs: dict[str, Any] = {
            key: value
            for key, value in {
                "prediction_length": prediction_length,
                "target": target,
                "id_column": id_column,
                "timestamp_column": timestamp_column,
                "quantile_levels": quantile_levels,
            }.items()
            if value is not None
        }

        payload = AutoGluonSerializationWrapper(
            data=data,
            inference_kwargs=inference_kwargs,
            static_features=static_features,
            known_covariates=known_covariates,
        )
        return invoke_endpoint(
            self._endpoint_name,
            self._session,
            payload,
            serializer=AutoGluonSerializer(),
            deserializer=PandasDeserializer(),
            accept="application/x-parquet",
        )
