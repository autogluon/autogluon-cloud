from pathlib import Path
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
      ``quantile_levels``, and ``target`` set at fit time, and reject requests that set them to different values.
    * **Foundation model endpoints** (:meth:`TimeSeriesFoundationModel.deploy`) read them from each request.
    """

    def predict(
        self,
        data: str | Path | pd.DataFrame,
        *,
        known_covariates: str | Path | pd.DataFrame | None = None,
        static_features: str | Path | pd.DataFrame | None = None,
        prediction_length: int | None = None,
        target: str | None = None,
        id_column: str | None = None,
        timestamp_column: str | None = None,
        quantile_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Forecast the future values of ``data`` with the deployed endpoint.

        On trained predictor endpoints, ``prediction_length``, ``target``, ``id_column``, ``timestamp_column``, and
        ``quantile_levels`` default to the values set at fit time, and setting ``prediction_length``, ``target``, or
        ``quantile_levels`` to a different value raises an error.

        Parameters
        ----------
        data: str | pathlib.Path | pd.DataFrame
            Historical time series to forecast from, in long format, as a ``pd.DataFrame`` or local/S3 path to a data file.
            See the `TimeSeriesPredictor docs <https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html>`_
            for the expected format.
        known_covariates: str | pathlib.Path | pd.DataFrame | None, default = None
            Future values of the known covariates over the forecast horizon, as a ``pd.DataFrame`` or local/S3 path to
            a data file.
        static_features: str | pathlib.Path | pd.DataFrame | None, default = None
            Static (time-independent) features describing each individual time series, as a ``pd.DataFrame`` or
            local/S3 path to a data file.
        prediction_length: int | None, default = None
            Number of time steps to forecast. Defaults to 1 on foundation model endpoints.
        target: str | None, default = None
            Name of the column with the values to forecast. Defaults to ``"target"`` on foundation model endpoints.
        id_column: str | None, default = None
            Name of the column with the ID of each time series. Defaults to ``"item_id"`` on foundation model
            endpoints.
        timestamp_column: str | None, default = None
            Name of the column with the observation timestamps. Defaults to ``"timestamp"`` on foundation model
            endpoints.
        quantile_levels: list[float] | None, default = None
            Quantiles to forecast, as floats between 0 and 1. Defaults to ``[0.1, 0.2, ..., 0.9]`` on foundation
            model endpoints.

        Returns
        -------
        pd.DataFrame
            Forecasts with ``item_id`` and ``timestamp`` columns, a ``mean`` column, and one column per quantile
            level.

        SageMaker API
        -------------
        * :sm-runtime-api:`InvokeEndpoint`: sends the data to the endpoint and returns the predictions. The payload is
          limited to 6 MB (4 MB for serverless endpoints).
        """
        if isinstance(data, (str, Path)):
            data = load_pd.load(str(data))
        if isinstance(known_covariates, (str, Path)):
            known_covariates = load_pd.load(str(known_covariates))
        if isinstance(static_features, (str, Path)):
            static_features = load_pd.load(str(static_features))

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
