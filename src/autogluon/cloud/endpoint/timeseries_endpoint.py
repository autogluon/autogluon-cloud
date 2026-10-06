from typing import Any

import boto3
import pandas as pd

from autogluon.common.loaders import load_pd

from ..utils.aws_utils import setup_sagemaker_session
from ..utils.deserializers import PandasDeserializer
from ..utils.sagemaker_api import delete_endpoint, invoke_endpoint
from ..utils.serializers import AutoGluonSerializationWrapper, AutoGluonSerializer


class TimeSeriesEndpoint:
    """High-level handle for an AutoGluon-Cloud time series inference endpoint.

    Wraps a SageMaker endpoint with the AutoGluon-Cloud serializer/deserializer pair, providing a clean
    :meth:`predict` interface. Use this to attach to an existing endpoint by name. To create a new endpoint, call
    :meth:`autogluon.cloud.TimeSeriesFoundationModel.deploy`, which returns a :class:`TimeSeriesEndpoint` already
    pointing at the new endpoint.
    """

    def __init__(self, endpoint_name: str, session: boto3.Session | None = None):
        """
        Parameters
        ----------
        endpoint_name
            Name of an existing SageMaker endpoint deployed via AutoGluon-Cloud (e.g. through
            :meth:`autogluon.cloud.TimeSeriesFoundationModel.deploy`). The endpoint must understand the AutoGluon-Cloud
            request payload format.
        session
            ``boto3.Session`` used to invoke and delete the endpoint. If ``None``, the default ambient session is used.
        """
        self._endpoint_name = endpoint_name
        self._session = setup_sagemaker_session(boto_session=session)

    @property
    def endpoint_name(self) -> str:
        return self._endpoint_name

    def predict(
        self,
        data: str | pd.DataFrame,
        known_covariates: str | pd.DataFrame | None = None,
        static_features: str | pd.DataFrame | None = None,
        prediction_length: int = 1,
        target: str = "target",
        id_column: str = "item_id",
        timestamp_column: str = "timestamp",
        quantile_levels: list[float] | None = None,
    ) -> pd.DataFrame:
        """
        Run real-time prediction on the deployed endpoint.

        Parameters
        ----------
        data
            Historical time series to forecast from, in long format, as a DataFrame or local/S3 path to a data file.
            See the `TimeSeriesPredictor docs <https://auto.gluon.ai/stable/api/autogluon.timeseries.TimeSeriesPredictor.html>`_
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
            List of increasing decimals between 0 and 1 specifying which quantiles to estimate. Defaults to
            ``[0.1, 0.2, ..., 0.9]``.

        Returns
        -------
        pd.DataFrame

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

        inference_kwargs: dict[str, Any] = {
            "prediction_length": prediction_length,
            "target": target,
            "id_column": id_column,
            "timestamp_column": timestamp_column,
        }
        if quantile_levels is not None:
            inference_kwargs["quantile_levels"] = quantile_levels

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

    def delete_endpoint(self) -> None:
        """Delete the endpoint and its backing model + endpoint config.

        SageMaker API
        -------------
        * :sm-api:`DeleteEndpoint`, :sm-api:`DeleteEndpointConfig` and :sm-api:`DeleteModel`: delete the endpoint and
          the endpoint config and model created with it.
        """
        delete_endpoint(self._endpoint_name, self._session)
