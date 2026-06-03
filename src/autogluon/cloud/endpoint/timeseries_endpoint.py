from typing import Any, Dict, List, Optional, Union

import boto3
import pandas as pd
import sagemaker
from sagemaker.predictor import Predictor

from autogluon.common.loaders import load_pd

from ..utils.deserializers import PandasDeserializer
from ..utils.serializers import AutoGluonSerializationWrapper, AutoGluonSerializer


class TimeSeriesEndpoint:
    """High-level handle for an AutoGluon-Cloud time series inference endpoint.

    Wraps a SageMaker endpoint with the AutoGluon-Cloud serializer/deserializer pair, providing a clean
    :meth:`predict` interface. Use this to attach to an existing endpoint by name. To create a new endpoint, call
    :meth:`autogluon.cloud.TimeSeriesFoundationModel.deploy`, which returns a :class:`TimeSeriesEndpoint` already
    pointing at the new endpoint.
    """

    def __init__(self, endpoint_name: str, session: Optional[boto3.Session] = None):
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
        boto_session = session or boto3.Session()
        sagemaker_session = sagemaker.Session(boto_session=boto_session)
        self._predictor = Predictor(
            endpoint_name=endpoint_name,
            sagemaker_session=sagemaker_session,
            serializer=AutoGluonSerializer(),
            deserializer=PandasDeserializer(),
        )

    @property
    def endpoint_name(self) -> str:
        return self._predictor.endpoint_name

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

        payload = AutoGluonSerializationWrapper(
            data=data,
            inference_kwargs=inference_kwargs,
            static_features=static_features,
            known_covariates=known_covariates,
        )
        return self._predictor.predict(payload, initial_args={"Accept": accept})

    def delete_endpoint(self) -> None:
        """Delete the endpoint and its backing model + endpoint config."""
        self._predictor.delete_model()
        self._predictor.delete_endpoint(delete_endpoint_config=True)
