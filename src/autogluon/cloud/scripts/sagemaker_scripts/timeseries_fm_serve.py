"""Serve script for time series foundation models (Chronos, etc.) on SageMaker.

Config comes from the AG_SERVE_CONFIG env var (set by the backend at deploy time):
    {"model_name": "Chronos", "hyperparameters": {"model_path": "amazon/chronos-bolt-base", ...}}
"""

import json
import logging
import os
import pickle
from io import BytesIO, StringIO

import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models import ModelRegistry

logger = logging.getLogger(__name__)

_SERVE_CONFIG = json.loads(os.environ.get("AG_SERVE_CONFIG", "{}"))


def model_fn(model_dir):
    """Instantiate the foundation model and load weights into memory."""
    model_name = _SERVE_CONFIG["model_name"]
    hyperparameters = _SERVE_CONFIG.get("hyperparameters", {})

    model_cls = ModelRegistry.get_model_class(model_name)
    model = model_cls(
        path=model_name,
        freq=None,
        prediction_length=1,
        hyperparameters=hyperparameters,
    )
    model.persist()
    return model


def _build_tsdf(df, id_column, timestamp_column, target):
    """Convert a long-format DataFrame to TimeSeriesDataFrame.

    Column convention (same as TimeSeriesSagemakerBackend._preprocess_data):
    id at position 0, timestamp at position 1. If target is not the last column,
    trailing columns are treated as static features.
    """
    df[timestamp_column] = pd.to_datetime(df[timestamp_column])
    cols = df.columns.to_list()
    static_features = None
    if target != cols[-1]:
        target_index = cols.index(target)
        static_columns = cols[target_index + 1 :]
        static_features = df[[id_column] + static_columns].groupby([id_column], sort=False).head(1)
        static_features.set_index(id_column, inplace=True)
        df = df.drop(columns=static_columns)
    tsdf = TimeSeriesDataFrame.from_data_frame(df, id_column=id_column, timestamp_column=timestamp_column)
    if static_features is not None:
        tsdf.static_features = static_features
    return tsdf


def transform_fn(model, request_body, input_content_type, output_content_type="application/x-parquet"):
    """Run inference with per-request prediction_length, quantile_levels, etc."""
    inference_kwargs = {}
    if input_content_type == "application/x-autogluon":
        payload = pickle.loads(bytes(request_body))
        data = pd.read_parquet(BytesIO(payload["data"]))
        inference_kwargs = payload.get("inference_kwargs", {})
    elif input_content_type == "application/x-parquet":
        data = pd.read_parquet(BytesIO(request_body))
    elif input_content_type == "text/csv":
        data = pd.read_csv(StringIO(request_body))
    elif input_content_type == "application/json":
        data = pd.read_json(StringIO(request_body))
    else:
        raise ValueError(f"{input_content_type} input content type not supported.")

    cols = data.columns.to_list()
    id_column = inference_kwargs.pop("id_column", cols[0])
    timestamp_column = inference_kwargs.pop("timestamp_column", cols[1])
    target = inference_kwargs.pop("target", cols[-1])
    prediction_length = inference_kwargs.pop("prediction_length", 1)
    quantile_levels = inference_kwargs.pop("quantile_levels", None)

    tsdf = _build_tsdf(data, id_column, timestamp_column, target)

    model.prediction_length = prediction_length
    if quantile_levels is not None:
        model.quantile_levels = sorted(quantile_levels)

    predictions = model.predict(tsdf, **inference_kwargs)
    predictions = pd.DataFrame(predictions)

    if "application/x-parquet" in output_content_type:
        predictions.columns = predictions.columns.astype(str)
        output = predictions.to_parquet()
        output_content_type = "application/x-parquet"
    elif "text/csv" in output_content_type:
        output = predictions.to_csv()
        output_content_type = "text/csv"
    elif "application/json" in output_content_type:
        output = predictions.to_json()
        output_content_type = "application/json"
    else:
        raise ValueError(f"{output_content_type} content type not supported")

    return output, output_content_type
