"""Serve script for time series foundation models (Chronos, etc.) on SageMaker endpoints.

Config comes from the AG_SERVE_CONFIG env var (set by the backend at deploy time):
    {"model_name": "Chronos", "hyperparameters": {"model_path": "amazon/chronos-bolt-base", ...}}
"""

import json
import os
import pickle
from io import BytesIO

import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame

_SERVE_CONFIG = json.loads(os.environ.get("AG_SERVE_CONFIG", "{}"))


def model_fn(model_dir):
    """Instantiate the foundation model and load weights into memory."""
    import numpy as np

    from autogluon.timeseries.models import ModelRegistry

    model_name = _SERVE_CONFIG["model_name"]
    hyperparameters = _SERVE_CONFIG.get("hyperparameters", {})

    model_cls = ModelRegistry.get_model_class(model_name)
    model = model_cls(
        path=model_name,
        freq=None,
        prediction_length=1,
        hyperparameters=hyperparameters,
    )

    # fit() on dummy data to initialize model internals and load weights
    dummy_df = TimeSeriesDataFrame(
        pd.DataFrame(
            {"item_id": [0] * 10, "timestamp": pd.date_range("2020-01-01", periods=10), "target": np.zeros(10)}
        )
    )
    model.fit(train_data=dummy_df)
    model.persist()
    return model


def _build_tsdf(df, id_column, timestamp_column, target):
    """Convert a long-format DataFrame to TimeSeriesDataFrame.

    Static features are encoded as columns after the target column (same convention
    as TimeSeriesSagemakerBackend._preprocess_data).
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


def _parse_jumpstart_payload(payload: dict):
    """Parse JumpStart-style JSON input into DataFrames + inference kwargs.

    Expected schema:
        {
            "inputs": [{"target": [...], "item_id": "...", "start": "...", ...}, ...],
            "parameters": {"prediction_length": 12, "quantile_levels": [...]}
        }
    """
    # TODO: implement JumpStart schema parsing
    raise NotImplementedError("JumpStart JSON input schema is not yet supported")


def _format_jumpstart_response(predictions: pd.DataFrame) -> str:
    """Format predictions into JumpStart-style JSON output.

    Expected output schema:
        {
            "forecasts": [{"item_id": "...", "mean": [...], "quantiles": {"0.1": [...], ...}}, ...]
        }
    """
    # TODO: implement JumpStart response formatting
    raise NotImplementedError("JumpStart JSON output schema is not yet supported")


def _parse_autogluon_payload(request_body):
    """Parse application/x-autogluon pickle payload.

    Same format as timeseries_serve.py:
        pickle({"data": parquet_bytes, "inference_kwargs": {...}})
    """
    payload = pickle.loads(bytes(request_body))
    data = pd.read_parquet(BytesIO(payload["data"]))
    inference_kwargs = payload.get("inference_kwargs", {})
    return data, inference_kwargs


def transform_fn(model, request_body, input_content_type, output_content_type="application/x-parquet"):
    """Run inference with per-request prediction_length, quantile_levels, etc."""
    if input_content_type == "application/x-autogluon":
        data, inference_kwargs = _parse_autogluon_payload(request_body)
    elif input_content_type == "application/json":
        payload = json.loads(request_body)
        data, inference_kwargs = _parse_jumpstart_payload(payload)
    else:
        raise ValueError(f"{input_content_type} input content type not supported.")

    # Extract TS-specific params from inference_kwargs
    cols = data.columns.to_list()
    id_column = inference_kwargs.pop("id_column", cols[0])
    timestamp_column = inference_kwargs.pop("timestamp_column", cols[1])
    target = inference_kwargs.pop("target", cols[2])
    prediction_length = inference_kwargs.pop("prediction_length", 1)
    quantile_levels = inference_kwargs.pop("quantile_levels", None)

    tsdf = _build_tsdf(data, id_column, timestamp_column, target)

    model.prediction_length = prediction_length
    if quantile_levels is not None:
        model.quantile_levels = sorted(quantile_levels)

    predictions = model.predict(tsdf, **inference_kwargs)
    predictions = pd.DataFrame(predictions)

    # Serialize response — output_content_type may be a comma-separated accept list
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
