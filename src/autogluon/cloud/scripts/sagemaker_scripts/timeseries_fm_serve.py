"""Serve script for time series foundation models (Chronos, etc.) on SageMaker endpoints.

Config comes from the AG_SERVE_CONFIG env var (set by the backend at deploy time):
    {"model_name": "Chronos", "hyperparameters": {"model_path": "amazon/chronos-bolt-base", ...}}
"""

import json
import os

import numpy as np
import pandas as pd
from serving_utils.timeseries import (
    parse_jumpstart_payload,
    parse_x_autogluon_payload,
    render_dataframe,
    render_jumpstart,
)

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.models import ModelRegistry

_SERVE_CONFIG = json.loads(os.environ.get("AG_SERVE_CONFIG", "{}"))


def model_fn(model_dir):
    """Instantiate the foundation model and load weights into memory."""
    model_name = _SERVE_CONFIG["model_name"]
    hyperparameters = _SERVE_CONFIG.get("hyperparameters", {})

    model_cls = ModelRegistry.get_model_class(model_name)
    # freq and prediction_length are overridden per-request in transform_fn
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


def transform_fn(model, request_body, input_content_type, output_content_type="application/x-parquet"):
    """Run inference with per-request prediction_length, quantile_levels, etc."""
    if input_content_type == "application/x-autogluon":
        tsdf, known_covariates, parameters = parse_x_autogluon_payload(request_body)
    elif input_content_type == "application/json":
        tsdf, known_covariates, parameters = parse_jumpstart_payload(request_body)
    else:
        raise ValueError(f"{input_content_type} input content type not supported.")

    model.target = parameters.get("target", "target")
    model.freq = parameters.get("freq", "D")
    model.prediction_length = parameters.get("prediction_length", 1)
    if "quantile_levels" in parameters:
        model.quantile_levels = sorted(parameters["quantile_levels"])

    predictions = model.predict(tsdf, known_covariates=known_covariates)
    if input_content_type == "application/json":
        return render_jumpstart(predictions)
    return render_dataframe(predictions.to_data_frame().reset_index(), output_content_type)
