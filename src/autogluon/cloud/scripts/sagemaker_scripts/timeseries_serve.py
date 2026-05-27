# flake8: noqa
import json
import os
import pickle
import shutil
from io import BytesIO, StringIO

import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame, TimeSeriesPredictor


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    # TSPredictor will write to the model file during inference while the default model_dir is read only
    # Copy the model file to a writable location as a temporary workaround
    tmp_model_dir = os.path.join("/tmp", "model")
    try:
        shutil.copytree(model_dir, tmp_model_dir, dirs_exist_ok=False)
    except:
        # model already copied
        pass
    model = TimeSeriesPredictor.load(tmp_model_dir)
    if hasattr(model, "persist"):  # timeseries added persist in v1.1
        model.persist()

    metadata_path = os.path.join(tmp_model_dir, "predictor_metadata.json")
    if os.path.exists(metadata_path):
        with open(metadata_path) as f:
            metadata = json.load(f)
        model._id_column = metadata["id_column"]
        model._timestamp_column = metadata["timestamp_column"]
    else:
        model._id_column = "item_id"
        model._timestamp_column = "timestamp"
    return model


def _parse_autogluon_payload(request_body, *, id_column, timestamp_column):
    """Parse x-autogluon payload. Returns (data, known_covariates, inference_kwargs)."""
    payload = pickle.loads(request_body)
    inference_kwargs = payload.get("inference_kwargs") or {}

    data = pd.read_parquet(BytesIO(payload["data"]))
    static_features = payload.get("static_features")
    if static_features is not None:
        static_features = pd.read_parquet(BytesIO(static_features))

    tsdf = TimeSeriesDataFrame.from_data_frame(
        data, id_column=id_column, timestamp_column=timestamp_column, static_features_df=static_features
    )

    known_covariates = payload.get("known_covariates")
    if known_covariates is not None:
        known_covariates = TimeSeriesDataFrame.from_data_frame(
            pd.read_parquet(BytesIO(known_covariates)), id_column=id_column, timestamp_column=timestamp_column
        )

    return tsdf, known_covariates, inference_kwargs


def _parse_simple_payload(request_body, content_type, *, id_column, timestamp_column):
    """Parse plain parquet/csv/json payloads using the column names recorded at fit time."""
    if content_type == "application/x-parquet":
        data = pd.read_parquet(BytesIO(request_body))
    elif content_type == "text/csv":
        data = pd.read_csv(StringIO(request_body))
    elif content_type == "application/json":
        data = pd.read_json(StringIO(request_body))
    elif content_type == "application/jsonl":
        data = pd.read_json(StringIO(request_body), orient="records", lines=True)
    else:
        raise ValueError(f"{content_type} input content type not supported.")

    tsdf = TimeSeriesDataFrame.from_data_frame(data, id_column=id_column, timestamp_column=timestamp_column)
    return tsdf, None, {}


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    id_column = model._id_column
    timestamp_column = model._timestamp_column
    if input_content_type == "application/x-autogluon":
        tsdf, known_covariates, inference_kwargs = _parse_autogluon_payload(
            request_body, id_column=id_column, timestamp_column=timestamp_column
        )
    else:
        tsdf, known_covariates, inference_kwargs = _parse_simple_payload(
            request_body, input_content_type, id_column=id_column, timestamp_column=timestamp_column
        )

    prediction = model.predict(tsdf, known_covariates=known_covariates, **inference_kwargs)
    prediction = pd.DataFrame(prediction)

    if "application/x-parquet" in output_content_type:
        prediction.columns = prediction.columns.astype(str)
        output = prediction.to_parquet()
        output_content_type = "application/x-parquet"
    elif "application/json" in output_content_type:
        output = prediction.to_json()
        output_content_type = "application/json"
    elif "text/csv" in output_content_type:
        output = prediction.to_csv(index=None)
        output_content_type = "text/csv"
    else:
        raise ValueError(f"{output_content_type} content type not supported")

    return output, output_content_type
