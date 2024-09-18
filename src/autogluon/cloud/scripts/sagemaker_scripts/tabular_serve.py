# flake8: noqa
import base64
import hashlib
import os
import pickle
from io import BytesIO, StringIO

import pandas as pd
from PIL import Image

from autogluon.core.constants import QUANTILE, REGRESSION
from autogluon.core.utils import get_pred_from_proba_df
from autogluon.tabular import TabularPredictor

image_dir = os.path.join("/tmp", "ag_images")


def _save_image_and_update_dataframe_column(bytes):
    os.makedirs(image_dir, exist_ok=True)
    im_bytes = base64.b85decode(bytes)
    # nosec B303 - not a cryptographic use
    im_hash = hashlib.sha1(im_bytes).hexdigest()
    im = Image.open(BytesIO(im_bytes))
    im_name = f"tabular_image_{im_hash}.png"
    im_path = os.path.join(image_dir, im_name)
    im.save(im_path)
    print(f"Image saved as {im_path}")

    return im_path


def model_fn(model_dir):
    """loads model from previously saved artifact"""
    model = TabularPredictor.load(model_dir)
    model.persist_models()
    globals()["column_names"] = model.original_features

    return model


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    inference_kwargs = {}

    if input_content_type == "application/x-parquet":
        buf = BytesIO(request_body)
        data = pd.read_parquet(buf)
        data = _align_columns(data, column_names)

    elif input_content_type == "text/csv":
        buf = StringIO(request_body)
        data = _read_with_fallback(pd.read_csv, buf, column_names)

    elif input_content_type == "application/json":
        buf = StringIO(request_body)
        data = _read_with_fallback(pd.read_json, buf, column_names)

    elif input_content_type == "application/jsonl":
        buf = StringIO(request_body)
        data = _read_with_fallback(lambda b: pd.read_json(b, orient="records", lines=True), buf, column_names)

    elif input_content_type == "application/x-autogluon":
        buf = bytes(request_body)
        payload = pickle.loads(buf)
        data = pd.read_parquet(BytesIO(payload["data"]))
        inference_kwargs = payload.get("inference_kwargs", {})
        data = _align_columns(data, column_names)

    else:
        raise ValueError(f"{input_content_type} input content type not supported.")

    # Find and process image column if present
    image_column = None
    for column_name, special_types in model.feature_metadata.get_type_map_special().items():
        if "image_path" in special_types:
            image_column = column_name
            break

    if image_column is not None:
        print(f"Detected image column {image_column}")
        data[image_column] = [_save_image_and_update_dataframe_column(bytes) for bytes in data[image_column]]

    # Ensure inference_kwargs is a dictionary right before use
    if inference_kwargs is None:
        inference_kwargs = {}

    # Make predictions
    if model.problem_type not in [REGRESSION, QUANTILE]:
        pred_proba = model.predict_proba(data, as_pandas=True, **inference_kwargs)
        pred = get_pred_from_proba_df(pred_proba, problem_type=model.problem_type)
        pred_proba.columns = [str(c) + "_proba" for c in pred_proba.columns]
        pred.name = model.label
        prediction = pd.concat([pred, pred_proba], axis=1)
    else:
        prediction = model.predict(data, as_pandas=True, **inference_kwargs)

    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame()

    # Format output
    output_content_type = output_content_type.lower()
    if "application/x-parquet" in output_content_type:
        prediction.columns = prediction.columns.astype(str)
        output = prediction.to_parquet(index=False)
        output_content_type = "application/x-parquet"
    elif "application/json" in output_content_type:
        output = prediction.to_json(orient="records")
        output_content_type = "application/json"
    elif "text/csv" in output_content_type:
        output = prediction.to_csv(index=False)
        output_content_type = "text/csv"
    else:
        raise ValueError(f"{output_content_type} content type not supported")

    return output, output_content_type


def _read_with_fallback(read_func, buf, expected_columns):
    """
    Attempts to read data with headers. If columns don't match expected_columns,
    re-reads without headers and assigns expected_columns.

    Parameters
    ----------
    read_func : callable
        Function to read the data (e.g., pd.read_csv, pd.read_json).
    buf : IO buffer
        Buffer containing the data.
    expected_columns : list
        List of expected column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns aligned to expected_columns.
    """
    # Attempt to read with headers
    data = read_func(buf)
    if set(data.columns) != set(expected_columns):
        # Reset buffer and read without headers
        buf.seek(0)
        data = read_func(buf, header=None)
        # Assign expected column names
        data.columns = expected_columns
    else:
        # Reorder columns to match expected_columns
        data = data[expected_columns]
    return data


def _align_columns(data, expected_columns):
    """
    Aligns DataFrame columns to expected_columns.
    Removes extra columns and reorders existing ones.

    Parameters
    ----------
    data : pd.DataFrame
        Input DataFrame.
    expected_columns : list
        List of expected column names.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns aligned to expected_columns.
    """
    if set(data.columns) != set(expected_columns):
        missing_columns = set(expected_columns) - set(data.columns)
        if missing_columns:
            raise ValueError(f"Missing columns in input data: {missing_columns}")
    # Remove extra columns and reorder
    data = data[expected_columns]
    return data
