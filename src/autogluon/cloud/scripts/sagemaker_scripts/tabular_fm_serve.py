"""Serve script for tabular foundation models (Mitra, etc.) on SageMaker endpoints.

Each request contains labeled ``train_data`` used as in-context examples and ``data`` to score.
Configuration comes from the ``AG_FM_SERVE_CONFIG`` environment variable set during deployment.
"""

import base64
import copy
import json
import os
import tempfile
from io import BytesIO

import pandas as pd
from huggingface_hub import snapshot_download

from autogluon.tabular import TabularPredictor

_FM_SERVE_CONFIG = json.loads(os.environ.get("AG_FM_SERVE_CONFIG", "{}"))
_SUPPORTED_INPUT_CONTENT_TYPES = {"application/x-autogluon"}


def model_fn(model_dir):
    """Download remote weights during container startup and return model configuration."""
    model_config = copy.deepcopy(_FM_SERVE_CONFIG)
    hyperparameters = model_config.get("hyperparameters", {})
    for source_key in ("hf_cls_model", "hf_reg_model", "hf_general_model", "hf_model"):
        source = hyperparameters.get(source_key)
        if source is not None and not os.path.isdir(source):
            snapshot_download(
                repo_id=source,
                allow_patterns=["config.json", "model.safetensors"],
            )
            break
    return model_config


def _read_parquet(payload, key):
    encoded = payload.get(key)
    if encoded is None:
        raise ValueError(f"Missing required field {key!r} in x-autogluon payload.")
    return pd.read_parquet(BytesIO(base64.b64decode(encoded)))


def _parse_payload(request_body, input_content_type):
    if input_content_type not in _SUPPORTED_INPUT_CONTENT_TYPES:
        raise ValueError(
            f"{input_content_type} input content type not supported. "
            f"Supported: {sorted(_SUPPORTED_INPUT_CONTENT_TYPES)}"
        )

    payload = json.loads(request_body)
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported x-autogluon payload version: {payload.get('version')}. Expected 1.")

    data = _read_parquet(payload, "data")
    train_data = _read_parquet(payload, "train_data")
    inference_kwargs = payload.get("inference_kwargs") or {}
    return data, train_data, inference_kwargs


def _render_response(prediction, output_content_type):
    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame()

    output_content_type = output_content_type.lower()
    if "application/x-parquet" in output_content_type:
        prediction.columns = prediction.columns.astype(str)
        return prediction.to_parquet(index=False), "application/x-parquet"
    if "application/json" in output_content_type:
        return prediction.to_json(orient="records"), "application/json"
    if "text/csv" in output_content_type:
        return prediction.to_csv(index=False), "text/csv"
    raise ValueError(f"{output_content_type} content type not supported")


def transform_fn(model_config, request_body, input_content_type, output_content_type="application/json"):
    """Fit a request-scoped TabularPredictor and score the request's prediction data."""
    data, train_data, inference_kwargs = _parse_payload(request_body, input_content_type)
    inference_kwargs = dict(inference_kwargs)
    label = inference_kwargs.pop("label", None)
    if label is None:
        raise ValueError("`inference_kwargs` must contain the training label column name under `label`.")
    if label not in train_data.columns:
        raise ValueError(f"Label column {label!r} is not present in `train_data`.")

    ag_model_key = model_config["ag_model_key"]
    hyperparameters = model_config.get("hyperparameters", {})
    problem_type = model_config["problem_type"]

    with tempfile.TemporaryDirectory(prefix="ag_tabular_fm_") as temp_dir:
        predictor = TabularPredictor(
            label=label,
            problem_type=problem_type,
            path=os.path.join(temp_dir, "predictor"),
        ).fit(
            train_data,
            hyperparameters={ag_model_key: hyperparameters},
            fit_weighted_ensemble=False,
        )

        pred = predictor.predict(data, as_pandas=True, **inference_kwargs)
        if predictor.can_predict_proba:
            pred_proba = predictor.predict_proba(data, as_pandas=True, **inference_kwargs)
            pred_proba.columns = [f"{column}_proba" for column in pred_proba.columns]
            pred.name = predictor.label
            prediction = pd.concat([pred, pred_proba], axis=1)
        else:
            prediction = pred

    return _render_response(prediction, output_content_type)
