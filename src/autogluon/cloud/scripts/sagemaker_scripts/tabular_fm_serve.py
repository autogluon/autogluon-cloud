"""Serve script for tabular foundation models (Mitra, TabICL, etc.) on SageMaker endpoints.

Config comes from the AG_SERVE_CONFIG env var (set by the backend at deploy time):
    {"model_name": "MITRA", "hyperparameters": {...}}

Tabular foundation models perform in-context learning, so each request must include both the
labeled few-shot context (``train_data``) and the unlabeled rows to predict on (``data``).
The endpoint instantiates a fresh ``TabularPredictor`` per request, fits it on ``train_data``
with the configured AG model, and returns combined ``[<label>, <class>_proba, ...]`` predictions
matching the format produced by the standard tabular serve script.
"""

import json
import os
import shutil
import tempfile

import pandas as pd
from serving_utils.tabular import parse_payload, render_response

from autogluon.core.constants import QUANTILE, REGRESSION
from autogluon.core.utils import get_pred_from_proba_df
from autogluon.tabular import TabularPredictor

_SERVE_CONFIG = json.loads(os.environ.get("AG_SERVE_CONFIG", "{}"))


def model_fn(model_dir):
    """Return the serve config; the predictor is fit per-request from the request's train_data."""
    return _SERVE_CONFIG


def _resolve_label(train_df: pd.DataFrame, requested_label: str) -> str:
    if requested_label in train_df.columns:
        return requested_label
    raise ValueError(
        f"Label column {requested_label!r} not found in train_data. Available columns: {list(train_df.columns)}"
    )


def transform_fn(model, request_body, input_content_type, output_content_type="application/json"):
    train_df, test_df, inference_kwargs = parse_payload(request_body, input_content_type)

    model_name = model["model_name"]
    hyperparameters = model.get("hyperparameters", {})
    label = _resolve_label(train_df, inference_kwargs.get("label", "target"))

    tmpdir = tempfile.mkdtemp(prefix="ag_fm_predictor_")
    try:
        predictor = TabularPredictor(label=label, path=tmpdir).fit(
            train_data=train_df,
            hyperparameters={model_name: hyperparameters},
            fit_weighted_ensemble=False,
            calibrate_decision_threshold=False,
        )

        if predictor.problem_type not in [REGRESSION, QUANTILE]:
            pred_proba = predictor.predict_proba(test_df, as_pandas=True)
            pred = get_pred_from_proba_df(pred_proba, problem_type=predictor.problem_type)
            pred_proba.columns = [str(c) + "_proba" for c in pred_proba.columns]
            pred.name = predictor.label
            prediction = pd.concat([pred, pred_proba], axis=1)
        else:
            prediction = predictor.predict(test_df, as_pandas=True)
            if isinstance(prediction, pd.Series):
                prediction = prediction.to_frame()
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)

    return render_response(prediction, output_content_type)
