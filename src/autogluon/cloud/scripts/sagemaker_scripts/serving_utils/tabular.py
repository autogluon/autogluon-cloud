"""Serde helpers used by the tabular foundation-model serve script."""

import base64
import json
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import pandas as pd

ParsedPayload = Tuple[pd.DataFrame, pd.DataFrame, Dict[str, Any]]


def parse_payload(request_body, content_type: str) -> ParsedPayload:
    """Parse a request body into ``(train_data, test_data, inference_kwargs)``.

    Tabular foundation models need both the few-shot context (``train_data``) and the rows
    to predict on (``test_data``) on every request.
    """
    if isinstance(request_body, str):
        request_body = request_body.encode()
    if content_type == "application/x-autogluon":
        return _parse_x_autogluon(request_body)
    raise ValueError(
        f"{content_type} input content type not supported. "
        f"Tabular foundation-model endpoints accept only 'application/x-autogluon'."
    )


def _parse_x_autogluon(request_body: bytes) -> ParsedPayload:
    payload = json.loads(request_body)
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported x-autogluon payload version: {payload.get('version')}. Expected 1.")

    train_df = _decode_parquet(payload.get("train_data"))
    if train_df is None:
        raise ValueError("Tabular foundation-model payload must include `train_data` (few-shot context).")
    test_df = _decode_parquet(payload["data"])
    inference_kwargs = payload.get("inference_kwargs") or {}
    return train_df, test_df, inference_kwargs


def _decode_parquet(b64: Optional[str]) -> Optional[pd.DataFrame]:
    if b64 is None:
        return None
    return pd.read_parquet(BytesIO(base64.b64decode(b64)))


def render_response(prediction: pd.DataFrame, accept: str) -> Tuple[Any, str]:
    """Serialize predictions per the request's ``Accept`` header."""
    accept = accept.lower()
    if "application/x-parquet" in accept:
        prediction = prediction.copy()
        prediction.columns = prediction.columns.astype(str)
        return prediction.to_parquet(index=False), "application/x-parquet"
    if "application/json" in accept:
        return prediction.to_json(orient="records"), "application/json"
    if "text/csv" in accept:
        return prediction.to_csv(index=False), "text/csv"
    raise ValueError(f"{accept} content type not supported")
