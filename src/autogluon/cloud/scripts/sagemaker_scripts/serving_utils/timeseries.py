"""Serde helpers used by the timeseries serve scripts."""

import base64
import json
from io import BytesIO
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.utils.forecast import make_future_data_frame

ParsedPayload = Tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame], Dict[str, Any]]


def parse_payload(
    request_body,
    content_type: str,
    *,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
    target_column: str = "target",
) -> ParsedPayload:
    """Parse a request body into ``(past_data, known_covariates, inference_kwargs)``."""
    # SageMaker passes ``str`` for text content types (csv/json) and ``bytes`` for binary ones.
    if isinstance(request_body, str):
        request_body = request_body.encode()
    if content_type == "application/x-autogluon":
        return _parse_x_autogluon(request_body, id_column=id_column, timestamp_column=timestamp_column)
    elif content_type == "application/json":
        return _parse_jumpstart(request_body, target_column=target_column)
    elif content_type == "application/x-parquet":
        data = pd.read_parquet(BytesIO(request_body))
    elif content_type == "text/csv":
        data = pd.read_csv(BytesIO(request_body))
    elif content_type == "application/jsonl":
        data = pd.read_json(BytesIO(request_body), orient="records", lines=True)
    else:
        raise ValueError(f"{content_type} input content type not supported.")
    tsdf = TimeSeriesDataFrame.from_data_frame(data, id_column=id_column, timestamp_column=timestamp_column)
    return tsdf, None, {}


def _parse_x_autogluon(request_body: bytes, *, id_column: str, timestamp_column: str) -> ParsedPayload:
    payload = json.loads(request_body)
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported x-autogluon payload version: {payload.get('version')}. Expected 1.")

    inference_kwargs = payload.get("inference_kwargs") or {}
    id_column = inference_kwargs.pop("id_column", id_column)
    timestamp_column = inference_kwargs.pop("timestamp_column", timestamp_column)

    static_features = _decode_parquet(payload.get("static_features"))
    tsdf = TimeSeriesDataFrame.from_data_frame(
        _decode_parquet(payload["data"]),
        id_column=id_column,
        timestamp_column=timestamp_column,
        static_features_df=static_features,
    )

    known_covariates_df = _decode_parquet(payload.get("known_covariates"))
    if known_covariates_df is not None:
        known_covariates = TimeSeriesDataFrame.from_data_frame(
            known_covariates_df, id_column=id_column, timestamp_column=timestamp_column
        )
    else:
        known_covariates = None

    return tsdf, known_covariates, inference_kwargs


def _parse_jumpstart(request_body: bytes, *, target_column: str = "target") -> ParsedPayload:
    payload = json.loads(request_body)
    inputs = payload["inputs"]
    inference_kwargs = payload.get("parameters") or {}
    freq = inference_kwargs.get("freq", "D")
    prediction_length = inference_kwargs.get("prediction_length", 1)

    past_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "item_id": ts.get("item_id", str(i)),
                    "timestamp": pd.date_range(
                        start=pd.Timestamp(ts.get("start", "2020-01-01")), periods=len(ts["target"]), freq=freq
                    ),
                    target_column: ts["target"],
                    **(ts.get("past_covariates") or {}),
                }
            )
            for i, ts in enumerate(inputs)
        ],
        ignore_index=True,
    )
    tsdf = TimeSeriesDataFrame.from_data_frame(past_df)

    if any("future_covariates" in ts for ts in inputs):
        future_index_df = make_future_data_frame(tsdf, prediction_length=prediction_length, freq=freq)
        future_values = pd.concat([pd.DataFrame(ts["future_covariates"]) for ts in inputs], ignore_index=True)
        known_covariates = TimeSeriesDataFrame.from_data_frame(pd.concat([future_index_df, future_values], axis=1))
    else:
        known_covariates = None

    return tsdf, known_covariates, inference_kwargs


def _decode_parquet(b64: Optional[str]) -> Optional[pd.DataFrame]:
    if b64 is None:
        return None
    else:
        return pd.read_parquet(BytesIO(base64.b64decode(b64)))


def render_response(predictions: TimeSeriesDataFrame, accept: str) -> Tuple[Any, str]:
    """Serialize predictions per the request's ``Accept`` header."""
    if "application/json" in accept:
        return _render_jumpstart(predictions), "application/json"
    df = pd.DataFrame(predictions).reset_index()
    df.columns = df.columns.astype(str)
    if "application/x-parquet" in accept:
        return df.to_parquet(), "application/x-parquet"
    elif "text/csv" in accept:
        return df.to_csv(index=False), "text/csv"
    else:
        raise ValueError(f"{accept} content type not supported")


def _render_jumpstart(predictions: TimeSeriesDataFrame) -> bytes:
    forecast_list = []
    for item_id, group in predictions.groupby("item_id", sort=False):
        forecast = {col: group[col].tolist() for col in group.columns}
        forecast["item_id"] = str(item_id)
        forecast["start"] = group.index.get_level_values("timestamp")[0].isoformat()
        forecast_list.append(forecast)
    return json.dumps({"predictions": forecast_list}).encode("utf-8")
