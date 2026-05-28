"""Serde helpers used by the timeseries serve scripts."""

import base64
import json
from io import BytesIO, StringIO
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame
from autogluon.timeseries.utils.forecast import make_future_data_frame


def parse_x_autogluon_payload(
    request_body: bytes,
    *,
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> Tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame], Dict[str, Any]]:
    payload = json.loads(request_body)
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported x-autogluon payload version: {payload.get('version')}. Expected 1.")
    inference_kwargs = payload.get("inference_kwargs") or {}
    id_column = inference_kwargs.pop("id_column", id_column)
    timestamp_column = inference_kwargs.pop("timestamp_column", timestamp_column)

    data = pd.read_parquet(BytesIO(base64.b64decode(payload["data"])))
    static_features = payload.get("static_features")
    if static_features is not None:
        static_features = pd.read_parquet(BytesIO(base64.b64decode(static_features)))

    tsdf = TimeSeriesDataFrame.from_data_frame(
        data, id_column=id_column, timestamp_column=timestamp_column, static_features_df=static_features
    )

    known_covariates = payload.get("known_covariates")
    if known_covariates is not None:
        known_covariates = TimeSeriesDataFrame.from_data_frame(
            pd.read_parquet(BytesIO(base64.b64decode(known_covariates))),
            id_column=id_column,
            timestamp_column=timestamp_column,
        )

    return tsdf, known_covariates, inference_kwargs


def parse_jumpstart_payload(
    request_body: bytes,
) -> Tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame], Dict[str, Any]]:
    payload = json.loads(request_body)
    inputs = payload["inputs"]
    parameters = payload.get("parameters") or {}
    freq = parameters.get("freq", "D")
    prediction_length = parameters.get("prediction_length", 1)

    past_df = pd.concat(
        [
            pd.DataFrame(
                {
                    "item_id": ts.get("item_id", str(i)),
                    "timestamp": pd.date_range(
                        start=pd.Timestamp(ts.get("start", "2020-01-01")),
                        periods=len(ts["target"]),
                        freq=freq,
                    ),
                    "target": ts["target"],
                    **(ts.get("past_covariates") or {}),
                }
            )
            for i, ts in enumerate(inputs)
        ],
        ignore_index=True,
    )
    tsdf = TimeSeriesDataFrame.from_data_frame(past_df)

    if any("future_covariates" in ts for ts in inputs):
        future_df = make_future_data_frame(tsdf, prediction_length=prediction_length, freq=freq)
        future_covariates = pd.concat([pd.DataFrame(ts["future_covariates"]) for ts in inputs], ignore_index=True)
        known_covariates = TimeSeriesDataFrame.from_data_frame(pd.concat([future_df, future_covariates], axis=1))
    else:
        known_covariates = None

    return tsdf, known_covariates, parameters


def parse_dataframe_payload(
    request_body: bytes, content_type: str, *, id_column: str, timestamp_column: str
) -> Tuple[TimeSeriesDataFrame, None, Dict[str, Any]]:
    if content_type == "application/x-parquet":
        data = pd.read_parquet(BytesIO(request_body))
    elif content_type == "text/csv":
        data = pd.read_csv(StringIO(request_body))
    elif content_type == "application/jsonl":
        data = pd.read_json(StringIO(request_body), orient="records", lines=True)
    else:
        raise ValueError(f"{content_type} input content type not supported.")

    tsdf = TimeSeriesDataFrame.from_data_frame(data, id_column=id_column, timestamp_column=timestamp_column)
    return tsdf, None, {}


def render_jumpstart(predictions: TimeSeriesDataFrame) -> Tuple[bytes, str]:
    forecast_list = []
    for item_id, group in predictions.groupby("item_id", sort=False):
        forecast = {col: group[col].tolist() for col in group.columns}
        forecast["item_id"] = str(item_id)
        forecast["start"] = group.index.get_level_values("timestamp")[0].isoformat()
        forecast_list.append(forecast)
    return json.dumps({"predictions": forecast_list}).encode("utf-8"), "application/json"


def render_dataframe(predictions_df: pd.DataFrame, accept: str) -> Tuple[Any, str]:
    if "application/x-parquet" in accept:
        predictions_df = predictions_df.copy()
        predictions_df.columns = predictions_df.columns.astype(str)
        return predictions_df.to_parquet(), "application/x-parquet"
    if "application/json" in accept:
        return predictions_df.to_json(), "application/json"
    if "text/csv" in accept:
        return predictions_df.to_csv(index=False), "text/csv"
    raise ValueError(f"{accept} content type not supported")
