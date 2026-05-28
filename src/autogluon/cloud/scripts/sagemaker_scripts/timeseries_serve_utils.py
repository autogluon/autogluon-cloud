# flake8: noqa
"""Shared serde helpers for time series serve scripts.

This module is bundled into the model tarball under ``code/`` so it can be
imported as a sibling of the entry-point script (``timeseries_serve.py`` or
``timeseries_fm_serve.py``).

Two payload formats are first-class:
  * ``application/x-autogluon`` — JSON envelope with base64-encoded parquet
    payloads, produced by ``AutoGluonSerializer`` on the client side.
  * ``application/json`` — the JumpStart payload format. Univariate target
    only (matches AutoGluon's data model). Covariates in the JumpStart payload
    are not yet supported.

JumpStart input schema::

    {
        "inputs": [
            {"target": [<float>, ...], "item_id": <str, optional>, "start": <ISO-8601, optional>},
            ...
        ],
        "parameters": {
            "prediction_length": <int, optional>,
            "freq":              <str, optional>,
            "quantile_levels":   [<float>, ...],
        }
    }

JumpStart response schema::

    {
        "predictions": [
            {
                "item_id": <str>,
                "start":   <ISO-8601>,
                "mean":    [<float>, ...],
                "0.1":     [<float>, ...],
                ...
            },
            ...
        ]
    }
"""

import base64
import json
from io import BytesIO, StringIO
from typing import Any, Dict, Optional, Tuple

import pandas as pd

from autogluon.timeseries import TimeSeriesDataFrame

X_AUTOGLUON = "application/x-autogluon"
APPLICATION_JSON = "application/json"
APPLICATION_PARQUET = "application/x-parquet"
APPLICATION_JSONL = "application/jsonl"
TEXT_CSV = "text/csv"

# Used only when the JumpStart payload omits ``freq`` / ``start`` — the foundation
# models don't depend on absolute timestamps, so the choice doesn't affect forecasts.
_JUMPSTART_DEFAULT_FREQ = "D"
_JUMPSTART_DEFAULT_START = "2020-01-01"


def parse_x_autogluon_payload(
    request_body: bytes,
    *,
    id_column: Optional[str] = None,
    timestamp_column: Optional[str] = None,
) -> Tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame], Dict[str, Any]]:
    """Parse the ``application/x-autogluon`` payload into AutoGluon objects.

    ``id_column`` / ``timestamp_column`` may be passed by the caller (the
    TimeSeriesPredictor records them at fit time). If they are omitted, the
    parser expects them inside ``inference_kwargs`` (FoundationModel path).
    """
    payload = json.loads(request_body)
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported x-autogluon payload version: {payload.get('version')}. Expected 1.")
    inference_kwargs = payload.get("inference_kwargs") or {}

    if id_column is None or timestamp_column is None:
        try:
            id_column = inference_kwargs.pop("id_column")
            timestamp_column = inference_kwargs.pop("timestamp_column")
        except KeyError as e:
            raise ValueError(
                f"`application/x-autogluon` payload must include {e.args[0]!r} in inference_kwargs."
            ) from e

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
    *,
    target_column: str = "target",
    id_column: str = "item_id",
    timestamp_column: str = "timestamp",
) -> Tuple[TimeSeriesDataFrame, Optional[TimeSeriesDataFrame], Dict[str, Any]]:
    """Parse the JumpStart ``application/json`` payload.

    Per-item ``past_covariates`` are attached to the past data alongside the target;
    per-item ``future_covariates`` are returned as a separate ``known_covariates``
    TimeSeriesDataFrame whose rows cover ``prediction_length`` steps after the
    last past timestamp. The two sets of covariate names must match — AutoGluon
    requires past values for every known covariate.
    """
    payload = json.loads(request_body)
    if "inputs" not in payload:
        raise ValueError("JumpStart payload must contain an 'inputs' field.")
    inputs = payload["inputs"]
    if not isinstance(inputs, list) or not inputs:
        raise ValueError("'inputs' must be a non-empty list.")

    parameters = payload.get("parameters") or {}
    freq = parameters.get("freq", _JUMPSTART_DEFAULT_FREQ)
    prediction_length = parameters.get("prediction_length", 1)
    quantile_levels = parameters.get("quantile_levels")
    freq_offset = pd.tseries.frequencies.to_offset(freq)

    past_rows, future_rows = [], []
    for i, ts in enumerate(inputs):
        _validate_jumpstart_target(ts, i)
        item_id = ts.get("item_id", str(i))
        start = pd.Timestamp(ts.get("start", _JUMPSTART_DEFAULT_START))
        past_timestamps = pd.date_range(start=start, periods=len(ts["target"]), freq=freq)
        past_rows.append(
            pd.DataFrame(
                {
                    id_column: item_id,
                    timestamp_column: past_timestamps,
                    target_column: ts["target"],
                    **(ts.get("past_covariates") or {}),
                }
            )
        )

        future_covariates = ts.get("future_covariates")
        if future_covariates:
            future_timestamps = pd.date_range(
                start=past_timestamps[-1] + freq_offset, periods=prediction_length, freq=freq
            )
            future_rows.append(
                pd.DataFrame(
                    {
                        id_column: item_id,
                        timestamp_column: future_timestamps,
                        **future_covariates,
                    }
                )
            )

    if future_rows and len(future_rows) != len(inputs):
        raise ValueError("future_covariates must be provided for all inputs or none.")

    tsdf = TimeSeriesDataFrame.from_data_frame(
        pd.concat(past_rows, ignore_index=True), id_column=id_column, timestamp_column=timestamp_column
    )
    known_covariates = (
        TimeSeriesDataFrame.from_data_frame(
            pd.concat(future_rows, ignore_index=True), id_column=id_column, timestamp_column=timestamp_column
        )
        if future_rows
        else None
    )

    inference_kwargs: Dict[str, Any] = {"prediction_length": prediction_length}
    if quantile_levels is not None:
        inference_kwargs["quantile_levels"] = quantile_levels

    return tsdf, known_covariates, inference_kwargs


def _validate_jumpstart_target(ts: Dict[str, Any], index: int) -> None:
    target = ts.get("target")
    if not isinstance(target, list) or not target:
        raise ValueError(f"inputs[{index}].target must be a non-empty list of numbers.")
    if isinstance(target[0], list):
        raise ValueError(f"inputs[{index}].target must be univariate (a flat list of numbers).")


def parse_dataframe_payload(
    request_body: bytes, content_type: str, *, id_column: str, timestamp_column: str
) -> Tuple[TimeSeriesDataFrame, None, Dict[str, Any]]:
    """Parse plain parquet/csv/jsonl payloads using fit-time column names."""
    if content_type == APPLICATION_PARQUET:
        data = pd.read_parquet(BytesIO(request_body))
    elif content_type == TEXT_CSV:
        data = pd.read_csv(StringIO(request_body))
    elif content_type == APPLICATION_JSONL:
        data = pd.read_json(StringIO(request_body), orient="records", lines=True)
    else:
        raise ValueError(f"{content_type} input content type not supported.")

    tsdf = TimeSeriesDataFrame.from_data_frame(data, id_column=id_column, timestamp_column=timestamp_column)
    return tsdf, None, {}


def render_jumpstart(
    predictions_df: pd.DataFrame, *, id_column: str = "item_id", timestamp_column: str = "timestamp"
) -> Tuple[bytes, str]:
    """Render predictions to a JumpStart-shaped JSON response.

    ``predictions_df`` is the long-format dataframe produced by
    ``predictions.to_data_frame().reset_index()``: it has ``id_column``,
    ``timestamp_column``, and one column per forecast quantity.
    """
    forecast_list = []
    for item_id, group in predictions_df.groupby(id_column, sort=False):
        forecast = {col: group[col].tolist() for col in group.columns if col not in (id_column, timestamp_column)}
        forecast["item_id"] = str(item_id)
        forecast["start"] = group[timestamp_column].iloc[0].isoformat()
        forecast_list.append(forecast)

    return json.dumps({"predictions": forecast_list}).encode("utf-8"), APPLICATION_JSON


def render_dataframe(predictions_df: pd.DataFrame, accept: str) -> Tuple[Any, str]:
    """Serialize a prediction DataFrame as parquet / CSV / JSON per the Accept header."""
    if APPLICATION_PARQUET in accept:
        predictions_df.columns = predictions_df.columns.astype(str)
        return predictions_df.to_parquet(), APPLICATION_PARQUET
    if APPLICATION_JSON in accept:
        return predictions_df.to_json(), APPLICATION_JSON
    if TEXT_CSV in accept:
        return predictions_df.to_csv(index=None), TEXT_CSV
    raise ValueError(f"{accept} content type not supported")
