"""Tests for the shared serve-script helpers used by both timeseries serve scripts."""

import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from autogluon.cloud.utils.serializers import AutoGluonSerializationWrapper, AutoGluonSerializer

# The helpers live next to the entry-point scripts (they are loaded by SageMaker
# from a flat ``code/`` directory in the model tarball, so they are imported by
# bare module name rather than as part of a package).
_SAGEMAKER_SCRIPTS_DIR = (
    Path(__file__).resolve().parents[3] / "src" / "autogluon" / "cloud" / "scripts" / "sagemaker_scripts"
)
sys.path.insert(0, str(_SAGEMAKER_SCRIPTS_DIR))
import timeseries_serve_utils as u  # noqa: E402


@pytest.fixture
def jumpstart_payload():
    return {
        "inputs": [
            {"target": [1.0, 2.0, 3.0, 4.0, 5.0], "item_id": "A", "start": "2020-01-01"},
            {"target": [10.0, 20.0, 30.0], "item_id": "B", "start": "2020-02-01"},
        ],
        "parameters": {"prediction_length": 2, "freq": "D", "quantile_levels": [0.1, 0.9]},
    }


# --- parse_jumpstart_payload ---


def test_when_jumpstart_payload_valid_then_tsdf_and_kwargs_returned(jumpstart_payload):
    tsdf, known_covariates, kwargs = u.parse_jumpstart_payload(json.dumps(jumpstart_payload).encode("utf-8"))

    assert known_covariates is None
    assert kwargs == {"prediction_length": 2, "quantile_levels": [0.1, 0.9]}
    assert tsdf.item_ids.tolist() == ["A", "B"]
    assert tsdf.loc["A", "target"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert tsdf.loc["B", "target"].tolist() == [10.0, 20.0, 30.0]
    assert tsdf.loc["A"].index[0] == pd.Timestamp("2020-01-01")
    assert tsdf.loc["B"].index[0] == pd.Timestamp("2020-02-01")


def test_when_jumpstart_payload_omits_item_id_then_index_used():
    payload = {"inputs": [{"target": [1.0, 2.0]}, {"target": [3.0, 4.0]}]}
    tsdf, _, _ = u.parse_jumpstart_payload(json.dumps(payload).encode("utf-8"))
    assert tsdf.item_ids.tolist() == ["0", "1"]


def test_when_jumpstart_payload_uses_custom_target_then_column_named_accordingly():
    payload = {"inputs": [{"target": [1.0, 2.0, 3.0], "item_id": "A"}]}
    tsdf, _, _ = u.parse_jumpstart_payload(json.dumps(payload).encode("utf-8"), target_column="sales")
    assert "sales" in tsdf.columns
    assert tsdf["sales"].tolist() == [1.0, 2.0, 3.0]


def test_when_jumpstart_payload_omits_inputs_then_raises():
    with pytest.raises(ValueError, match="must contain an 'inputs' field"):
        u.parse_jumpstart_payload(b'{"parameters": {}}')


def test_when_jumpstart_inputs_empty_then_raises():
    with pytest.raises(ValueError, match="non-empty list"):
        u.parse_jumpstart_payload(b'{"inputs": []}')


def test_when_jumpstart_target_is_multivariate_then_raises():
    payload = {"inputs": [{"target": [[1.0, 2.0], [3.0, 4.0]]}]}
    with pytest.raises(ValueError, match="must be univariate"):
        u.parse_jumpstart_payload(json.dumps(payload).encode("utf-8"))


def test_when_jumpstart_target_missing_then_raises():
    with pytest.raises(ValueError, match="non-empty list of numbers"):
        u.parse_jumpstart_payload(b'{"inputs": [{"item_id": "A"}]}')


def test_when_jumpstart_payload_has_covariates_then_known_covariates_returned():
    payload = {
        "inputs": [
            {
                "target": [1.0, 2.0, 3.0, 4.0],
                "item_id": "A",
                "start": "2020-01-01",
                "past_covariates": {"promo": [0.0, 1.0, 0.0, 1.0]},
                "future_covariates": {"promo": [1.0, 0.0]},
            }
        ],
        "parameters": {"prediction_length": 2, "freq": "D"},
    }
    tsdf, known_covariates, _ = u.parse_jumpstart_payload(json.dumps(payload).encode("utf-8"))
    assert "promo" in tsdf.columns
    assert tsdf["promo"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert known_covariates is not None
    assert known_covariates["promo"].tolist() == [1.0, 0.0]
    assert known_covariates.loc["A"].index.tolist() == [pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-06")]


def test_when_jumpstart_future_covariates_partial_then_raises():
    payload = {
        "inputs": [
            {"target": [1.0, 2.0], "future_covariates": {"a": [1.0]}, "past_covariates": {"a": [0.0, 0.0]}},
            {"target": [3.0, 4.0]},
        ],
        "parameters": {"prediction_length": 1},
    }
    with pytest.raises(ValueError, match="for all inputs or none"):
        u.parse_jumpstart_payload(json.dumps(payload).encode("utf-8"))


def test_when_jumpstart_past_covariate_length_mismatch_then_pandas_raises():
    payload = {"inputs": [{"target": [1.0, 2.0, 3.0], "past_covariates": {"x": [1.0]}}]}
    with pytest.raises(ValueError):
        u.parse_jumpstart_payload(json.dumps(payload).encode("utf-8"))


# --- render_jumpstart ---


def test_when_predictions_rendered_then_jumpstart_shape_returned():
    predictions_df = pd.DataFrame(
        {
            "item_id": ["A", "A", "B", "B"],
            "timestamp": pd.to_datetime(["2020-01-06", "2020-01-07", "2020-02-04", "2020-02-05"]),
            "mean": [6.0, 7.0, 40.0, 50.0],
            "0.1": [5.5, 6.5, 35.0, 45.0],
            "0.9": [6.5, 7.5, 45.0, 55.0],
        }
    )
    body, content_type = u.render_jumpstart(predictions_df)
    assert content_type == u.APPLICATION_JSON
    parsed = json.loads(body)
    assert parsed == {
        "predictions": [
            {"mean": [6.0, 7.0], "0.1": [5.5, 6.5], "0.9": [6.5, 7.5], "item_id": "A", "start": "2020-01-06T00:00:00"},
            {
                "mean": [40.0, 50.0],
                "0.1": [35.0, 45.0],
                "0.9": [45.0, 55.0],
                "item_id": "B",
                "start": "2020-02-04T00:00:00",
            },
        ]
    }


# --- parse_x_autogluon_payload ---


def test_when_x_autogluon_payload_uses_fit_time_columns_then_parses(jumpstart_payload):
    df = pd.DataFrame({"item_id": ["A"] * 3, "timestamp": pd.date_range("2020-01-01", periods=3), "target": [1, 2, 3]})
    payload = AutoGluonSerializer().serialize(
        AutoGluonSerializationWrapper(data=df, inference_kwargs={"prediction_length": 1})
    )
    tsdf, kc, kwargs = u.parse_x_autogluon_payload(payload, id_column="item_id", timestamp_column="timestamp")
    assert kc is None
    assert kwargs == {"prediction_length": 1}
    assert tsdf.item_ids.tolist() == ["A"]


def test_when_x_autogluon_payload_relies_on_kwargs_columns_then_parses():
    df = pd.DataFrame({"id": ["A"] * 3, "ts": pd.date_range("2020-01-01", periods=3), "target": [1, 2, 3]})
    payload = AutoGluonSerializer().serialize(
        AutoGluonSerializationWrapper(
            data=df, inference_kwargs={"id_column": "id", "timestamp_column": "ts", "prediction_length": 1}
        )
    )
    tsdf, _, kwargs = u.parse_x_autogluon_payload(payload)
    assert tsdf.item_ids.tolist() == ["A"]
    assert "id_column" not in kwargs and "timestamp_column" not in kwargs


def test_when_x_autogluon_payload_missing_columns_then_raises():
    df = pd.DataFrame({"item_id": ["A"], "timestamp": pd.date_range("2020-01-01", periods=1), "target": [1]})
    payload = AutoGluonSerializer().serialize(AutoGluonSerializationWrapper(data=df, inference_kwargs={}))
    with pytest.raises(ValueError, match="must include 'id_column'"):
        u.parse_x_autogluon_payload(payload)


def test_when_x_autogluon_version_unknown_then_rejects():
    body = json.dumps({"version": 99, "data": "", "inference_kwargs": {}}).encode("utf-8")
    with pytest.raises(ValueError, match="Unsupported x-autogluon payload version: 99"):
        u.parse_x_autogluon_payload(body, id_column="item_id", timestamp_column="timestamp")


# --- render_dataframe ---


@pytest.mark.parametrize(
    ("accept", "expected_ct"),
    [
        ("application/x-parquet", "application/x-parquet"),
        ("application/json", "application/json"),
        ("text/csv", "text/csv"),
    ],
)
def test_when_render_dataframe_then_uses_accept_header(accept, expected_ct):
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    _, ct = u.render_dataframe(df, accept)
    assert ct == expected_ct


def test_when_render_dataframe_unsupported_then_raises():
    with pytest.raises(ValueError, match="not supported"):
        u.render_dataframe(pd.DataFrame({"a": [1]}), "application/x-bogus")
