import json
import sys
from pathlib import Path

import pandas as pd
import pytest

from autogluon.cloud.utils.serializers import AutoGluonSerializationWrapper, AutoGluonSerializer
from autogluon.timeseries import TimeSeriesDataFrame

# serving_utils/ is bundled into the model tarball as a sibling of the entry-point script;
# replicate that import path here so we test the same module the serve scripts load.
sys.path.insert(
    0, str(Path(__file__).resolve().parents[3] / "src" / "autogluon" / "cloud" / "scripts" / "sagemaker_scripts")
)
from serving_utils import timeseries as u  # noqa: E402


@pytest.fixture
def jumpstart_payload():
    return {
        "inputs": [
            {"target": [1.0, 2.0, 3.0, 4.0, 5.0], "item_id": "A", "start": "2020-01-01"},
            {"target": [10.0, 20.0, 30.0], "item_id": "B", "start": "2020-02-01"},
        ],
        "parameters": {"prediction_length": 2, "freq": "D", "quantile_levels": [0.1, 0.9]},
    }


def test_when_jumpstart_payload_valid_then_tsdf_and_inference_kwargs_returned(jumpstart_payload):
    tsdf, known_covariates, inference_kwargs = u.parse_payload(
        json.dumps(jumpstart_payload).encode("utf-8"), "application/json"
    )

    assert known_covariates is None
    assert inference_kwargs == {"prediction_length": 2, "freq": "D", "quantile_levels": [0.1, 0.9]}
    assert tsdf.item_ids.tolist() == ["A", "B"]
    assert tsdf.loc["A", "target"].tolist() == [1.0, 2.0, 3.0, 4.0, 5.0]
    assert tsdf.loc["B", "target"].tolist() == [10.0, 20.0, 30.0]


def test_when_jumpstart_payload_omits_item_id_then_index_used():
    payload = {"inputs": [{"target": [1.0, 2.0]}, {"target": [3.0, 4.0]}]}
    tsdf, _, _ = u.parse_payload(json.dumps(payload).encode("utf-8"), "application/json")
    assert tsdf.item_ids.tolist() == ["0", "1"]


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
    tsdf, known_covariates, _ = u.parse_payload(json.dumps(payload).encode("utf-8"), "application/json")
    assert "promo" in tsdf.columns
    assert tsdf["promo"].tolist() == [0.0, 1.0, 0.0, 1.0]
    assert known_covariates is not None
    assert known_covariates["promo"].tolist() == [1.0, 0.0]
    assert known_covariates.loc["A"].index.tolist() == [pd.Timestamp("2020-01-05"), pd.Timestamp("2020-01-06")]


def test_when_jumpstart_past_covariate_length_mismatch_then_pandas_raises():
    payload = {"inputs": [{"target": [1.0, 2.0, 3.0], "past_covariates": {"x": [1.0]}}]}
    with pytest.raises(ValueError):
        u.parse_payload(json.dumps(payload).encode("utf-8"), "application/json")


def test_when_x_autogluon_payload_uses_fit_time_columns_then_parses():
    df = pd.DataFrame({"item_id": ["A"] * 3, "timestamp": pd.date_range("2020-01-01", periods=3), "target": [1, 2, 3]})
    payload = AutoGluonSerializer().serialize(
        AutoGluonSerializationWrapper(data=df, inference_kwargs={"prediction_length": 1})
    )
    tsdf, kc, kwargs = u.parse_payload(
        payload, "application/x-autogluon", id_column="item_id", timestamp_column="timestamp"
    )
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
    tsdf, _, kwargs = u.parse_payload(payload, "application/x-autogluon")
    assert tsdf.item_ids.tolist() == ["A"]
    assert "id_column" not in kwargs and "timestamp_column" not in kwargs


def test_when_x_autogluon_version_unknown_then_rejects():
    body = json.dumps({"version": 99, "data": "", "inference_kwargs": {}}).encode("utf-8")
    with pytest.raises(ValueError, match="Unsupported x-autogluon payload version: 99"):
        u.parse_payload(body, "application/x-autogluon")


def _predictions_tsdf():
    df = pd.DataFrame(
        {
            "item_id": ["A", "A", "B", "B"],
            "timestamp": pd.to_datetime(["2020-01-06", "2020-01-07", "2020-02-04", "2020-02-05"]),
            "mean": [6.0, 7.0, 40.0, 50.0],
            "0.1": [5.5, 6.5, 35.0, 45.0],
            "0.9": [6.5, 7.5, 45.0, 55.0],
        }
    )
    return TimeSeriesDataFrame.from_data_frame(df)


def test_when_render_response_json_then_jumpstart_shape_returned():
    body, content_type = u.render_response(_predictions_tsdf(), "application/json")
    assert content_type == "application/json"
    assert json.loads(body) == {
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


@pytest.mark.parametrize(
    ("accept", "expected_ct"),
    [("application/x-parquet", "application/x-parquet"), ("text/csv", "text/csv")],
)
def test_when_render_response_dataframe_then_uses_accept_header(accept, expected_ct):
    _, ct = u.render_response(_predictions_tsdf(), accept)
    assert ct == expected_ct


def test_when_render_response_unsupported_then_raises():
    with pytest.raises(ValueError, match="not supported"):
        u.render_response(_predictions_tsdf(), "application/x-bogus")
