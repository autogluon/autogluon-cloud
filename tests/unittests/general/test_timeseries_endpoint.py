from unittest import mock

import pandas as pd
import pytest

from autogluon.cloud.endpoint.timeseries_endpoint import TimeSeriesEndpoint


@pytest.fixture
def invoke_endpoint():
    with mock.patch("autogluon.cloud.endpoint.timeseries_endpoint.invoke_endpoint") as invoke:
        yield invoke


@pytest.fixture
def endpoint():
    endpoint = TimeSeriesEndpoint.__new__(TimeSeriesEndpoint)
    endpoint._endpoint_name = "ts-endpoint"
    endpoint._session = mock.sentinel.session
    return endpoint


def test_predict_omits_unset_args_so_endpoint_defaults_apply(endpoint, invoke_endpoint):
    # Trained predictor endpoints read id/timestamp columns from fit-time metadata unless the request overrides them.
    endpoint.predict(pd.DataFrame({"id": [1], "ts": ["2020-01-01"], "target": [0.0]}))

    assert invoke_endpoint.call_args.args[2].inference_kwargs == {}


def test_predict_forwards_set_args(endpoint, invoke_endpoint):
    endpoint.predict(
        pd.DataFrame({"id": [1], "ts": ["2020-01-01"], "y": [0.0]}),
        prediction_length=3,
        target="y",
        id_column="id",
        timestamp_column="ts",
        quantile_levels=[0.5],
    )

    assert invoke_endpoint.call_args.args[2].inference_kwargs == {
        "prediction_length": 3,
        "target": "y",
        "id_column": "id",
        "timestamp_column": "ts",
        "quantile_levels": [0.5],
    }
