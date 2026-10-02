from unittest import mock

import pandas as pd
import pytest

from autogluon.cloud.endpoint.tabular_endpoint import TabularEndpoint
from autogluon.cloud.utils.serializers import AutoGluonSerializationWrapper

TE = "autogluon.cloud.endpoint.tabular_endpoint"


@pytest.fixture(autouse=True)
def invoke_endpoint():
    with mock.patch(f"{TE}.invoke_endpoint") as invoke:
        yield invoke


@pytest.fixture
def make_endpoint(invoke_endpoint):
    def make(response):
        endpoint = TabularEndpoint.__new__(TabularEndpoint)
        endpoint._endpoint_name = "tabular-fm-endpoint"
        endpoint._session = mock.sentinel.session
        invoke_endpoint.return_value = response
        return endpoint

    return make


def test_predict_sends_train_data_and_returns_prediction_series(make_endpoint, invoke_endpoint):
    train_data = pd.DataFrame({"feature": [0, 1], "label": ["a", "b"]})
    data = pd.DataFrame({"feature": [2, 3]})
    response = pd.DataFrame({"label": ["a", "b"], "a_proba": [0.8, 0.2], "b_proba": [0.2, 0.8]})
    endpoint = make_endpoint(response)

    pred = endpoint.predict(data=data, train_data=train_data, label="label")

    assert pred.tolist() == ["a", "b"]
    payload = invoke_endpoint.call_args.args[2]
    assert isinstance(payload, AutoGluonSerializationWrapper)
    pd.testing.assert_frame_equal(payload.data, data)
    pd.testing.assert_frame_equal(payload.train_data, train_data)
    assert payload.inference_kwargs == {"label": "label"}


def test_predict_proba_matches_batch_result_shape(make_endpoint, invoke_endpoint):
    response = pd.DataFrame({"label": ["a"], "a_proba": [0.7], "b_proba": [0.3]})
    endpoint = make_endpoint(response)
    train_data = pd.DataFrame({"feature": [0, 1], "label": ["a", "b"]})
    data = pd.DataFrame({"feature": [2]})

    pred, proba = endpoint.predict_proba(data=data, train_data=train_data, label="label")

    assert pred.tolist() == ["a"]
    assert proba.columns.tolist() == ["a", "b"]
    assert proba.iloc[0].tolist() == [0.7, 0.3]


def test_regression_predict_proba_equals_prediction(make_endpoint, invoke_endpoint):
    endpoint = make_endpoint(pd.DataFrame({"target": [1.5, 2.5]}))
    train_data = pd.DataFrame({"feature": [0, 1], "target": [0.0, 1.0]})
    data = pd.DataFrame({"feature": [2, 3]})

    pred, proba = endpoint.predict_proba(data=data, train_data=train_data, label="target")

    pd.testing.assert_series_equal(pred, proba)


def test_predict_validates_label_and_feature_columns(make_endpoint, invoke_endpoint):
    endpoint = make_endpoint(pd.DataFrame())
    train_data = pd.DataFrame({"feature": [0], "label": ["a"]})

    with pytest.raises(ValueError, match="Label column"):
        endpoint.predict(data=pd.DataFrame({"feature": [1]}), train_data=train_data, label="missing")

    with pytest.raises(ValueError, match="missing feature columns"):
        endpoint.predict(data=pd.DataFrame({"other": [1]}), train_data=train_data, label="label")


def test_delete_endpoint_removes_model_endpoint_and_config(make_endpoint, invoke_endpoint):
    endpoint = make_endpoint(pd.DataFrame())
    with mock.patch(f"{TE}.delete_endpoint") as delete_endpoint:
        endpoint.delete_endpoint()

    delete_endpoint.assert_called_once_with("tabular-fm-endpoint", mock.sentinel.session)
