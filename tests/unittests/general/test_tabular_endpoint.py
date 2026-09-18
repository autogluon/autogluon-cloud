from unittest import mock

import pandas as pd
import pytest

from autogluon.cloud.endpoint.tabular_endpoint import TabularEndpoint
from autogluon.cloud.utils.serializers import AutoGluonSerializationWrapper


def _make_endpoint(response):
    endpoint = TabularEndpoint.__new__(TabularEndpoint)
    endpoint._predictor = mock.MagicMock(endpoint_name="tabular-fm-endpoint")
    endpoint._predictor.predict.return_value = response
    return endpoint


def test_predict_sends_train_data_and_returns_prediction_series():
    train_data = pd.DataFrame({"feature": [0, 1], "label": ["a", "b"]})
    data = pd.DataFrame({"feature": [2, 3]})
    response = pd.DataFrame({"label": ["a", "b"], "a_proba": [0.8, 0.2], "b_proba": [0.2, 0.8]})
    endpoint = _make_endpoint(response)

    pred = endpoint.predict(data=data, train_data=train_data, label="label")

    assert pred.tolist() == ["a", "b"]
    payload = endpoint._predictor.predict.call_args.args[0]
    assert isinstance(payload, AutoGluonSerializationWrapper)
    pd.testing.assert_frame_equal(payload.data, data)
    pd.testing.assert_frame_equal(payload.train_data, train_data)
    assert payload.inference_kwargs == {"label": "label"}


def test_predict_proba_matches_batch_result_shape():
    response = pd.DataFrame({"label": ["a"], "a_proba": [0.7], "b_proba": [0.3]})
    endpoint = _make_endpoint(response)
    train_data = pd.DataFrame({"feature": [0, 1], "label": ["a", "b"]})
    data = pd.DataFrame({"feature": [2]})

    pred, proba = endpoint.predict_proba(data=data, train_data=train_data, label="label")

    assert pred.tolist() == ["a"]
    assert proba.columns.tolist() == ["a", "b"]
    assert proba.iloc[0].tolist() == [0.7, 0.3]


def test_regression_predict_proba_equals_prediction():
    endpoint = _make_endpoint(pd.DataFrame({"target": [1.5, 2.5]}))
    train_data = pd.DataFrame({"feature": [0, 1], "target": [0.0, 1.0]})
    data = pd.DataFrame({"feature": [2, 3]})

    pred, proba = endpoint.predict_proba(data=data, train_data=train_data, label="target")

    pd.testing.assert_series_equal(pred, proba)


def test_predict_validates_label_and_feature_columns():
    endpoint = _make_endpoint(pd.DataFrame())
    train_data = pd.DataFrame({"feature": [0], "label": ["a"]})

    with pytest.raises(ValueError, match="Label column"):
        endpoint.predict(data=pd.DataFrame({"feature": [1]}), train_data=train_data, label="missing")

    with pytest.raises(ValueError, match="missing feature columns"):
        endpoint.predict(data=pd.DataFrame({"other": [1]}), train_data=train_data, label="label")


def test_delete_endpoint_removes_model_endpoint_and_config():
    endpoint = _make_endpoint(pd.DataFrame())
    endpoint.delete_endpoint()

    endpoint._predictor.delete_model.assert_called_once_with()
    endpoint._predictor.delete_endpoint.assert_called_once_with(delete_endpoint_config=True)
