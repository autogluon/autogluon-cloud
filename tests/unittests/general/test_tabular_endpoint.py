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


def test_predict_without_train_data_sends_only_data(make_endpoint, invoke_endpoint):
    data = pd.DataFrame({"feature": [2, 3]})
    endpoint = make_endpoint(pd.DataFrame({"label": ["a", "b"], "a_proba": [0.8, 0.2], "b_proba": [0.2, 0.8]}))

    pred = endpoint.predict(data, model="LightGBM")

    assert pred.tolist() == ["a", "b"]
    payload = invoke_endpoint.call_args.args[2]
    pd.testing.assert_frame_equal(payload.data, data)
    assert payload.train_data is None
    assert payload.inference_kwargs == {"model": "LightGBM"}


def test_predict_requires_train_data_and_label_together(make_endpoint, invoke_endpoint):
    endpoint = make_endpoint(pd.DataFrame())
    data = pd.DataFrame({"feature": [1]})

    with pytest.raises(ValueError, match="must be passed together"):
        endpoint.predict(data, train_data=pd.DataFrame({"feature": [0], "label": ["a"]}))

    with pytest.raises(ValueError, match="must be passed together"):
        endpoint.predict(data, label="label")

    invoke_endpoint.assert_not_called()


def test_image_column_is_encoded_and_not_forwarded(make_endpoint, invoke_endpoint):
    endpoint = make_endpoint(pd.DataFrame({"target": [1.5]}))
    data = pd.DataFrame({"image": ["/abs/img.png"]})

    with mock.patch(f"{TE}.convert_image_path_to_encoded_bytes_in_dataframe") as convert:
        endpoint.predict(data, image_column="image")

    convert.assert_called_once_with(data, "image")
    payload = invoke_endpoint.call_args.args[2]
    assert payload.data is convert.return_value
    assert payload.inference_kwargs == {}


def test_image_column_with_train_data_raises(make_endpoint, invoke_endpoint):
    endpoint = make_endpoint(pd.DataFrame({"label": ["a"]}))
    train_data = pd.DataFrame({"image": ["/abs/img.png"], "label": ["a"]})

    with pytest.raises(ValueError, match="`image_column` is only supported"):
        endpoint.predict(pd.DataFrame({"image": ["/abs/img.png"]}), train_data, "label", image_column="image")

    invoke_endpoint.assert_not_called()


@pytest.mark.parametrize(
    "train_data, label", [(None, None), (pd.DataFrame({"feature": [0], "label": ["a"]}), "label")]
)
def test_as_pandas_is_not_forwarded(make_endpoint, invoke_endpoint, train_data, label):
    # The serve scripts pass as_pandas=True themselves; forwarding it raises a duplicate-keyword TypeError.
    endpoint = make_endpoint(pd.DataFrame({"label": ["a"], "a_proba": [1.0]}))

    endpoint.predict(pd.DataFrame({"feature": [1]}), train_data, label, as_pandas=True, model="LightGBM")

    assert "as_pandas" not in invoke_endpoint.call_args.args[2].inference_kwargs


def test_delete_endpoint_removes_model_endpoint_and_config(make_endpoint, invoke_endpoint):
    endpoint = make_endpoint(pd.DataFrame())
    with mock.patch("autogluon.cloud.endpoint.endpoint.delete_endpoint") as delete_endpoint:
        endpoint.delete_endpoint()

    delete_endpoint.assert_called_once_with("tabular-fm-endpoint", mock.sentinel.session)
