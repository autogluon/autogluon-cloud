import base64
import datetime
import json
from io import BytesIO

import numpy as np
import pandas as pd
import pytest

from autogluon.cloud.utils.serializers import (
    AUTOGLUON_SERDE_VERSION,
    AutoGluonSerializationWrapper,
    AutoGluonSerializer,
    MultiModalSerializer,
)


@pytest.fixture
def df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


@pytest.fixture
def static_features():
    return pd.DataFrame({"item_id": [0, 1], "category": ["A", "B"]})


@pytest.fixture
def known_covariates():
    return pd.DataFrame({"item_id": [0, 0, 1, 1], "promo": [1.0, 0.0, 0.0, 1.0]})


def _decode_parquet(b64_str: str) -> pd.DataFrame:
    return pd.read_parquet(BytesIO(base64.b64decode(b64_str)))


def _server_parse_autogluon(serialized):
    """Mirrors _parse_autogluon_payload in timeseries_serve.py / timeseries_fm_serve.py."""
    payload = json.loads(serialized)
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported x-autogluon payload version: {payload.get('version')}. Expected 1.")
    inference_kwargs = payload.get("inference_kwargs") or {}
    data = pd.read_parquet(BytesIO(base64.b64decode(payload["data"])))
    static_features = payload.get("static_features")
    if static_features is not None:
        static_features = pd.read_parquet(BytesIO(base64.b64decode(static_features)))
    known_covariates = payload.get("known_covariates")
    if known_covariates is not None:
        known_covariates = pd.read_parquet(BytesIO(base64.b64decode(known_covariates)))
    return data, static_features, known_covariates, inference_kwargs


def _server_parse_multimodal_npy(serialized):
    """Mirrors the x-autogluon-npy path in multimodal_serve.py."""
    payload = json.loads(serialized)
    if payload.get("version") != 1:
        raise ValueError(f"Unsupported x-autogluon payload version: {payload.get('version')}. Expected 1.")
    image_bytearrays = [base64.b85decode(_bytes) for _bytes in payload["data"]]
    inference_kwargs = payload.get("inference_kwargs") or {}
    return image_bytearrays, inference_kwargs


# --- AutoGluonSerializer ---


def test_when_all_fields_provided_then_payload_contains_version_and_data(df, static_features, known_covariates):
    wrapper = AutoGluonSerializationWrapper(
        data=df,
        inference_kwargs={"prediction_length": 3, "quantile_levels": [0.1, 0.9]},
        static_features=static_features,
        known_covariates=known_covariates,
    )
    payload = json.loads(AutoGluonSerializer().serialize(wrapper))

    assert payload["version"] == AUTOGLUON_SERDE_VERSION
    assert payload["inference_kwargs"] == {"prediction_length": 3, "quantile_levels": [0.1, 0.9]}
    pd.testing.assert_frame_equal(df, _decode_parquet(payload["data"]))
    pd.testing.assert_frame_equal(static_features, _decode_parquet(payload["static_features"]))
    pd.testing.assert_frame_equal(known_covariates, _decode_parquet(payload["known_covariates"]))


def test_when_no_optional_fields_then_payload_omits_them(df):
    wrapper = AutoGluonSerializationWrapper(data=df, inference_kwargs={})
    payload = json.loads(AutoGluonSerializer().serialize(wrapper))

    assert "static_features" not in payload
    assert "known_covariates" not in payload
    pd.testing.assert_frame_equal(df, _decode_parquet(payload["data"]))


def test_when_inference_kwargs_is_none_then_treated_as_empty(df):
    wrapper = AutoGluonSerializationWrapper(data=df, inference_kwargs=None)
    payload = json.loads(AutoGluonSerializer().serialize(wrapper))
    assert payload["inference_kwargs"] == {}


def test_when_inference_kwargs_not_json_serializable_then_raises(df):
    wrapper = AutoGluonSerializationWrapper(data=df, inference_kwargs={"bad": datetime.datetime.now()})
    with pytest.raises(ValueError, match="JSON-serializable"):
        AutoGluonSerializer().serialize(wrapper)


# --- MultiModalSerializer ---


def test_given_dataframe_then_payload_has_base64_parquet(df):
    wrapper = AutoGluonSerializationWrapper(data=df, inference_kwargs={"temperature": 0.5})
    payload = json.loads(MultiModalSerializer().serialize(wrapper))

    assert payload["version"] == AUTOGLUON_SERDE_VERSION
    assert payload["inference_kwargs"] == {"temperature": 0.5}
    pd.testing.assert_frame_equal(df, _decode_parquet(payload["data"]))


def test_given_numpy_images_then_payload_has_json_list():
    images = np.array(["b85_encoded_img_1", "b85_encoded_img_2"], dtype="object")
    wrapper = AutoGluonSerializationWrapper(data=images, inference_kwargs={"realtime": True})
    payload = json.loads(MultiModalSerializer().serialize(wrapper))

    assert payload["version"] == AUTOGLUON_SERDE_VERSION
    assert payload["data"] == ["b85_encoded_img_1", "b85_encoded_img_2"]
    assert payload["inference_kwargs"] == {"realtime": True}


def test_given_multimodal_none_kwargs_then_treated_as_empty(df):
    wrapper = AutoGluonSerializationWrapper(data=df, inference_kwargs=None)
    payload = json.loads(MultiModalSerializer().serialize(wrapper))
    assert payload["inference_kwargs"] == {}


def test_given_unsupported_data_type_then_raises():
    wrapper = AutoGluonSerializationWrapper(data=["not", "a", "df"], inference_kwargs={})
    with pytest.raises(ValueError, match="format is not supported"):
        MultiModalSerializer().serialize(wrapper)


# --- Server-side round-trip ---


def test_given_full_payload_then_server_recovers_all_fields(df, static_features, known_covariates):
    wrapper = AutoGluonSerializationWrapper(
        data=df, inference_kwargs={"target": "b"}, static_features=static_features, known_covariates=known_covariates
    )
    data, sf, kc, kwargs = _server_parse_autogluon(AutoGluonSerializer().serialize(wrapper))

    pd.testing.assert_frame_equal(df, data)
    pd.testing.assert_frame_equal(static_features, sf)
    pd.testing.assert_frame_equal(known_covariates, kc)
    assert kwargs == {"target": "b"}


def test_given_minimal_payload_then_optionals_are_none(df):
    wrapper = AutoGluonSerializationWrapper(data=df, inference_kwargs={})
    data, sf, kc, kwargs = _server_parse_autogluon(AutoGluonSerializer().serialize(wrapper))

    pd.testing.assert_frame_equal(df, data)
    assert sf is None
    assert kc is None
    assert kwargs == {}


def test_given_image_payload_then_server_decodes_to_original_bytes():
    raw_images = [b"\x89PNG_fake_image_1", b"\x89PNG_fake_image_2"]
    b85_images = np.array([base64.b85encode(img).decode("utf-8") for img in raw_images], dtype="object")

    wrapper = AutoGluonSerializationWrapper(data=b85_images, inference_kwargs={"k": 5})
    image_bytearrays, kwargs = _server_parse_multimodal_npy(MultiModalSerializer().serialize(wrapper))

    assert image_bytearrays == raw_images
    assert kwargs == {"k": 5}


def test_when_version_wrong_then_server_rejects(df):
    payload = json.loads(AutoGluonSerializer().serialize(AutoGluonSerializationWrapper(data=df, inference_kwargs={})))
    payload["version"] = 99

    with pytest.raises(ValueError, match="Unsupported x-autogluon payload version: 99"):
        _server_parse_autogluon(json.dumps(payload).encode("utf-8"))


def test_when_version_missing_then_server_rejects():
    payload = json.dumps({"data": "irrelevant", "inference_kwargs": {}}).encode("utf-8")
    with pytest.raises(ValueError, match="Unsupported x-autogluon payload version: None"):
        _server_parse_autogluon(payload)


def test_when_body_not_json_then_server_rejects():
    with pytest.raises(json.JSONDecodeError):
        _server_parse_autogluon(b"not json at all")
