import base64
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
def sample_df():
    return pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})


@pytest.fixture
def sample_static_features():
    return pd.DataFrame({"item_id": [0, 1], "category": ["A", "B"]})


@pytest.fixture
def sample_known_covariates():
    return pd.DataFrame({"item_id": [0, 0, 1, 1], "promo": [1.0, 0.0, 0.0, 1.0]})


def _parse_payload(serialized: bytes) -> dict:
    return json.loads(serialized)


def _decode_parquet(b64_str: str) -> pd.DataFrame:
    return pd.read_parquet(BytesIO(base64.b64decode(b64_str)))


class TestAutoGluonSerializer:
    def test_full_payload(self, sample_df, sample_static_features, sample_known_covariates):
        wrapper = AutoGluonSerializationWrapper(
            data=sample_df,
            inference_kwargs={"prediction_length": 3, "quantile_levels": [0.1, 0.9]},
            static_features=sample_static_features,
            known_covariates=sample_known_covariates,
        )
        result = AutoGluonSerializer().serialize(wrapper)
        payload = _parse_payload(result)

        assert payload["version"] == AUTOGLUON_SERDE_VERSION
        assert payload["inference_kwargs"] == {"prediction_length": 3, "quantile_levels": [0.1, 0.9]}
        pd.testing.assert_frame_equal(sample_df, _decode_parquet(payload["data"]))
        pd.testing.assert_frame_equal(sample_static_features, _decode_parquet(payload["static_features"]))
        pd.testing.assert_frame_equal(sample_known_covariates, _decode_parquet(payload["known_covariates"]))

    def test_minimal_payload(self, sample_df):
        wrapper = AutoGluonSerializationWrapper(data=sample_df, inference_kwargs={})
        result = AutoGluonSerializer().serialize(wrapper)
        payload = _parse_payload(result)

        assert payload["version"] == AUTOGLUON_SERDE_VERSION
        assert payload["inference_kwargs"] == {}
        assert "static_features" not in payload
        assert "known_covariates" not in payload
        pd.testing.assert_frame_equal(sample_df, _decode_parquet(payload["data"]))

    def test_none_inference_kwargs(self, sample_df):
        wrapper = AutoGluonSerializationWrapper(data=sample_df, inference_kwargs=None)
        result = AutoGluonSerializer().serialize(wrapper)
        payload = _parse_payload(result)

        assert payload["inference_kwargs"] == {}

    def test_non_serializable_inference_kwargs_raises(self, sample_df):
        import datetime

        wrapper = AutoGluonSerializationWrapper(data=sample_df, inference_kwargs={"bad": datetime.datetime.now()})
        with pytest.raises(ValueError, match="JSON-serializable"):
            AutoGluonSerializer().serialize(wrapper)


class TestMultiModalSerializer:
    def test_dataframe_payload(self, sample_df):
        wrapper = AutoGluonSerializationWrapper(data=sample_df, inference_kwargs={"temperature": 0.5})
        result = MultiModalSerializer().serialize(wrapper)
        payload = _parse_payload(result)

        assert payload["version"] == AUTOGLUON_SERDE_VERSION
        assert payload["inference_kwargs"] == {"temperature": 0.5}
        pd.testing.assert_frame_equal(sample_df, _decode_parquet(payload["data"]))

    def test_numpy_image_payload(self):
        images = np.array(["b85_encoded_img_1", "b85_encoded_img_2"], dtype="object")
        wrapper = AutoGluonSerializationWrapper(data=images, inference_kwargs={"realtime": True})
        result = MultiModalSerializer().serialize(wrapper)
        payload = _parse_payload(result)

        assert payload["version"] == AUTOGLUON_SERDE_VERSION
        assert payload["data"] == ["b85_encoded_img_1", "b85_encoded_img_2"]
        assert payload["inference_kwargs"] == {"realtime": True}

    def test_none_inference_kwargs(self, sample_df):
        wrapper = AutoGluonSerializationWrapper(data=sample_df, inference_kwargs=None)
        result = MultiModalSerializer().serialize(wrapper)
        payload = _parse_payload(result)

        assert payload["inference_kwargs"] == {}

    def test_unsupported_data_type_raises(self):
        wrapper = AutoGluonSerializationWrapper(data=["not", "a", "df", "or", "array"], inference_kwargs={})
        with pytest.raises(ValueError, match="format is not supported"):
            MultiModalSerializer().serialize(wrapper)


class TestServerSideParsing:
    """Simulate the server-side parsing logic from the serve scripts."""

    def _server_parse_autogluon(self, serialized: bytes):
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

    def _server_parse_multimodal_npy(self, serialized: bytes):
        """Mirrors the x-autogluon-npy path in multimodal_serve.py."""
        payload = json.loads(serialized)
        if payload.get("version") != 1:
            raise ValueError(f"Unsupported x-autogluon payload version: {payload.get('version')}. Expected 1.")
        image_bytearrays = [base64.b85decode(_bytes) for _bytes in payload["data"]]
        inference_kwargs = payload.get("inference_kwargs") or {}
        return image_bytearrays, inference_kwargs

    def test_round_trip_full(self, sample_df, sample_static_features, sample_known_covariates):
        wrapper = AutoGluonSerializationWrapper(
            data=sample_df,
            inference_kwargs={"target": "b"},
            static_features=sample_static_features,
            known_covariates=sample_known_covariates,
        )
        serialized = AutoGluonSerializer().serialize(wrapper)
        data, sf, kc, kwargs = self._server_parse_autogluon(serialized)

        pd.testing.assert_frame_equal(sample_df, data)
        pd.testing.assert_frame_equal(sample_static_features, sf)
        pd.testing.assert_frame_equal(sample_known_covariates, kc)
        assert kwargs == {"target": "b"}

    def test_round_trip_minimal(self, sample_df):
        wrapper = AutoGluonSerializationWrapper(data=sample_df, inference_kwargs={})
        serialized = AutoGluonSerializer().serialize(wrapper)
        data, sf, kc, kwargs = self._server_parse_autogluon(serialized)

        pd.testing.assert_frame_equal(sample_df, data)
        assert sf is None
        assert kc is None
        assert kwargs == {}

    def test_round_trip_multimodal_images(self):
        raw_images = [b"\x89PNG_fake_image_1", b"\x89PNG_fake_image_2"]
        b85_images = np.array([base64.b85encode(img).decode("utf-8") for img in raw_images], dtype="object")

        wrapper = AutoGluonSerializationWrapper(data=b85_images, inference_kwargs={"k": 5})
        serialized = MultiModalSerializer().serialize(wrapper)
        image_bytearrays, kwargs = self._server_parse_multimodal_npy(serialized)

        assert image_bytearrays == raw_images
        assert kwargs == {"k": 5}

    def test_wrong_version_raises(self, sample_df):
        wrapper = AutoGluonSerializationWrapper(data=sample_df, inference_kwargs={})
        serialized = AutoGluonSerializer().serialize(wrapper)
        # Tamper with the version
        payload = json.loads(serialized)
        payload["version"] = 99
        tampered = json.dumps(payload).encode("utf-8")

        with pytest.raises(ValueError, match="Unsupported x-autogluon payload version: 99"):
            self._server_parse_autogluon(tampered)

    def test_missing_version_raises(self, sample_df):
        payload = json.dumps({"data": "irrelevant", "inference_kwargs": {}}).encode("utf-8")
        with pytest.raises(ValueError, match="Unsupported x-autogluon payload version: None"):
            self._server_parse_autogluon(payload)

    def test_malformed_json_raises(self):
        with pytest.raises(json.JSONDecodeError):
            self._server_parse_autogluon(b"not json at all")
