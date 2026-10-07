"""Unit tests for the ``**kwargs`` catch-all of CloudPredictor and FoundationModel methods."""

import logging
from unittest import mock

import pandas as pd
import pytest

from autogluon.cloud import (
    MultiModalCloudPredictor,
    TabularCloudPredictor,
    TabularFoundationModel,
    TimeSeriesCloudPredictor,
    TimeSeriesFoundationModel,
)

TRAIN_DATA = pd.DataFrame({"x": [1, 2], "y": [0, 1]})
TS_DATA = pd.DataFrame({"item_id": ["A", "A"], "timestamp": ["2024-01-01", "2024-01-02"], "target": [1.0, 2.0]})


@pytest.fixture(autouse=True)
def _stub_aws(monkeypatch):
    """Avoid touching AWS / config files during construction."""
    for module in ("predictor.cloud_predictor", "model.foundation_model"):
        monkeypatch.setattr(
            f"autogluon.cloud.{module}.resolve_cloud_output_path",
            lambda path, backend_name: path or "s3://stub/output",
        )
    monkeypatch.setattr(
        "autogluon.cloud.backend.backend_factory.BackendFactory.get_backend",
        lambda **kwargs: mock.MagicMock(is_fit=False, endpoint_name="ep"),
    )
    monkeypatch.setattr("autogluon.cloud.endpoint.endpoint.setup_sagemaker_session", lambda boto_session: boto_session)
    # autogluon.common sets propagate=False on the `autogluon` logger, which hides records from caplog.
    monkeypatch.setattr(logging.getLogger("autogluon"), "propagate", True)


@pytest.fixture
def predictor(tmp_path):
    return TabularCloudPredictor(local_output_path=str(tmp_path))


def test_fit_forwards_backend_kwargs(predictor):
    predictor.fit(TRAIN_DATA, predictor_init_args={"label": "y"}, job_name="job", custom_image_uri="img", timeout=60)
    kwargs = predictor.backend.fit.call_args.kwargs
    assert (kwargs["job_name"], kwargs["custom_image_uri"], kwargs["timeout"]) == ("job", "img", 60)


@pytest.mark.parametrize("name, value", [("instance_count", 2), ("leaderboard", False)])
def test_fit_warns_and_drops_ignored_kwargs(predictor, caplog, name, value):
    predictor.fit(TRAIN_DATA, predictor_init_args={"label": "y"}, **{name: value})
    assert f"`{name}` is no longer supported by fit() and is ignored" in caplog.text
    assert name not in predictor.backend.fit.call_args.kwargs


@pytest.mark.parametrize(
    "call",
    [
        lambda p: p.fit(TRAIN_DATA, predictor_init_args={"label": "y"}, instance_typo="ml.m5.xlarge"),
        lambda p: p.fit_predict(TRAIN_DATA, TRAIN_DATA, predictor_init_args={"label": "y"}, instance_typo="x"),
        lambda p: p.predict(TRAIN_DATA, volume_size=10),  # batch transform has no volume_size
        lambda p: p.predict_proba(TRAIN_DATA, test_data_image_column="image"),  # multimodal only
        lambda p: p.deploy(timeout=10),
    ],
    ids=["fit", "fit_predict", "predict", "predict_proba", "deploy"],
)
def test_unknown_backend_kwargs_raise(predictor, call):
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        call(predictor)


def test_tabular_fit_rejects_image_column(predictor):
    with pytest.raises(ValueError, match="`image_column` is no longer supported for tabular predictors"):
        predictor.fit(TRAIN_DATA, predictor_init_args={"label": "y"}, image_column="image")


@pytest.mark.filterwarnings("ignore:AutoGluon Multimodal is on a deprecation path")
def test_multimodal_forwards_image_columns(tmp_path):
    predictor = MultiModalCloudPredictor(local_output_path=str(tmp_path))
    predictor.fit(TRAIN_DATA, predictor_init_args={"label": "y"}, image_column="image")
    assert predictor.backend.fit.call_args.kwargs["image_column"] == "image"
    predictor.predict(TRAIN_DATA, test_data_image_column="image", instance_count=2)
    kwargs = predictor.backend.predict.call_args.kwargs
    assert (kwargs["test_data_image_column"], kwargs["instance_count"]) == ("image", 2)


def test_timeseries_fit_forwards_timeout(tmp_path):
    predictor = TimeSeriesCloudPredictor(local_output_path=str(tmp_path))
    predictor.fit(TS_DATA, predictor_init_args={"prediction_length": 1}, timeout=60)
    assert predictor.backend.fit.call_args.kwargs["timeout"] == 60


@pytest.mark.parametrize(
    "call",
    [
        lambda p: p.predict(TRAIN_DATA, "s3://bucket/predictor.tar.gz"),
        lambda p: p.deploy("s3://bucket/predictor.tar.gz"),
    ],
    ids=["predict", "deploy"],
)
def test_options_are_keyword_only(predictor, call):
    with pytest.raises(TypeError, match="positional argument"):
        call(predictor)


def test_foundation_model_deploy_forwards_backend_kwargs():
    fm = TimeSeriesFoundationModel("chronos-2", cloud_output_path="s3://b")
    fm.deploy(initial_instance_count=2, custom_image_uri="img", backend_overrides={"CreateEndpoint": {}})
    kwargs = fm._backend.deploy.call_args.kwargs
    assert (kwargs["initial_instance_count"], kwargs["custom_image_uri"]) == (2, "img")
    assert kwargs["backend_overrides"] == {"CreateEndpoint": {}}


def test_foundation_model_predict_rejects_unknown_backend_kwargs():
    fm = TabularFoundationModel("mitra-classifier", cloud_output_path="s3://b")
    with pytest.raises(TypeError, match="unexpected keyword argument"):
        fm.predict(TRAIN_DATA, TRAIN_DATA, label="y", instance_count=2)
    fm._backend.fit.assert_not_called()


def test_timeseries_foundation_model_predict_is_keyword_only():
    fm = TimeSeriesFoundationModel("chronos-2", cloud_output_path="s3://b")
    with pytest.raises(TypeError, match="positional argument"):
        fm.predict(TS_DATA, "target")
