"""Unit tests for CloudPredictor.deploy() returning an endpoint and the deprecated endpoint methods."""

from unittest import mock

import pandas as pd
import pytest

from autogluon.cloud import (
    MultiModalCloudPredictor,
    MultiModalEndpoint,
    TabularCloudPredictor,
    TabularEndpoint,
    TimeSeriesCloudPredictor,
    TimeSeriesEndpoint,
)


@pytest.fixture(autouse=True)
def _stub_aws(monkeypatch):
    """Avoid touching AWS / config files during construction."""
    monkeypatch.setattr(
        "autogluon.cloud.predictor.cloud_predictor.resolve_cloud_output_path",
        lambda path, backend_name: path or "s3://stub/output",
    )
    monkeypatch.setattr(
        "autogluon.cloud.backend.backend_factory.BackendFactory.get_backend",
        lambda **kwargs: mock.MagicMock(endpoint_name="ep"),
    )
    monkeypatch.setattr("autogluon.cloud.endpoint.endpoint.setup_sagemaker_session", lambda boto_session: boto_session)


@pytest.mark.filterwarnings("ignore:AutoGluon Multimodal is on a deprecation path")
@pytest.mark.parametrize(
    "predictor_cls, endpoint_cls",
    [
        (TabularCloudPredictor, TabularEndpoint),
        (TimeSeriesCloudPredictor, TimeSeriesEndpoint),
        (MultiModalCloudPredictor, MultiModalEndpoint),
    ],
)
def test_deploy_returns_endpoint_matching_predictor_type(tmp_path, predictor_cls, endpoint_cls):
    predictor = predictor_cls(local_output_path=str(tmp_path))
    predictor.backend.sagemaker_session.boto_session = mock.sentinel.boto_session

    endpoint = predictor.deploy()

    predictor.backend.deploy.assert_called_once()
    assert type(endpoint) is endpoint_cls
    assert endpoint.endpoint_name == "ep"
    assert endpoint._session is mock.sentinel.boto_session


@pytest.mark.parametrize(
    "call",
    [
        lambda p: p.predict_real_time(pd.DataFrame({"a": [1]})),
        lambda p: p.predict_proba_real_time(pd.DataFrame({"a": [1]})),
        lambda p: p.attach_endpoint("ep"),
        lambda p: p.detach_endpoint(),
        lambda p: p.cleanup_deployment(),
    ],
    ids=["predict_real_time", "predict_proba_real_time", "attach_endpoint", "detach_endpoint", "cleanup_deployment"],
)
def test_legacy_endpoint_methods_warn_and_delegate_to_backend(tmp_path, call, request):
    predictor = TabularCloudPredictor(local_output_path=str(tmp_path))
    method = request.node.callspec.id

    with pytest.warns(FutureWarning, match=f"TabularCloudPredictor.{method}` is deprecated") as record:
        call(predictor)

    assert record[0].filename == __file__
    getattr(predictor.backend, method).assert_called_once()


def test_timeseries_predict_real_time_warns(tmp_path):
    predictor = TimeSeriesCloudPredictor(local_output_path=str(tmp_path))

    with pytest.warns(FutureWarning, match="endpoint.predict") as record:
        predictor.predict_real_time(pd.DataFrame({"a": [1]}))

    assert record[0].filename == __file__
    predictor.backend.predict_real_time.assert_called_once()
