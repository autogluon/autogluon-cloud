"""Unit tests for CloudPredictor.deploy() returning an endpoint and the deprecated endpoint methods."""

import logging
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
    # autogluon.common sets propagate=False on the `autogluon` logger, which hides records from caplog.
    monkeypatch.setattr(logging.getLogger("autogluon"), "propagate", True)


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


def _describe_endpoint_returning(predictor, status=None, error=None):
    describe = predictor.backend.sagemaker_session.sagemaker_client.describe_endpoint
    describe.return_value = {"EndpointStatus": status}
    describe.side_effect = error
    return describe


@pytest.mark.parametrize("status", ["InService", "Creating"])
def test_deploy_warns_when_previous_endpoint_is_active(tmp_path, caplog, status):
    predictor = TabularCloudPredictor(local_output_path=str(tmp_path))
    describe = _describe_endpoint_returning(predictor, status=status)

    predictor.deploy()

    describe.assert_called_once_with(EndpointName="ep")
    assert "already deployed endpoint ep" in caplog.text
    assert "TabularEndpoint('ep').delete_endpoint()" in caplog.text


@pytest.mark.parametrize(
    "status, error",
    [("Deleting", None), ("Failed", None), (None, Exception("Could not find endpoint"))],
    ids=["deleting", "failed", "not_found"],
)
def test_deploy_does_not_warn_when_previous_endpoint_is_gone(tmp_path, caplog, status, error):
    predictor = TabularCloudPredictor(local_output_path=str(tmp_path))
    _describe_endpoint_returning(predictor, status=status, error=error)

    predictor.deploy()

    assert "already deployed endpoint" not in caplog.text
    predictor.backend.deploy.assert_called_once()


def test_first_deploy_does_not_check_endpoint(tmp_path):
    predictor = TabularCloudPredictor(local_output_path=str(tmp_path))
    predictor.backend.endpoint_name = None

    predictor.deploy()

    predictor.backend.sagemaker_session.sagemaker_client.describe_endpoint.assert_not_called()
