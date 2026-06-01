"""Verify that ``inference_mode`` translates to the right ``sagemaker.Model.deploy(...)`` kwargs."""

from unittest import mock

import pytest
from sagemaker.serverless import ServerlessInferenceConfig

from autogluon.cloud.backend.sagemaker_backend import SagemakerBackend


@pytest.fixture
def deploy_kwargs():
    """Run ``SagemakerBackend.deploy(...)`` with AWS calls and the SDK Model class mocked,
    and return the kwargs that reached ``model.deploy(...)``."""
    sb = "autogluon.cloud.backend.sagemaker_backend"
    with (
        mock.patch(f"{sb}.setup_sagemaker_session", return_value=mock.MagicMock(boto_region_name="us-east-1")),
        mock.patch(f"{sb}.resolve_execution_role", return_value="arn:aws:iam::000000000000:role/test"),
        mock.patch(f"{sb}.AutoGluonNonRepackInferenceModel") as model_cls,
        mock.patch.object(SagemakerBackend, "_create_serve_script_tarball", return_value="s3://stub/m.tar.gz"),
    ):
        backend = SagemakerBackend(
            local_output_path="/tmp/test",
            cloud_output_path="s3://bucket/run",
            predictor_type="timeseries",
        )
        backend._fit_job = None  # deploy a serve-script tarball, not a fit-job artifact

        def run(**kwargs):
            backend.endpoint = None  # allow re-deploy across cases
            backend.deploy(endpoint_name="ep", model_kwargs={"entry_point": "stub.py"}, **kwargs)
            return model_cls.return_value.deploy.call_args.kwargs

        yield run


def test_when_inference_mode_realtime_then_instance_kwargs_are_passed(deploy_kwargs):
    captured = deploy_kwargs(instance_type="ml.m5.xlarge", initial_instance_count=2)
    assert captured["instance_type"] == "ml.m5.xlarge"
    assert captured["initial_instance_count"] == 2
    assert "serverless_inference_config" not in captured


def test_when_inference_mode_serverless_then_preset_serverless_config_is_used(deploy_kwargs):
    captured = deploy_kwargs(inference_mode="serverless")
    cfg = captured["serverless_inference_config"]
    assert isinstance(cfg, ServerlessInferenceConfig)
    assert cfg.memory_size_in_mb == 4096
    assert cfg.max_concurrency == 5
    assert "instance_type" not in captured


def test_when_inference_config_provided_then_user_values_override_preset(deploy_kwargs):
    captured = deploy_kwargs(inference_mode="serverless", inference_config={"memory_size_in_mb": 8192})
    cfg = captured["serverless_inference_config"]
    assert cfg.memory_size_in_mb == 8192
    assert cfg.max_concurrency == 5  # preset wins for keys the user didn't override


def test_when_inference_mode_is_unknown_then_value_error_is_raised(deploy_kwargs):
    with pytest.raises(ValueError, match="Unsupported inference_mode"):
        deploy_kwargs(inference_mode="batch")
