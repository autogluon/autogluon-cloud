"""Verify that ``inference_mode`` translates to the right SageMaker SDK kwargs.

These tests pin the contract between :class:`SagemakerBackend.deploy` and
``sagemaker.Model.deploy(...)`` — what kwargs reach SageMaker for each mode.
"""

from unittest import mock

import pytest
from sagemaker.serverless import ServerlessInferenceConfig

from autogluon.cloud.backend.sagemaker_backend import SagemakerBackend


@pytest.fixture
def deploy_kwargs():
    """Run ``SagemakerBackend.deploy(...)`` against a stubbed SageMaker SDK and return the kwargs
    that reached ``sagemaker.Model.deploy(...)``.

    The backend's real ``__init__`` calls ``setup_sagemaker_session()`` (hits AWS), so we bypass it
    and inject the minimal state ``deploy()`` reads.
    """
    backend = SagemakerBackend.__new__(SagemakerBackend)
    backend.endpoint = None
    backend.role_arn = "arn:aws:iam::000000000000:role/test"
    backend._region = "us-east-1"
    backend._cloud_output_path = "s3://bucket/run"
    backend._fit_job = None  # FM-style: no prior training job, deploy from a serve-script tarball

    captured: dict = {}

    def fake_deploy(self, **kwargs):
        captured.update(kwargs)
        return mock.Mock(endpoint_name="ep")

    sb = "autogluon.cloud.backend.sagemaker_backend"
    with (
        mock.patch.object(SagemakerBackend, "_create_serve_script_tarball", return_value="s3://stub/m.tar.gz"),
        mock.patch(f"{sb}.parse_framework_version", return_value=("1.0", "py3")),
        mock.patch(f"{sb}.ScriptManager.get_serve_script", return_value="stub.py"),
        mock.patch(f"{sb}.AutoGluonNonRepackInferenceModel.__init__", return_value=None),
        mock.patch(f"{sb}.AutoGluonNonRepackInferenceModel.deploy", fake_deploy),
    ):

        def run(**kwargs):
            backend.deploy(endpoint_name="ep", **kwargs)
            return captured

        yield run


def test_realtime_passes_instance_kwargs(deploy_kwargs):
    captured = deploy_kwargs(instance_type="ml.m5.xlarge", initial_instance_count=2)
    assert captured["instance_type"] == "ml.m5.xlarge"
    assert captured["initial_instance_count"] == 2
    assert "serverless_inference_config" not in captured


def test_serverless_uses_preset(deploy_kwargs):
    captured = deploy_kwargs(inference_mode="serverless")
    cfg = captured["serverless_inference_config"]
    assert isinstance(cfg, ServerlessInferenceConfig)
    assert cfg.memory_size_in_mb == 4096
    assert cfg.max_concurrency == 5
    assert "instance_type" not in captured


def test_serverless_user_overrides_merge_over_preset(deploy_kwargs):
    captured = deploy_kwargs(inference_mode="serverless", inference_config={"memory_size_in_mb": 8192})
    cfg = captured["serverless_inference_config"]
    assert cfg.memory_size_in_mb == 8192
    assert cfg.max_concurrency == 5  # preset wins for keys the user didn't override


def test_unknown_inference_mode_raises(deploy_kwargs):
    with pytest.raises(ValueError, match="Unsupported inference_mode"):
        deploy_kwargs(inference_mode="batch")
