"""Verify that ``inference_mode`` translates to the right SageMaker endpoint config production variant."""

from unittest import mock

import pytest

from autogluon.cloud.backend.sagemaker_backend import SagemakerBackend

GPU_IMAGE_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com/autogluon:1.6-cu133-amzn2023"
SB = "autogluon.cloud.backend.sagemaker_backend"


@pytest.fixture
def deploy_requests():
    """Run ``SagemakerBackend.deploy(...)`` with AWS calls and sagemaker-core resources mocked,
    and return the requests that reached ``Model.create`` / ``EndpointConfig.create`` / ``Endpoint.create``."""
    with (
        mock.patch(f"{SB}.setup_sagemaker_session", return_value=mock.MagicMock(boto_region_name="us-east-1")),
        mock.patch(f"{SB}.resolve_execution_role", return_value="arn:aws:iam::000000000000:role/test"),
        mock.patch(f"{SB}.bind_core_session"),
        mock.patch(f"{SB}.Model") as model_cls,
        mock.patch(f"{SB}.EndpointConfig") as endpoint_config_cls,
        mock.patch(f"{SB}.Endpoint") as endpoint_cls,
        mock.patch.object(SagemakerBackend, "_create_serve_script_tarball", return_value="s3://stub/m.tar.gz"),
    ):
        backend = SagemakerBackend(
            local_output_path="/tmp/test",
            cloud_output_path="s3://bucket/run",
            predictor_type="timeseries",
        )
        backend._fit_job = None  # deploy a serve-script tarball, not a fit-job artifact

        def run(**kwargs):
            backend.endpoint_name = None  # allow re-deploy across cases
            backend.deploy(endpoint_name="ep", entry_point="stub.py", **kwargs)
            return {
                "model": model_cls.create.call_args.kwargs,
                "endpoint_config": endpoint_config_cls.create.call_args.kwargs,
                "endpoint": endpoint_cls.create.call_args.kwargs,
            }

        yield run


def _variant(requests):
    (variant,) = requests["endpoint_config"]["production_variants"]
    return variant


def test_when_inference_mode_realtime_then_instance_settings_are_in_variant(deploy_requests):
    variant = _variant(deploy_requests(instance_type="ml.m5.xlarge", initial_instance_count=2))
    assert variant["instance_type"] == "ml.m5.xlarge"
    assert variant["initial_instance_count"] == 2
    assert "serverless_config" not in variant


def test_when_deployed_then_model_endpoint_config_and_endpoint_are_linked(deploy_requests):
    requests = deploy_requests(instance_type="ml.m5.xlarge")
    assert _variant(requests)["model_name"] == requests["model"]["model_name"]
    assert requests["endpoint"]["endpoint_config_name"] == requests["endpoint_config"]["endpoint_config_name"]
    assert requests["endpoint"]["endpoint_name"] == "ep"
    environment = requests["model"]["primary_container"]["environment"]
    assert environment["SAGEMAKER_PROGRAM"] == "stub.py"
    assert environment["SAGEMAKER_SUBMIT_DIRECTORY"] == "/opt/ml/model/code"


def test_when_cuda_13_custom_image_then_inference_ami_is_inferred(deploy_requests):
    variant = _variant(deploy_requests(instance_type="ml.g4dn.xlarge", image_uri=GPU_IMAGE_URI))
    assert variant["inference_ami_version"] == "al2023-ami-sagemaker-inference-gpu-4-1"


def test_when_inference_ami_is_overridden_then_override_wins(deploy_requests):
    variant = _variant(
        deploy_requests(
            instance_type="ml.g4dn.xlarge",
            image_uri=GPU_IMAGE_URI,
            sagemaker_overrides={"production_variant": {"inference_ami_version": "custom-ami"}},
        )
    )
    assert variant["inference_ami_version"] == "custom-ami"


def test_when_inference_mode_serverless_then_preset_serverless_config_is_used(deploy_requests):
    variant = _variant(deploy_requests(inference_mode="serverless"))
    assert variant["serverless_config"] == {"memory_size_in_mb": 4096, "max_concurrency": 5}
    assert "instance_type" not in variant


def test_when_inference_config_provided_then_user_values_override_preset(deploy_requests):
    variant = _variant(deploy_requests(inference_mode="serverless", inference_config={"memory_size_in_mb": 8192}))
    assert variant["serverless_config"]["memory_size_in_mb"] == 8192
    assert variant["serverless_config"]["max_concurrency"] == 5  # preset wins for keys the user didn't override


def test_when_inference_mode_is_unknown_then_value_error_is_raised(deploy_requests):
    with pytest.raises(ValueError, match="Unsupported inference_mode"):
        deploy_requests(inference_mode="batch")


def test_when_environment_given_then_it_reaches_the_container(deploy_requests):
    requests = deploy_requests(instance_type="ml.m5.xlarge", environment={"FOO": "bar"})
    environment = requests["model"]["primary_container"]["environment"]
    assert environment["FOO"] == "bar"
    assert environment["SAGEMAKER_MODEL_SERVER_WORKERS"] == "1"


def test_when_override_targets_training_job_then_deploy_rejects_it(deploy_requests):
    with pytest.raises(ValueError, match="Unsupported `sagemaker_overrides` key"):
        deploy_requests(sagemaker_overrides={"create_training_job": {}})
