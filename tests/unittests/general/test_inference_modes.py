"""Verify that ``inference_mode`` translates to the right SageMaker endpoint config production variant."""

from unittest import mock

import pytest

from autogluon.cloud.backend.sagemaker_backend import SagemakerBackend

GPU_IMAGE_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com/autogluon:1.6-cu133-amzn2023"
SB = "autogluon.cloud.backend.sagemaker_backend"


@pytest.fixture
def deploy_requests(assert_valid_request):
    """Run ``SagemakerBackend.deploy(...)`` with AWS calls mocked, and return the validated requests sent to
    ``create_model`` / ``create_endpoint_config`` / ``create_endpoint``."""
    with (
        mock.patch(f"{SB}.setup_sagemaker_session", return_value=mock.MagicMock(boto_region_name="us-east-1")),
        mock.patch(f"{SB}.resolve_execution_role", return_value="arn:aws:iam::000000000000:role/test"),
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
            client = backend.sagemaker_session.sagemaker_client
            requests = {
                "model": client.create_model.call_args.kwargs,
                "endpoint_config": client.create_endpoint_config.call_args.kwargs,
                "endpoint": client.create_endpoint.call_args.kwargs,
            }
            assert_valid_request("CreateModel", requests["model"])
            assert_valid_request("CreateEndpointConfig", requests["endpoint_config"])
            assert_valid_request("CreateEndpoint", requests["endpoint"])
            return requests

        run.backend = backend
        yield run


def _variant(requests):
    (variant,) = requests["endpoint_config"]["ProductionVariants"]
    return variant


def test_when_inference_mode_realtime_then_instance_settings_are_in_variant(deploy_requests):
    variant = _variant(deploy_requests(instance_type="ml.m5.xlarge", initial_instance_count=2))
    assert variant["InstanceType"] == "ml.m5.xlarge"
    assert variant["InitialInstanceCount"] == 2
    assert "ServerlessConfig" not in variant


def test_when_deployed_then_model_endpoint_config_and_endpoint_are_linked(deploy_requests):
    requests = deploy_requests(instance_type="ml.m5.xlarge")
    assert _variant(requests)["ModelName"] == requests["model"]["ModelName"]
    assert requests["endpoint"]["EndpointConfigName"] == requests["endpoint_config"]["EndpointConfigName"]
    assert requests["endpoint"]["EndpointName"] == "ep"
    environment = requests["model"]["PrimaryContainer"]["Environment"]
    assert environment["SAGEMAKER_PROGRAM"] == "stub.py"
    assert environment["SAGEMAKER_SUBMIT_DIRECTORY"] == "/opt/ml/model/code"


def test_when_cuda_13_custom_image_then_inference_ami_is_inferred(deploy_requests):
    variant = _variant(deploy_requests(instance_type="ml.g4dn.xlarge", custom_image_uri=GPU_IMAGE_URI))
    assert variant["InferenceAmiVersion"] == "al2023-ami-sagemaker-inference-gpu-4-1"


def test_when_inference_ami_is_overridden_then_override_wins(deploy_requests):
    variant = _variant(
        deploy_requests(
            instance_type="ml.g4dn.xlarge",
            custom_image_uri=GPU_IMAGE_URI,
            backend_overrides={"production_variant": {"InferenceAmiVersion": "custom-ami"}},
        )
    )
    assert variant["InferenceAmiVersion"] == "custom-ami"


def test_when_inference_mode_serverless_then_preset_serverless_config_is_used(deploy_requests):
    variant = _variant(deploy_requests(inference_mode="serverless"))
    assert variant["ServerlessConfig"] == {"MemorySizeInMB": 4096, "MaxConcurrency": 5}
    assert "InstanceType" not in variant


def test_when_inference_config_provided_then_user_values_override_preset(deploy_requests):
    variant = _variant(deploy_requests(inference_mode="serverless", inference_config={"memory_size_in_mb": 8192}))
    assert variant["ServerlessConfig"]["MemorySizeInMB"] == 8192
    assert variant["ServerlessConfig"]["MaxConcurrency"] == 5  # preset wins for keys the user didn't override


def test_when_inference_config_has_unknown_key_then_value_error_is_raised(deploy_requests):
    with pytest.raises(ValueError, match="memory_size"):
        deploy_requests(inference_mode="serverless", inference_config={"memory_size": 8192})
    deploy_requests.backend.sagemaker_session.sagemaker_client.create_model.assert_not_called()


@pytest.mark.parametrize("wait", [True, False])
def test_endpoint_waiter_is_used_only_when_waiting(deploy_requests, wait):
    deploy_requests(instance_type="ml.m5.xlarge", wait=wait)
    client = deploy_requests.backend.sagemaker_session.sagemaker_client
    if wait:
        client.get_waiter.assert_called_once_with("endpoint_in_service")
        client.get_waiter.return_value.wait.assert_called_once_with(EndpointName="ep")
    else:
        client.get_waiter.assert_not_called()


def test_when_inference_mode_is_unknown_then_value_error_is_raised(deploy_requests):
    with pytest.raises(ValueError, match="Unsupported inference_mode"):
        deploy_requests(inference_mode="batch")


def test_when_container_environment_overridden_then_it_merges_with_defaults(deploy_requests):
    requests = deploy_requests(
        instance_type="ml.m5.xlarge",
        backend_overrides={"create_model": {"PrimaryContainer": {"Environment": {"FOO": "bar"}}}},
    )
    environment = requests["model"]["PrimaryContainer"]["Environment"]
    assert environment["FOO"] == "bar"
    assert environment["SAGEMAKER_MODEL_SERVER_WORKERS"] == "1"


def test_when_override_targets_training_job_then_deploy_rejects_it(deploy_requests):
    with pytest.raises(ValueError, match="Unsupported `backend_overrides` key"):
        deploy_requests(backend_overrides={"create_training_job": {}})


def test_when_endpoint_creation_fails_then_model_and_config_are_deleted(deploy_requests):
    client = deploy_requests.backend.sagemaker_session.sagemaker_client
    client.create_endpoint.side_effect = RuntimeError("boom")
    with pytest.raises(RuntimeError, match="boom"):
        deploy_requests(instance_type="ml.m5.xlarge")
    config_name = client.create_endpoint_config.call_args.kwargs["EndpointConfigName"]
    assert config_name.startswith("ep-")  # unique per deploy, so a leftover config can't block a redeploy
    client.delete_endpoint_config.assert_called_once_with(EndpointConfigName=config_name)
    client.delete_model.assert_called_once_with(ModelName=client.create_model.call_args.kwargs["ModelName"])
    assert deploy_requests.backend.endpoint_name is None
