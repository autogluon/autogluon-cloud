from unittest import mock

import pandas as pd
import pytest

from autogluon.cloud import SageMakerConfig
from autogluon.cloud.backend.tabular_sagemaker_backend import TabularSagemakerBackend
from autogluon.cloud.utils.sagemaker_api import (
    check_override_keys,
    deep_merge,
    delete_endpoint,
    invoke_endpoint,
    reject_legacy_kwargs,
)

SB = "autogluon.cloud.backend.sagemaker_backend"


def test_deep_merge_merges_dicts_and_replaces_other_values():
    base = {"a": {"x": 1, "y": 2}, "b": [1, 2], "c": 1}
    merged = deep_merge(base, {"a": {"y": 3}, "b": [3], "d": 4})
    assert merged == {"a": {"x": 1, "y": 3}, "b": [3], "c": 1, "d": 4}
    assert base["a"]["y"] == 2  # input is not mutated


def test_check_override_keys_rejects_unknown_keys():
    with pytest.raises(ValueError, match="create_model"):
        check_override_keys({"create_model": {}}, ("create_training_job",))
    assert check_override_keys(None, ("create_training_job",)) == {}


def test_reject_legacy_kwargs_points_to_replacement():
    @reject_legacy_kwargs
    def fit(**kwargs):
        return kwargs

    assert fit(image_uri="x") == {"image_uri": "x"}
    with pytest.raises(TypeError, match="Use `image_uri` instead"):
        fit(custom_image_uri="x")
    with pytest.raises(TypeError, match="backend_overrides"):
        fit(backend_kwargs={})


def test_delete_endpoint_removes_endpoint_config_and_models():
    session = mock.MagicMock()
    client = session.sagemaker_client
    client.describe_endpoint.return_value = {"EndpointConfigName": "ep-config"}
    client.describe_endpoint_config.return_value = {"ProductionVariants": [{"ModelName": "m-1"}, {"ModelName": "m-2"}]}
    delete_endpoint("ep", session)
    client.delete_endpoint.assert_called_once_with(EndpointName="ep")
    client.delete_endpoint_config.assert_called_once_with(EndpointConfigName="ep-config")
    assert [c.kwargs for c in client.delete_model.call_args_list] == [{"ModelName": "m-1"}, {"ModelName": "m-2"}]


def test_invoke_endpoint_uses_the_session_runtime_client():
    session = mock.MagicMock()
    runtime = session.sagemaker_runtime_client
    runtime.invoke_endpoint.return_value = {"Body": mock.sentinel.body, "ContentType": "text/csv"}
    serializer = mock.Mock(CONTENT_TYPE="text/csv")
    deserializer = mock.Mock(ACCEPT=("application/json",))
    result = invoke_endpoint("ep", session, "payload", serializer=serializer, deserializer=deserializer)
    runtime.invoke_endpoint.assert_called_once_with(
        EndpointName="ep",
        Body=serializer.serialize.return_value,
        ContentType="text/csv",
        Accept="application/json",
    )
    deserializer.deserialize.assert_called_once_with(mock.sentinel.body, "text/csv")
    assert result is deserializer.deserialize.return_value


@pytest.fixture
def fit_request(tmp_path, assert_valid_request):
    """Run ``SagemakerBackend.fit(...)`` with uploads mocked and return the ``CreateTrainingJob`` request."""
    with (
        mock.patch(f"{SB}.setup_sagemaker_session", return_value=mock.MagicMock(boto_region_name="us-east-1")),
        mock.patch(f"{SB}.resolve_execution_role", return_value="arn:aws:iam::000000000000:role/test"),
        mock.patch(f"{SB}.upload_training_code", return_value="s3://bucket/run/code/job/source/sourcedir.tar.gz"),
        mock.patch(f"{SB}.SageMakerFitJob") as fit_job_cls,
        mock.patch.object(
            TabularSagemakerBackend, "_upload_fit_artifact", return_value={"train_data": "s3://b/train.csv"}
        ),
    ):

        def run(backend_config=None, **fit_kwargs):
            backend = TabularSagemakerBackend(
                local_output_path=str(tmp_path),
                cloud_output_path="s3://bucket/run",
                predictor_type="tabular",
                config=backend_config,
            )
            backend.fit(
                predictor_init_args={"label": "y"},
                predictor_fit_args={},
                data_channels={"train_data": pd.DataFrame({"x": [1], "y": [0]})},
                job_name="job",
                image_uri="example.com/autogluon:train",
                **fit_kwargs,
            )
            request = fit_job_cls.return_value.run.call_args.kwargs["training_job_request"]
            assert_valid_request("CreateTrainingJob", request)
            return request

        yield run


def test_fit_builds_script_mode_training_job(fit_request):
    request = fit_request(timeout=3600)
    assert request["TrainingJobName"] == "job"
    assert request["AlgorithmSpecification"]["TrainingImage"] == "example.com/autogluon:train"
    assert request["HyperParameters"]["sagemaker_program"] == '"train.py"'
    assert request["HyperParameters"]["sagemaker_submit_directory"] == (
        '"s3://bucket/run/code/job/source/sourcedir.tar.gz"'
    )
    assert request["InputDataConfig"][0]["ChannelName"] == "train_data"
    assert request["StoppingCondition"] == {"MaxRuntimeInSeconds": 3600}
    assert request["OutputDataConfig"] == {"S3OutputPath": "s3://bucket/run/model"}
    assert {"Key": "autogluon-cloud-module", "Value": "tabular"} in request["Tags"]
    assert "VpcConfig" not in request


def test_fit_applies_infra_settings_spot_and_overrides(fit_request):
    request = fit_request(
        backend_config=SageMakerConfig(
            vpc_config={"subnets": ["s-1"], "security_group_ids": ["sg-1"]},
            output_kms_key="output-key",
            volume_kms_key="volume-key",
            tags={"team": "ts"},
        ),
        timeout=3600,
        environment={"FOO": "bar"},
        use_spot_instances=True,
        backend_overrides={"create_training_job": {"RetryStrategy": {"MaximumRetryAttempts": 2}}},
    )
    assert request["VpcConfig"] == {"Subnets": ["s-1"], "SecurityGroupIds": ["sg-1"]}
    assert request["OutputDataConfig"]["KmsKeyId"] == "output-key"
    assert request["ResourceConfig"]["VolumeKmsKeyId"] == "volume-key"
    assert {"Key": "team", "Value": "ts"} in request["Tags"]
    assert request["Environment"] == {"FOO": "bar"}
    assert request["EnableManagedSpotTraining"] is True
    assert request["StoppingCondition"]["MaxWaitTimeInSeconds"] == 3600
    assert request["RetryStrategy"] == {"MaximumRetryAttempts": 2}


@pytest.mark.parametrize(
    "vpc_config",
    [{"subnets": ["s-1"]}, {"subnets": ["s-1"], "security_group_ids": ["sg-1"], "security_groups": ["sg-2"]}],
)
def test_fit_rejects_malformed_vpc_config(fit_request, vpc_config):
    with pytest.raises(ValueError, match="security_group"):
        fit_request(backend_config=SageMakerConfig(vpc_config=vpc_config))


def test_output_encryption_does_not_set_volume_key_on_nvme_instance(fit_request):
    request = fit_request(
        backend_config=SageMakerConfig(output_kms_key="output-key"),
        instance_type="ml.g5.xlarge",
    )
    assert request["OutputDataConfig"]["KmsKeyId"] == "output-key"
    assert "VolumeKmsKeyId" not in request["ResourceConfig"]


def test_fit_rejects_local_mode_and_max_wait_without_spot(fit_request):
    with pytest.raises(ValueError, match="local mode"):
        fit_request(instance_type="local")
    with pytest.raises(ValueError, match="use_spot_instances"):
        fit_request(max_wait=100)


def test_misspelled_override_field_fails_request_validation(fit_request):
    from botocore.exceptions import ParamValidationError

    with pytest.raises(ParamValidationError, match="RetryStrategyy"):
        fit_request(backend_overrides={"create_training_job": {"RetryStrategyy": {"MaximumRetryAttempts": 2}}})
