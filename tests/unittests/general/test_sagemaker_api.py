from unittest import mock

import boto3
import pandas as pd
import pytest

from autogluon.cloud.backend.tabular_sagemaker_backend import TabularSagemakerBackend
from autogluon.cloud.utils.sagemaker_api import check_override_keys, deep_merge, reject_legacy_kwargs
from autogluon.cloud.utils.sagemaker_core_workarounds import bind_core_session

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
    with pytest.raises(TypeError, match="sagemaker_overrides"):
        fit(backend_kwargs={})


def test_bind_core_session_rebinds_when_session_changes():
    from sagemaker.core.utils.utils import SageMakerClient, SingletonMeta

    first = boto3.Session(region_name="us-east-1")
    second = boto3.Session(region_name="eu-west-1")
    bind_core_session(first)
    cached = SingletonMeta._instances[SageMakerClient]
    assert cached.session is first
    bind_core_session(first)
    assert SingletonMeta._instances[SageMakerClient] is cached  # no rebuild for the same session
    bind_core_session(second)
    assert SageMakerClient().session is second
    assert SageMakerClient().region_name == "eu-west-1"


@pytest.fixture
def fit_request(tmp_path):
    """Run ``SagemakerBackend.fit(...)`` with uploads mocked and return the ``TrainingJob.create`` request."""
    with (
        mock.patch(f"{SB}.setup_sagemaker_session", return_value=mock.MagicMock(boto_region_name="us-east-1")),
        mock.patch(f"{SB}.resolve_execution_role", return_value="arn:aws:iam::000000000000:role/test"),
        mock.patch(f"{SB}.upload_training_code", return_value="s3://bucket/run/code/job/source/sourcedir.tar.gz"),
        mock.patch.object(
            TabularSagemakerBackend, "_upload_fit_artifact", return_value={"train_data": "s3://b/train.csv"}
        ),
    ):

        def run(backend_kwargs=None, **fit_kwargs):
            backend = TabularSagemakerBackend(
                local_output_path=str(tmp_path),
                cloud_output_path="s3://bucket/run",
                predictor_type="tabular",
                **(backend_kwargs or {}),
            )
            backend._fit_job = mock.MagicMock()
            backend.fit(
                predictor_init_args={"label": "y"},
                predictor_fit_args={},
                data_channels={"train_data": pd.DataFrame({"x": [1], "y": [0]})},
                job_name="job",
                image_uri="example.com/autogluon:train",
                **fit_kwargs,
            )
            return backend._fit_job.run.call_args.kwargs["training_job_request"]

        yield run


def test_fit_builds_script_mode_training_job(fit_request):
    request = fit_request(timeout=3600)
    assert request["training_job_name"] == "job"
    assert request["algorithm_specification"]["training_image"] == "example.com/autogluon:train"
    assert request["hyper_parameters"]["sagemaker_program"] == '"train.py"'
    assert request["hyper_parameters"]["sagemaker_submit_directory"] == (
        '"s3://bucket/run/code/job/source/sourcedir.tar.gz"'
    )
    assert request["input_data_config"][0]["channel_name"] == "train_data"
    assert request["stopping_condition"] == {"max_runtime_in_seconds": 3600}
    assert request["output_data_config"] == {"s3_output_path": "s3://bucket/run/model"}
    assert {"key": "autogluon-cloud-module", "value": "tabular"} in request["tags"]
    assert "vpc_config" not in request


def test_fit_applies_infra_settings_spot_and_overrides(fit_request):
    request = fit_request(
        backend_kwargs={
            "vpc_config": {"subnets": ["s-1"], "security_group_ids": ["sg-1"]},
            "kms_key": "kms-1",
            "tags": {"team": "ts"},
        },
        timeout=3600,
        environment={"FOO": "bar"},
        use_spot_instances=True,
        sagemaker_overrides={"create_training_job": {"retry_strategy": {"maximum_retry_attempts": 2}}},
    )
    assert request["vpc_config"] == {"subnets": ["s-1"], "security_group_ids": ["sg-1"]}
    assert request["output_data_config"]["kms_key_id"] == "kms-1"
    assert request["resource_config"]["volume_kms_key_id"] == "kms-1"
    assert {"key": "team", "value": "ts"} in request["tags"]
    assert request["environment"] == {"FOO": "bar"}
    assert request["enable_managed_spot_training"] is True
    assert request["stopping_condition"]["max_wait_time_in_seconds"] == 3600
    assert request["retry_strategy"] == {"maximum_retry_attempts": 2}


def test_fit_rejects_malformed_vpc_config(fit_request):
    with pytest.raises(ValueError, match="security_group_ids"):
        fit_request(backend_kwargs={"vpc_config": {"subnets": ["s-1"]}})


def test_fit_rejects_local_mode_and_max_wait_without_spot(fit_request):
    with pytest.raises(ValueError, match="local mode"):
        fit_request(instance_type="local")
    with pytest.raises(ValueError, match="use_spot_instances"):
        fit_request(max_wait=100)


def test_core_serializes_acronym_field_names_with_api_casing():
    from sagemaker.core.shapes import ProductionVariant
    from sagemaker.core.utils.utils import serialize

    variant = ProductionVariant(
        variant_name="AllTraffic",
        serverless_config={"memory_size_in_mb": 4096, "max_concurrency": 5},
        enable_ssm_access=True,
    )
    assert serialize(variant) == {
        "VariantName": "AllTraffic",
        "ServerlessConfig": {"MemorySizeInMB": 4096, "MaxConcurrency": 5},
        "EnableSSMAccess": True,
    }
