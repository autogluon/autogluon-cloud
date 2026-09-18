from unittest import mock

import pytest

from autogluon.cloud.job.sagemaker_job import SageMakerBatchTransformationJob
from autogluon.cloud.utils.ag_sagemaker import _TransformAmiVersionSession
from autogluon.cloud.utils.dlc_utils import infer_sagemaker_ami_version

GPU_IMAGE_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com/autogluon:1.6-cu133-amzn2023"
GPU_UBUNTU_IMAGE_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com/autogluon:1.6-cu133-ubuntu24.04"
CPU_IMAGE_URI = "123456789012.dkr.ecr.us-east-1.amazonaws.com/autogluon:1.6-cpu-amzn2023"


@pytest.mark.parametrize("image_uri", [GPU_IMAGE_URI, GPU_UBUNTU_IMAGE_URI])
@pytest.mark.parametrize(
    ("image_scope", "expected"),
    [
        ("inference", "al2023-ami-sagemaker-inference-gpu-4-1"),
        ("transform", "al2-ami-sagemaker-batch-gpu-535"),
    ],
)
def test_infer_sagemaker_ami_version_for_cuda_13_image(image_uri, image_scope, expected):
    assert infer_sagemaker_ami_version(image_uri, "ml.g4dn.xlarge", image_scope) == expected


@pytest.mark.parametrize(
    ("image_uri", "instance_type"),
    [
        (CPU_IMAGE_URI, "ml.m5.xlarge"),
        (GPU_IMAGE_URI, "ml.m5.xlarge"),
        (GPU_IMAGE_URI, "local_gpu"),
        ("example.com/autogluon:1.5-gpu-py310", "ml.g4dn.xlarge"),
    ],
)
def test_infer_sagemaker_ami_version_ignores_other_images(image_uri, instance_type):
    assert infer_sagemaker_ami_version(image_uri, instance_type, "inference") is None


def test_infer_batch_ami_ignores_unsupported_instance_family():
    assert infer_sagemaker_ami_version(GPU_IMAGE_URI, "ml.p5.xlarge", "transform") is None


@pytest.mark.parametrize("instance_type", ["ml.p3.2xlarge", "ml.p6-b200.48xlarge", "ml.g7e.48xlarge"])
def test_infer_realtime_ami_ignores_unsupported_or_already_compatible_instance_family(instance_type):
    assert infer_sagemaker_ami_version(GPU_IMAGE_URI, instance_type, "inference") is None


def test_transform_ami_session_injects_ami_without_mutating_input():
    session = mock.MagicMock()
    wrapper = _TransformAmiVersionSession(session, "al2-ami-sagemaker-batch-gpu-535")
    resource_config = {"InstanceCount": 1, "InstanceType": "ml.g4dn.xlarge"}

    wrapper.transform(resource_config=resource_config, job_name="job")

    assert "TransformAmiVersion" not in resource_config
    assert session.transform.call_args.kwargs["resource_config"]["TransformAmiVersion"] == (
        "al2-ami-sagemaker-batch-gpu-535"
    )


@pytest.mark.parametrize(
    ("transformer_kwargs", "expected"),
    [
        ({}, "al2-ami-sagemaker-batch-gpu-535"),
        ({"transform_ami_version": "custom-ami"}, "custom-ami"),
    ],
)
def test_batch_transform_job_sets_inferred_ami_without_overriding_user_value(transformer_kwargs, expected):
    sj = "autogluon.cloud.job.sagemaker_job"
    transformer = mock.MagicMock(output_path="s3://output")
    transformer.latest_transform_job.name = "job"
    with mock.patch(f"{sj}.AutoGluonNonRepackInferenceModel") as model_cls:
        model_cls.return_value.transformer.return_value = transformer
        job = SageMakerBatchTransformationJob(session=mock.MagicMock())
        job.run(
            model_data="s3://bucket/model.tar.gz",
            role="role",
            region="us-east-1",
            framework_version=None,
            py_version=None,
            instance_count=1,
            instance_type="ml.g4dn.xlarge",
            entry_point="serve.py",
            predictor_cls=mock.MagicMock(),
            output_path="s3://output",
            test_input="s3://input/data.csv",
            job_name="job",
            split_type="Line",
            content_type="text/csv",
            custom_image_uri=GPU_IMAGE_URI,
            wait=False,
            model_kwargs={},
            transformer_kwargs=transformer_kwargs,
        )

    assert model_cls.return_value.transformer.call_args.kwargs["transform_ami_version"] == expected
