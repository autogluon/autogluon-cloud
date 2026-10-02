from unittest import mock

import pandas as pd
import pytest

from autogluon.cloud import SageMakerConfig
from autogluon.cloud.backend.tabular_sagemaker_backend import TabularSagemakerBackend
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


SB = "autogluon.cloud.backend.sagemaker_backend"


@pytest.mark.parametrize(
    ("backend_overrides", "expected"),
    [
        (None, "al2-ami-sagemaker-batch-gpu-535"),
        ({"create_transform_job": {"transform_resources": {"transform_ami_version": "custom-ami"}}}, "custom-ami"),
    ],
)
def test_batch_transform_job_sets_inferred_ami_without_overriding_user_value(backend_overrides, expected):
    with (
        mock.patch(f"{SB}.setup_sagemaker_session", return_value=mock.MagicMock(boto_region_name="us-east-1")),
        mock.patch(f"{SB}.resolve_execution_role", return_value="arn:aws:iam::000000000000:role/test"),
        mock.patch(f"{SB}.SageMakerBatchTransformationJob") as job_cls,
        mock.patch.object(TabularSagemakerBackend, "_upload_predictor", side_effect=lambda path, _: path),
        mock.patch.object(TabularSagemakerBackend, "_upload_batch_predict_data", return_value="s3://input/data.csv"),
        mock.patch.object(TabularSagemakerBackend, "_prepare_model_data", return_value="s3://bucket/model.tar.gz"),
        mock.patch.object(TabularSagemakerBackend, "_create_model", return_value="job"),
    ):
        backend = TabularSagemakerBackend(
            local_output_path="/tmp/test",
            cloud_output_path="s3://bucket/run",
            predictor_type="tabular",
            config=SageMakerConfig(output_kms_key="output-key"),
        )
        backend._fit_job = mock.MagicMock()
        backend._predict(
            test_data=pd.DataFrame({"x": [1]}),
            predictor_path="s3://bucket/model.tar.gz",
            job_name="job",
            instance_type="ml.g4dn.xlarge",
            image_uri=GPU_IMAGE_URI,
            wait=False,
            download=False,
            persist=False,
            backend_overrides=backend_overrides,
        )

    request = job_cls.return_value.run.call_args.kwargs["transform_job_request"]
    assert request["transform_resources"]["transform_ami_version"] == expected
    assert request["transform_resources"]["instance_type"] == "ml.g4dn.xlarge"
    assert request["transform_output"]["kms_key_id"] == "output-key"
    assert "volume_kms_key_id" not in request["transform_resources"]
