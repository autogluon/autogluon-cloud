from unittest import mock

import pandas as pd
import pytest

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


@pytest.fixture
def transform_request(assert_valid_request):
    """Run ``TabularSagemakerBackend._predict(...)`` with AWS calls mocked and return the ``CreateTransformJob`` request."""

    def run(**predict_kwargs):
        with (
            mock.patch(f"{SB}.setup_sagemaker_session", return_value=mock.MagicMock(boto_region_name="us-east-1")),
            mock.patch(f"{SB}.resolve_execution_role", return_value="arn:aws:iam::000000000000:role/test"),
            mock.patch(f"{SB}.SageMakerBatchTransformationJob") as job_cls,
            mock.patch.object(TabularSagemakerBackend, "_upload_predictor", side_effect=lambda path, _: path),
            mock.patch.object(
                TabularSagemakerBackend, "_upload_batch_predict_data", return_value="s3://input/data.csv"
            ),
            mock.patch.object(TabularSagemakerBackend, "_prepare_model_data", return_value="s3://bucket/model.tar.gz"),
            mock.patch.object(TabularSagemakerBackend, "_create_model", return_value="job"),
        ):
            backend = TabularSagemakerBackend(
                local_output_path="/tmp/test",
                cloud_output_path="s3://bucket/run",
                predictor_type="tabular",
            )
            backend._fit_job = mock.MagicMock()
            backend._predict(
                test_data=pd.DataFrame({"x": [1]}),
                predictor_path="s3://bucket/model.tar.gz",
                job_name="job",
                wait=False,
                **predict_kwargs,
            )
        request = job_cls.return_value.run.call_args.kwargs["transform_job_request"]
        assert_valid_request("CreateTransformJob", request)
        return request

    return run


@pytest.mark.parametrize(
    ("backend_overrides", "expected"),
    [
        (None, "al2-ami-sagemaker-batch-gpu-535"),
        ({"create_transform_job": {"TransformResources": {"TransformAmiVersion": "custom-ami"}}}, "custom-ami"),
    ],
)
def test_batch_transform_job_sets_inferred_ami_without_overriding_user_value(
    transform_request, backend_overrides, expected
):
    request = transform_request(
        instance_type="ml.g4dn.xlarge",
        custom_image_uri=GPU_IMAGE_URI,
        backend_overrides=backend_overrides,
    )
    assert request["TransformResources"]["TransformAmiVersion"] == expected
    assert request["TransformResources"]["InstanceType"] == "ml.g4dn.xlarge"


def test_batch_transform_writes_results_to_predictions_path(transform_request):
    request = transform_request(predictions_path="s3://my-bucket/preds/")
    assert request["TransformOutput"]["S3OutputPath"] == "s3://my-bucket/preds"


def test_batch_transform_results_default_to_cloud_output_path(transform_request):
    output_path = transform_request()["TransformOutput"]["S3OutputPath"]
    assert output_path.startswith("s3://bucket/run/batch_transform/")
    assert output_path.endswith("/results")


def test_batch_transform_rejects_non_s3_predictions_path(transform_request):
    with pytest.raises(ValueError, match="S3 URL"):
        transform_request(predictions_path="/tmp/preds")
