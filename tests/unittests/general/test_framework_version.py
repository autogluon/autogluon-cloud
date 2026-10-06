from unittest import mock

import pytest
from packaging.version import Version

from autogluon.cloud.backend.sagemaker_backend import SagemakerBackend
from autogluon.cloud.job.sagemaker_job import SageMakerFitJob
from autogluon.cloud.utils.constants import DEFAULT_FRAMEWORK_VERSION
from autogluon.cloud.utils.dlc_utils import (
    infer_framework_version_from_image_uri,
    resolve_framework_version,
    retrieve_available_framework_versions,
    retrieve_image_uri,
)


@pytest.mark.parametrize(
    ("framework_version", "expected"),
    [("latest", "1.6.3"), ("1.6", "1.6.3"), ("1.6.3", "1.6.3"), ("1.1", "1.1.1"), ("1.1.0", "1.1.0")],
)
def test_resolve_framework_version(framework_version, expected):
    assert resolve_framework_version(framework_version) == expected


def test_resolve_framework_version_falls_back_to_newest_patch(caplog):
    assert resolve_framework_version("1.6.0") == "1.6.3"
    assert "using 1.6.3 instead" in caplog.text


@pytest.mark.parametrize("framework_version", ["0.1", "99.0", "1", "foo"])
def test_resolve_framework_version_rejects_unsupported(framework_version):
    with pytest.raises(ValueError, match="1.6"):
        resolve_framework_version(framework_version)


@pytest.mark.parametrize("framework_type", ["training", "inference"])
def test_default_framework_version_is_newest_release(framework_type):
    newest = max(retrieve_available_framework_versions(framework_type), key=Version)
    assert Version(newest).release[:2] == Version(DEFAULT_FRAMEWORK_VERSION).release
    assert resolve_framework_version(DEFAULT_FRAMEWORK_VERSION, framework_type) == newest


@pytest.mark.parametrize("framework_version", ["1.5.0", "1.6.3"])
def test_infer_framework_version_from_official_image_uri(framework_version):
    image_uri = retrieve_image_uri(framework_version, "us-east-1", "training", "ml.m5.xlarge")
    assert infer_framework_version_from_image_uri(image_uri) == framework_version


@pytest.mark.parametrize(
    "image_uri",
    [
        "369469875935.dkr.ecr.us-east-1.amazonaws.com/autogluon-nightly-training:cpu-latest",
        "123456789012.dkr.ecr.us-east-1.amazonaws.com/my-image:1.6.3-cpu-amzn2023",
    ],
)
def test_infer_framework_version_ignores_custom_image_uri(image_uri):
    assert infer_framework_version_from_image_uri(image_uri) is None


def test_attach_recovers_framework_version():
    image_uri = retrieve_image_uri("1.5.0", "us-east-1", "training", "ml.m5.xlarge")
    session = mock.MagicMock()
    session.sagemaker_client.describe_training_job.return_value = {
        "AlgorithmSpecification": {"TrainingImage": image_uri}
    }
    with mock.patch.object(SageMakerFitJob, "_wait_until_completed"):
        job = SageMakerFitJob.attach("job", session=session)
    assert job.framework_version == "1.5.0"


@pytest.mark.parametrize(
    ("framework_version", "uses_fit_output", "expected"),
    [
        (None, True, "1.5.0"),
        (None, False, DEFAULT_FRAMEWORK_VERSION),
        ("1.6", True, "1.6"),
    ],
)
def test_inference_defaults_to_fit_framework_version(framework_version, uses_fit_output, expected):
    backend = SagemakerBackend.__new__(SagemakerBackend)
    backend._fit_job = mock.MagicMock(framework_version="1.5.0")
    assert backend._default_inference_framework_version(framework_version, uses_fit_output) == expected
