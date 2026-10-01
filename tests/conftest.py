import os
from datetime import datetime, timezone

import boto3
import pandas as pd
import pytest

from autogluon.cloud.backend import sagemaker_backend

# Must match .github/workflow_scripts/cleanup_endpoints.py
CI_RUN_TAG = "autogluon-cloud-ci-run"


class CloudTestHelper:
    cpu_training_image = "369469875935.dkr.ecr.us-east-1.amazonaws.com/autogluon-nightly-training:cpu-latest"
    gpu_training_image = "369469875935.dkr.ecr.us-east-1.amazonaws.com/autogluon-nightly-training:gpu-latest"
    cpu_inference_image = "369469875935.dkr.ecr.us-east-1.amazonaws.com/autogluon-nightly-inference:cpu-latest"
    gpu_inference_image = "369469875935.dkr.ecr.us-east-1.amazonaws.com/autogluon-nightly-inference:gpu-latest"

    @staticmethod
    def get_custom_image_uri(framework_version="source", type="training", gpu=False):
        assert type in ["training", "inference"]
        if type == "training":
            if gpu:
                custom_image_uri = CloudTestHelper.gpu_training_image
            else:
                custom_image_uri = CloudTestHelper.cpu_training_image
        else:
            if gpu:
                custom_image_uri = CloudTestHelper.gpu_inference_image
            else:
                custom_image_uri = CloudTestHelper.cpu_inference_image
        if framework_version != "source":
            custom_image_uri = None

        return custom_image_uri

    @staticmethod
    def prepare_data(*args):
        # TODO: make this handle more general structured directory format
        """
        Download files specified by args from cloud CI s3 bucket

        args: str
            names of files to download
        """
        s3 = boto3.client("s3")
        for arg in args:
            s3.download_file("autogluon-cloud", arg, os.path.basename(arg))

    @staticmethod
    def get_utc_timestamp_now():
        return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")

    @staticmethod
    def shared_training_output_path(module: str, framework_version: str, job_name: str) -> str:
        return f"s3://autogluon-cloud-ci/test-{module}/{framework_version}/{job_name}"

    @staticmethod
    def shared_followup_output_path(module: str, framework_version: str, job_name: str, test_name: str) -> str:
        return f"s3://autogluon-cloud-ci/test-{module}-followups/{framework_version}/{job_name}/{test_name}"

    @staticmethod
    def shared_training_artifact_path(job_name: str) -> str:
        job = boto3.client("sagemaker").describe_training_job(TrainingJobName=job_name)
        return job["ModelArtifacts"]["S3ModelArtifacts"]

    @staticmethod
    def attach_shared_training_job(predictor_cls, module, framework_version, job_name, test_name):
        predictor = predictor_cls(
            cloud_output_path=CloudTestHelper.shared_followup_output_path(
                module, framework_version, job_name, test_name
            ),
            local_output_path=f"test_{module}_{test_name}",
        )
        predictor.attach_job(job_name)
        assert predictor.get_fit_job_status() == "Completed"
        return predictor

    @staticmethod
    def assert_ag_cloud_tags(arn: str, *, module: str, model_id: str = None):
        """Assert the resource at ``arn`` carries ``autogluon-cloud-module`` (and optionally ``autogluon-cloud-model-id``).

        Works on any tagged SageMaker resource (training/transform jobs, models, endpoints).
        """
        tags = {t["Key"]: t["Value"] for t in boto3.client("sagemaker").list_tags(ResourceArn=arn)["Tags"]}
        assert tags.get("autogluon-cloud-module") == module, f"missing/wrong module tag on {arn}: {tags}"
        if model_id is not None:
            assert tags.get("autogluon-cloud-model-id") == model_id, f"missing/wrong model-id tag on {arn}: {tags}"

    @staticmethod
    def test_endpoint(cloud_predictor, test_data, inference_kwargs=None, **predict_real_time_kwargs):
        if inference_kwargs is None:
            inference_kwargs = {}
        try:
            pred = cloud_predictor.predict_real_time(test_data, **inference_kwargs, **predict_real_time_kwargs)
            assert isinstance(pred, pd.Series)
            pred_proba = cloud_predictor.predict_proba_real_time(
                test_data, **inference_kwargs, **predict_real_time_kwargs
            )
            assert isinstance(pred_proba, pd.DataFrame)
        except Exception as e:
            cloud_predictor.cleanup_deployment()  # cleanup endpoint if test failed
            raise e

    @staticmethod
    def test_timeseries_endpoint(cloud_predictor, test_data, **predict_real_time_kwargs):
        try:
            pred = cloud_predictor.predict_real_time(test_data, **predict_real_time_kwargs)
            assert isinstance(pred, pd.DataFrame)
        except Exception as e:
            cloud_predictor.cleanup_deployment()  # cleanup endpoint if test failed
            raise e


def pytest_addoption(parser):
    parser.addoption("--framework_version", action="store", default="source")


@pytest.fixture(scope="session")
def framework_version(pytestconfig):
    return pytestconfig.getoption("framework_version")


@pytest.fixture(scope="session", autouse=True)
def tag_resources_with_ci_run():
    """Tag every SageMaker resource created in CI with the run id, so the cleanup job can find leaked endpoints."""
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not run_id:
        yield
        return
    build_tags = sagemaker_backend.build_tags

    def build_tags_with_ci_run(*args, **kwargs):
        return build_tags(*args, **kwargs) + [{"Key": CI_RUN_TAG, "Value": run_id}]

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(sagemaker_backend, "build_tags", build_tags_with_ci_run)
        yield


@pytest.fixture(scope="session")
def shared_training_job_name():
    job_name = os.environ.get("AG_CLOUD_SHARED_TRAINING_JOB_NAME")
    if not job_name:
        pytest.fail(
            "AG_CLOUD_SHARED_TRAINING_JOB_NAME is not set. Tabular and timeseries cloud tests run in two phases "
            "(train once, then follow-ups attach to that job); see .github/workflow_scripts/test_cloud.sh."
        )
    return job_name


@pytest.fixture
def test_helper():
    return CloudTestHelper
