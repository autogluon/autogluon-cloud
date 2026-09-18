import os
import zipfile
from datetime import datetime, timezone

import boto3
import pandas as pd
import pytest


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
    def extract_images(image_zip_file):
        with zipfile.ZipFile(image_zip_file, "r") as zip_ref:
            zip_ref.extractall(".")

    @staticmethod
    def replace_image_abspath(data, image_column):
        data = pd.read_csv(data)
        data[image_column] = data[image_column].apply(os.path.abspath)
        return data

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

    @staticmethod
    def test_basic_functionality(
        cloud_predictor,
        train_data,
        predictor_init_args,
        predictor_fit_args,
        test_data,
        tuning_data=None,
        fit_kwargs=None,
        deploy_kwargs=None,
        predict_real_time_kwargs=None,
        predict_kwargs=None,
    ):
        if fit_kwargs is None:
            fit_kwargs = dict(instance_type="ml.m5.2xlarge")
        cloud_predictor.fit(
            train_data=train_data,
            tuning_data=tuning_data,
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            **fit_kwargs,
        )
        info = cloud_predictor.info()
        assert info["local_output_path"] is not None
        assert info["cloud_output_path"] is not None
        assert info["fit_job"]["name"] is not None
        assert info["fit_job"]["status"] == "Completed"

        if deploy_kwargs is None:
            deploy_kwargs = dict()
        if predict_real_time_kwargs is None:
            predict_real_time_kwargs = dict()
        cloud_predictor.deploy(**deploy_kwargs)
        CloudTestHelper.test_endpoint(cloud_predictor, test_data, **predict_real_time_kwargs)
        cloud_predictor.cleanup_deployment()

        info = cloud_predictor.info()
        assert info["local_output_path"] is not None
        assert info["cloud_output_path"] is not None
        assert info["fit_job"]["name"] is not None
        assert info["fit_job"]["status"] == "Completed"

        if predict_kwargs is None:
            predict_kwargs = dict()
        pred, pred_proba = cloud_predictor.predict_proba(test_data, **predict_kwargs)
        assert isinstance(pred, pd.Series) and isinstance(pred_proba, pd.DataFrame)
        info = cloud_predictor.info()
        assert info["recent_batch_inference_job"]["status"] == "Completed"


def pytest_addoption(parser):
    parser.addoption("--framework_version", action="store", default="source")


@pytest.fixture(scope="session")
def framework_version(pytestconfig):
    return pytestconfig.getoption("framework_version")


@pytest.fixture(scope="session")
def shared_training_job_name():
    return os.environ["AG_CLOUD_SHARED_TRAINING_JOB_NAME"]


@pytest.fixture
def test_helper():
    return CloudTestHelper
