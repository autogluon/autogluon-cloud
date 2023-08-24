import os
import zipfile
from datetime import datetime, timezone

import boto3
import pandas as pd
import pytest

from autogluon.cloud import TimeSeriesCloudPredictor


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
    def extract_images(image_zip_file):
        with zipfile.ZipFile(image_zip_file, "r") as zip_ref:
            zip_ref.extractall(".")

    @staticmethod
    def replace_image_abspath(data, image_column):
        data = pd.read_csv(data)
        data[image_column] = data[image_column].apply(os.path.abspath)
        return data

    @staticmethod
    def test_endpoint(cloud_predictor, test_data, inference_kwargs=None, **predict_real_time_kwargs):
        if inference_kwargs is None:
            inference_kwargs = {}
        try:
            if isinstance(cloud_predictor, TimeSeriesCloudPredictor):
                pred = cloud_predictor.predict_real_time(test_data, **inference_kwargs, **predict_real_time_kwargs)
                assert isinstance(pred, pd.DataFrame)
            else:
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
    def test_basic_functionality(
        cloud_predictor,
        predictor_init_args,
        predictor_fit_args,
        test_data,
        fit_kwargs=None,
        deploy_kwargs=None,
        predict_real_time_kwargs=None,
        predict_kwargs=None,
    ):
        if fit_kwargs is None:
            fit_kwargs = dict(instance_type="ml.m5.2xlarge")
        cloud_predictor.fit(
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
        if isinstance(cloud_predictor, TimeSeriesCloudPredictor):
            pred = cloud_predictor.predict(test_data, **predict_kwargs)
            assert isinstance(pred, pd.DataFrame)
        else:
            pred, pred_proba = cloud_predictor.predict_proba(test_data, **predict_kwargs)
            assert isinstance(pred, pd.Series) and isinstance(pred_proba, pd.DataFrame)
        info = cloud_predictor.info()
        assert info["recent_batch_inference_job"]["status"] == "Completed"

    @staticmethod
    def test_functionality(
        cloud_predictor,
        predictor_init_args,
        predictor_fit_args,
        cloud_predictor_no_train,
        test_data,
        fit_kwargs=None,
        deploy_kwargs=None,
        predict_real_time_kwargs=None,
        inference_kwargs=None,
        predict_kwargs=None,
    ):
        if fit_kwargs is None:
            fit_kwargs = dict(instance_type="ml.m5.2xlarge")
        cloud_predictor.fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            **fit_kwargs,
        )
        info = cloud_predictor.info()
        job_name = info["fit_job"]["name"]
        assert info["local_output_path"] is not None
        assert info["cloud_output_path"] is not None
        assert job_name is not None
        assert info["fit_job"]["status"] == "Completed"

        cloud_predictor.attach_job(job_name)

        if deploy_kwargs is None:
            deploy_kwargs = dict()
        if predict_real_time_kwargs is None:
            predict_real_time_kwargs = dict()
        cloud_predictor.deploy(**deploy_kwargs)
        CloudTestHelper.test_endpoint(
            cloud_predictor, test_data, inference_kwargs=inference_kwargs, **predict_real_time_kwargs
        )
        detached_endpoint = cloud_predictor.detach_endpoint()
        cloud_predictor.attach_endpoint(detached_endpoint)
        CloudTestHelper.test_endpoint(cloud_predictor, test_data, **predict_real_time_kwargs)
        cloud_predictor.save()
        cloud_predictor = cloud_predictor.__class__.load(cloud_predictor.local_output_path)
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

        # Test deploy with already trained predictor
        trained_predictor_path = cloud_predictor.get_fit_job_output_path()
        cloud_predictor_no_train.deploy(predictor_path=trained_predictor_path, **deploy_kwargs)
        CloudTestHelper.test_endpoint(cloud_predictor_no_train, test_data, **predict_real_time_kwargs)
        cloud_predictor_no_train.cleanup_deployment()

        pred, pred_proba = cloud_predictor_no_train.predict_proba(
            test_data, predictor_path=trained_predictor_path, **predict_kwargs
        )
        assert isinstance(pred, pd.Series) and isinstance(pred_proba, pd.DataFrame)
        info = cloud_predictor_no_train.info()
        assert info["recent_batch_inference_job"]["status"] == "Completed"


def pytest_addoption(parser):
    parser.addoption("--framework_version", action="store", default="source")


@pytest.fixture(scope="session")
def framework_version(pytestconfig):
    return pytestconfig.getoption("framework_version")


@pytest.fixture
def test_helper():
    return CloudTestHelper
