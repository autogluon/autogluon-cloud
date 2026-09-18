import os
import tempfile

import boto3
import pandas as pd
import pytest

from autogluon.cloud import TabularCloudPredictor

_TRAIN_DATA = "tabular_train.csv"
_TUNE_DATA = "tabular_tune.csv"
_TEST_DATA = "tabular_test.csv"


def _shared_training_job_name() -> str:
    return os.environ["AG_CLOUD_SHARED_TRAINING_JOB_NAME"]


def _training_cloud_output_path(framework_version: str, job_name: str) -> str:
    return f"s3://autogluon-cloud-ci/test-tabular/{framework_version}/{job_name}"


def _followup_cloud_output_path(framework_version: str, job_name: str, test_name: str) -> str:
    return f"s3://autogluon-cloud-ci/test-tabular-followups/{framework_version}/{job_name}/{test_name}"


def _prepare_data(test_helper) -> None:
    test_helper.prepare_data(_TRAIN_DATA, _TUNE_DATA, _TEST_DATA)


def _attached_predictor(framework_version: str, test_name: str):
    job_name = _shared_training_job_name()
    predictor = TabularCloudPredictor(
        cloud_output_path=_followup_cloud_output_path(framework_version, job_name, test_name),
        local_output_path=f"test_tabular_{test_name}",
    )
    predictor.attach_job(job_name)
    assert predictor.get_fit_job_status() == "Completed"
    return predictor


def test_full_functionality_train(test_helper, framework_version):
    """Train the predictor once; follow-up tests attach to this completed SageMaker job."""
    job_name = _shared_training_job_name()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        _prepare_data(test_helper)

        predictor_init_args = dict(label="class", eval_metric="roc_auc")
        predictor_fit_args = dict(time_limit=60)
        with pytest.raises(ValueError, match="No `cloud_output_path` was provided"):
            TabularCloudPredictor().fit(
                train_data=_TRAIN_DATA,
                predictor_init_args=predictor_init_args,
                predictor_fit_args=predictor_fit_args,
            )

        predictor = TabularCloudPredictor(
            cloud_output_path=_training_cloud_output_path(framework_version, job_name),
            local_output_path="test_tabular_training",
        )
        predictor.fit(
            train_data=_TRAIN_DATA,
            tuning_data=_TUNE_DATA,
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            framework_version=framework_version,
            custom_image_uri=test_helper.get_custom_image_uri(framework_version, type="training", gpu=False),
            job_name=job_name,
        )
        info = predictor.info()
        assert info["fit_job"]["name"] == job_name
        assert info["fit_job"]["status"] == "Completed"
        job_arn = boto3.client("sagemaker").describe_training_job(TrainingJobName=job_name)["TrainingJobArn"]
        test_helper.assert_ag_cloud_tags(job_arn, module="tabular")


def test_full_functionality_endpoint_lifecycle(test_helper, framework_version):
    """Deploy the shared predictor and exercise detach, attach, save, and load."""
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        _prepare_data(test_helper)
        predictor = _attached_predictor(framework_version, "endpoint-lifecycle")

        predictor.deploy(
            framework_version=framework_version,
            custom_image_uri=test_helper.get_custom_image_uri(framework_version, type="inference", gpu=False),
        )
        endpoint_arn = boto3.client("sagemaker").describe_endpoint(EndpointName=predictor.endpoint_name)["EndpointArn"]
        test_helper.assert_ag_cloud_tags(endpoint_arn, module="tabular")
        test_helper.test_endpoint(predictor, _TEST_DATA, inference_kwargs=dict(model="LightGBM"))

        detached_endpoint = predictor.detach_endpoint()
        predictor.attach_endpoint(detached_endpoint)
        test_helper.test_endpoint(predictor, _TEST_DATA)

        predictor.save()
        predictor = TabularCloudPredictor.load(predictor.local_output_path)
        test_helper.test_endpoint(predictor, _TEST_DATA)
        predictor.cleanup_deployment()


def test_full_functionality_batch_predict(test_helper, framework_version):
    """Run batch prediction from a predictor attached to the shared training job."""
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        _prepare_data(test_helper)
        predictor = _attached_predictor(framework_version, "batch-predict")

        pred, pred_proba = predictor.predict_proba(
            _TEST_DATA,
            framework_version=framework_version,
            custom_image_uri=test_helper.get_custom_image_uri(framework_version, type="inference", gpu=False),
        )
        assert isinstance(pred, pd.Series)
        assert isinstance(pred_proba, pd.DataFrame)
        assert predictor.info()["recent_batch_inference_job"]["status"] == "Completed"


def test_full_functionality_deploy_trained_artifact(test_helper, framework_version):
    """Deploy the shared model artifact from a fresh CloudPredictor."""
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        _prepare_data(test_helper)
        job_name = _shared_training_job_name()
        artifact_path = boto3.client("sagemaker").describe_training_job(TrainingJobName=job_name)["ModelArtifacts"][
            "S3ModelArtifacts"
        ]
        predictor = TabularCloudPredictor(
            cloud_output_path=_followup_cloud_output_path(framework_version, job_name, "deploy-trained-artifact"),
            local_output_path="test_tabular_deploy_trained_artifact",
        )

        predictor.deploy(
            predictor_path=artifact_path,
            framework_version=framework_version,
            custom_image_uri=test_helper.get_custom_image_uri(framework_version, type="inference", gpu=False),
        )
        test_helper.test_endpoint(predictor, _TEST_DATA)
        predictor.cleanup_deployment()


def test_full_functionality_predict_trained_artifact(test_helper, framework_version):
    """Run batch prediction from the shared model artifact with a fresh CloudPredictor."""
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        _prepare_data(test_helper)
        job_name = _shared_training_job_name()
        artifact_path = boto3.client("sagemaker").describe_training_job(TrainingJobName=job_name)["ModelArtifacts"][
            "S3ModelArtifacts"
        ]
        predictor = TabularCloudPredictor(
            cloud_output_path=_followup_cloud_output_path(framework_version, job_name, "predict-trained-artifact"),
            local_output_path="test_tabular_predict_trained_artifact",
        )

        pred, pred_proba = predictor.predict_proba(
            _TEST_DATA,
            predictor_path=artifact_path,
            framework_version=framework_version,
            custom_image_uri=test_helper.get_custom_image_uri(framework_version, type="inference", gpu=False),
        )
        assert isinstance(pred, pd.Series)
        assert isinstance(pred_proba, pd.DataFrame)
        assert predictor.info()["recent_batch_inference_job"]["status"] == "Completed"
