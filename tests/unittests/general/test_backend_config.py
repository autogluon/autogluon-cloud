"""Shared backend configuration and job-bound foundation-model results, without AWS calls."""

import pickle
from pathlib import Path
from unittest import mock

import pandas as pd
import pytest

from autogluon.cloud import FoundationModel, SageMakerConfig, TabularCloudPredictor, TimeSeriesCloudPredictor
from autogluon.cloud.backend.backend_factory import BackendFactory
from autogluon.cloud.backend.sagemaker_backend import SagemakerBackend
from autogluon.cloud.backend.tabular_sagemaker_backend import TabularSagemakerBackend
from autogluon.cloud.backend.timeseries_sagemaker_backend import TimeSeriesSagemakerBackend
from autogluon.cloud.job.sagemaker_job import SageMakerFitJob

SB = "autogluon.cloud.backend.sagemaker_backend"


@pytest.fixture(autouse=True)
def stub_aws(monkeypatch):
    session_factory = mock.Mock(
        side_effect=lambda *, region=None: mock.MagicMock(boto_region_name=region or "us-east-1")
    )
    monkeypatch.setattr(f"{SB}.setup_sagemaker_session", session_factory)
    monkeypatch.setattr(
        f"{SB}.resolve_execution_role",
        lambda role, backend_name, session=None: role or "arn:aws:iam::0:role/default",
    )
    monkeypatch.setattr("autogluon.cloud.utils.aws_utils._s3_prefix_has_objects", lambda *_: False)
    return session_factory


@pytest.mark.parametrize(
    "predictor_cls,model_id",
    [(TabularCloudPredictor, "mitra-classifier"), (TimeSeriesCloudPredictor, "chronos-2")],
)
def test_shared_config_creates_independent_predictor_and_model_backends(tmp_path, predictor_cls, model_id):
    config = SageMakerConfig(
        region="eu-west-1",
        role_arn="arn:aws:iam::0:role/custom",
        vpc_config={"subnets": ["subnet-1"], "security_group_ids": ["sg-1"]},
        output_kms_key="output-key",
        tags={"team": "forecasting"},
    )
    predictor = predictor_cls(backend=config, cloud_output_path="s3://b/predictor", local_output_path=str(tmp_path))
    model = FoundationModel(model_id, backend=config, cloud_output_path="s3://b/model")

    assert predictor.backend.config == model._backend.config == config
    assert predictor.backend is not model._backend
    assert predictor.backend._fit_job is not model._backend._fit_job
    assert predictor.backend.sagemaker_session is not model._backend.sagemaker_session

    predictor.backend.config.tags["team"] = "other"
    predictor.backend.config.vpc_config["subnets"].append("subnet-2")
    predictor.backend.attach_endpoint("endpoint")
    assert config.tags == model._backend.config.tags == {"team": "forecasting"}
    assert (
        config.vpc_config
        == model._backend.config.vpc_config
        == {
            "subnets": ["subnet-1"],
            "security_group_ids": ["sg-1"],
        }
    )
    assert model._backend.endpoint_name is None


def test_default_backend_name_uses_the_same_config_resolver(tmp_path):
    predictor = TabularCloudPredictor(local_output_path=str(tmp_path), cloud_output_path="s3://b/run")
    model = FoundationModel("chronos-2", cloud_output_path="s3://b/model")
    assert predictor.backend.config == model._backend.config
    assert predictor.backend.config.role_arn == "arn:aws:iam::0:role/default"
    assert predictor.backend.config.region == "us-east-1"


@pytest.mark.parametrize("backend", ["unknown", "ray", "ray_aws"])
def test_invalid_backend_rejected_by_both_entry_points(tmp_path, backend):
    with pytest.raises(ValueError):
        TabularCloudPredictor(backend=backend, local_output_path=str(tmp_path))
    with pytest.raises(ValueError):
        FoundationModel("chronos-2", backend=backend)


def test_backend_resolver_rejects_untyped_dict():
    with pytest.raises(TypeError, match="SageMakerConfig"):
        BackendFactory.resolve_config({"region": "us-east-1"})


def test_predictor_reload_keeps_the_original_region(tmp_path, stub_aws):
    predictor = TabularCloudPredictor(
        backend=SageMakerConfig(region="eu-west-1"),
        cloud_output_path="s3://b/run",
        local_output_path=str(tmp_path),
    )
    restored = pickle.loads(pickle.dumps(predictor))
    assert restored.backend.config == predictor.backend.config
    stub_aws.assert_called_with(region="eu-west-1")


def test_real_training_submissions_keep_job_objects_and_inputs_separate(tmp_path, monkeypatch):
    """Later predictions must not overwrite the first job's handle, config or data channels."""
    backend = TabularSagemakerBackend(
        local_output_path=str(tmp_path), cloud_output_path="s3://b/run", predictor_type="tabular"
    )
    uploads = {}

    def upload(path, bucket, key_prefix):
        uri = f"s3://{bucket}/{key_prefix}/{Path(path).name}"
        if Path(path).is_file():
            uploads[uri] = Path(path).read_bytes()
        return uri

    def run(job, training_job_request, framework_version, wait):
        job._job_name = training_job_request["training_job_name"]
        job.request = training_job_request

    backend.sagemaker_session.upload_data.side_effect = upload
    monkeypatch.setattr(SageMakerFitJob, "run", run)
    monkeypatch.setattr(SageMakerFitJob, "_get_job_status", lambda self: "Completed")
    monkeypatch.setattr(SB + ".upload_training_code", lambda **kwargs: "s3://b/code")
    monkeypatch.setattr(backend, "_load_fit_predict_results", lambda job: pd.DataFrame({"job": [job.job_name]}))
    futures, requests = [], []
    for name, value in [("first", 1), ("second", 2)]:
        backend.fit(
            predictor_init_args={"label": "y"},
            predictor_fit_args={},
            data_channels={"train_data": pd.DataFrame({"x": [value], "y": [0]})},
            job_name=name,
            image_uri="example.com/autogluon:train",
            wait=False,
            extra_ag_args={"predict_after_fit": True, "save_predictor": False},
        )
        futures.append(backend.get_prediction_future())
        requests.append(backend._fit_job.request)
    assert [future.job_name for future in futures] == ["first", "second"]
    assert [future.result()["job"].iloc[0] for future in futures] == ["first", "second"]
    channels = [
        {
            channel["channel_name"]: channel["data_source"]["s3_data_source"]["s3_uri"]
            for channel in request["input_data_config"]
        }
        for request in requests
    ]
    assert channels[0]["ag_args"] != channels[1]["ag_args"]
    assert channels[0]["train_data"] != channels[1]["train_data"]
    assert b"/first/predictions.csv" in uploads[channels[0]["ag_args"]]
    assert b"/second/predictions.csv" in uploads[channels[1]["ag_args"]]
    assert uploads[channels[0]["train_data"]] != uploads[channels[1]["train_data"]]


@pytest.mark.parametrize(
    "model_id,operation,include_predict",
    [
        ("chronos-2", "predict", None),
        ("mitra-classifier", "predict", None),
        ("mitra-classifier", "predict_proba", True),
        ("mitra-classifier", "predict_proba", False),
    ],
)
def test_async_model_results_stay_bound_to_the_submitted_job(monkeypatch, model_id, operation, include_predict):
    model = FoundationModel(model_id, cloud_output_path="s3://b/model")
    frames = [
        pd.DataFrame({"target": ["a"], "a_proba": [0.8], "b_proba": [0.2]}),
        pd.DataFrame({"target": ["b"], "a_proba": [0.1], "b_proba": [0.9]}),
    ]
    jobs = []

    def submit(self, **kwargs):
        assert kwargs["backend_overrides"] == {"create_training_job": {"retry_strategy": {}}}
        job = mock.Mock(job_name=f"job-{len(jobs)}", completed=True)
        job.frame = frames[len(jobs)]
        jobs.append(job)
        self._fit_job = job

    monkeypatch.setattr(TabularSagemakerBackend, "fit", submit)
    monkeypatch.setattr(TimeSeriesSagemakerBackend, "fit", submit)
    monkeypatch.setattr(SagemakerBackend, "_load_fit_predict_results", lambda self, job: job.frame)
    kwargs = {"wait": False, "backend_overrides": {"create_training_job": {"retry_strategy": {}}}}
    if model_id == "chronos-2":
        kwargs["data"] = pd.DataFrame({"target": [1.0]})
    else:
        kwargs.update(
            train_data=pd.DataFrame({"feature": [1], "target": ["a"]}),
            test_data=pd.DataFrame({"feature": [2]}),
            label="target",
        )
    if include_predict is not None:
        kwargs["include_predict"] = include_predict

    first = getattr(model, operation)(**kwargs)
    second = getattr(model, operation)(**kwargs)
    assert first.job_name == "job-0"
    assert second.job_name == "job-1"
    first_result, second_result = first.result(), second.result()
    if model_id == "chronos-2":
        pd.testing.assert_frame_equal(first_result, frames[0])
        pd.testing.assert_frame_equal(second_result, frames[1])
    elif operation == "predict":
        assert first_result.tolist() == ["a"]
        assert second_result.tolist() == ["b"]
    else:
        if include_predict:
            first_pred, first_result = first_result
            second_pred, second_result = second_result
            assert first_pred.tolist() == ["a"]
            assert second_pred.tolist() == ["b"]
        assert first_result["a"].tolist() == [0.8]
        assert second_result["a"].tolist() == [0.1]
