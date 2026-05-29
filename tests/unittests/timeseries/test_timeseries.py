import os
import tempfile

import pandas as pd
import pytest

from autogluon.cloud import TimeSeriesCloudPredictor
from autogluon.cloud.model import FoundationModel


def _assert_timeseries_predictions(predictions: pd.DataFrame, expected_item_ids: list, prediction_length: int) -> None:
    """Validate shape and content of time series prediction DataFrame."""
    assert isinstance(predictions, pd.DataFrame)
    assert {"item_id", "timestamp", "mean"} <= set(predictions.columns)
    assert sorted(predictions["item_id"].astype(str).unique()) == sorted(map(str, expected_item_ids))
    counts = predictions.groupby("item_id").size()
    assert (counts == prediction_length).all()
    assert len(predictions) == len(expected_item_ids) * prediction_length


@pytest.fixture(scope="module")
def retail_sales_dataset():
    """Public retail-sales dataset with train data, known covariates, and synthetic static features."""
    target = "Sales"
    id_column = "id"
    timestamp_column = "timestamp"
    prediction_length = 13
    train_df = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/retail_sales/train.parquet")
    train_df[timestamp_column] = pd.to_datetime(train_df[timestamp_column])
    test_df = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/retail_sales/test.parquet")
    test_df[timestamp_column] = pd.to_datetime(test_df[timestamp_column])
    known_covariates_df = test_df.drop(columns=target)
    known_covariates_names = [c for c in known_covariates_df.columns if c not in (id_column, timestamp_column)]
    unique_ids = train_df[id_column].unique()
    static_features_df = pd.DataFrame(
        {id_column: unique_ids, "category": [f"cat_{i % 3}" for i in range(len(unique_ids))]}
    )
    return {
        "train_data": train_df,
        "known_covariates": known_covariates_df,
        "known_covariates_names": known_covariates_names,
        "static_features": static_features_df,
        "target": target,
        "id_column": id_column,
        "timestamp_column": timestamp_column,
        "prediction_length": prediction_length,
    }


def test_timeseries(test_helper, framework_version, retail_sales_dataset):
    ds = retail_sales_dataset
    timestamp = test_helper.get_utc_timestamp_now()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)

        predictor_init_args = dict(
            target=ds["target"],
            prediction_length=ds["prediction_length"],
            known_covariates_names=ds["known_covariates_names"],
        )
        predictor_fit_args = dict(
            presets="medium_quality",
            time_limit=60,
        )

        cloud_predictor = TimeSeriesCloudPredictor(
            cloud_output_path=f"s3://autogluon-cloud-ci/test-timeseries/{framework_version}/{timestamp}",
            local_output_path="test_timeseries_cloud_predictor",
        )
        cloud_predictor_no_train = TimeSeriesCloudPredictor(
            cloud_output_path=f"s3://autogluon-cloud-ci/test-timeseries-no-train/{framework_version}/{timestamp}",
            local_output_path="test_timeseries_cloud_predictor_no_train",
        )

        training_custom_image_uri = test_helper.get_custom_image_uri(framework_version, type="training", gpu=False)
        inference_custom_image_uri = test_helper.get_custom_image_uri(framework_version, type="inference", gpu=False)

        test_helper.test_functionality(
            cloud_predictor,
            ds["train_data"],
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            ds["train_data"],
            fit_kwargs=dict(
                id_column=ds["id_column"],
                timestamp_column=ds["timestamp_column"],
                static_features=ds["static_features"],
                framework_version=framework_version,
                custom_image_uri=training_custom_image_uri,
            ),
            deploy_kwargs=dict(framework_version=framework_version, custom_image_uri=inference_custom_image_uri),
            predict_kwargs=dict(
                static_features=ds["static_features"],
                known_covariates=ds["known_covariates"],
                framework_version=framework_version,
                custom_image_uri=inference_custom_image_uri,
            ),
            predict_real_time_kwargs=dict(
                static_features=ds["static_features"],
                known_covariates=ds["known_covariates"],
            ),
        )


@pytest.mark.parametrize(
    "model_name, hyperparameters, with_covariates",
    [
        ("chronos", {"Chronos": {"model_path": "tiny"}}, False),
        ("chronos_bolt", {"Chronos": {"model_path": "bolt_small"}}, False),
        ("chronos2", {"Chronos2": {"model_path": "autogluon/chronos-2-small"}}, False),
        ("chronos2_with_covs", {"Chronos2": {"model_path": "autogluon/chronos-2-small"}}, True),
    ],
)
def test_timeseries_fit_predict_chronos(
    test_helper, framework_version, retail_sales_dataset, model_name, hyperparameters, with_covariates
):
    import boto3

    ds = retail_sales_dataset
    timestamp = test_helper.get_utc_timestamp_now()
    known_covariates = ds["known_covariates"] if with_covariates else None
    predictor_init_args = dict(target=ds["target"], prediction_length=ds["prediction_length"])
    if with_covariates:
        predictor_init_args["known_covariates_names"] = ds["known_covariates_names"]

    bucket = "autogluon-cloud-ci"
    predictions_key = (
        f"test-timeseries-fit-predict-{model_name}/{framework_version}/{timestamp}/custom_predictions.parquet"
    )
    predictions_path = f"s3://{bucket}/{predictions_key}"

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        expected_item_ids = sorted(ds["train_data"][ds["id_column"]].unique())

        training_custom_image_uri = test_helper.get_custom_image_uri(framework_version, type="training", gpu=False)

        cloud_predictor = TimeSeriesCloudPredictor(
            cloud_output_path=(
                f"s3://{bucket}/test-timeseries-fit-predict-{model_name}/{framework_version}/{timestamp}"
            ),
            local_output_path=f"test_timeseries_fit_predict_{model_name}_cloud_predictor",
        )

        predictions = cloud_predictor.fit_predict(
            train_data=ds["train_data"],
            known_covariates=known_covariates,
            predictor_init_args=predictor_init_args,
            predictor_fit_args=dict(hyperparameters=hyperparameters),
            id_column=ds["id_column"],
            timestamp_column=ds["timestamp_column"],
            framework_version=framework_version,
            custom_image_uri=training_custom_image_uri,
            predictions_path=predictions_path,
        )

        _assert_timeseries_predictions(predictions, expected_item_ids, ds["prediction_length"])

        head = boto3.client("s3").head_object(Bucket=bucket, Key=predictions_key)
        assert head["ContentLength"] > 0, "predictions file on S3 should not be empty"

        info = cloud_predictor.info()
        assert info["local_output_path"] is not None
        assert info["cloud_output_path"] is not None
        assert info["fit_job"]["name"] is not None
        assert info["fit_job"]["status"] == "Completed"


def test_foundation_model_predict(test_helper, framework_version, retail_sales_dataset):
    """Test FoundationModel batch prediction via the fit_predict training job pattern."""
    ds = retail_sales_dataset
    timestamp = test_helper.get_utc_timestamp_now()

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        expected_item_ids = sorted(ds["train_data"][ds["id_column"]].unique())

        model = FoundationModel(
            "chronos-2",
            cloud_output_path=f"s3://autogluon-cloud-ci/test-fm-predict/{framework_version}/{timestamp}",
        )

        predictions = model.predict(
            data=ds["train_data"],
            target=ds["target"],
            id_column=ds["id_column"],
            timestamp_column=ds["timestamp_column"],
            prediction_length=ds["prediction_length"],
            known_covariates=ds["known_covariates"],
            instance_type="ml.m5.2xlarge",
        )

        _assert_timeseries_predictions(predictions, expected_item_ids, ds["prediction_length"])


def test_foundation_model_deploy(test_helper, framework_version, retail_sales_dataset):
    """Test FoundationModel deploy to a real-time endpoint and predict."""
    ds = retail_sales_dataset
    timestamp = test_helper.get_utc_timestamp_now()

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)

        inference_custom_image_uri = test_helper.get_custom_image_uri(framework_version, type="inference", gpu=True)

        model = FoundationModel(
            "chronos-bolt-tiny",
            cloud_output_path=f"s3://autogluon-cloud-ci/test-fm-deploy/{framework_version}/{timestamp}",
        )

        endpoint = model.deploy(
            custom_image_uri=inference_custom_image_uri,
        )

        try:
            expected_item_ids = sorted(ds["train_data"][ds["id_column"]].unique())
            predictions = endpoint.predict(
                data=ds["train_data"],
                target=ds["target"],
                id_column=ds["id_column"],
                timestamp_column=ds["timestamp_column"],
                prediction_length=ds["prediction_length"],
            )
            _assert_timeseries_predictions(predictions, expected_item_ids, ds["prediction_length"])
        finally:
            endpoint.delete_endpoint()
