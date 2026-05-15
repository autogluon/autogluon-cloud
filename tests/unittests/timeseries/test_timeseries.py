import os
import tempfile

import pandas as pd
import pytest

from autogluon.cloud import TimeSeriesCloudPredictor


def test_timeseries(test_helper, framework_version):
    train_data = "timeseries_train.csv"
    static_features = "timeseries_static_features.csv"
    timestamp = test_helper.get_utc_timestamp_now()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        test_helper.prepare_data(train_data, static_features)
        time_limit = 60

        predictor_init_args = dict(target="target", prediction_length=3)
        predictor_fit_args = dict(
            train_data=train_data,
            presets="medium_quality",
            time_limit=time_limit,
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
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            train_data,
            fit_kwargs=dict(
                static_features=static_features,
                framework_version=framework_version,
                custom_image_uri=training_custom_image_uri,
            ),
            deploy_kwargs=dict(framework_version=framework_version, custom_image_uri=inference_custom_image_uri),
            predict_kwargs=dict(
                static_features=static_features,
                framework_version=framework_version,
                custom_image_uri=inference_custom_image_uri,
            ),
            predict_real_time_kwargs=dict(static_features=static_features),
        )


def _load_retail_sales_with_covariates():
    """Fetch the public retail-sales fixture and split it into train + future-covariates frames."""
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
    predictor_init_args = dict(
        target=target,
        prediction_length=prediction_length,
        known_covariates_names=known_covariates_names,
    )
    return train_df, known_covariates_df, predictor_init_args, id_column, timestamp_column, prediction_length


@pytest.mark.parametrize(
    "model_name, hyperparameters, with_covariates",
    [
        ("chronos", {"Chronos": {"model_path": "tiny"}}, False),
        ("chronos_bolt", {"Chronos": {"model_path": "bolt_small"}}, False),
        ("chronos2", {"Chronos2": {"model_path": "autogluon/chronos-2-small"}}, False),
        ("chronos2_with_covs", {"Chronos2": {"model_path": "autogluon/chronos-2-small"}}, True),
    ],
)
def test_timeseries_fit_predict_chronos(test_helper, framework_version, model_name, hyperparameters, with_covariates):
    timestamp = test_helper.get_utc_timestamp_now()
    train_data, known_covariates, predictor_init_args, id_column, timestamp_column, prediction_length = (
        _load_retail_sales_with_covariates()
    )
    if not with_covariates:
        known_covariates = None
        predictor_init_args.pop("known_covariates_names")

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        expected_item_ids = sorted(train_data[id_column].unique())

        training_custom_image_uri = test_helper.get_custom_image_uri(framework_version, type="training", gpu=False)

        cloud_predictor = TimeSeriesCloudPredictor(
            cloud_output_path=(
                f"s3://autogluon-cloud-ci/test-timeseries-fit-predict-{model_name}/{framework_version}/{timestamp}"
            ),
            local_output_path=f"test_timeseries_fit_predict_{model_name}_cloud_predictor",
        )

        predictions = cloud_predictor.fit_predict(
            train_data=train_data,
            known_covariates=known_covariates,
            predictor_init_args=predictor_init_args,
            predictor_fit_args=dict(hyperparameters=hyperparameters),
            id_column=id_column,
            timestamp_column=timestamp_column,
            framework_version=framework_version,
            custom_image_uri=training_custom_image_uri,
        )

        assert isinstance(predictions, pd.DataFrame), (
            f"Expected predictions to be a DataFrame, got {type(predictions).__name__}"
        )
        # fit_predict returns predictions with AutoGluon's normalized "item_id" / "timestamp" column names,
        # matching the contract of real-time and batch transform endpoints.
        assert {"item_id", "timestamp", "mean"} <= set(predictions.columns), (
            f"predictions is missing required columns; got {sorted(predictions.columns)}"
        )
        assert sorted(predictions["item_id"].astype(str).unique()) == sorted(map(str, expected_item_ids)), (
            f"predictions ids do not match train_data; "
            f"expected {expected_item_ids}, got {sorted(predictions['item_id'].unique())}"
        )
        counts = predictions.groupby("item_id").size()
        assert (counts == prediction_length).all(), (
            f"Expected {prediction_length} rows per item, got {counts.to_dict()}"
        )
        assert len(predictions) == len(expected_item_ids) * prediction_length, (
            f"Expected {len(expected_item_ids) * prediction_length} rows total, got {len(predictions)}"
        )

        info = cloud_predictor.info()
        assert info["local_output_path"] is not None, "info()['local_output_path'] is unexpectedly None"
        assert info["cloud_output_path"] is not None, "info()['cloud_output_path'] is unexpectedly None"
        assert info["fit_job"]["name"] is not None, "info()['fit_job']['name'] is unexpectedly None"
        assert info["fit_job"]["status"] == "Completed", (
            f"Expected fit_job status 'Completed', got {info['fit_job']['status']!r}"
        )
