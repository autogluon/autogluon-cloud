import os
import tempfile

import pandas as pd
import pytest

from autogluon.cloud.model import FoundationModel


def _load_retail_sales():
    """Load the public retail-sales fixture."""
    target = "Sales"
    id_column = "id"
    timestamp_column = "timestamp"
    prediction_length = 13
    train_df = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/retail_sales/train.parquet")
    train_df[timestamp_column] = pd.to_datetime(train_df[timestamp_column])
    return train_df, target, id_column, timestamp_column, prediction_length


def _load_retail_sales_with_covariates():
    """Load the public retail-sales fixture with known covariates."""
    train_df, target, id_column, timestamp_column, prediction_length = _load_retail_sales()
    test_df = pd.read_parquet("https://autogluon.s3.amazonaws.com/datasets/timeseries/retail_sales/test.parquet")
    test_df[timestamp_column] = pd.to_datetime(test_df[timestamp_column])
    known_covariates_df = test_df.drop(columns=target)
    return train_df, known_covariates_df, target, id_column, timestamp_column, prediction_length


@pytest.mark.parametrize(
    "model_id, with_covariates",
    [
        ("chronos-bolt-tiny", False),
        ("chronos-2", False),
        ("chronos-2", True),
    ],
)
def test_foundation_model_predict(test_helper, framework_version, model_id, with_covariates):
    """Test batch prediction via the fit_predict training job pattern."""
    timestamp = test_helper.get_utc_timestamp_now()
    training_custom_image_uri = test_helper.get_custom_image_uri(framework_version, type="training", gpu=False)

    if with_covariates:
        train_data, known_covariates, target, id_column, timestamp_column, prediction_length = (
            _load_retail_sales_with_covariates()
        )
    else:
        train_data, target, id_column, timestamp_column, prediction_length = _load_retail_sales()
        known_covariates = None

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        expected_item_ids = sorted(train_data[id_column].unique())

        model = FoundationModel(
            model_id,
            cloud_output_path=(f"s3://autogluon-cloud-ci/test-fm-predict-{model_id}/{framework_version}/{timestamp}"),
        )

        predictions = model.predict(
            data=train_data,
            target=target,
            id_column=id_column,
            timestamp_column=timestamp_column,
            prediction_length=prediction_length,
            known_covariates=known_covariates,
            framework_version=framework_version,
            custom_image_uri=training_custom_image_uri,
        )

        assert isinstance(predictions, pd.DataFrame)
        assert {"item_id", "timestamp", "mean"} <= set(predictions.columns)
        assert sorted(predictions["item_id"].astype(str).unique()) == sorted(map(str, expected_item_ids))
        counts = predictions.groupby("item_id").size()
        assert (counts == prediction_length).all()
        assert len(predictions) == len(expected_item_ids) * prediction_length


def test_foundation_model_deploy(test_helper, framework_version):
    """Test deploy to a real-time endpoint and invoke it."""
    timestamp = test_helper.get_utc_timestamp_now()
    train_data, target, id_column, timestamp_column, prediction_length = _load_retail_sales()
    inference_custom_image_uri = test_helper.get_custom_image_uri(framework_version, type="inference", gpu=False)

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)

        model = FoundationModel(
            "chronos-bolt-tiny",
            cloud_output_path=f"s3://autogluon-cloud-ci/test-fm-deploy/{framework_version}/{timestamp}",
        )

        endpoint = model.deploy(
            framework_version=framework_version,
            custom_image_uri=inference_custom_image_uri,
        )

        try:
            pred = endpoint.predict(train_data)
            assert isinstance(pred, pd.DataFrame)
        finally:
            endpoint.delete_endpoint()
