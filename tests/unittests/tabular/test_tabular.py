import os
import tempfile

import pandas as pd


def test_tabular_foundation_model_predict(test_helper, framework_version):
    import boto3

    from autogluon.cloud.model import TabularFoundationModel

    train_data = "tabular_train.csv"
    test_data = "tabular_test.csv"
    timestamp = test_helper.get_utc_timestamp_now()

    bucket = "autogluon-cloud-ci"
    predictions_key = f"test-tabular-fm-predict/{framework_version}/{timestamp}/custom_predictions.csv"
    predictions_path = f"s3://{bucket}/{predictions_key}"

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        test_helper.prepare_data(train_data, test_data)
        n_test_rows = len(pd.read_csv(test_data))

        training_custom_image_uri = test_helper.get_custom_image_uri(framework_version, type="training", gpu=False)

        model = TabularFoundationModel(
            "mitra-classifier",
            cloud_output_path=f"s3://{bucket}/test-tabular-fm-predict/{framework_version}/{timestamp}",
        )

        pred, pred_proba = model.predict_proba(
            train_data=train_data,
            test_data=test_data,
            label="class",
            include_predict=True,
            framework_version=framework_version,
            custom_image_uri=training_custom_image_uri,
            predictions_path=predictions_path,
        )

        assert isinstance(pred, pd.Series)
        assert len(pred) == n_test_rows
        assert isinstance(pred_proba, pd.DataFrame)
        assert len(pred_proba) == n_test_rows

        head = boto3.client("s3").head_object(Bucket=bucket, Key=predictions_key)
        assert head["ContentLength"] > 0, "predictions file on S3 should not be empty"
