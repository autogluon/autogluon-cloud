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


def test_tabular_foundation_model_deploy(test_helper, framework_version):
    """Test TabularFoundationModel deploy to a real-time CPU endpoint and predict."""
    import boto3

    from autogluon.cloud.model import TabularFoundationModel

    timestamp = test_helper.get_utc_timestamp_now()
    train_data = pd.DataFrame(
        {
            "feature_a": list(range(40)),
            "feature_b": [value % 3 for value in range(40)],
            "class": ["negative"] * 20 + ["positive"] * 20,
        }
    )
    test_data = pd.DataFrame(
        {
            "feature_a": [5, 35],
            "feature_b": [2, 2],
        }
    )

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        inference_custom_image_uri = test_helper.get_custom_image_uri(framework_version, type="inference", gpu=False)

        model = TabularFoundationModel(
            "mitra-classifier",
            cloud_output_path=(f"s3://autogluon-cloud-ci/test-tabular-fm-deploy/{framework_version}/{timestamp}"),
        )
        endpoint = model.deploy(custom_image_uri=inference_custom_image_uri)
        endpoint_arn = boto3.client("sagemaker").describe_endpoint(EndpointName=endpoint.endpoint_name)["EndpointArn"]
        test_helper.assert_ag_cloud_tags(endpoint_arn, module="tabular", model_id="mitra-classifier")

        try:
            pred, pred_proba = endpoint.predict_proba(
                data=test_data,
                train_data=train_data,
                label="class",
            )
            assert isinstance(pred, pd.Series)
            assert len(pred) == len(test_data)
            assert isinstance(pred_proba, pd.DataFrame)
            assert len(pred_proba) == len(test_data)
            assert set(pred_proba.columns) == {"negative", "positive"}
        finally:
            endpoint.delete_endpoint()
