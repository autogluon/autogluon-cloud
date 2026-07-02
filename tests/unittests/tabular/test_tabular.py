import os
import tempfile

import pandas as pd

from autogluon.cloud import TabularCloudPredictor
from autogluon.common.features.feature_metadata import FeatureMetadata


def test_tabular_tabular_text_image(test_helper, framework_version):
    train_data = "tabular_text_image_train.csv"
    test_data = "tabular_text_image_test.csv"
    images = "tabular_text_image_images.zip"
    image_column = "Images"
    timestamp = test_helper.get_utc_timestamp_now()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        test_helper.prepare_data(train_data, test_data, images)
        test_helper.extract_images(images)
        train_data = test_helper.replace_image_abspath(train_data, image_column)
        test_data = test_helper.replace_image_abspath(test_data, image_column)
        feature_metadata = FeatureMetadata.from_df(train_data)
        feature_metadata = feature_metadata.add_special_types({"Images": ["image_path"]})

        time_limit = 600

        predictor_init_args = dict(
            label="AdoptionSpeed",
        )
        text_model = "AG_TEXT_NN"
        image_model = "AG_IMAGE_NN"
        predictor_fit_args = dict(
            time_limit=time_limit,
            hyperparameters={
                "XGB": {},
                text_model: {"presets": "medium_quality_faster_train"},
                image_model: {},
            },
            feature_metadata=feature_metadata,
        )
        cloud_predictor = TabularCloudPredictor(
            cloud_output_path=f"s3://autogluon-cloud-ci/test-tabular-tabular-text-image/{framework_version}/{timestamp}",
            local_output_path="test_tabular_tabular_text_image_cloud_predictor",
        )
        training_custom_image_uri = test_helper.get_custom_image_uri(framework_version, type="training", gpu=True)
        inference_custom_image_uri = test_helper.get_custom_image_uri(framework_version, type="inference", gpu=False)
        test_helper.test_basic_functionality(
            cloud_predictor,
            train_data,
            predictor_init_args,
            predictor_fit_args,
            test_data,
            fit_kwargs=dict(
                instance_type="ml.g4dn.2xlarge",
                volume_size=100,
                image_column=image_column,
                framework_version=framework_version,
                custom_image_uri=training_custom_image_uri,
            ),
            deploy_kwargs=dict(framework_version=framework_version, custom_image_uri=inference_custom_image_uri),
            predict_real_time_kwargs=dict(
                test_data_image_column=image_column,
            ),
            predict_kwargs=dict(
                test_data_image_column=image_column,
                framework_version=framework_version,
                custom_image_uri=inference_custom_image_uri,
            ),
        )
        local_predictor = cloud_predictor.to_local_predictor(
            require_version_match=False, require_py_version_match=False
        )
        models = local_predictor.model_names()
        assert "ImagePredictor" in models


def test_tabular_fit_predict(test_helper, framework_version):
    """fit + in-job batch predict in a single SageMaker training job.

    Only the classification path is exercised end-to-end (it is the superset: it produces both the
    prediction and the proba frame). The regression path — where proba mirrors pred — is covered by the
    pure-unit tests in test_tabular_fit_predict.py, so it does not warrant a second SageMaker job.
    """
    import boto3

    train_data = "tabular_train.csv"
    test_data = "tabular_test.csv"
    timestamp = test_helper.get_utc_timestamp_now()

    bucket = "autogluon-cloud-ci"
    predictions_key = f"test-tabular-fit-predict/{framework_version}/{timestamp}/custom_predictions.csv"
    predictions_path = f"s3://{bucket}/{predictions_key}"

    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        test_helper.prepare_data(train_data, test_data)
        n_test_rows = len(pd.read_csv(test_data))

        training_custom_image_uri = test_helper.get_custom_image_uri(framework_version, type="training", gpu=False)

        cloud_predictor = TabularCloudPredictor(
            cloud_output_path=f"s3://{bucket}/test-tabular-fit-predict/{framework_version}/{timestamp}",
            local_output_path="test_tabular_fit_predict_cloud_predictor",
        )

        pred, pred_proba = cloud_predictor.fit_predict_proba(
            train_data=train_data,
            test_data=test_data,
            predictor_init_args=dict(label="class"),
            predictor_fit_args=dict(time_limit=60),
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

        info = cloud_predictor.info()
        assert info["fit_job"]["name"] is not None
        assert info["fit_job"]["status"] == "Completed"
