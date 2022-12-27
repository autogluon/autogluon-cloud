import tempfile

from autogluon.cloud import TabularCloudPredictor


def test_full_functionality(test_helper, framework_version):
    """
    Use tabular as an example to test full functionality.
    Those functionalities shouldn't differ between modality
    """
    train_data = "tabular_train.csv"
    tune_data = "tabular_tune.csv"
    test_data = "tabular_test.csv"
    timestamp = test_helper.get_utc_timestamp_now()
    with tempfile.TemporaryDirectory() as _:
        test_helper.prepare_data(train_data, tune_data, test_data)
        time_limit = 60

        predictor_init_args = dict(label="class", eval_metric="roc_auc")
        predictor_fit_args = dict(
            train_data=train_data,
            tuning_data=tune_data,
            time_limit=time_limit,
        )
        cloud_predictor = TabularCloudPredictor(
            cloud_output_path=f"s3://autogluon-cloud-ci/test-tabular/{timestamp}",
            local_output_path="test_tabular_cloud_predictor",
        )
        cloud_predictor_no_train = TabularCloudPredictor(
            cloud_output_path=f"s3://autogluon-cloud-ci/test-tabular-no-train/{timestamp}",
            local_output_path="test_tabular_cloud_predictor_no_train",
        )
        training_custom_image_uri, inference_custom_image_uri = test_helper.get_custom_image_uri(framework_version)
        test_helper.test_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            cloud_predictor_no_train,
            test_data,
            fit_kwargs=dict(framework_version=framework_version, custom_image_uri=training_custom_image_uri),
            deploy_kwargs=dict(framework_version=framework_version, custom_image_uri=inference_custom_image_uri),
            predict_kwargs=dict(framework_version=framework_version, custom_image_uri=inference_custom_image_uri),
        )
