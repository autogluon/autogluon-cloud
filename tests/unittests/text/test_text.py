import tempfile

from autogluon.cloud import MultiModalCloudPredictor, TextCloudPredictor


def test_text(test_helper, framework_version="latest"):
    train_data = "text_train.csv"
    tune_data = "text_tune.csv"
    test_data = "text_test.csv"
    timestamp = test_helper.get_utc_timestamp_now()
    with tempfile.TemporaryDirectory() as _:
        test_helper.prepare_data(train_data, tune_data, test_data)
        time_limit = 60

        predictor_init_args = dict(label="label", eval_metric="acc")
        predictor_fit_args = dict(train_data=train_data, tuning_data=tune_data, time_limit=time_limit)
        cloud_predictor = TextCloudPredictor(
            cloud_output_path=f"s3://autogluon-cloud-ci/test-text/{timestamp}",
            local_output_path="test_text_cloud_predictor",
        )
        training_custom_image_uri, inference_custom_image_uri = test_helper.get_custom_image_uri(framework_version)
        test_helper.test_basic_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            test_data,
            fit_kwargs=dict(
                instance_type="ml.g4dn.2xlarge",
                framework_version=framework_version,
                custom_image_uri=training_custom_image_uri
            ),
            deploy_kwargs=dict(
                framework_version=framework_version,
                custom_image_uri=inference_custom_image_uri
            ),
            predict_kwargs=dict(
                framework_version=framework_version,
                custom_image_uri=inference_custom_image_uri
            ),
        )


def test_multimodal_text_only(test_helper, framework_version="latest"):
    train_data = "text_train.csv"
    tune_data = "text_tune.csv"
    test_data = "text_test.csv"
    timestamp = test_helper.get_utc_timestamp_now()
    with tempfile.TemporaryDirectory() as _:
        test_helper.prepare_data(train_data, tune_data, test_data)
        time_limit = 60

        predictor_init_args = dict(label="label", eval_metric="acc")
        predictor_fit_args = dict(train_data=train_data, tuning_data=tune_data, time_limit=time_limit)
        cloud_predictor = MultiModalCloudPredictor(
            cloud_output_path=f"s3://autogluon-cloud-ci/test-multimodal-text/{timestamp}",
            local_output_path="test_multimodal_text_cloud_predictor",
        )
        training_custom_image_uri, inference_custom_image_uri = test_helper.get_custom_image_uri(framework_version)
        test_helper.test_basic_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            test_data,
            fit_kwargs=dict(
                instance_type="ml.g4dn.2xlarge",
                framework_version=framework_version,
                custom_image_uri=training_custom_image_uri
            ),
            deploy_kwargs=dict(
                framework_version=framework_version,
                custom_image_uri=inference_custom_image_uri
            ),
            predict_kwargs=dict(
                framework_version=framework_version,
                custom_image_uri=inference_custom_image_uri
            ),
        )
