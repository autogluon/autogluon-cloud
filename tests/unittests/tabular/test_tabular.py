import tempfile

from autogluon.cloud import TabularCloudPredictor


def test_tabular_tabular_text_image(test_helper):
    train_data = "tabular_text_image_train.csv"
    test_data = "tabular_text_image_test.csv"
    images = "tabular_text_image_images.zip"
    image_column = "Images"
    timestamp = test_helper.get_utc_timestamp_now()
    with tempfile.TemporaryDirectory() as _:
        test_helper.prepare_data(train_data, test_data, images)
        test_helper.extract_images(images)
        train_data = test_helper.replace_image_abspath(train_data, image_column)
        test_data = test_helper.replace_image_abspath(test_data, image_column)
        time_limit = 600

        predictor_init_args = dict(
            label="AdoptionSpeed",
        )
        predictor_fit_args = dict(
            train_data=train_data,
            time_limit=time_limit,
            hyperparameters={
                "XGB": {},
                "AG_TEXT_NN": {"presets": "medium_quality_faster_train"},
                "AG_IMAGE_NN": {},
            },
        )
        cloud_predictor = TabularCloudPredictor(
            cloud_output_path=f"s3://autogluon-cloud-ci/test-tabular-tabular-text-image/{timestamp}",
            local_output_path="test_tabular_tabular_text_image_cloud_predictor",
        )
        test_helper.test_basic_functionality(
            cloud_predictor,
            predictor_init_args,
            predictor_fit_args,
            test_data,
            fit_kwargs=dict(
                instance_type="ml.g4dn.2xlarge",
                image_column=image_column,
                custom_image_uri=test_helper.gpu_training_image,
            ),
            deploy_kwargs=dict(custom_image_uri=test_helper.cpu_inference_image),
            predict_real_time_kwargs=dict(
                test_data_image_column="Images",
            ),
            predict_kwargs=dict(
                test_data_image_column="Images",
                custom_image_uri=test_helper.cpu_inference_image,
            ),
        )
