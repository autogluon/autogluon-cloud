import os
import tempfile

from autogluon.cloud import TabularCloudPredictor


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
        time_limit = 600

        predictor_init_args = dict(
            label="AdoptionSpeed",
        )
        text_model = "AG_TEXT_NN"
        image_model = "AG_IMAGE_NN"
        predictor_fit_args = dict(
            train_data=train_data,
            time_limit=time_limit,
            hyperparameters={
                "XGB": {},
                text_model: {"presets": "medium_quality_faster_train"},
                image_model: {},
            },
        )
        cloud_predictor = TabularCloudPredictor(
            cloud_output_path=f"s3://autogluon-cloud-ci/test-tabular-tabular-text-image/{framework_version}/{timestamp}",
            local_output_path="test_tabular_tabular_text_image_cloud_predictor",
        )
        training_custom_image_uri = test_helper.get_custom_image_uri(framework_version, type="training", gpu=True)
        inference_custom_image_uri = test_helper.get_custom_image_uri(framework_version, type="inference", gpu=False)
        test_helper.test_basic_functionality(
            cloud_predictor,
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
        models = local_predictor.get_model_names()
        assert "ImagePredictor" in models
