import os
import tempfile

from autogluon.cloud import MultiModalCloudPredictor


def test_multimodal_image_only(test_helper, framework_version):
    train_data = "image_train_relative.csv"
    train_image = "shopee-iet.zip"
    test_data = "test_images/BabyPants_1035.jpg"
    image_column = "image"
    timestamp = test_helper.get_utc_timestamp_now()
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        test_helper.prepare_data(train_data, train_image, test_data)
        test_helper.extract_images(train_image)
        train_data = test_helper.replace_image_abspath(train_data, image_column)
        test_data = "BabyPants_1035.jpg"
        time_limit = 60

        predictor_init_args = dict(label="label", eval_metric="acc")
        predictor_fit_args = dict(train_data=train_data, time_limit=time_limit)
        cloud_predictor = MultiModalCloudPredictor(
            cloud_output_path=f"s3://autogluon-cloud-ci/test-multimodal-image/{framework_version}/{timestamp}",
            local_output_path="test_multimodal_image_cloud_predictor",
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
            predict_kwargs=dict(framework_version=framework_version, custom_image_uri=inference_custom_image_uri),
        )
