import boto3
import os
import tempfile

import pandas as pd

from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix
from autogluon.cloud import TabularCloudPredictor


def test_distributed_training(test_helper, framework_version):
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        cp = TabularCloudPredictor(
            cloud_output_path="s3://autogluon-cloud-ci/test-tabular-distributed/",
            local_output_path="test_tabular_distributed",
            backend="ray_aws",
        )

        train_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
        predictor_init_args = {"label": "class"}
        predictor_fit_args = {
            "train_data": train_data,
            "hyperparameters": {
                "GBM": {},
            },
            "num_bag_folds": 2,
            "num_bag_sets": 1,
        }

        image_uri = test_helper.get_custom_image_uri(framework_version, type="training", gpu=False)

        cp.fit(
            predictor_init_args=predictor_init_args, predictor_fit_args=predictor_fit_args, custom_image_uri=image_uri
        )
        
        model_artifact_path = cp.get_fit_job_output_path()
        bucket, prefix = s3_path_to_bucket_prefix(model_artifact_path)
        s3_client = boto3.client("s3")
        s3_client.head_object(Bucket=bucket, Key=prefix)
