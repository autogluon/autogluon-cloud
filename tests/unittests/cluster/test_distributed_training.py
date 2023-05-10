import os
import tempfile

import boto3
import pandas as pd

from autogluon.cloud import TabularCloudPredictor
from autogluon.common import space
from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix


def test_distributed_training(test_helper, framework_version):
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        timestamp = test_helper.get_utc_timestamp_now()
        cp = TabularCloudPredictor(
            cloud_output_path=f"s3://autogluon-cloud-ci/test-tabular-distributed/{framework_version}/{timestamp}",
            local_output_path="test_tabular_distributed",
            backend="ray_aws",
        )

        train_data = pd.read_csv("https://autogluon.s3.amazonaws.com/datasets/Inc/train.csv")
        subsample_size = 1000
        if subsample_size is not None and subsample_size < len(train_data):
            train_data = train_data.sample(n=subsample_size, random_state=0)
        predictor_init_args = {"label": "class"}
        predictor_fit_args = {
            "train_data": train_data,
            "hyperparameters": {
                "GBM": {"num_leaves": space.Int(lower=26, upper=66, default=36)},
            },
            "num_bag_folds": 2,
            "num_bag_sets": 1,
            "hyperparameter_tune_kwargs": {  # HPO is not performed unless hyperparameter_tune_kwargs is specified
                "num_trials": 2,
                "scheduler": "local",
                "searcher": "auto",
            },
        }

        image_uri = test_helper.get_custom_image_uri(framework_version, type="training", gpu=False)

        cp.fit(
            predictor_init_args=predictor_init_args,
            predictor_fit_args=predictor_fit_args,
            custom_image_uri=image_uri,
            backend_kwargs={
                "initialization_commands": [
                    "aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com",
                    "aws ecr get-login-password --region us-east-1 | docker login --username AWS --password-stdin 369469875935.dkr.ecr.us-east-1.amazonaws.com",
                ]
            },
        )

        model_artifact_path = cp.get_fit_job_output_path()
        bucket, prefix = s3_path_to_bucket_prefix(model_artifact_path)
        s3_client = boto3.client("s3")
        s3_client.head_object(Bucket=bucket, Key=prefix)
