import argparse
import json
import os
import shutil
import yaml
from datetime import datetime, timezone
from typing import Optional

import boto3
from botocore.exceptions import ClientError

from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix
from autogluon.tabular import TabularDataset, TabularPredictor


def get_utc_timestamp_now():
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def upload_file(file_name: str, bucket: str, prefix: Optional[str] = None):
    """
    Upload a file to an S3 bucket

    Parameters
    ----------
    file_name: str,
        File to upload
    bucket: str,
        Bucket to upload to
    prefix: Optional[str], default = None
        S3 prefix. If not specified then will upload to the root of the bucket
    """
    object_name = os.path.basename(file_name)
    if len(prefix) == 0:
        prefix = None
    if prefix is not None:
        object_name = prefix + "/" + object_name

    # Upload the file
    s3_client = boto3.client("s3")
    try:
        s3_client.upload_file(file_name, bucket, object_name)
    except ClientError as e:
        raise e


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ag_args_path", type=str, required=True)
    # parser.add_argument("--predictor_init_args", type=str, required=True)
    # parser.add_argument("--predictor_fit_args", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--tune_data", type=str, required=False)
    parser.add_argument("--model_output_path", type=str, required=True)
    parser.add_argument("--leaderboard", action="store_true")
    parser.set_defaults(leaderboard=False)
    args = parser.parse_args()

    train_data = TabularDataset(args.train_data)
    tune_data = None
    if args.tune_data is not None:
        tune_data = TabularDataset(args.tune_data)
    with open(args.ag_args_path) as f:
        ag_args = yaml.safe_load(f)
    predictor_init_args = ag_args["predictor_init_args"]
    predictor_fit_args = ag_args["predictor_fit_args"]
    # predictor_init_args = json.loads(args.predictor_init_args)
    # predictor_fit_args = json.loads(args.predictor_fit_args)
    save_path = f"ag_distributed_training_{get_utc_timestamp_now()}"
    predictor_init_args["path"] = save_path

    print("Start training TabularPredictor")
    predictor = TabularPredictor(**predictor_init_args).fit(
        train_data=train_data, tuning_data=tune_data, **predictor_fit_args
    )
    print("Compressing the model artifacts")
    model_artifact = shutil.make_archive("model", "zip", save_path)
    cloud_bucket, cloud_prefix = s3_path_to_bucket_prefix(args.model_output_path)
    print(f"Uploading model artifact to {args.model_output_path}")
    upload_file(file_name=model_artifact, bucket=cloud_bucket, prefix=cloud_prefix)

    if args.leaderboard:
        lb = predictor.leaderboard(silent=False)
        leaderboard_file = "leaderboard.csv"
        lb.to_csv(leaderboard_file)
        upload_file(file_name=leaderboard_file, bucket=cloud_bucket, prefix=cloud_prefix)
    print("Training finished")