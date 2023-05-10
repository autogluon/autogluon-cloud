import argparse
import os
import pickle
import shutil
import time
from datetime import datetime, timezone
from typing import Optional

import boto3
import ray

from autogluon.common.utils.s3_utils import s3_path_to_bucket_prefix
from autogluon.tabular import TabularDataset, TabularPredictor


def wait_for_nodes_to_be_ready():
    expected_num_nodes = int(os.environ.get("AG_NUM_NODES"))
    ray.init("auto")
    print("Waiting for worker nodes to be ready")
    while len(ray.nodes()) < expected_num_nodes:
        time.sleep(5)
    ray.shutdown()
    print("All nodes ready")


def get_utc_timestamp_now():
    return datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S")


def upload_log_file(job_id: str, model_output_path: str):
    from ray.job_submission import JobSubmissionClient

    client = JobSubmissionClient("http://127.0.0.1:8265")
    logs = client.get_job_logs(job_id=job_id)
    log_file = "training.log"
    with open(log_file, "w") as f:
        f.write(logs)
    cloud_bucket, cloud_prefix = s3_path_to_bucket_prefix(model_output_path)
    upload_file(file_name=log_file, bucket=cloud_bucket, prefix=cloud_prefix)


def tear_down_cluster(cluster_config_file: str):
    import subprocess

    print("Will tear down the cluster in 10 secs")
    time.sleep(10)
    cmd = f"ray stop --force; ray down {cluster_config_file} -y"
    subprocess.Popen(cmd, shell=True, preexec_fn=os.setpgrp)  # Avoid being terminated because ray runtime is down


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
    if prefix is not None and len(prefix) == 0:
        prefix = None
    if prefix is not None:
        object_name = prefix + "/" + object_name

    # Upload the file
    s3_client = boto3.client("s3")
    s3_client.upload_file(file_name, bucket, object_name)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ag_args_path", type=str, required=True)
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--tune_data", type=str, required=False)
    parser.add_argument("--model_output_path", type=str, required=True)
    parser.add_argument("--ray_job_id", type=str, required=True)
    parser.add_argument("--leaderboard", action="store_true")
    parser.add_argument("--cluster_config_file", type=str, required=False)
    parser.set_defaults(leaderboard=False)
    args = parser.parse_args()

    try:
        wait_for_nodes_to_be_ready()
        train_data = TabularDataset(args.train_data)
        tune_data = None
        if args.tune_data is not None:
            tune_data = TabularDataset(args.tune_data)
        with open(args.ag_args_path, "rb") as f:
            ag_args = pickle.load(f)
        predictor_init_args = ag_args["predictor_init_args"]
        predictor_fit_args = ag_args["predictor_fit_args"]
        save_path = f"ag_distributed_training_{get_utc_timestamp_now()}"
        predictor_init_args["path"] = save_path

        print("Start training TabularPredictor")
        predictor = TabularPredictor(**predictor_init_args).fit(
            train_data=train_data, tuning_data=tune_data, **predictor_fit_args
        )
        print("Compressing the model artifacts")
        model_artifact = shutil.make_archive("model", "zip", save_path)
        model_output_path = os.path.dirname(args.model_output_path)
        cloud_bucket, cloud_prefix = s3_path_to_bucket_prefix(model_output_path)
        print(f"Uploading model artifact to {model_output_path}")
        upload_file(file_name=model_artifact, bucket=cloud_bucket, prefix=cloud_prefix)

        if args.leaderboard:
            lb = predictor.leaderboard(silent=False)
            leaderboard_file = "leaderboard.csv"
            lb.to_csv(leaderboard_file)
            upload_file(file_name=leaderboard_file, bucket=cloud_bucket, prefix=cloud_prefix)
        print("Training finished")
    except Exception as e:
        raise e
    finally:
        upload_log_file(job_id=args.ray_job_id, model_output_path=os.path.dirname(args.model_output_path))

        cluster_config_file = args.cluster_config_file
        if cluster_config_file is not None:
            tear_down_cluster(cluster_config_file)
