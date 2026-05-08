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

        import threading
        import time
        import boto3

        # Create a safety timer that will terminate any instances from this test after 5 minutes
        def safety_termination():
            time.sleep(300)  # 5 minutes for testing
            try:
                print("Safety termination triggered - checking for running instances...")
                ec2 = boto3.client("ec2", region_name="us-east-1")

                # Find instances that might be from this test run
                response = ec2.describe_instances(
                    Filters=[
                        {"Name": "instance-state-name", "Values": ["running", "pending"]},
                        {"Name": "tag:ray-cluster-name", "Values": ["*ag_ray_aws_default*"]},
                    ]
                )

                instance_ids = []
                for reservation in response["Reservations"]:
                    for instance in reservation["Instances"]:
                        # Only terminate recent instances (launched within last 10 minutes)
                        launch_time = instance["LaunchTime"].timestamp()
                        current_time = time.time()
                        if (current_time - launch_time) < 600:  # 10 minutes
                            instance_ids.append(instance["InstanceId"])

                if instance_ids:
                    print(f"Safety termination: Found {len(instance_ids)} instances to terminate: {instance_ids}")
                    ec2.terminate_instances(InstanceIds=instance_ids)
                    print("Safety termination: Instances terminated successfully")
                else:
                    print("Safety termination: No matching instances found")

            except Exception as e:
                print(f"Safety termination failed: {e}")

        # Start the safety timer in background
        safety_thread = threading.Thread(target=safety_termination, daemon=True)
        safety_thread.start()
        print("Started safety termination timer (5 minutes)")

        try:
            cp.fit(
                predictor_init_args=predictor_init_args,
                predictor_fit_args=predictor_fit_args,
                custom_image_uri=image_uri,
                framework_version=framework_version,
                backend_kwargs={
                    "initialization_commands": [
                        "aws ecr get-login-password --region us-east-1 | docker login --username AWS "
                        "--password-stdin 763104351884.dkr.ecr.us-east-1.amazonaws.com",
                        "aws ecr get-login-password --region us-east-1 | docker login --username AWS "
                        "--password-stdin 369469875935.dkr.ecr.us-east-1.amazonaws.com",
                        # Auto-terminate after 20 minutes as safety for CI
                        "echo '#!/bin/bash' > /tmp/auto_terminate.sh",
                        "echo 'exec > >(tee -a /tmp/auto_terminate.log) 2>&1' >> /tmp/auto_terminate.sh",  # Log everything
                        "echo 'echo \"[$(date)] Auto-termination script started. Will terminate in 5 minutes.\"' >> /tmp/auto_terminate.sh",
                        "echo 'echo \"[$(date)] Process ID: $$\"' >> /tmp/auto_terminate.sh",
                        "echo 'sleep 300' >> /tmp/auto_terminate.sh",  # 5 minutes for testing
                        "echo 'echo \"[$(date)] 5 minutes elapsed. Initiating termination...\"' >> /tmp/auto_terminate.sh",
                        'echo \'TOKEN=$(curl -X PUT "http://169.254.169.254/latest/api/token" '
                        '-H "X-aws-ec2-metadata-token-ttl-seconds: 21600" 2>/dev/null || echo "failed")\' >> /tmp/auto_terminate.sh',
                        'echo \'if [ "$TOKEN" = "failed" ]; then\' >> /tmp/auto_terminate.sh',
                        "echo '  echo \"[$(date)] Failed to get metadata token, trying without token...\"' >> /tmp/auto_terminate.sh",
                        "echo '  INSTANCE_ID=$(curl -s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo \"unknown\")' >> /tmp/auto_terminate.sh",
                        "echo 'else' >> /tmp/auto_terminate.sh",
                        'echo \'  INSTANCE_ID=$(curl -H "X-aws-ec2-metadata-token: $TOKEN" '
                        '-s http://169.254.169.254/latest/meta-data/instance-id 2>/dev/null || echo "unknown")\' >> /tmp/auto_terminate.sh',
                        "echo 'fi' >> /tmp/auto_terminate.sh",
                        "echo 'echo \"[$(date)] Instance ID: \\$INSTANCE_ID\"' >> /tmp/auto_terminate.sh",
                        "echo 'echo \"[$(date)] Instance ID: \\$INSTANCE_ID\"' >> /tmp/auto_terminate.sh",
                        'echo \'if [ "\\$INSTANCE_ID" != "unknown" ] && [ -n "\\$INSTANCE_ID" ]; then\' >> /tmp/auto_terminate.sh',
                        "echo 'echo \"[$(date)] Checking IAM permissions...\"' >> /tmp/auto_terminate.sh",
                        "echo 'aws sts get-caller-identity 2>&1' >> /tmp/auto_terminate.sh",
                        "echo 'echo \"[$(date)] Attempting termination...\"' >> /tmp/auto_terminate.sh",
                        "echo 'aws ec2 terminate-instances --instance-ids \\$INSTANCE_ID "
                        "--region us-east-1 2>&1' >> /tmp/auto_terminate.sh",
                        "echo 'echo \"[$(date)] Termination command sent.\"' >> /tmp/auto_terminate.sh",
                        "echo 'else' >> /tmp/auto_terminate.sh",
                        "echo 'echo \"[$(date)] ERROR: Could not determine instance ID, cannot terminate\"' >> /tmp/auto_terminate.sh",
                        "echo 'fi' >> /tmp/auto_terminate.sh",
                        "chmod +x /tmp/auto_terminate.sh",
                        # Create a systemd-style service for better persistence
                        "echo 'Starting auto-termination script...'",
                        "setsid /tmp/auto_terminate.sh &",  # Use setsid instead of nohup for better process isolation
                        "echo 'Auto-termination script PID:' $!",
                        "sleep 2",  # Give script time to start
                        "ps aux | grep auto_terminate | grep -v grep || echo 'Warning: auto_terminate script not found in process list'",
                        "ls -la /tmp/auto_terminate.*",
                        "echo 'Check /tmp/auto_terminate.log for status'",
                    ]
                },
            )

            model_artifact_path = cp.get_fit_job_output_path()
            bucket, prefix = s3_path_to_bucket_prefix(model_artifact_path)
            s3_client = boto3.client("s3")
            s3_client.head_object(Bucket=bucket, Key=prefix)
        except Exception as e:
            # In case of any failure, try to cleanup the Ray cluster
            try:
                if hasattr(cp.backend, "_cluster_manager") and cp.backend._cluster_manager is not None:
                    cp.backend._cluster_manager.down()
            except Exception as cleanup_error:
                print(f"Warning: Failed to cleanup cluster after test failure: {cleanup_error}")
            raise e
