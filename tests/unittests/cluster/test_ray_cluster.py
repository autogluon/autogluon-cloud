import os
import tempfile
import time

from autogluon.cloud.cluster import RayAWSClusterConfigGenerator, RayAWSClusterManager


def test_ray_aws_cluster(test_helper):
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        config_generator = RayAWSClusterConfigGenerator(
            cluster_name=f"ag-ray-cluster-{test_helper.get_utc_timestamp_now()}"
        )
        config_path = os.path.join(temp_dir, "config.yaml")
        config_generator.save_config(config_path)
        cluster_manager = RayAWSClusterManager(config_path)
        try:
            cluster_manager.up()
            # sleep to give some time for the worker nodes to scale up
            time.sleep(180)
            cluster_manager.setup_connection()
        finally:
            cluster_manager.down()
