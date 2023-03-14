import os
import subprocess
import tempfile
import time

from autogluon.cloud.job.ray_job import RayJob


def test_ray_job():
    # Create a local cluster to test ray job
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        try:
            os.environ.pop("RAY_ADDRESS", None)
            dashboard_port = "8266"  # not using the default 8265 to avoid conflicts
            address = f"http://127.0.0.1:{dashboard_port}"
            subprocess.run(["ray", "start", "--head", "--dashboard-port", dashboard_port], check=True)
            job = RayJob(address=address)
            job.run(entry_point="echo hi", runtime_env=None, wait=True)
            info = job.info()
            assert info["status"] == "SUCCEEDED"
            job2 = RayJob(address=address)
            job2_name = "dummy"
            job2.run(entry_point="echo hi", runtime_env=None, wait=False, job_name=job2_name)
            job3 = RayJob.attach(job_name=job2_name, address=address)
            info = job3.info()
            assert info["status"] == "SUCCEEDED"
            timeout_job = RayJob(address=address)
            start = time.time()
            timeout_job.run(entry_point="sleep 60", runtime_env=None, wait=True, timeout=2)
            end = time.time()
            assert end - start < 60
        finally:
            subprocess.run(["ray", "stop"])
