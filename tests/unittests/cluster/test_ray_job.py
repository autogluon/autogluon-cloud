import os
import subprocess
import tempfile

from autogluon.cloud.job.ray_job import RayJob


def test_ray_job():
    # Create a local cluster to test ray job
    with tempfile.TemporaryDirectory() as temp_dir:
        os.chdir(temp_dir)
        try:
            os.environ.pop("RAY_ADDRESS", None)
            dashboard_port = "8266"  # not using the default 8265 to avoid conflicts
            address = f"http://127.0.0.1:{dashboard_port}"
            result = subprocess.run(
                ["ray", "start", "--head", "--dashboard-port", dashboard_port], capture_output=True, check=True
            )
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
        finally:
            subprocess.run(["ray", "stop"])
