import subprocess

# import pytest

from autogluon.cloud.job.ray_job import RayJob


# @pytest.mark.local_only
def test_ray_job():
    # Create a local cluster to test ray job
    try:
        result = subprocess.run(["ray", "start", "--head"], capture_output=True, check=True)
        print(result.stdout)
        job = RayJob()
        job.run(entry_point="echo hi", runtime_env=None, wait=True)
        info = job.info()
        assert info["status"] == "SUCCEEDED"
        job2 = RayJob()
        job2_name = "dummy"
        job2.run(entry_point="echo hi", runtime_env=None, wait=False, job_name=job2_name)
        job3 = RayJob.attach(job_name=job2_name)
        info = job3.info()
        assert info["status"] == "SUCCEEDED"
    finally:
        subprocess.run(["ray", "stop"])
