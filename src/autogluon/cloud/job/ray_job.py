import asyncio
import logging
import math
import time
from typing import Any, Dict, Optional

from ray.job_submission import JobStatus, JobSubmissionClient

from ..utils.utils import get_utc_timestamp_now
from .remote_job import RemoteJob

logger = logging.getLogger(__name__)


class RayJob(RemoteJob):
    def __init__(self, address: str = "http://127.0.0.1:8265", output_path: Optional[str] = None, **kwargs) -> None:
        """
        Parameters
        ----------
            address: str. Default http://127.0.0.1:8265, which is the default RAY_ADDRESS after connection being setup
                Either (1) the address of the Ray cluster,
                    or (2) the HTTP address of the dashboard server on the head node, e.g. “http://<head-node-ip>:8265”.
                In case (1) it must be specified as an address that can be passed to ray.init(),
                    e.g. a Ray Client address (ray://<head_node_host>:10001), or “auto”, or “localhost:<port>”.
                This argument is always overridden by the RAY_ADDRESS environment variable.
            output_path: Optional[str]. Default None
                Remote output_path to store the job artifacts, if any.
        """
        self.client = None
        self._job_name = None
        self._output_path = output_path
        self._address = address

    @property
    def job_name(self):
        return self._job_name

    @property
    def completed(self):
        if not self.job_name:
            return False
        return self.get_job_status() in ["STOPPED", "SUCCEEDED", "FAILED"]

    @classmethod
    def attach(cls, job_name: str, **kwargs):
        """
        Reattach to a job given its name.

        Parameters:
        -----------
        job_name: str
            Name of the job to be attached.
        """
        obj = cls(**kwargs)
        obj.client = JobSubmissionClient(obj._address)
        obj._wait_until_status(
            job_name=job_name,
            status_to_wait_for={JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED},
            timeout=math.inf,  # TODO: do we add timeout to attach api too?
        )
        logs = obj.client.get_job_logs(job_id=job_name)
        obj._job_name = job_name
        logger.log(20, logs)

        return obj

    def info(self) -> Dict:
        """
        Give general information about the job.

        Returns:
        ------
        dict
            A dictionary containing the general information about the job.
        """
        assert self.job_name is not None, "No job detected. Please submit a job first"
        info = dict(name=self.job_name, status=self.get_job_status(), artifact_path=self.get_output_path())
        return info

    def run(
        self,
        entry_point: str,
        runtime_env: Optional[Dict[str, Any]] = None,
        job_name: Optional[str] = None,
        wait: bool = True,
        timeout: int = 24 * 60 * 60,
        ray_submit_job_args: Optional[Dict[str, Any]] = None,
        **kwargs,
    ):
        """
        Execute the job

        Parameters
        ----------
        entry_point: str
            The shell command to run for this job.
        runtime_env: Dict[str, Any]. Default None
            The runtime environment to install and run this job in.
            To learn more,
                https://docs.ray.io/en/latest/cluster/running-applications/job-submission/sdk.html
        job_name: Optional[str]. Default None
            Name of the job being submitted. If not specified, will create one with pattern `ag-ray-{timestamp}`
        wait: bool. Default True
            Whether to wait
        timeout: int. Default 3600
            Timeout in second. If `wait` is True, will stop the job once timeout is reached.
        ray_submit_job_args : Optional[Dict[str, Any]]. Default None
            Additional args to be passed to ray JobSubmissionClient.submit_job call.
            To learn more,
                https://docs.ray.io/en/latest/cluster/running-applications/job-submission/doc/ray.job_submission.JobSubmissionClient.submit_job.html#ray.job_submission.JobSubmissionClient.submit_job
        """
        if job_name is None:
            job_name = f"ag-ray-{get_utc_timestamp_now()}"
        if ray_submit_job_args is None:
            ray_submit_job_args = {}
        if self.client is None:
            self.client = JobSubmissionClient(self._address)
        self.client.submit_job(
            entrypoint=entry_point, runtime_env=runtime_env, submission_id=job_name, **ray_submit_job_args
        )
        logger.log(20, f"Submitted job {job_name} to the cluster")
        self._job_name = job_name
        if wait:
            self._wait_until_status(
                job_name=job_name,
                status_to_wait_for={JobStatus.SUCCEEDED, JobStatus.STOPPED, JobStatus.FAILED},
                timeout=timeout,
            )

    def get_job_status(self) -> Optional[str]:
        """
        Get job status

        Returns:
        --------
        str:
            Valid Values: PENDING | RUNNING | STOPPED | SUCCEEDED | FAILED | NotCreated
        """
        if not self.job_name:
            return "NotCreated"
        return str(self.client.get_job_status(job_id=self.job_name))

    def get_output_path(self):
        """
        Get the output path of the job generated artifacts if any.
        """
        if self._output_path is not None:
            output_path = (
                self._output_path + "/" + "model.zip"
                if not self._output_path.endswith("/")
                else self._output_path + "model.zip"
            )
            return output_path
        return None

    def _wait_until_status(self, job_name, status_to_wait_for, timeout, log_frequency=10):
        start = time.time()
        finished = False
        asyncio.run(self._stream_log(job_name))
        while time.time() - start <= timeout:
            status = self.client.get_job_status(job_name)
            if status in status_to_wait_for:
                finished = True
                break
            time.sleep(log_frequency)

        if not finished:
            logger.log(20, f"timeout: {timeout} secs reached. Will stop the job")
            self.client.stop_job(job_id=job_name)

    async def _stream_log(self, job_name):
        async for lines in self.client.tail_job_logs(job_name):
            logger.log(20, lines.strip())


class RayFitJob(RayJob):
    pass
