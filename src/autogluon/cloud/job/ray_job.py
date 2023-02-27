from typing import Dict, Optional

from .remote_job import RemoteJob


class RayJob(RemoteJob):
    def __init__(self) -> None:
        pass

    def attach(cls, job_name):
        """
        Reattach to a job given its name.

        Parameters:
        -----------
        job_name: str
            Name of the job to be attached.
        """
        raise NotImplementedError

    def info(self) -> Dict:
        """
        Give general information about the job.

        Returns:
        ------
        dict
            A dictionary containing the general information about the job.
        """
        raise NotImplementedError

    def run(self, **kwargs):
        """Execute the job"""
        raise NotImplementedError

    def get_job_status(self) -> Optional[str]:
        """
        Get job status
        """
        raise NotImplementedError

    def get_output_path(self):
        """
        Get the output path of the job generated artifacts if any.
        """
        raise NotImplementedError
