from abc import ABC, abstractmethod


class RemoteJob(ABC):
    @classmethod
    @abstractmethod
    def attach(cls, job_name):
        """
        Reattach to a job given its name.

        Parameters:
        -----------
        job_name: str
            Name of the job to be attached.
        """
        raise NotImplementedError

    @abstractmethod
    def info(self) -> dict:
        """
        Give general information about the job.

        Returns:
        ------
        dict
            A dictionary containing the general information about the job.
        """
        raise NotImplementedError

    @abstractmethod
    def run(self, **kwargs):
        """Execute the job"""
        raise NotImplementedError

    @abstractmethod
    def get_job_status(self) -> str | None:
        """
        Get job status
        """
        raise NotImplementedError

    @abstractmethod
    def get_output_path(self) -> str | None:
        """
        Get the output path of the job generated artifacts if any.
        """
        raise NotImplementedError
