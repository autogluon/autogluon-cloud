"""Pending-prediction handle for job-backed inference (e.g. ``predict(wait=False)``)."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:
    from ..job.sagemaker_job import SageMakerFitJob

PredictionStatus = Literal["InProgress", "Completed", "Failed"]


class JobPredictionFuture:
    """Handle to the pending result of a SageMaker job.

    Returned by the foundation model ``predict()`` / ``predict_proba()`` methods when called with ``wait=False``. Poll
    it with :meth:`status` and get the predictions with :meth:`result`.
    """

    def __init__(self, job: SageMakerFitJob, result_loader: Callable[[], Any]) -> None:
        self._job = job
        self._result_loader = result_loader

    @property
    def output_path(self) -> str:
        """S3 path of the job's output artifact, or ``""`` if the job hasn't completed."""
        return self._job.get_output_path() or ""

    @property
    def job_name(self) -> str:
        """Name of the SageMaker training job."""
        return self._job.job_name

    def status(self) -> PredictionStatus:
        """
        Get the status of the job.

        Returns
        -------
        str
            ``"InProgress"``, ``"Completed"``, or ``"Failed"`` (also returned for stopped jobs).
        """
        raw = self._job.get_job_status()
        if raw == "Completed":
            return "Completed"
        if raw in ("Failed", "Stopped"):
            return "Failed"
        return "InProgress"

    def result(self) -> Any:
        """
        Wait for the job to complete, streaming its logs, and return the predictions.

        Returns
        -------
        Any
            Same as the return value of the ``predict()`` / ``predict_proba()`` call with ``wait=True``.

        Raises
        ------
        RuntimeError
            If the job failed or was stopped.
        """
        if not self._job.completed:
            self._job.wait(logs=True)
        if self.status() == "Failed":
            raise RuntimeError(
                f"Prediction job {self._job.job_name!r} did not complete successfully "
                f"(status={self._job.get_job_status()!r}). Check the SageMaker console for details."
            )
        return self._result_loader()
