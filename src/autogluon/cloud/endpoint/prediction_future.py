"""Pending-prediction handle for job-backed inference (e.g. ``predict(wait=False)``)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Literal

import pandas as pd
from sagemaker.estimator import Estimator

if TYPE_CHECKING:
    from ..job.sagemaker_job import SageMakerFitJob

PredictionStatus = Literal["InProgress", "Completed", "Failed"]


class JobPredictionFuture:
    """Pending result from a SageMaker job (e.g. ``predict(wait=False)``).

    Wraps the underlying job and exposes a small future-like surface: ``output_path``,
    ``status()``, and ``result()``.
    """

    def __init__(self, job: "SageMakerFitJob", result_loader: Callable[[], pd.DataFrame]) -> None:
        self._job = job
        self._result_loader = result_loader

    @property
    def output_path(self) -> str:
        return self._job.get_output_path() or ""

    @property
    def job_name(self) -> str:
        return self._job.job_name

    def status(self) -> PredictionStatus:
        raw = self._job.get_job_status()
        if raw == "Completed":
            return "Completed"
        if raw in ("Failed", "Stopped"):
            return "Failed"
        return "InProgress"

    def result(self) -> pd.DataFrame:
        if not self._job.completed:
            Estimator.attach(self._job.job_name, sagemaker_session=self._job.session).logs()
        if self.status() == "Failed":
            raise RuntimeError(
                f"Prediction job {self._job.job_name!r} did not complete successfully "
                f"(status={self._job.get_job_status()!r}). Check the SageMaker console for details."
            )
        return self._result_loader()
