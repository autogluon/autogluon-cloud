"""Pending-prediction handle for asynchronous and job-backed inference."""

from __future__ import annotations

import io
from abc import ABC, abstractmethod
from io import BytesIO
from typing import TYPE_CHECKING, Callable, Literal, Optional

import pandas as pd
from sagemaker.async_inference.async_inference_response import AsyncInferenceResponse
from sagemaker.async_inference.waiter_config import WaiterConfig

if TYPE_CHECKING:
    from ..job.sagemaker_job import SageMakerFitJob

PredictionStatus = Literal["InProgress", "Completed", "Failed"]


class PredictionFuture(ABC):
    """A pending prediction whose result becomes available later."""

    @property
    @abstractmethod
    def output_path(self) -> str: ...

    @abstractmethod
    def status(self) -> PredictionStatus: ...

    @abstractmethod
    def result(self, timeout: Optional[float] = None) -> pd.DataFrame: ...


class AsyncPredictionFuture(PredictionFuture):
    """Pending result from a SageMaker async endpoint invocation."""

    def __init__(self, response: AsyncInferenceResponse, accept: str) -> None:
        self._response = response
        self._accept = accept

    @property
    def output_path(self) -> str:
        return self._response.output_path

    @property
    def failure_path(self) -> Optional[str]:
        return self._response.failure_path

    def status(self) -> PredictionStatus:
        import boto3
        from botocore.exceptions import ClientError

        s3 = boto3.client("s3")

        def exists(url: Optional[str]) -> bool:
            if url is None:
                return False
            bucket, key = _parse_s3(url)
            try:
                s3.head_object(Bucket=bucket, Key=key)
                return True
            except ClientError as e:
                if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
                    return False
                raise

        if exists(self.output_path):
            return "Completed"
        if exists(self.failure_path):
            return "Failed"
        return "InProgress"

    def result(self, timeout: Optional[float] = None) -> pd.DataFrame:
        waiter = None if timeout is None else WaiterConfig(max_attempts=max(1, int(timeout // 5)), delay=5)
        return _deserialize(self._response.get_result(waiter_config=waiter), self._accept)


class JobPredictionFuture(PredictionFuture):
    """Pending result from a SageMaker job (e.g. ``predict(wait=False)``)."""

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

    def result(self, timeout: Optional[float] = None) -> pd.DataFrame:
        from sagemaker.estimator import Estimator

        if not self._job.completed:
            Estimator.attach(self._job.job_name, sagemaker_session=self._job.session).logs()
        if self.status() == "Failed":
            raise RuntimeError(
                f"Prediction job {self._job.job_name!r} did not complete successfully "
                f"(status={self._job.get_job_status()!r}). Check the SageMaker console for details."
            )
        return self._result_loader()


def _parse_s3(url: str) -> tuple[str, str]:
    assert url.startswith("s3://"), f"Not an S3 URL: {url}"
    bucket, _, key = url[len("s3://") :].partition("/")
    return bucket, key


def _deserialize(raw: object, accept: str) -> pd.DataFrame:
    if isinstance(raw, pd.DataFrame):
        return raw
    if isinstance(raw, (bytes, bytearray)):
        buf: object = BytesIO(raw)
    elif hasattr(raw, "read"):
        buf = raw
    else:
        raise TypeError(f"Cannot deserialize async response of type {type(raw).__name__}")
    if accept == "application/x-parquet":
        return pd.read_parquet(buf)
    if accept == "text/csv":
        if isinstance(buf, BytesIO):
            buf = io.StringIO(buf.read().decode("utf-8"))
        return pd.read_csv(buf)
    raise ValueError(f"Unsupported accept={accept!r}; use 'application/x-parquet' or 'text/csv'.")
