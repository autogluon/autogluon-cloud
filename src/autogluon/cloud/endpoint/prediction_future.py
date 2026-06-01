"""Pending-prediction handle for asynchronous and job-backed inference.

A single public class — :class:`PredictionFuture` — wraps a result that becomes
available later. The internal strategy differs by source:

* Async endpoint invocations poll S3 (via :class:`AsyncInferenceResponse`).
* ``predict(wait=False)`` polls the underlying SageMaker job.

Callers should only need ``status()``, ``result()``, and ``output_path``.
"""

from __future__ import annotations

import io
import logging
from io import BytesIO
from typing import TYPE_CHECKING, Callable, Literal, Optional, Protocol

import pandas as pd
from sagemaker.async_inference.async_inference_response import AsyncInferenceResponse
from sagemaker.async_inference.waiter_config import WaiterConfig

if TYPE_CHECKING:
    from ..job.sagemaker_job import SageMakerFitJob

logger = logging.getLogger(__name__)

PredictionStatus = Literal["InProgress", "Completed", "Failed"]


class _Backing(Protocol):
    output_path: str

    def status(self) -> PredictionStatus: ...

    def result(self, timeout: Optional[float]) -> pd.DataFrame: ...


class PredictionFuture:
    """A pending prediction whose result becomes available later.

    Returned by both :meth:`Endpoint.predict_async` (async-endpoint inference) and
    by ``predict(wait=False)`` (job-backed inference). Both paths expose the same
    surface, so callers can write code generic over either source.

    Attributes
    ----------
    output_path
        S3 URL where the result will be written. Useful for retrieving results
        out-of-band (e.g. via the AWS console or another tool).

    Methods
    -------
    status()
        Non-blocking check of the current state.
    result(timeout=None)
        Blocks until the result is available and returns it as a DataFrame.
    """

    def __init__(self, backing: _Backing) -> None:
        self._backing = backing

    @property
    def output_path(self) -> str:
        return self._backing.output_path

    def status(self) -> PredictionStatus:
        return self._backing.status()

    def result(self, timeout: Optional[float] = None) -> pd.DataFrame:
        return self._backing.result(timeout)

    @classmethod
    def _from_async_response(cls, response: AsyncInferenceResponse, accept: str) -> "PredictionFuture":
        return cls(_AsyncBacking(response=response, accept=accept))

    @classmethod
    def _from_job(
        cls,
        job: "SageMakerFitJob",
        result_loader: Callable[[], pd.DataFrame],
    ) -> "PredictionFuture":
        return cls(_JobBacking(job=job, result_loader=result_loader))


class _AsyncBacking:
    """Backing for async-endpoint inference. Polls S3 for the response object."""

    def __init__(self, response: AsyncInferenceResponse, accept: str) -> None:
        self._response = response
        self._accept = accept
        self.output_path: str = response.output_path

    def status(self) -> PredictionStatus:
        from botocore.exceptions import ClientError

        s3 = self._response.predictor_async.s3_client
        bucket, key = _parse_s3(self.output_path)
        try:
            s3.head_object(Bucket=bucket, Key=key)
            return "Completed"
        except ClientError as e:
            if e.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                raise

        if self._response.failure_path is not None:
            f_bucket, f_key = _parse_s3(self._response.failure_path)
            try:
                s3.head_object(Bucket=f_bucket, Key=f_key)
                return "Failed"
            except ClientError as e:
                if e.response["Error"]["Code"] not in ("404", "NoSuchKey"):
                    raise

        return "InProgress"

    def result(self, timeout: Optional[float]) -> pd.DataFrame:
        waiter_config = _build_waiter_config(timeout)
        raw = self._response.get_result(waiter_config=waiter_config)
        return _deserialize(raw, self._accept)


class _JobBacking:
    """Backing for job-backed inference (e.g. ``predict(wait=False)`` running a SageMaker job)."""

    def __init__(self, job: "SageMakerFitJob", result_loader: Callable[[], pd.DataFrame]) -> None:
        self._job = job
        self._result_loader = result_loader
        self.output_path: str = job.get_output_path() or ""

    def status(self) -> PredictionStatus:
        raw = self._job.get_job_status()
        if raw == "Completed":
            return "Completed"
        if raw in ("Failed", "Stopped"):
            return "Failed"
        return "InProgress"

    def result(self, timeout: Optional[float]) -> pd.DataFrame:
        from sagemaker.estimator import Estimator

        if not self._job.completed:
            estimator = Estimator.attach(self._job.job_name, sagemaker_session=self._job.session)
            estimator.logs()
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


def _build_waiter_config(timeout: Optional[float]) -> Optional[WaiterConfig]:
    if timeout is None:
        return None
    delay = 5
    max_attempts = max(1, int(timeout // delay))
    return WaiterConfig(max_attempts=max_attempts, delay=delay)


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
    raise ValueError(
        f"Unsupported accept type for async deserialization: {accept!r}. "
        f"Expected 'application/x-parquet' or 'text/csv'."
    )
