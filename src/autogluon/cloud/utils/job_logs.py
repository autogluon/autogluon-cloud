"""Wait for SageMaker jobs while streaming their CloudWatch logs through the caller's own boto3 session.

Polling uses the clients we pass in, so logs always come from the job's account and region.
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, Optional

from botocore.exceptions import BotoCoreError, ClientError

logger = logging.getLogger(__name__)

TRAINING_JOB_LOG_GROUP = "/aws/sagemaker/TrainingJobs"
TRANSFORM_JOB_LOG_GROUP = "/aws/sagemaker/TransformJobs"
TERMINAL_JOB_STATUSES = ("Completed", "Failed", "Stopped")


class LogTailer:
    """Print new events from every log stream ``<job_name>/...`` in ``log_group``, oldest first."""

    def __init__(self, logs_client: Any, log_group: str, job_name: str) -> None:
        self._client = logs_client
        self._log_group = log_group
        self._prefix = job_name + "/"
        self._next_tokens: Dict[str, Optional[str]] = {}
        self._enabled = True

    def poll(self) -> None:
        """Print all events that arrived since the previous call. Never raises: log access problems disable tailing."""
        if not self._enabled:
            return
        try:
            self._discover_streams()
            for stream_name in sorted(self._next_tokens):
                self._print_new_events(stream_name)
        except ClientError as e:
            if e.response.get("Error", {}).get("Code") == "ResourceNotFoundException":
                return  # The log group / streams appear once the container starts writing output.
            self._disable(e)
        except BotoCoreError as e:
            self._disable(e)

    def _disable(self, error: Exception) -> None:
        logger.warning(f"Unable to read job logs from CloudWatch ({error}). Waiting for the job without logs.")
        self._enabled = False

    def _discover_streams(self) -> None:
        kwargs = {"logGroupName": self._log_group, "logStreamNamePrefix": self._prefix}
        while True:
            response = self._client.describe_log_streams(**kwargs)
            for stream in response.get("logStreams", []):
                self._next_tokens.setdefault(stream["logStreamName"], None)
            if not response.get("nextToken"):
                return
            kwargs["nextToken"] = response["nextToken"]

    def _print_new_events(self, stream_name: str) -> None:
        # Prefix lines with the stream (e.g. instance id) only when there is more than one stream to tell apart.
        label = f"[{stream_name[len(self._prefix) :]}] " if len(self._next_tokens) > 1 else ""
        while True:
            kwargs = {"logGroupName": self._log_group, "logStreamName": stream_name, "startFromHead": True}
            if self._next_tokens[stream_name]:
                kwargs["nextToken"] = self._next_tokens[stream_name]
            response = self._client.get_log_events(**kwargs)
            self._next_tokens[stream_name] = response["nextForwardToken"]
            if not response["events"]:
                return
            for event in response["events"]:
                print(f"{label}{event['message']}")


def wait_for_job(
    get_status: Callable[[], str],
    job_name: str,
    log_group: str,
    logs_client: Optional[Any] = None,
    poll: float = 10,
) -> str:
    """Poll ``get_status`` until the job reaches a terminal state and return that state.

    If ``logs_client`` (a boto3 ``logs`` client) is given, the job's CloudWatch logs are printed while waiting.
    """
    tailer = LogTailer(logs_client, log_group, job_name) if logs_client is not None else None
    while True:
        status = get_status()
        if tailer is not None:
            tailer.poll()
        if status in TERMINAL_JOB_STATUSES:
            if tailer is not None:
                # The last log lines can land in CloudWatch shortly after the status flips.
                time.sleep(poll)
                tailer.poll()
            return status
        time.sleep(poll)
