from unittest import mock

import pytest
from botocore.exceptions import ClientError

from autogluon.cloud.utils.job_logs import LogTailer, wait_for_job


class FakeLogsClient:
    """In-memory CloudWatch Logs client: ``streams`` maps stream name -> list of messages written so far."""

    def __init__(self):
        self.streams = {}

    def describe_log_streams(self, logGroupName, logStreamNamePrefix, nextToken=None):
        if not self.streams:
            raise ClientError({"Error": {"Code": "ResourceNotFoundException"}}, "DescribeLogStreams")
        names = sorted(n for n in self.streams if n.startswith(logStreamNamePrefix))
        return {"logStreams": [{"logStreamName": n} for n in names]}

    def get_log_events(self, logGroupName, logStreamName, startFromHead, nextToken=None):
        start = int(nextToken or 0)
        messages = self.streams[logStreamName][start:]
        return {"events": [{"message": m} for m in messages], "nextForwardToken": str(start + len(messages))}


def test_log_tailer_prints_each_event_once(capsys):
    client = FakeLogsClient()
    tailer = LogTailer(client, "/aws/sagemaker/TrainingJobs", "job")

    tailer.poll()  # no log group yet
    client.streams["job/algo-1"] = ["a", "b"]
    tailer.poll()
    client.streams["job/algo-1"].append("c")
    client.streams["other-job/algo-1"] = ["not mine"]
    tailer.poll()

    assert capsys.readouterr().out.splitlines() == ["a", "b", "c"]


def test_log_tailer_labels_lines_when_there_are_multiple_streams(capsys):
    client = FakeLogsClient()
    client.streams = {"job/i-1": ["x"], "job/i-1/data-log": ["y"]}
    LogTailer(client, "/aws/sagemaker/TransformJobs", "job").poll()
    assert capsys.readouterr().out.splitlines() == ["[i-1] x", "[i-1/data-log] y"]


def test_log_tailer_disables_itself_on_access_errors(capsys):
    client = mock.MagicMock()
    client.describe_log_streams.side_effect = ClientError({"Error": {"Code": "AccessDeniedException"}}, "Describe")
    tailer = LogTailer(client, "/aws/sagemaker/TrainingJobs", "job")
    tailer.poll()
    tailer.poll()
    assert client.describe_log_streams.call_count == 1


@pytest.mark.parametrize("final_status", ["Completed", "Failed", "Stopped"])
def test_wait_for_job_returns_terminal_status_and_drains_logs(final_status, capsys):
    client = FakeLogsClient()
    statuses = iter(["InProgress", "InProgress", final_status])

    def get_status():
        status = next(statuses)
        client.streams.setdefault("job/algo-1", []).append(status)
        return status

    with mock.patch("autogluon.cloud.utils.job_logs.time.sleep") as sleep:
        assert wait_for_job(get_status, job_name="job", log_group="g", logs_client=client, poll=3) == final_status

    assert capsys.readouterr().out.splitlines() == ["InProgress", "InProgress", final_status]
    assert all(call.args == (3,) for call in sleep.call_args_list)


def test_wait_for_job_without_logs_only_polls_status():
    statuses = iter(["InProgress", "Completed"])
    with mock.patch("autogluon.cloud.utils.job_logs.time.sleep") as sleep:
        assert wait_for_job(lambda: next(statuses), job_name="job", log_group="g") == "Completed"
    assert sleep.call_count == 1
