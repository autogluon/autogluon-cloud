import os
import re
import tarfile
from unittest import mock

import boto3
import pytest
from botocore.exceptions import ClientError
from moto import mock_aws

from autogluon.cloud.utils.ag_sagemaker import repack_model_with_serving_code
from autogluon.cloud.utils.aws_utils import AwsSession, get_execution_role
from autogluon.cloud.utils.misc import sagemaker_timestamp, unique_name_from_base

BUCKET = "test-bucket"


@pytest.fixture
def session(monkeypatch):
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "testing")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "testing")
    with mock_aws():
        boto_session = boto3.Session(region_name="us-east-1")
        boto_session.client("s3").create_bucket(Bucket=BUCKET)
        yield AwsSession(boto_session)


def _keys(session):
    return sorted(obj["Key"] for obj in session.s3_client.list_objects_v2(Bucket=BUCKET).get("Contents", []))


def _write(path, content="x"):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        f.write(content)


def test_upload_file_returns_object_uri(session, tmp_path):
    _write(tmp_path / "data.csv")
    uri = session.upload_data(str(tmp_path / "data.csv"), BUCKET, "run/utils")
    assert uri == f"s3://{BUCKET}/run/utils/data.csv"
    assert _keys(session) == ["run/utils/data.csv"]


def test_upload_directory_keeps_structure_and_returns_prefix_uri(session, tmp_path):
    _write(tmp_path / "code" / "serve.py")
    _write(tmp_path / "code" / "serving_utils" / "a.py")
    uri = session.upload_data(str(tmp_path / "code"), BUCKET, "run/serving")
    assert uri == f"s3://{BUCKET}/run/serving"
    assert _keys(session) == ["run/serving/serve.py", "run/serving/serving_utils/a.py"]


def test_download_prefix_skips_siblings_sharing_the_string_prefix(session, tmp_path):
    for key in ["results/a.out", "results/sub/b.out", "results-other/c.out"]:
        session.s3_client.put_object(Bucket=BUCKET, Key=key, Body=b"x")
    downloaded = session.download_data(str(tmp_path), BUCKET, "results")
    assert sorted(os.path.relpath(p, tmp_path) for p in downloaded) == ["a.out", os.path.join("sub", "b.out")]


def test_download_single_object_uses_its_file_name(session, tmp_path):
    session.s3_client.put_object(Bucket=BUCKET, Key="bt/results/test.csv.out", Body=b"pred")
    downloaded = session.download_data(str(tmp_path), BUCKET, "bt/results/test.csv.out")
    assert downloaded == [os.path.join(os.path.realpath(tmp_path), "test.csv.out")]


def test_repack_replaces_code_dir_and_keeps_model_files(session, tmp_path):
    model_dir = tmp_path / "model"
    _write(model_dir / "predictor.pkl", "weights")
    _write(model_dir / "code" / "stale.py")
    with tarfile.open(tmp_path / "model.tar.gz", "w:gz") as tar:
        for name in os.listdir(model_dir):
            tar.add(model_dir / name, arcname=name)
    session.s3_client.upload_file(str(tmp_path / "model.tar.gz"), BUCKET, "fit/model.tar.gz")
    entry_point = tmp_path / "my_serve.py"
    _write(entry_point)

    uri = repack_model_with_serving_code(
        model_data=f"s3://{BUCKET}/fit/model.tar.gz",
        entry_point=str(entry_point),
        repacked_model_uri=f"s3://{BUCKET}/endpoints/ep/model/model.tar.gz",
        sagemaker_session=session,
    )

    assert uri == f"s3://{BUCKET}/endpoints/ep/model/model.tar.gz"
    session.s3_client.download_file(BUCKET, "endpoints/ep/model/model.tar.gz", str(tmp_path / "repacked.tar.gz"))
    with tarfile.open(tmp_path / "repacked.tar.gz") as tar:
        names = set(tar.getnames())
    assert "predictor.pkl" in names
    assert "code/my_serve.py" in names
    assert any(name.startswith("code/serving_utils/") for name in names)
    assert "code/stale.py" not in names


def _session_with_caller(arn, get_role_result=None):
    clients = {"sts": mock.MagicMock(), "iam": mock.MagicMock()}
    clients["sts"].get_caller_identity.return_value = {"Arn": arn}
    if isinstance(get_role_result, Exception):
        clients["iam"].get_role.side_effect = get_role_result
    else:
        clients["iam"].get_role.return_value = {"Role": {"Arn": get_role_result}}
    session = mock.MagicMock()
    session.boto_session.client.side_effect = lambda name, **kwargs: clients[name]
    return session, clients


def test_execution_role_resolves_path_through_iam():
    session, clients = _session_with_caller(
        "arn:aws:sts::123456789012:assumed-role/MyRole/botocore-session-1",
        get_role_result="arn:aws:iam::123456789012:role/team/MyRole",
    )
    assert get_execution_role(session) == "arn:aws:iam::123456789012:role/team/MyRole"
    clients["iam"].get_role.assert_called_once_with(RoleName="MyRole")


@pytest.mark.parametrize(
    "role_name, expected_path",
    [("MyRole", ""), ("AmazonSageMaker-ExecutionRole-20240101T000000", "service-role/")],
)
def test_execution_role_without_iam_access_falls_back_to_sts_arn(role_name, expected_path):
    denied = ClientError({"Error": {"Code": "AccessDenied", "Message": "denied"}}, "GetRole")
    session, _ = _session_with_caller(f"arn:aws:sts::123456789012:assumed-role/{role_name}/SageMaker", denied)
    assert get_execution_role(session) == f"arn:aws:iam::123456789012:role/{expected_path}{role_name}"


def test_execution_role_rejects_iam_users():
    session, _ = _session_with_caller("arn:aws:iam::123456789012:user/alice")
    with pytest.raises(ValueError, match="role=<arn>"):
        get_execution_role(session)


def test_resource_names():
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}-\d{2}-\d{2}-\d{2}-\d{3}", sagemaker_timestamp())
    name = unique_name_from_base("a" * 100)
    assert len(name) == 63
    assert re.fullmatch(r"a+-\d+-[0-9a-f]{4}", name)
    assert unique_name_from_base("ag") != unique_name_from_base("ag")
    assert re.fullmatch(r"ag-cloud-toto-2-0-4m-\d+-[0-9a-f]{4}", unique_name_from_base("ag-cloud-toto-2.0-4m"))
