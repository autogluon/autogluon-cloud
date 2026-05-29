import logging
from unittest import mock

import pytest

from autogluon.cloud.config import (
    CONFIG_DIR_ENV,
    BackendConfig,
    CloudConfig,
    save_config,
)
from autogluon.cloud.utils.aws_utils import resolve_cloud_output_path, resolve_execution_role


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    yield tmp_path


def _save_role_in_config(backend_name: str, role_arn: str) -> None:
    save_config(
        CloudConfig(
            backends={
                backend_name: BackendConfig(
                    region="us-east-1",
                    role_arn=role_arn,
                    bucket="ag-cloud-bucket",
                    stack_name="ag-cloud",
                )
            }
        )
    )


def test_explicit_role_wins_over_config_and_env():
    _save_role_in_config("sagemaker", "arn:aws:iam::111111111111:role/from-config")
    explicit = "arn:aws:iam::222222222222:role/explicit"
    with mock.patch("autogluon.cloud.utils.aws_utils.sagemaker.get_execution_role") as mock_env:
        assert resolve_execution_role(explicit, backend_name="sagemaker") == explicit
        mock_env.assert_not_called()


def test_config_role_used_when_no_explicit():
    config_role = "arn:aws:iam::111111111111:role/from-config"
    _save_role_in_config("sagemaker", config_role)
    aws_utils_logger = logging.getLogger("autogluon.cloud.utils.aws_utils")
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.setLevel(logging.INFO)
    handler.emit = records.append
    aws_utils_logger.addHandler(handler)
    aws_utils_logger.setLevel(logging.INFO)
    try:
        with mock.patch("autogluon.cloud.utils.aws_utils.sagemaker.get_execution_role") as mock_env:
            assert resolve_execution_role(None, backend_name="sagemaker") == config_role
            mock_env.assert_not_called()
    finally:
        aws_utils_logger.removeHandler(handler)
    assert any(config_role in r.getMessage() for r in records), (
        f"expected log mentioning {config_role!r}, got {[r.getMessage() for r in records]}"
    )


def test_falls_back_to_env_when_no_config_or_explicit():
    env_role = "arn:aws:iam::333333333333:role/from-env"
    with mock.patch(
        "autogluon.cloud.utils.aws_utils.sagemaker.get_execution_role",
        return_value=env_role,
    ) as mock_env:
        assert resolve_execution_role(None, backend_name="sagemaker") == env_role
        mock_env.assert_called_once()


def test_falls_back_to_env_when_backend_missing_in_config():
    _save_role_in_config("ray_aws", "arn:aws:iam::111111111111:role/ray")
    env_role = "arn:aws:iam::333333333333:role/from-env"
    with mock.patch(
        "autogluon.cloud.utils.aws_utils.sagemaker.get_execution_role",
        return_value=env_role,
    ) as mock_env:
        assert resolve_execution_role(None, backend_name="sagemaker") == env_role
        mock_env.assert_called_once()


def _save_bucket_in_config(backend_name: str, bucket: str) -> None:
    save_config(
        CloudConfig(
            backends={
                backend_name: BackendConfig(
                    region="us-east-1",
                    role_arn="arn:aws:iam::111111111111:role/r",
                    bucket=bucket,
                    stack_name=None,
                )
            }
        )
    )


@pytest.fixture
def no_s3_check(monkeypatch):
    """Stub the S3 emptiness check so tests don't make AWS calls."""
    monkeypatch.setattr(
        "autogluon.cloud.utils.aws_utils._s3_prefix_has_objects",
        lambda bucket, prefix: False,
    )


def test_resolve_path_bucket_only_appends_timestamp(no_s3_check):
    resolved = resolve_cloud_output_path("s3://my-bucket", backend_name="sagemaker")
    assert resolved.startswith("s3://my-bucket/ag-")
    assert resolved.count("/") == 3  # s3://my-bucket/ag-<ts>


def test_resolve_path_bucket_only_no_scheme_appends_timestamp(no_s3_check):
    resolved = resolve_cloud_output_path("my-bucket", backend_name="sagemaker")
    assert resolved.startswith("s3://my-bucket/ag-")


def test_resolve_path_bucket_with_prefix_used_verbatim(no_s3_check):
    resolved = resolve_cloud_output_path("s3://my-bucket/exp-1", backend_name="sagemaker")
    assert resolved == "s3://my-bucket/exp-1"


def test_resolve_path_strips_trailing_slash_then_appends_timestamp(no_s3_check):
    resolved = resolve_cloud_output_path("s3://my-bucket/", backend_name="sagemaker")
    assert resolved.startswith("s3://my-bucket/ag-")


def test_resolve_path_uses_bucket_from_config_when_path_is_none(no_s3_check):
    _save_bucket_in_config("sagemaker", "config-bucket")
    resolved = resolve_cloud_output_path(None, backend_name="sagemaker")
    assert resolved.startswith("s3://config-bucket/ag-")


def test_resolve_path_raises_when_path_none_and_no_config(no_s3_check):
    with pytest.raises(ValueError, match="No `cloud_output_path`"):
        resolve_cloud_output_path(None, backend_name="sagemaker")


def test_resolve_path_raises_when_backend_not_in_config(no_s3_check):
    _save_bucket_in_config("ray_aws", "config-bucket")
    with pytest.raises(ValueError, match="No `cloud_output_path`"):
        resolve_cloud_output_path(None, backend_name="sagemaker")


def _capture_aws_utils_logs(level: int) -> tuple[list[logging.LogRecord], logging.Handler]:
    records: list[logging.LogRecord] = []
    handler = logging.Handler()
    handler.setLevel(level)
    handler.emit = records.append
    aws_logger = logging.getLogger("autogluon.cloud.utils.aws_utils")
    aws_logger.addHandler(handler)
    aws_logger.setLevel(level)
    return records, handler


def test_resolve_path_warns_when_prefix_already_has_objects(monkeypatch):
    monkeypatch.setattr(
        "autogluon.cloud.utils.aws_utils._s3_prefix_has_objects",
        lambda bucket, prefix: True,
    )
    records, handler = _capture_aws_utils_logs(logging.WARNING)
    try:
        resolved = resolve_cloud_output_path("s3://my-bucket/exp-1", backend_name="sagemaker")
    finally:
        logging.getLogger("autogluon.cloud.utils.aws_utils").removeHandler(handler)
    assert resolved == "s3://my-bucket/exp-1"
    assert any("already contains objects" in r.getMessage() for r in records)


def test_resolve_path_no_warning_when_prefix_empty(no_s3_check):
    records, handler = _capture_aws_utils_logs(logging.WARNING)
    try:
        resolve_cloud_output_path("s3://my-bucket/exp-1", backend_name="sagemaker")
    finally:
        logging.getLogger("autogluon.cloud.utils.aws_utils").removeHandler(handler)
    assert not any("already contains objects" in r.getMessage() for r in records)


def test_resolve_path_no_warning_for_bucket_only(monkeypatch):
    # Even if list_objects_v2 returns truthy, bucket-only inputs build a fresh timestamped
    # prefix that did not exist a moment ago, so we should not run the emptiness check on it.
    called = []
    monkeypatch.setattr(
        "autogluon.cloud.utils.aws_utils._s3_prefix_has_objects",
        lambda bucket, prefix: called.append((bucket, prefix)) or True,
    )
    records, handler = _capture_aws_utils_logs(logging.WARNING)
    try:
        resolve_cloud_output_path("s3://my-bucket", backend_name="sagemaker")
    finally:
        logging.getLogger("autogluon.cloud.utils.aws_utils").removeHandler(handler)
    assert called == []
    assert not any("already contains objects" in r.getMessage() for r in records)
