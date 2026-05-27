import logging
from unittest import mock

import pytest

from autogluon.cloud.config import (
    CONFIG_DIR_ENV,
    BackendConfig,
    CloudConfig,
    save_config,
)
from autogluon.cloud.utils.aws_utils import resolve_execution_role


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
