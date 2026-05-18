import os

import pytest
import yaml

from autogluon.cloud.config import (
    CONFIG_DIR_ENV,
    CloudConfig,
    delete_config,
    get_config_path,
    load_config,
    save_config,
)


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    yield tmp_path


def _make_config(**overrides):
    defaults = dict(
        region="us-east-1",
        role_arn="arn:aws:iam::123456789012:role/ag-cloud-execution-role",
        bucket="ag-cloud-bucket-abc",
        backend="sagemaker",
        stack_name="ag-cloud",
    )
    defaults.update(overrides)
    return CloudConfig(**defaults)


def test_load_config_missing_returns_none():
    assert load_config() is None


def test_save_load_roundtrip():
    config = _make_config()
    save_config(config)

    loaded = load_config()
    assert loaded is not None
    assert loaded.region == config.region
    assert loaded.role_arn == config.role_arn
    assert loaded.bucket == config.bucket
    assert loaded.backend == config.backend
    assert loaded.stack_name == config.stack_name


def test_saved_file_is_user_only_readable():
    save_config(_make_config())
    mode = os.stat(get_config_path()).st_mode & 0o777
    # 0o600 — user read/write, nothing for group/other.
    assert mode == 0o600


def test_save_overwrites_previous():
    save_config(_make_config(bucket="b1"))
    save_config(_make_config(bucket="b2"))
    assert load_config().bucket == "b2"


def test_delete_config_removes_file():
    save_config(_make_config())
    assert delete_config() is True
    assert load_config() is None


def test_delete_config_returns_false_when_missing():
    assert delete_config() is False


def test_config_file_is_yaml():
    save_config(_make_config())
    with open(get_config_path()) as f:
        data = yaml.safe_load(f)
    assert data["bucket"] == "ag-cloud-bucket-abc"
    assert data["region"] == "us-east-1"


def test_load_config_handles_empty_file(isolated_config_dir):
    """A file that exists but is empty should be treated as missing."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    assert load_config() is None
