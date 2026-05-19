import os

import pytest
import yaml

from autogluon.cloud.config import (
    CONFIG_DIR_ENV,
    BackendConfig,
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


def _make_backend_config(**overrides):
    defaults = dict(
        region="us-east-1",
        role_arn="arn:aws:iam::123456789012:role/ag-cloud-execution-role",
        bucket="ag-cloud-bucket-abc",
        stack_name="ag-cloud",
    )
    defaults.update(overrides)
    return BackendConfig(**defaults)


def test_load_config_missing_returns_none():
    assert load_config() is None


def test_save_load_roundtrip():
    config = CloudConfig(backends={"sagemaker": _make_backend_config()})
    save_config(config)

    loaded = load_config()
    assert loaded is not None
    assert "sagemaker" in loaded.backends
    sage = loaded.backends["sagemaker"]
    assert sage.region == "us-east-1"
    assert sage.bucket == "ag-cloud-bucket-abc"


def test_save_supports_multiple_backends():
    config = CloudConfig(
        backends={
            "sagemaker": _make_backend_config(bucket="sage-bucket"),
            "ray_aws": _make_backend_config(bucket="ray-bucket", stack_name="ag-cloud-ray"),
        }
    )
    save_config(config)

    loaded = load_config()
    assert set(loaded.backends) == {"sagemaker", "ray_aws"}
    assert loaded.backends["sagemaker"].bucket == "sage-bucket"
    assert loaded.backends["ray_aws"].bucket == "ray-bucket"


def test_saved_file_is_user_only_readable():
    save_config(CloudConfig(backends={"sagemaker": _make_backend_config()}))
    mode = os.stat(get_config_path()).st_mode & 0o777
    # 0o600 — user read/write, nothing for group/other.
    assert mode == 0o600


def test_save_overwrites_previous():
    save_config(CloudConfig(backends={"sagemaker": _make_backend_config(bucket="b1")}))
    save_config(CloudConfig(backends={"sagemaker": _make_backend_config(bucket="b2")}))
    assert load_config().backends["sagemaker"].bucket == "b2"


def test_delete_config_removes_file():
    save_config(CloudConfig(backends={"sagemaker": _make_backend_config()}))
    assert delete_config() is True
    assert load_config() is None


def test_delete_config_returns_false_when_missing():
    assert delete_config() is False


def test_config_file_is_keyed_by_backend():
    save_config(CloudConfig(backends={"sagemaker": _make_backend_config()}))
    with open(get_config_path()) as f:
        data = yaml.safe_load(f)
    # Top-level keys are backend names; per-backend dict has the resource fields.
    assert list(data.keys()) == ["sagemaker"]
    assert data["sagemaker"]["bucket"] == "ag-cloud-bucket-abc"
    assert data["sagemaker"]["region"] == "us-east-1"


def test_load_config_handles_empty_file():
    """A file that exists but is empty should be treated as missing."""
    path = get_config_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()
    assert load_config() is None
