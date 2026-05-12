import os

import pytest
import yaml

from autogluon.cloud.config import (
    CONFIG_DIR_ENV,
    CloudConfig,
    Profile,
    delete_profile,
    get_config_path,
    load_config,
    save_config,
    upsert_profile,
)


@pytest.fixture(autouse=True)
def isolated_config_dir(tmp_path, monkeypatch):
    monkeypatch.setenv(CONFIG_DIR_ENV, str(tmp_path))
    yield tmp_path


def _make_profile(**overrides):
    defaults = dict(
        region="us-east-1",
        role_arn="arn:aws:iam::123456789012:role/ag-cloud-execution-role",
        bucket="ag-cloud-bucket-abc",
        backend="sagemaker",
        stack_name="ag-cloud",
    )
    defaults.update(overrides)
    return Profile(**defaults)


def test_load_config_missing_returns_none():
    assert load_config() is None


def test_save_load_roundtrip():
    profile = _make_profile()
    config = CloudConfig(active_profile="default", profiles={"default": profile})
    save_config(config)

    loaded = load_config()
    assert loaded is not None
    assert loaded.active_profile == "default"
    assert loaded.profiles["default"].bucket == profile.bucket
    assert loaded.profiles["default"].role_arn == profile.role_arn


def test_saved_file_is_user_only_readable(isolated_config_dir):
    save_config(CloudConfig(profiles={"default": _make_profile()}))
    path = get_config_path()
    mode = os.stat(path).st_mode & 0o777
    # 0o600 — user read/write, nothing for group/other.
    assert mode == 0o600


def test_upsert_profile_sets_active_for_first_profile():
    upsert_profile("prod", _make_profile())
    config = load_config()
    assert config.active_profile == "prod"


def test_upsert_profile_can_switch_active():
    upsert_profile("dev", _make_profile())
    upsert_profile("prod", _make_profile(), set_active=True)
    config = load_config()
    assert config.active_profile == "prod"
    assert set(config.profiles) == {"dev", "prod"}


def test_delete_profile_falls_back_to_first_remaining():
    upsert_profile("dev", _make_profile())
    upsert_profile("prod", _make_profile(), set_active=True)
    assert delete_profile("prod") is True
    config = load_config()
    assert "prod" not in config.profiles
    assert config.active_profile == "dev"


def test_delete_profile_returns_false_when_missing():
    assert delete_profile("nope") is False


def test_get_profile_respects_env_override(monkeypatch):
    upsert_profile("default", _make_profile(region="us-east-1"))
    upsert_profile("other", _make_profile(region="us-west-2"), set_active=False)
    monkeypatch.setenv("AUTOGLUON_CLOUD_PROFILE", "other")
    config = load_config()
    assert config.get_profile().region == "us-west-2"


def test_config_file_is_yaml(isolated_config_dir):
    upsert_profile("default", _make_profile())
    with open(get_config_path()) as f:
        data = yaml.safe_load(f)
    assert data["version"] == 1
    assert "default" in data["profiles"]
    assert data["profiles"]["default"]["bucket"] == "ag-cloud-bucket-abc"
